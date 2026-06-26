"""Full-tile inference — produce an occlusion-robust road mask from a REAL image.

Runs the trained model over a large image with overlapping 512 tiles and Hann-window
blending (so tile seams don't show), thresholds, and applies light morphological closing.
The output binary mask is the artifact Phase 2 (skeletonization) consumes.

Usage:
  python src/predict.py --ckpt runs/phase1/best.pt --image path/to/xxx_sat.jpg --out runs/phase1/pred
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import yaml
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from augment import IMAGENET_MEAN, IMAGENET_STD, build_channels, channel_norm, RGB
from phase3_heal import canopy_evidence
from scipy.ndimage import binary_propagation


def _hann2d(n):
    w = np.hanning(n)
    win = np.outer(w, w)
    return (win / win.max()).astype(np.float32)


def hysteresis_mask(prob, image_rgb=None, thr_hi=0.45, thr_lo=None, canopy_lo=None):
    """Double-threshold road extraction for MAXIMUM connected coverage without noise.

    Keep confident road 'seeds' (prob >= thr_hi), then grow into faint pixels (prob >= thr_lo)
    ONLY where they connect to a seed — this recovers occluded road continuations (which fade
    to weak probability under canopy/shadow) while dropping isolated low-confidence blobs.
    Under the adaptive canopy mask the low threshold drops further (canopy_lo), since roads
    beneath trees are the faintest signal of all. Falls back to a single threshold if asked.
    """
    hi = thr_hi
    lo = thr_lo if thr_lo is not None else max(0.15, thr_hi - 0.22)
    seed = prob >= hi
    cand = prob >= lo
    ev = canopy_evidence(image_rgb) if image_rgb is not None else None
    if ev is not None:
        clo = canopy_lo if canopy_lo is not None else max(0.08, lo - 0.08)
        cand = cand | ((prob >= clo) & ev["dense"])          # reach deeper under canopy
    return binary_propagation(seed, mask=cand).astype(np.uint8)


def _fwd_prob(net, t, device):
    """Single forward -> sigmoid prob map (H, W) numpy."""
    if device == "cuda":
        with autocast("cuda"):
            return torch.sigmoid(net(t))[0, 0].float().cpu().numpy()
    return torch.sigmoid(net(t))[0, 0].float().cpu().numpy()


def _d4_prob(net, t, device):
    """8-way D4 test-time augmentation: mean sigmoid prob over rot{0,90,180,270} x {no-flip, h-flip},
    each mapped back to input space. Matches the shipped inference recipe."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            x = torch.rot90(t, k, [2, 3])
            if flip:
                x = torch.flip(x, [3])
            if device == "cuda":
                with autocast("cuda"):
                    p = torch.sigmoid(net(x))
            else:
                p = torch.sigmoid(net(x))
            if flip:
                p = torch.flip(p, [3])
            p = torch.rot90(p, (4 - k) % 4, [2, 3])
            acc = p if acc is None else acc + p
    return (acc / 8.0)[0, 0].float().cpu().numpy()


@torch.no_grad()
def predict_full(net, image_rgb, device, tile=512, overlap=64, thr=0.475, tta=True,
                 hysteresis=True, thr_lo=None, channels=None):
    H, W = image_rgb.shape[:2]
    step = tile - overlap
    ch = channels or getattr(net, "_channels", RGB)          # channel spec recorded on the net
    cm, cs = channel_norm(ch)
    mean = np.array(cm, np.float32); std = np.array(cs, np.float32)
    prob = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = _hann2d(tile) + 1e-3
    ys = list(range(0, max(1, H - tile + 1), step)) + ([H - tile] if H > tile else [0])
    xs = list(range(0, max(1, W - tile + 1), step)) + ([W - tile] if W > tile else [0])
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            patch = image_rgb[y:y + tile, x:x + tile]
            ph, pw = patch.shape[:2]
            if (ph, pw) != (tile, tile):
                patch = cv2.copyMakeBorder(patch, 0, tile - ph, 0, tile - pw, cv2.BORDER_REFLECT_101)
            patch = build_channels(patch, ch)                # RGB (+ derived/NIR) -> HxWxC uint8
            t = ((patch.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            t = torch.from_numpy(np.ascontiguousarray(t)).unsqueeze(0).to(device)
            p = _d4_prob(net, t, device) if tta else _fwd_prob(net, t, device)
            prob[y:y + ph, x:x + pw] += (p * win)[:ph, :pw]
            wsum[y:y + ph, x:x + pw] += win[:ph, :pw]
    prob /= np.maximum(wsum, 1e-6)
    if hysteresis:
        mask = hysteresis_mask(prob, image_rgb, thr_hi=thr, thr_lo=thr_lo)
    else:
        mask = (prob > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, prob


@torch.no_grad()
def predict_multiscale(net, image_rgb, device, scales=(0.6, 0.8, 1.0, 1.3, 1.6),
                       tile=512, overlap=64, thr=0.40, tta=True, fuse="max",
                       thr_lo=None, channels=None):
    """Run inference at several image scales and fuse the probability maps to DISCOVER roads
    the single-scale model misses.

    A road whose width sits outside the receptive field at 1x (narrow residential streets,
    or wide arterials) is often clearly detected once the image is up- or down-scaled. We run
    each scale, resize its prob back to native resolution, and combine:
        fuse="max"  -> recall-first: a road seen at ANY scale survives (best for "find more roads"),
        fuse="mean" -> balanced: trades a little recall for precision.
    Then ONE hysteresis threshold + closing yields the mask. Returns (mask, prob_fused).

    Roads fully under tree canopy stay invisible at every scale (an RGB limit, not a scale issue);
    this recovers the OPEN-ground misses, which is where most of the recall gap lives.
    """
    H, W = image_rgb.shape[:2]
    probs = []
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            im = image_rgb
        else:
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            im = cv2.resize(image_rgb, (max(tile, int(round(W * s))), max(tile, int(round(H * s)))),
                            interpolation=interp)
        _, p = predict_full(net, im, device, tile=tile, overlap=overlap, thr=thr,
                            tta=tta, hysteresis=False, channels=channels)
        if p.shape != (H, W):
            p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
        probs.append(p)
    prob = np.maximum.reduce(probs) if fuse == "max" else np.mean(probs, axis=0)
    mask = hysteresis_mask(prob, image_rgb, thr_hi=thr, thr_lo=thr_lo)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, prob.astype(np.float32)


def load_net(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    channels = cfg["model"].get("channels", RGB)             # default RGB for legacy checkpoints
    net = M.build_model(cfg["model"]["arch"], cfg["model"]["encoder"], None,
                        cfg["model"].get("in_channels", len(channels)), cfg["model"]["classes"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()
    net._channels = channels                                 # so predict_full builds the right bands
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1/best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="runs/phase1/pred")
    ap.add_argument("--thr", type=float, default=0.475)   # IoU-optimal on real held-out val
    ap.add_argument("--no-tta", action="store_true", help="disable 8-way D4 TTA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = load_net(args.ckpt, device)
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    mask, prob = predict_full(net, img, device, thr=args.thr, tta=not args.no_tta)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_pred.png"), mask * 255)
    overlay = img.copy(); overlay[mask > 0] = (0.4 * overlay[mask > 0] + 0.6 * np.array([255, 60, 60])).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved {stem}_pred.png + {stem}_overlay.png to {args.out}/  (road px: {int(mask.sum())})")


if __name__ == "__main__":
    main()
