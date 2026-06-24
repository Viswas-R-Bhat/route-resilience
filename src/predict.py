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
from augment import IMAGENET_MEAN, IMAGENET_STD


def _hann2d(n):
    w = np.hanning(n)
    win = np.outer(w, w)
    return (win / win.max()).astype(np.float32)


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
def predict_full(net, image_rgb, device, tile=512, overlap=64, thr=0.475, tta=True):
    H, W = image_rgb.shape[:2]
    step = tile - overlap
    mean = np.array(IMAGENET_MEAN, np.float32); std = np.array(IMAGENET_STD, np.float32)
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
            t = ((patch.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            t = torch.from_numpy(np.ascontiguousarray(t)).unsqueeze(0).to(device)
            p = _d4_prob(net, t, device) if tta else _fwd_prob(net, t, device)
            prob[y:y + ph, x:x + pw] += (p * win)[:ph, :pw]
            wsum[y:y + ph, x:x + pw] += win[:ph, :pw]
    prob /= np.maximum(wsum, 1e-6)
    mask = (prob > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, prob


def load_net(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    net = M.build_model(cfg["model"]["arch"], cfg["model"]["encoder"], None,
                        cfg["model"]["in_channels"], cfg["model"]["classes"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()
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
