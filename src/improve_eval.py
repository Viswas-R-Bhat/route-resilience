"""Quick-win evaluation (NO retraining) on the real held-out val split.

Measures how much three free levers improve the trained model:
  1) threshold tuning      (sweep decision threshold, pick best val IoU)
  2) test-time augmentation (average predictions over flips)
  3) post-processing       (remove tiny blobs + close small gaps)

Reports a baseline-vs-improved table and saves a side-by-side visual.
"""
import os, sys, argparse, json
import numpy as np, cv2, torch, yaml
from torch.utils.data import DataLoader
from torch.amp import autocast
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_closing

sys.path.insert(0, os.path.dirname(__file__))
import augment, dataset as DS, model as M
from train import split_pairs

GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def tta_prob(net, x):
    """Average sigmoid prob over {identity, hflip, vflip, rot180}."""
    views = [(lambda t: t, lambda t: t),
             (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
             (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
             (lambda t: torch.flip(t, [2, 3]), lambda t: torch.flip(t, [2, 3]))]
    acc = 0
    for fwd, inv in views:
        with autocast("cuda"):
            p = torch.sigmoid(net(fwd(x)))
        acc = acc + inv(p).float()
    return (acc / len(views))


def postproc(binmask, min_size=80, close=3):
    b = binmask.astype(bool)
    if close:
        b = binary_closing(b, structure=np.ones((close, close)))
    b = remove_small_objects(b, min_size=min_size)
    return b.astype(np.uint8)


def agg_iou_dice(probs, gts, thr, post=False):
    tp = fp = fn = 0.0
    for p, g in zip(probs, gts):
        m = (p > thr).astype(np.uint8)
        if post:
            m = postproc(m)
        tp += np.logical_and(m, g).sum()
        fp += np.logical_and(m, 1 - g).sum()
        fn += np.logical_and(1 - m, g).sum()
    eps = 1e-6
    return tp / (tp + fp + fn + eps), 2 * tp / (2 * tp + fp + fn + eps)


def agg_relaxed(probs, gts, thr, post=False, buf=3):
    k = 2 * buf + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    rtp = rfp = rfn = 0.0
    for p, g in zip(probs, gts):
        m = (p > thr).astype(np.uint8)
        if post:
            m = postproc(m)
        gd = cv2.dilate(g, ker); pd = cv2.dilate(m, ker)
        rtp += np.logical_and(m, gd).sum()
        rfp += np.logical_and(m, np.logical_not(gd)).sum()
        rfn += np.logical_and(g, np.logical_not(pd)).sum()
    eps = 1e-6
    return rtp / (rtp + rfp + rfn + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="C:/Users/VISWAS/route_data/runs/phase1_full/best.pt")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--out", default="runs/phase1_full")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ck.get("cfg", cfg)["model"]                 # eval with the CKPT's own arch/channels
    channels = mcfg.get("channels", augment.RGB)
    print(f"  ckpt channels ({len(channels)}): {channels}")

    sats, masks = DS.list_pairs(cfg["data"]["train_dir"])
    _, (va_s, va_m) = split_pairs(sats, masks, cfg["data"]["train_split"], cfg["seed"], None)
    crop = cfg["data"]["crop_size"]
    ds = DS.DeepGlobeRoads(va_s, va_m, augment.val_tf(crop, channels), cfg["data"]["road_thresh"])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    net = M.build_model(mcfg["arch"], mcfg["encoder"], None,
                        len(channels), mcfg["classes"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()

    probs_tta, gts = [], []
    base_tp = base_fp = base_fn = 0.0          # baseline: identity prob @ thr 0.5
    vis = []
    print(f"Evaluating {len(va_s)} held-out val tiles (TTA x4)...")
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            with autocast("cuda"):
                p_id = torch.sigmoid(net(x)).float()
            p_tta = tta_prob(net, x)
            yb = y.numpy().astype(np.uint8)[:, 0]
            mid = (p_id.cpu().numpy()[:, 0] > 0.5).astype(np.uint8)
            base_tp += np.logical_and(mid, yb).sum()
            base_fp += np.logical_and(mid, 1 - yb).sum()
            base_fn += np.logical_and(1 - mid, yb).sum()
            for b in range(x.size(0)):
                probs_tta.append(p_tta.cpu().numpy()[b, 0].astype(np.float16))
                gts.append(yb[b])
                if len(vis) < 6:
                    vis.append((x[b].cpu().numpy(), yb[b], p_id.cpu().numpy()[b, 0], p_tta.cpu().numpy()[b, 0]))

    eps = 1e-6
    base_iou = base_tp / (base_tp + base_fp + base_fn + eps)
    base_dice = 2 * base_tp / (2 * base_tp + base_fp + base_fn + eps)
    base_rel = agg_relaxed(probs_tta, gts, 0.5)  # relaxed at 0.5 (approx baseline)

    # threshold sweep on TTA probs
    sweep = [(t,) + agg_iou_dice(probs_tta, gts, t) for t in GRID]
    best_t, best_iou, best_dice = max(sweep, key=lambda r: r[1])
    # + post-processing at best threshold
    post_iou, post_dice = agg_iou_dice(probs_tta, gts, best_t, post=True)
    post_rel = agg_relaxed(probs_tta, gts, best_t, post=True)

    print("\nthreshold sweep (TTA, no postproc):")
    for t, i, d in sweep:
        print(f"  thr {t:.2f}: IoU {i:.4f}  Dice {d:.4f}" + ("   <-- best" if t == best_t else ""))
    print("\n================ QUICK-WIN COMPARISON (held-out val) ================")
    print(f"  baseline      (thr0.50, no TTA, no post): IoU {base_iou:.4f}  Dice {base_dice:.4f}  RelaxedIoU {base_rel:.4f}")
    print(f"  + TTA + thr{best_t:.2f}                  : IoU {best_iou:.4f}  Dice {best_dice:.4f}")
    print(f"  + TTA + thr{best_t:.2f} + postproc        : IoU {post_iou:.4f}  Dice {post_dice:.4f}  RelaxedIoU {post_rel:.4f}")
    print(f"  IoU gain: {(post_iou-base_iou)*100:+.2f} pts | RelaxedIoU gain: {(post_rel-base_rel)*100:+.2f} pts")
    print("=====================================================================")

    json.dump(dict(baseline=dict(iou=base_iou, dice=base_dice, relaxed=base_rel),
                   best_threshold=best_t,
                   tta_thr=dict(iou=best_iou, dice=best_dice),
                   tta_thr_post=dict(iou=post_iou, dice=post_dice, relaxed=post_rel)),
              open(os.path.join(args.out, "quickwin_eval.json"), "w"), indent=2)

    # visual: satellite | GT | baseline(0.5) | improved(TTA+thr+post)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mean = np.array(augment.IMAGENET_MEAN); std = np.array(augment.IMAGENET_STD)
    n = len(vis)
    fig, ax = plt.subplots(n, 4, figsize=(13, 3 * n))
    cols = ["satellite (real)", "ground truth", "baseline (thr 0.50)", f"improved (TTA+thr{best_t:.2f}+post)"]
    for r, (xi, g, pid, ptt) in enumerate(vis):
        img = (xi[:3].transpose(1, 2, 0) * std + mean).clip(0, 1)   # first 3 ch = RGB preview
        base = (pid > 0.5).astype(np.uint8)
        imp = postproc((ptt > best_t).astype(np.uint8))
        for c, im, cmap in [(0, img, None), (1, g, "gray"), (2, base, "gray"), (3, imp, "gray")]:
            ax[r, c].imshow(im, cmap=cmap); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "quickwin_compare.png"), dpi=110); plt.close()
    print("saved", os.path.join(args.out, "quickwin_compare.png"))


if __name__ == "__main__":
    main()
