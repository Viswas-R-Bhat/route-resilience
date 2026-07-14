"""Classified gallery + per-tile IoU distribution over the REAL held-out val set.

For each of the 934 held-out tiles: TTA prediction, per-tile IoU, GT road fraction.
Tiles are classified into quality buckets so we can SEE where the model wins/fails
before deciding whether a heavyweight retrain is worth it.
Outputs: pertile_distribution.png, gallery_classified.png, gallery_stats.json
"""
import os, sys, json
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from torch.amp import autocast
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import augment, dataset as DS, model as M
from train import split_pairs
from improve_eval import tta_prob

CKPT = "models/best.pt"
OUT = "runs/phase1_full"
EMPTY_PX = 64          # GT road px below this -> "no-road" tile
K = 3                  # examples shown per bucket
BUCKETS = [("Excellent (>=0.70)", 0.70, 1.01),
           ("Good (0.55-0.70)", 0.55, 0.70),
           ("Fair (0.40-0.55)", 0.40, 0.55),
           ("Poor (<0.40)", -0.01, 0.40)]


def per_tile_iou(pred, gt, eps=1e-6):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, 1 - gt).sum()
    fn = np.logical_and(1 - pred, gt).sum()
    return tp / (tp + fp + fn + eps)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
    sats, masks = DS.list_pairs(cfg["data"]["train_dir"])
    _, (va_s, va_m) = split_pairs(sats, masks, cfg["data"]["train_split"], cfg["seed"], None)
    crop = cfg["data"]["crop_size"]
    ds = DS.DeepGlobeRoads(va_s, va_m, augment.val_tf(crop), cfg["data"]["road_thresh"])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    net = M.build_model(cfg["model"]["arch"], cfg["model"]["encoder"], None,
                        cfg["model"]["in_channels"], cfg["model"]["classes"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()

    mean = np.array(augment.IMAGENET_MEAN); std = np.array(augment.IMAGENET_STD)
    road_ious = []                       # per-tile IoU for road-containing tiles
    samples = {b[0]: [] for b in BUCKETS}; samples["No-road"] = []
    n_noroad = 0; noroad_correct = 0; noroad_fp = []
    print(f"Scoring {len(va_s)} held-out tiles (TTA x4)...")
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            prob = tta_prob(net, x).cpu().numpy()[:, 0]
            gt = y.numpy().astype(np.uint8)[:, 0]
            for b in range(x.shape[0]):
                g = gt[b]; p = (prob[b] > 0.5).astype(np.uint8)
                img = (x[b].cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
                if g.sum() < EMPTY_PX:                      # no-road tile
                    n_noroad += 1
                    fp_frac = p.mean()
                    noroad_fp.append(fp_frac)
                    if fp_frac < 0.005:                     # predicted essentially empty
                        noroad_correct += 1
                    if len(samples["No-road"]) < K:
                        samples["No-road"].append((img, g, p, fp_frac, "FP%%=%.2f" % (fp_frac * 100)))
                    continue
                iou = per_tile_iou(p, g)
                road_ious.append(iou)
                for name, lo, hi in BUCKETS:
                    if lo <= iou < hi and len(samples[name]) < K:
                        samples[name].append((img, g, p, iou, "IoU=%.2f" % iou))
                        break

    road_ious = np.array(road_ious)
    counts = {name: int(((road_ious >= lo) & (road_ious < hi)).sum()) for name, lo, hi in BUCKETS}
    nroad = len(road_ious)
    stats = dict(
        total_tiles=len(va_s), road_tiles=nroad, noroad_tiles=n_noroad,
        noroad_correct_pct=100 * noroad_correct / max(1, n_noroad),
        mean_pertile_iou=float(road_ious.mean()), median_pertile_iou=float(np.median(road_ious)),
        buckets={name: dict(count=counts[name], pct=100 * counts[name] / max(1, nroad)) for name, _, _ in BUCKETS},
    )
    print("\n================ CLASSIFIED VAL BREAKDOWN (934 real held-out tiles) ================")
    print(f"  road-containing tiles : {nroad}   |   no-road tiles : {n_noroad}")
    print(f"  no-road tiles predicted (near-)empty correctly : {stats['noroad_correct_pct']:.1f}%")
    print(f"  per-tile IoU (road tiles): mean {stats['mean_pertile_iou']:.3f} | median {stats['median_pertile_iou']:.3f}")
    for name, _, _ in BUCKETS:
        print(f"    {name:<20}: {counts[name]:4d} tiles  ({stats['buckets'][name]['pct']:5.1f}%)")
    print("====================================================================================")
    json.dump(stats, open(os.path.join(OUT, "gallery_stats.json"), "w"), indent=2)

    # ---- distribution figure ----
    fig, axd = plt.subplots(1, 2, figsize=(13, 4.2))
    axd[0].hist(road_ious, bins=20, range=(0, 1), color="#2e6fe0", edgecolor="white")
    axd[0].axvline(road_ious.mean(), color="#f26522", ls="--", lw=2, label=f"mean {road_ious.mean():.2f}")
    axd[0].set_title("Per-tile IoU distribution (road tiles)"); axd[0].set_xlabel("IoU"); axd[0].set_ylabel("tiles"); axd[0].legend()
    names = [b[0].split(" (")[0] for b in BUCKETS] + ["No-road"]
    vals = [counts[b[0]] for b in BUCKETS] + [n_noroad]
    cols = ["#12a594", "#2e6fe0", "#e0941a", "#d63b3b", "#7a5c9e"]
    axd[1].bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        axd[1].text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    axd[1].set_title("Tile classification counts"); axd[1].tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pertile_distribution.png"), dpi=120); plt.close()

    # ---- classified gallery ----
    order = [b[0] for b in BUCKETS] + ["No-road"]
    rows = [(cat, s) for cat in order for s in samples[cat]]
    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(10, 2.7 * n))
    if n == 1: ax = ax[None, :]
    for r, (cat, (img, g, p, score, label)) in enumerate(rows):
        ax[r, 0].imshow(img); ax[r, 0].axis("off")
        ax[r, 0].set_ylabel(cat, fontsize=10)
        ax[r, 0].text(-0.05, 0.5, f"{cat}\n{label}", transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=9, rotation=0)
        ax[r, 1].imshow(g, cmap="gray"); ax[r, 1].axis("off")
        ax[r, 2].imshow(p, cmap="gray"); ax[r, 2].axis("off")
        if r == 0:
            ax[r, 0].set_title("satellite (real)"); ax[r, 1].set_title("ground truth"); ax[r, 2].set_title("prediction")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gallery_classified.png"), dpi=110, bbox_inches="tight"); plt.close()
    print("saved pertile_distribution.png + gallery_classified.png to", OUT)


if __name__ == "__main__":
    main()
