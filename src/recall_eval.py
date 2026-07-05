"""Recall/completeness sweep on the REAL held-out val split (934 tiles).

Every prior eval optimized IoU, which balances FP against FN — it never reported
recall on its own, so "how many roads do we miss" was unknown. This measures, per
mask-extraction config:

  strict  precision / recall / IoU   (exact pixels)
  buffered correctness / completeness (3 px tolerance — the standard road-extraction
                                       pair; completeness ~= "fraction of GT road mapped".
                                       Width errors are free here, which matches the
                                       pipeline: P2 skeletonizes the mask anyway.)

Configs: plain threshold grid + the hysteresis mode run_pipeline.py actually ships.

Run:  python src/recall_eval.py            (route env, ~10 min on the 3050)
"""
import os, sys, argparse, json
import numpy as np, cv2, torch, yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import augment, dataset as DS, model as M
from train import split_pairs
from ensemble_eval import tta_prob
from predict import hysteresis_mask

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.475, 0.50]
HYSTERESIS = [(0.45, 0.23), (0.40, 0.20), (0.475, 0.25)]   # (hi, lo); first = pipeline default
BUF = 3


def eval_config(masks_fn, probs, gts, ker):
    tp = fp = fn = btp = bfp = bfn = 0.0
    for p, g in zip(probs, gts):
        m = masks_fn(p.astype(np.float32))
        tp += np.logical_and(m, g).sum(); fp += np.logical_and(m, 1 - g).sum()
        fn += np.logical_and(1 - m, g).sum()
        gd = cv2.dilate(g, ker); pd = cv2.dilate(m, ker)
        btp += np.logical_and(m, gd).sum()
        bfp += np.logical_and(m, np.logical_not(gd)).sum()
        bfn += np.logical_and(g, np.logical_not(pd)).sum()
    e = 1e-6
    return dict(precision=tp / (tp + fp + e), recall=tp / (tp + fn + e),
                iou=tp / (tp + fp + fn + e),
                correctness=btp / (btp + bfp + e),          # buffered precision
                completeness=btp / (btp + bfn + e))         # buffered recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="C:/Users/VISWAS/route_data/runs/phase1_full/best.pt")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--out", default="runs/phase1_full")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ck.get("cfg", cfg)["model"]
    channels = mcfg.get("channels", augment.RGB)
    sats, masks = DS.list_pairs(cfg["data"]["train_dir"])
    _, (va_s, va_m) = split_pairs(sats, masks, cfg["data"]["train_split"], cfg["seed"], None)
    ds = DS.DeepGlobeRoads(va_s, va_m, augment.val_tf(cfg["data"]["crop_size"], channels),
                           cfg["data"]["road_thresh"])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg["data"]["num_workers"],
                    pin_memory=True)
    net = M.build_model(mcfg["arch"], mcfg["encoder"], None, len(channels), mcfg["classes"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()

    probs, gts = [], []
    print(f"computing 8-view D4 TTA probs on {len(va_s)} held-out val tiles...")
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            p = tta_prob(net, x.to(device)).cpu().numpy()[:, 0]
            yb = y.numpy().astype(np.uint8)[:, 0]
            for b in range(x.size(0)):
                probs.append(p[b].astype(np.float16)); gts.append(yb[b])
            if i % 20 == 0:
                print(f"  batch {i}/{len(dl)}")

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BUF + 1, 2 * BUF + 1))
    rows = {}
    for t in THRESHOLDS:
        rows[f"thr {t:.3f}"] = eval_config(lambda p, t=t: (p > t).astype(np.uint8), probs, gts, ker)
    for hi, lo in HYSTERESIS:
        rows[f"hyst {hi:.3f}/{lo:.2f}"] = eval_config(
            lambda p, hi=hi, lo=lo: hysteresis_mask(p, None, thr_hi=hi, thr_lo=lo), probs, gts, ker)

    hdr = f"{'config':>16} | {'prec':>6} {'recall':>6} {'IoU':>6} | {'corr3':>6} {'compl3':>6}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:>16} | {r['precision']:.4f} {r['recall']:.4f} {r['iou']:.4f} "
              f"| {r['correctness']:.4f} {r['completeness']:.4f}")

    out = os.path.join(args.out, "recall_eval.json")
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(n_val=len(gts), buffer_px=BUF, rows=rows), open(out, "w"), indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
