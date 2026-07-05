"""Grid-search the Phase-3 healing parameters against GROUND TRUTH — not the (gameable)
raw connectivity metric.

Why GT-scored: connectivity_report only counts components merged / LCC growth, so "maximize
connectivity" degenerates to max_gap=inf, ang_tol=90 — connecting everything into one blob of
HALLUCINATED bridges. Instead we score every synthetic bridge against the real GT road mask:

  - a bridge whose straight span lies over (dilated) GT road  = TRUE  positive  (a real
    occluded gap the healer correctly reconnected)
  - a bridge over GT non-road                                 = FALSE positive  (invented road)

Objective: maximize TRUE bridges recovered at precision >= --min-precision. That rewards
recovering genuinely-occluded roads while punishing the over-healing the gates exist to prevent.

Loop engineering: model inference is the expensive part, so we predict each tile ONCE
(cached to an .npz), then sweep every param combo doing only the cheap CPU re-healing.

Usage:
  python src/heal_gridsearch.py --n 24 --device cuda --out runs/heal_grid
"""
import os, sys, json, argparse, itertools, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
import yaml
import predict
import dataset as DS
from train import split_pairs
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph
from phase3_heal import heal_graph, connectivity_report, _span_mean


def cache_predictions(ckpt, device, val_sats, val_masks, thr, cache_path, dilate_px=5):
    """Predict each val tile once; store rgb, soft prob, predicted graph inputs + GT mask.
    Returns list of per-tile dicts: dict(img, prob, mask, gt, gt_dil)."""
    if os.path.exists(cache_path):
        print(f"loading cached predictions: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return list(d["tiles"])
    net = predict.load_net(ckpt, device)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    tiles = []
    for i, (sp, mp) in enumerate(zip(val_sats, val_masks)):
        img = cv2.cvtColor(cv2.imread(sp), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mask, prob = predict.predict_full(net, img, device, tile=512, overlap=64, thr=thr, tta=True)
        tiles.append(dict(img=img, prob=prob.astype(np.float32), mask=mask.astype(np.uint8),
                          gt=gt, gt_dil=cv2.dilate(gt, ker)))
        print(f"  [{i+1}/{len(val_sats)}] predicted {os.path.basename(sp)}")
    np.savez_compressed(cache_path, tiles=np.array(tiles, dtype=object))
    return tiles


def score_healing(G, H, gt_dil, road_frac=0.5):
    """For each synthetic (healed) edge, is its straight span over real GT road?
    Returns (true_bridges, false_bridges)."""
    tp = fp = 0
    for u, v, d in H.edges(data=True):
        if not d.get("healed"):
            continue
        p1 = H.nodes[u]["pos"]; p2 = H.nodes[v]["pos"]
        cover = _span_mean(p1, p2, gt_dil.astype(np.float32))   # fraction of span over GT road
        if cover >= road_frac:
            tp += 1
        else:
            fp += 1
    return tp, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="number of held-out val tiles to score on")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--ckpt", default="C:/Users/VISWAS/route_data/runs/phase1_full/best.pt")
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min-precision", type=float, default=0.70,
                    help="reject param combos whose bridge precision falls below this")
    ap.add_argument("--road-frac", type=float, default=0.5,
                    help="a bridge counts as TRUE if >= this fraction of its span is over GT road")
    ap.add_argument("--out", default="runs/heal_grid")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
    sats, masks = DS.list_pairs(cfg["data"]["train_dir"])
    _, (va_s, va_m) = split_pairs(sats, masks, cfg["data"]["train_split"], cfg["seed"], None)
    va_s, va_m = va_s[:args.n], va_m[:args.n]
    print(f"Scoring healing on {len(va_s)} REAL held-out val tiles | min-precision {args.min_precision}")

    cache = os.path.join(args.out, f"pred_cache_n{args.n}.npz")
    tiles = cache_predictions(args.ckpt, args.device, va_s, va_m, args.thr, cache)

    # build the predicted graph once per tile (healing-independent)
    base = []
    for t in tiles:
        G = skeleton_to_graph(mask_to_skeleton(t["mask"]))
        base.append((G, t["img"], t["prob"], t["gt_dil"]))

    # ---- the grid: the three knobs that actually move healing ----
    grid = dict(
        max_gap_px=[40, 50, 60, 80, 100, 120],
        angular_tolerance_deg=[25, 35, 45, 55],
        prob_road_floor=[0.15, 0.20, 0.25],
    )
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"grid: {len(combos)} combos x {len(base)} tiles = {len(combos)*len(base)} heals")

    rows = []
    t0 = time.time()
    for ci, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        TP = FP = merged = 0
        for G, img, prob, gt_dil in base:
            H, _ = heal_graph(G, max_gap_px=p["max_gap_px"],
                              angular_tolerance_deg=p["angular_tolerance_deg"],
                              rgb=img, prob=prob, prob_road_floor=p["prob_road_floor"])
            tp, fp = score_healing(G, H, gt_dil, args.road_frac)
            TP += tp; FP += fp
            merged += connectivity_report(G, H)["components_merged"]
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rows.append(dict(**p, true_bridges=TP, false_bridges=FP, precision=round(prec, 4),
                         components_merged=merged))
        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(combos)} combos ({time.time()-t0:.0f}s)")

    # ---- pick the winner: most TRUE bridges among combos meeting the precision floor ----
    valid = [r for r in rows if r["precision"] >= args.min_precision]
    best = max(valid or rows, key=lambda r: (r["true_bridges"], r["precision"]))
    # the "naive max connectivity" combo, to show the trap it would have picked
    naive = max(rows, key=lambda r: r["components_merged"])

    rows.sort(key=lambda r: (-r["true_bridges"], -r["precision"]))
    result = dict(n_tiles=len(base), min_precision=args.min_precision, road_frac=args.road_frac,
                  current_defaults=dict(max_gap_px=50, angular_tolerance_deg=35, prob_road_floor=0.20),
                  best=best, naive_max_connectivity=naive, all_rows=rows)
    json.dump(result, open(os.path.join(args.out, "heal_gridsearch.json"), "w"), indent=2)

    print("\n================= HEALING GRID SEARCH (GT-scored) =================")
    print(f"  BEST (precision>={args.min_precision}): {best}")
    print(f"  naive max-connectivity would pick      : {naive}")
    print(f"  -> the naive pick recovers {naive['true_bridges']} real vs {naive['false_bridges']} FAKE "
          f"bridges (precision {naive['precision']}) — the over-healing trap.")
    print("==================================================================")
    print(f"saved {os.path.join(args.out, 'heal_gridsearch.json')}")


def _self_check():
    """span scorer: a bridge fully over road scores 1.0, over blank scores 0.0."""
    import networkx as nx
    road = np.zeros((50, 50), np.float32); road[25, :] = 1.0          # horizontal road row 25
    G = nx.Graph(); H = nx.Graph()
    H.add_node(0, pos=(5, 25)); H.add_node(1, pos=(45, 25))
    H.add_edge(0, 1, healed=True)
    tp, fp = score_healing(G, H, (road > 0).astype(np.uint8), road_frac=0.5)
    assert tp == 1 and fp == 0, (tp, fp)
    H2 = nx.Graph(); H2.add_node(0, pos=(5, 5)); H2.add_node(1, pos=(45, 5))
    H2.add_edge(0, 1, healed=True)
    tp, fp = score_healing(G, H2, (road > 0).astype(np.uint8), road_frac=0.5)
    assert tp == 0 and fp == 1, (tp, fp)
    print("self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
