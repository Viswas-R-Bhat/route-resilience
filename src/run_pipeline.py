"""End-to-end Route Resilience pipeline: REAL image -> mask -> graph -> healed -> criticality/resilience.

Chains Phase 1 (segmentation) -> Phase 2 (skeleton+graph) -> Phase 3 (healing) ->
Phase 4 (centrality + node-ablation Resilience Index). Saves: <stem>_mask.png,
<stem>_graph.gpickle, <stem>_report.json, <stem>_pipeline.png (composite).

Usage:
  python src/run_pipeline.py --image <sat.jpg> --ckpt <best.pt> --device cpu --out runs/pipeline
  python src/run_pipeline.py --mask <mask.png> --out runs/pipeline      # skip segmentation
(Default device=cpu so it won't disturb a GPU training run.)
"""
import os, sys, json, pickle, argparse
import numpy as np, cv2, torch, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import predict
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph, graph_stats
from phase3_heal import heal_graph, connectivity_report
from phase4_analysis import (compute_centrality, classify_nodes, ablation_simulation,
                             sample_node_elevations, flood_simulation, demand_weighted_criticality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--mask", help="use an existing binary mask and skip Phase 1")
    ap.add_argument("--ckpt", default="C:/Users/VISWAS/route_data/runs/phase1_full/best.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="runs/pipeline")
    ap.add_argument("--thr", type=float, default=0.45)    # hysteresis SEED (confident) threshold; faint roads grow from it
    ap.add_argument("--no-tta", action="store_true", help="disable 8-way D4 TTA")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-gap", type=float, default=50)
    ap.add_argument("--ang-tol", type=float, default=35)
    ap.add_argument("--dem", help="DEM .npy registered to the tile -> flood-resilience overlay")
    ap.add_argument("--osm", action="store_true", help="benchmark the graph vs OpenStreetMap (needs <stem>_geo.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- Phase 1: segmentation (or load given mask) ----
    if args.mask:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB) if args.image else None
        mask = (cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        prob = None                                       # no soft map when a pre-thresholded mask is given
        stem = os.path.splitext(os.path.basename(args.mask))[0]
        print(f"[1/4] using provided mask ({mask.shape}, {mask.mean()*100:.1f}% road)")
    else:
        assert args.image, "need --image or --mask"
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(args.image))[0].replace("_sat", "")
        net = predict.load_net(args.ckpt, args.device)
        print(f"[1/4] segmenting on {args.device} (D4 TTA={not args.no_tta}, thr={args.thr}) ...")
        mask, prob = predict.predict_full(net, img, args.device, tile=512, overlap=64,
                                          thr=args.thr, tta=not args.no_tta)
    cv2.imwrite(os.path.join(args.out, f"{stem}_mask.png"), mask * 255)
    if img is not None:
        cv2.imwrite(os.path.join(args.out, f"{stem}_sat.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ---- Phase 2: skeleton -> graph ----
    G = skeleton_to_graph(mask_to_skeleton(mask))
    raw_stats = graph_stats(G)
    print(f"[2/4] graph: {raw_stats['nodes']} nodes / {raw_stats['edges']} edges / {raw_stats['components']} components")

    # ---- Phase 3: healing (canopy- AND soft-probability-aware when the RGB tile / soft map is available) ----
    H, healed = heal_graph(G, args.max_gap, args.ang_tol, rgb=img, prob=prob)
    conn = connectivity_report(G, H)
    n_prob = sum(d.get("heal_kind") == "prob" for _, _, d in H.edges(data=True))
    n_canopy = sum(d.get("heal_kind") == "canopy" for _, _, d in H.edges(data=True))
    n_geom = sum(d.get("heal_kind") == "geom" for _, _, d in H.edges(data=True))
    n_tjct = sum(d.get("heal_via") == "tjunction" for _, _, d in H.edges(data=True))   # mid-edge T-junctions
    scene_green = H.graph.get("canopy_frac_scene", 0.0)
    saturated = H.graph.get("canopy_saturated", False)
    print(f"[3/4] healed: +{healed} bridges ({n_prob} faint-road / {n_canopy} under canopy / {n_geom} open; "
          f"{n_tjct} into T-junctions) | "
          f"scene canopy {scene_green*100:.0f}%{' [SATURATED-conservative]' if saturated else ''} | "
          f"components {conn['components_before']}->{conn['components_after']} | "
          f"LCC {conn['lcc_frac_before']*100:.0f}%->{conn['lcc_frac_after']*100:.0f}%")

    # ---- Phase 4: criticality + resilience ----
    report = dict(stem=stem, raw_graph=raw_stats, connectivity=conn, healed_bridges=healed,
                  healed_prob=n_prob, healed_canopy=n_canopy, healed_geom=n_geom, healed_tjunction=n_tjct,
                  scene_canopy_frac=scene_green, canopy_saturated=bool(saturated))
    if H.number_of_nodes() >= 4:
        ranked = compute_centrality(H)
        cls = classify_nodes(H)
        abl = ablation_simulation(H, args.top_k)
        svc = demand_weighted_criticality(H)        # betweenness x local demand
        report["gatekeepers"] = [dict(node=int(n), betweenness=round(b, 4), tier=cls[n]) for n, b in ranked[:10]]
        report["service_gatekeepers"] = [dict(node=int(n), service_crit=s,
            betweenness=round(H.nodes[n].get("betweenness", 0.0), 4),
            demand=H.nodes[n].get("demand", 0.0), tier=cls[n]) for n, s in svc[:10]]
        report["resilience"] = abl
        r3 = next((r["resilience_index"] for r in abl if r["step"] == min(3, len(abl) - 1)), None)
        print(f"[4/4] top gatekeeper node {ranked[0][0]} (BC={ranked[0][1]:.3f}) | "
              f"Resilience Index after {min(3,len(abl)-1)} removals = {r3}")
    else:
        print("[4/4] graph too small for criticality analysis"); abl = []; ranked = []; cls = {}

    # ---- flood-resilience overlay (optional: needs a DEM registered to the tile) ----
    if args.dem and os.path.exists(args.dem) and H.number_of_nodes() >= 2:
        dem = np.load(args.dem)
        node_elev = sample_node_elevations(H, dem)
        nx.set_node_attributes(H, node_elev, "elev")
        report["flood"] = flood_simulation(H, node_elev, n_steps=12)
        report["elev_min"] = round(min(node_elev.values()), 1)
        report["elev_max"] = round(max(node_elev.values()), 1)
        worst = min(report["flood"], key=lambda r: r["resilience_index"])
        print(f"[flood] elev {report['elev_min']}-{report['elev_max']} m | "
              f"at +{report['elev_max']-report['elev_min']:.0f}m peak, resilience floor {worst['resilience_index']}")

    pickle.dump(H, open(os.path.join(args.out, f"{stem}_graph.gpickle"), "wb"))  # after centrality -> attrs included

    # ---- OSM topological-accuracy benchmark (optional, best-effort: needs <stem>_geo.json + network) ----
    if args.osm:
        try:
            import osm_benchmark
            osm_res, osm_lines = osm_benchmark.run(stem, args.out, n_pairs=150)
            report["osm"] = osm_res; report["osm_lines"] = osm_lines
            c = osm_res["coverage"]; p = osm_res["path_length"] or {}
            print(f"[osm] recall {c['recall']*100:.0f}% precision {c['precision']*100:.0f}% | "
                  f"median path-len err {p.get('median_path_length_error_pct')}% | offset {osm_res['registration_offset_px']}px")
        except Exception as e:
            print(f"[osm] skipped ({type(e).__name__}: {e})")

    json.dump(report, open(os.path.join(args.out, f"{stem}_report.json"), "w"), indent=2)

    # ---- composite figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    if img is not None:
        ax[0, 0].imshow(img)
    ax[0, 0].set_title("input (real satellite)"); ax[0, 0].axis("off")
    ax[0, 1].imshow(mask, cmap="gray"); ax[0, 1].set_title("road mask (Phase 1)"); ax[0, 1].axis("off")
    base = img if img is not None else np.zeros((*mask.shape, 3), np.uint8)
    ax[1, 0].imshow(base)
    if ranked:
        ebs = [d.get("edge_betweenness", 0) for _, _, d in H.edges(data=True)] or [0]; emax = max(ebs) or 1
        for u, v, d in H.edges(data=True):
            eb = d.get("edge_betweenness", 0) / emax
            color = (min(1, .1 + eb), max(0, .8 - eb), max(0, .5 - eb))
            if d.get("healed"):
                p1 = H.nodes[u]["pos"]; p2 = H.nodes[v]["pos"]
                hc = {"canopy": "#7ed957", "prob": "#b388ff"}.get(d.get("heal_kind"), "#ffcc00")
                ax[1, 0].plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=hc, lw=2)
            else:
                p = d["pts"]; ax[1, 0].plot(p[:, 1], p[:, 0], "-", color=color, lw=1 + 5 * eb)
        cmap = {"critical": "#ff2b2b", "important": "#ffb347", "normal": "#00e6bd"}
        for n, d in H.nodes(data=True):
            x, y = d["pos"]; ax[1, 0].plot(x, y, "o", color=cmap[cls[n]], ms={"critical":8,"important":5,"normal":3}[cls[n]], mec="white", mew=.4)
    ax[1, 0].set_title("criticality map (red=gatekeeper, gold=bridged, violet=faint-road, green=under canopy)"); ax[1, 0].axis("off")
    if abl:
        xs = [r["step"] for r in abl]
        ax[1, 1].plot(xs, [r["resilience_index"] for r in abl], "o-", color="#2e6fe0", lw=2, label="Resilience Index")
        ax[1, 1].plot(xs, [r["lcc_fraction"] for r in abl], "s--", color="#f26522", lw=2, label="LCC fraction")
        ax[1, 1].set_ylim(0, 1.05); ax[1, 1].grid(alpha=.3); ax[1, 1].legend()
        ax[1, 1].set_xlabel("gatekeeper nodes removed"); ax[1, 1].set_ylabel("ratio vs baseline")
    ax[1, 1].set_title("resilience under stress test");
    plt.tight_layout(); plt.savefig(os.path.join(args.out, f"{stem}_pipeline.png"), dpi=110, bbox_inches="tight"); plt.close()
    print(f"DONE -> {args.out}/{stem}_{{mask.png, graph.gpickle, report.json, pipeline.png}}")


if __name__ == "__main__":
    main()
