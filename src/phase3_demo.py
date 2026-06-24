"""Phase 3 demo on REAL DeepGlobe masks: find fragmented road graphs, heal the gaps,
report the Connectivity Ratio improvement, and visualize the synthetic bridges.
"""
import os, sys, glob
import numpy as np, cv2, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph
from phase3_heal import heal_graph, connectivity_report

DG = "C:/Users/VISWAS/route_data/deepglobe/train"
OUT = "runs/phase3"
os.makedirs(OUT, exist_ok=True)
MAX_GAP, ANG_TOL = 60, 35


def main():
    # scan tiles, build graphs, keep the most-fragmented (most components) to show healing value
    scored = []
    for s in sorted(glob.glob(os.path.join(DG, "*_sat.jpg")))[:80]:
        m = s[:-len("_sat.jpg")] + "_mask.png"
        mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if mk is None or not (0.03 < (mk > 127).mean() < 0.15):
            continue
        G = skeleton_to_graph(mask_to_skeleton((mk > 127).astype(np.uint8)))
        c = nx.number_connected_components(G)
        if c >= 3 and G.number_of_nodes() > 8:
            scored.append((c, s, m, G))
    scored.sort(reverse=True)
    tiles = scored[:4]

    fig, ax = plt.subplots(len(tiles), 2, figsize=(9, 4.4 * len(tiles)))
    agg_before = agg_after = 0
    for i, (c, s, m, G) in enumerate(tiles):
        img = cv2.cvtColor(cv2.imread(s), cv2.COLOR_BGR2RGB)
        H, healed = heal_graph(G, MAX_GAP, ANG_TOL)
        rep = connectivity_report(G, H)
        agg_before += rep["lcc_frac_before"]; agg_after += rep["lcc_frac_after"]
        print(f"  {os.path.basename(s):>20}: components {rep['components_before']}->{rep['components_after']} "
              f"| LCC {rep['lcc_frac_before']*100:.0f}%->{rep['lcc_frac_after']*100:.0f}% | +{healed} bridges")
        for col, (g, title) in enumerate([(G, f"extracted graph - {rep['components_before']} fragments"),
                                          (H, f"healed - {rep['components_after']} comp, +{healed} bridges")]):
            ax[i, col].imshow(img)
            for u, v, d in g.edges(data=True):
                if d.get("healed"):
                    p1 = g.nodes[u]["pos"]; p2 = g.nodes[v]["pos"]
                    ax[i, col].plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="#ffcc00", lw=2.0)
                else:
                    p = d["pts"]; ax[i, col].plot(p[:, 1], p[:, 0], "-", color="#00e6bd", lw=1.2)
            ax[i, col].axis("off")
            if i == 0:
                ax[i, col].set_title(title, fontsize=10)
    n = len(tiles)
    print(f"\n  mean LCC fraction: {agg_before/n*100:.0f}% -> {agg_after/n*100:.0f}%  (Connectivity Ratio improvement)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase3_healing_demo.png"), dpi=110, bbox_inches="tight")
    print("saved", os.path.join(OUT, "phase3_healing_demo.png"))


if __name__ == "__main__":
    main()
