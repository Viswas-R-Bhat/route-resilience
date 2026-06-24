"""Phase 2 demo on REAL DeepGlobe ground-truth masks: mask -> skeleton -> routable graph.

Prints real graph stats per tile and saves a visual (satellite | GT mask | graph overlay).
Validates the skeleton->graph step on clean real labels before we feed model predictions in.
"""
import os, sys, glob
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph, graph_stats

DG = "C:/Users/VISWAS/route_data/deepglobe/train"
OUT = "runs/phase2"
os.makedirs(OUT, exist_ok=True)


def pick_tiles(n=4, lo=0.05, hi=0.13):
    out = []
    for s in sorted(glob.glob(os.path.join(DG, "*_sat.jpg"))):
        m = s[:-len("_sat.jpg")] + "_mask.png"
        mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if mk is None:
            continue
        if lo < (mk > 127).mean() < hi:
            out.append((s, m))
        if len(out) >= n:
            break
    return out


def main():
    tiles = pick_tiles()
    fig, ax = plt.subplots(len(tiles), 3, figsize=(12, 4 * len(tiles)))
    for i, (s, m) in enumerate(tiles):
        img = cv2.cvtColor(cv2.imread(s), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        skel = mask_to_skeleton(mask)
        G = skeleton_to_graph(skel)
        st = graph_stats(G)
        print(f"  {os.path.basename(s):>20}: {st['nodes']:4d} nodes | {st['edges']:4d} edges | "
              f"{st['components']:3d} comp | LCC {st['lcc_fraction']*100:4.0f}% | len {st['total_length_px']:.0f}px")
        ax[i, 0].imshow(img); ax[i, 0].axis("off")
        ax[i, 1].imshow(mask, cmap="gray"); ax[i, 1].axis("off")
        ax[i, 2].imshow(img)
        for u, v, d in G.edges(data=True):
            p = d["pts"]
            ax[i, 2].plot(p[:, 1], p[:, 0], "-", color="#00e6bd", lw=1.3)
        for nd, d in G.nodes(data=True):
            x, y = d["pos"]; deg = G.degree[nd]
            col = "#ff3b3b" if deg >= 3 else ("#ffb347" if deg == 1 else "#2e6fe0")
            ax[i, 2].plot(x, y, "o", color=col, ms=4)
        ax[i, 2].axis("off")
        if i == 0:
            ax[i, 0].set_title("real satellite"); ax[i, 1].set_title("road mask (GT)")
            ax[i, 2].set_title("routable graph (red=junction, amber=endpoint)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase2_graph_demo.png"), dpi=110, bbox_inches="tight")
    print("saved", os.path.join(OUT, "phase2_graph_demo.png"))


if __name__ == "__main__":
    main()
