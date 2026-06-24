"""Phase 4 demo on a REAL extracted road graph: criticality heatmap + resilience stress test.

Picks a road-dense real tile, builds + heals the graph, takes its largest connected component,
then computes Gatekeeper Nodes (betweenness) and a node-ablation Resilience Index curve.
Saves: criticality map (left) + resilience curve (right). All numbers are real.
"""
import os, sys, glob
import numpy as np, cv2, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph
from phase3_heal import heal_graph
from phase4_analysis import compute_centrality, classify_nodes, ablation_simulation, global_efficiency_weighted

DG = "C:/Users/VISWAS/route_data/deepglobe/train"
OUT = "runs/phase4"; os.makedirs(OUT, exist_ok=True)


def best_tile():
    """Scan tiles; return the one whose healed graph has the largest connected component."""
    best = None
    for s in sorted(glob.glob(os.path.join(DG, "*_sat.jpg")))[:60]:
        m = s[:-len("_sat.jpg")] + "_mask.png"
        mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if mk is None or not (0.05 < (mk > 127).mean() < 0.16):
            continue
        G, _ = heal_graph(skeleton_to_graph(mask_to_skeleton((mk > 127).astype(np.uint8))))
        comps = list(nx.connected_components(G))
        if not comps:
            continue
        lcc = G.subgraph(max(comps, key=len)).copy()
        if best is None or lcc.number_of_nodes() > best[0]:
            best = (lcc.number_of_nodes(), s, lcc)
    return best[1], best[2]


def main():
    s, G = best_tile()
    img = cv2.cvtColor(cv2.imread(s), cv2.COLOR_BGR2RGB)
    print(f"analysis tile: {os.path.basename(s)} | largest connected component: "
          f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    ranked = compute_centrality(G)
    cls = classify_nodes(G)
    print("  top gatekeeper nodes (betweenness):")
    for nid, sc in ranked[:6]:
        print(f"    node {nid:>4}  BC={sc:.4f}  [{cls[nid]}]")

    abl = ablation_simulation(G, top_k=8)
    print("  ablation stress test (remove highest-betweenness node each step):")
    for r in abl:
        print(f"    step {r['step']}: R={r['resilience_index']:.3f}  efficiency={r['global_efficiency']:.5f}  LCC={r['lcc_fraction']:.2f}")

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    # (1) criticality map
    ax[0].imshow(img)
    ebs = [d.get("edge_betweenness", 0) for _, _, d in G.edges(data=True)] or [0]
    emax = max(ebs) or 1
    for u, v, d in G.edges(data=True):
        eb = d.get("edge_betweenness", 0) / emax
        col = (min(1, 0.1 + eb), max(0, 0.8 - eb), max(0, 0.5 - eb))   # teal -> red
        p = d["pts"] if "pts" in d and d["pts"] is not None else np.array([G.nodes[u]["yx"], G.nodes[v]["yx"]])
        ax[0].plot(p[:, 1], p[:, 0], "-", color=col, lw=1 + 5 * eb)
    cmap = {"critical": "#ff2b2b", "important": "#ffb347", "normal": "#00e6bd"}
    rmap = {"critical": 9, "important": 6, "normal": 3}
    for nid, d in G.nodes(data=True):
        x, y = d["pos"]; t = cls[nid]
        ax[0].plot(x, y, "o", color=cmap[t], ms=rmap[t], mec="white", mew=0.5)
    top = ranked[0][0]
    tx, ty = G.nodes[top]["pos"]
    ax[0].annotate("top gatekeeper", (tx, ty), color="white", fontsize=9,
                   xytext=(tx + 18, ty - 18), arrowprops=dict(arrowstyle="->", color="white"))
    ax[0].set_title("Criticality map — red = high betweenness (Gatekeeper Nodes)"); ax[0].axis("off")
    # (2) resilience curve
    xs = [r["step"] for r in abl]
    ax[1].plot(xs, [r["resilience_index"] for r in abl], "o-", color="#2e6fe0", lw=2, label="Resilience Index R")
    ax[1].plot(xs, [r["lcc_fraction"] for r in abl], "s--", color="#f26522", lw=2, label="LCC fraction")
    ax[1].set_xlabel("gatekeeper nodes removed"); ax[1].set_ylabel("ratio vs baseline")
    ax[1].set_ylim(0, 1.05); ax[1].grid(alpha=0.3); ax[1].legend()
    ax[1].set_title("Network degradation under targeted node ablation (stress test)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase4_criticality_resilience.png"), dpi=120, bbox_inches="tight")
    print("saved", os.path.join(OUT, "phase4_criticality_resilience.png"))


if __name__ == "__main__":
    main()
