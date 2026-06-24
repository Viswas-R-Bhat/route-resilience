"""Export REAL pipeline artifacts -> web bundle for the premium frontend.

For each processed tile in runs/pipeline/, emits web/data/<stem>.json (nodes, edges with
polylines + weights, gatekeepers, resilience curve, stats) and a resized satellite jpg.
The frontend (Three.js) renders the real graph and recomputes resilience client-side.
"""
import os, sys, glob, json, pickle
import numpy as np, networkx as nx
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from phase4_analysis import classify_nodes, compute_centrality

ROOT = os.path.dirname(os.path.dirname(__file__))
PIPE = os.path.join(ROOT, "runs", "pipeline")
WEB = os.path.join(ROOT, "web", "data")
os.makedirs(WEB, exist_ok=True)
TEX = 1024            # satellite texture size for the 3D plane
MAXPTS = 16           # max points kept per edge polyline (downsample)


def downsample(pts, k=MAXPTS):
    pts = np.asarray(pts, dtype=float)
    if len(pts) <= k:
        idx = range(len(pts))
    else:
        idx = np.linspace(0, len(pts) - 1, k).astype(int)
    # sknw pts are (row, col) = (y, x) -> store [x, y]
    return [[round(float(pts[i][1]), 1), round(float(pts[i][0]), 1)] for i in idx]


def export_tile(stem):
    G = pickle.load(open(os.path.join(PIPE, f"{stem}_graph.gpickle"), "rb"))
    report = json.load(open(os.path.join(PIPE, f"{stem}_report.json")))
    if G.number_of_nodes() >= 4:               # ensure node/edge betweenness present
        compute_centrality(G)
    tiers = classify_nodes(G)
    # satellite -> resized jpg
    sat = Image.open(os.path.join(PIPE, f"{stem}_sat.png")).convert("RGB")
    W0, H0 = sat.size
    sat.resize((TEX, TEX), Image.LANCZOS).save(os.path.join(WEB, f"{stem}.jpg"), quality=88)

    nodes = [dict(id=int(n), x=round(float(d["pos"][0]), 1), y=round(float(d["pos"][1]), 1),
                  bc=round(float(d.get("betweenness", 0.0)), 4), tier=tiers[n], deg=int(G.degree[n]))
             for n, d in G.nodes(data=True)]
    edges = [dict(u=int(u), v=int(v), bc=round(float(d.get("edge_betweenness", 0.0)), 4),
                  w=round(float(d.get("weight", 1.0)), 1), healed=bool(d.get("healed", False)),
                  pts=downsample(d["pts"]) if (d.get("pts") is not None) else
                      [[G.nodes[u]["pos"][0], G.nodes[u]["pos"][1]], [G.nodes[v]["pos"][0], G.nodes[v]["pos"][1]]])
             for u, v, d in G.edges(data=True)]
    ranked = sorted(nodes, key=lambda n: n["bc"], reverse=True)
    out = dict(
        stem=stem, img_w=W0, img_h=H0, tex=f"{stem}.jpg",
        stats=report.get("raw_graph", {}), connectivity=report.get("connectivity", {}),
        healed_bridges=report.get("healed_bridges", 0),
        nodes=nodes, edges=edges,
        gatekeepers=[dict(id=n["id"], bc=n["bc"], tier=n["tier"]) for n in ranked[:10]],
        resilience=report.get("resilience", []),
    )
    json.dump(out, open(os.path.join(WEB, f"{stem}.json"), "w"))
    return out


def main():
    stems = sorted(os.path.basename(p).replace("_report.json", "")
                   for p in glob.glob(os.path.join(PIPE, "*_report.json")))
    manifest = []
    for stem in stems:
        o = export_tile(stem)
        manifest.append(dict(stem=stem, nodes=len(o["nodes"]), edges=len(o["edges"]),
                             gatekeeper_bc=o["gatekeepers"][0]["bc"] if o["gatekeepers"] else 0))
        print(f"  exported {stem}: {len(o['nodes'])} nodes / {len(o['edges'])} edges / "
              f"{len(o['resilience'])} resilience steps")
    json.dump(dict(tiles=manifest, default=stems[0] if stems else None),
              open(os.path.join(WEB, "manifest.json"), "w"), indent=2)
    print(f"manifest: {len(manifest)} tiles -> {WEB}")


if __name__ == "__main__":
    main()
