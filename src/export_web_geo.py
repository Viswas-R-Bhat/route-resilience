"""Export a GEOREFERENCED pipeline tile -> web bundle with real lat/lon, for the Leaflet map.

Reads runs/geo/<stem>_{graph.gpickle,report.json,geo.json}, converts every node/edge
vertex from pixel space to true WGS84 lat/lon (Web Mercator inverse, since the mosaic was
stitched from XYZ tiles), and writes web/data/<stem>.json with a `geo` block + bounds.
Then rebuilds web/data/manifest.json across all tiles (DeepGlobe 3D tiles + geo map tiles).

Usage:  python src/export_web_geo.py --stem blr_hsr --label "HSR Layout · Bengaluru"
"""
import os, sys, glob, json, pickle, math, argparse
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from phase4_analysis import classify_nodes, compute_centrality

ROOT = os.path.dirname(os.path.dirname(__file__))
GEO = os.path.join(ROOT, "runs", "geo")
WEB = os.path.join(ROOT, "web", "data")
os.makedirs(WEB, exist_ok=True)
TEX = 2048; MAXPTS = 16


def _merc_y(lat): return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
def _inv_merc_y(my): return math.degrees(2 * math.atan(math.exp(my)) - math.pi / 2)


def px_to_latlng(x, y, b, W, H):
    lon = b["west"] + (x / W) * (b["east"] - b["west"])
    myN, myS = _merc_y(b["north"]), _merc_y(b["south"])
    lat = _inv_merc_y(myN + (y / H) * (myS - myN))
    return round(lat, 6), round(lon, 6)


def downsample(pts, k=MAXPTS):
    import numpy as np
    pts = np.asarray(pts, dtype=float)
    idx = range(len(pts)) if len(pts) <= k else np.linspace(0, len(pts) - 1, k).astype(int)
    return [[round(float(pts[i][1]), 1), round(float(pts[i][0]), 1)] for i in idx]  # (row,col)->[x,y]


def export_geo(stem, label):
    geo = json.load(open(os.path.join(GEO, f"{stem}_geo.json")))
    b, W, H = geo["bounds"], geo["img_w"], geo["img_h"]
    G = pickle.load(open(os.path.join(GEO, f"{stem}_graph.gpickle"), "rb"))
    report = json.load(open(os.path.join(GEO, f"{stem}_report.json")))
    if G.number_of_nodes() >= 4:
        compute_centrality(G)
    tiers = classify_nodes(G)
    Image.open(os.path.join(GEO, f"{stem}_sat.png")).convert("RGB").resize((TEX, TEX), Image.LANCZOS)\
        .save(os.path.join(WEB, f"{stem}.jpg"), quality=88)

    nodes = []
    for n, d in G.nodes(data=True):
        x, y = float(d["pos"][0]), float(d["pos"][1])
        lat, lng = px_to_latlng(x, y, b, W, H)
        nodes.append(dict(id=int(n), x=round(x, 1), y=round(y, 1), lat=lat, lng=lng,
                          bc=round(float(d.get("betweenness", 0.0)), 4), tier=tiers[n], deg=int(G.degree[n]),
                          elev=(round(float(d["elev"]), 1) if d.get("elev") is not None else None),
                          demand=d.get("demand"), svc=d.get("service_crit")))
    edges = []
    for u, v, d in G.edges(data=True):
        pts = downsample(d["pts"]) if (d.get("pts") is not None) else \
            [[G.nodes[u]["pos"][0], G.nodes[u]["pos"][1]], [G.nodes[v]["pos"][0], G.nodes[v]["pos"][1]]]
        ll = [list(px_to_latlng(px, py, b, W, H)) for px, py in pts]
        edges.append(dict(u=int(u), v=int(v), bc=round(float(d.get("edge_betweenness", 0.0)), 4),
                          w=round(float(d.get("weight", 1.0)), 1), healed=bool(d.get("healed", False)),
                          hk=d.get("heal_kind"), conf=d.get("conf"), pts=pts, ll=ll))
    ranked = sorted(nodes, key=lambda n: n["bc"], reverse=True)
    out = dict(stem=stem, label=label, geo=True, bounds=b, img_w=W, img_h=H, tex=f"{stem}.jpg",
               res_m_per_px=geo.get("res_m_per_px"),
               stats=report.get("raw_graph", {}), connectivity=report.get("connectivity", {}),
               healed_bridges=report.get("healed_bridges", 0),
               healed_canopy=report.get("healed_canopy", 0), healed_geom=report.get("healed_geom", 0),
               scene_canopy_frac=report.get("scene_canopy_frac", 0.0),
               canopy_saturated=bool(report.get("canopy_saturated", False)),
               nodes=nodes, edges=edges,
               gatekeepers=[dict(id=n["id"], bc=n["bc"], tier=n["tier"]) for n in ranked[:10]],
               service_gatekeepers=report.get("service_gatekeepers", []),
               resilience=report.get("resilience", []),
               flood=report.get("flood", []),
               elev_min=report.get("elev_min"), elev_max=report.get("elev_max"))
    json.dump(out, open(os.path.join(WEB, f"{stem}.json"), "w"))
    print(f"  exported GEO {stem}: {len(nodes)} nodes / {len(edges)} edges | bounds {b}")
    return out


def rebuild_manifest():
    tiles = []
    for p in sorted(glob.glob(os.path.join(WEB, "*.json"))):
        base = os.path.basename(p)
        if base == "manifest.json" or base.startswith("live_"):   # skip index + ephemeral live tiles
            continue
        d = json.load(open(p))
        tiles.append(dict(stem=d["stem"], label=d.get("label", d["stem"]), geo=bool(d.get("geo", False)),
                          nodes=len(d["nodes"]), edges=len(d["edges"]),
                          gatekeeper_bc=d["gatekeepers"][0]["bc"] if d.get("gatekeepers") else 0))
    tiles.sort(key=lambda t: (t["geo"], t["stem"]))   # DeepGlobe 3D tiles first, geo map tiles after
    default = next((t["stem"] for t in tiles if not t["geo"]), tiles[0]["stem"] if tiles else None)
    json.dump(dict(tiles=tiles, default=default), open(os.path.join(WEB, "manifest.json"), "w"), indent=2)
    print(f"manifest: {len(tiles)} tiles ({sum(t['geo'] for t in tiles)} geo) | default {default}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-manifest", action="store_true", help="write the tile only; skip manifest rebuild")
    args = ap.parse_args()
    export_geo(args.stem, args.label)
    if not args.no_manifest:
        rebuild_manifest()


if __name__ == "__main__":
    main()
