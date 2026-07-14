"""OSM Topological-Accuracy benchmark — the PS-4 evaluation metric.

Grades the extracted graph against OpenStreetMap ground truth (Overpass API, no key):

  1. Average Path Length error : sample random point pairs, route them on BOTH the OSM graph
     and our model graph (endpoints snapped to the nearest model node), and compare shortest
     weighted path lengths. Reports mean/median relative error + routing-success rate (how often
     our network can route a trip OSM can) — exactly the rubric's "Topological Accuracy".
  2. Buffered coverage          : rasterize OSM roads and compare to the model mask with a
     tolerance buffer -> recall (% of known roads we recovered) and precision.

OSM lat/lon is projected into the tile's pixel space with the same Web-Mercator mapping the
dashboard uses, so both graphs live in one coordinate frame.

Usage:  python src/osm_benchmark.py --stem blr_hsr --out runs/geo
"""
import os, sys, io, json, math, time, pickle, argparse, urllib.request, urllib.parse, urllib.error
import numpy as np, cv2, networkx as nx
from scipy.spatial import cKDTree

OVERPASS = "https://overpass-api.de/api/interpreter"
# Drivable public-road hierarchy only. We exclude `service` (driveways, parking aisles) and
# `track`/`path` — these are mostly invisible at ~0.6 m and aren't what the model targets, so
# including them would unfairly depress recall against OSM.
ROAD_TYPES = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
              "residential", "living_street",
              "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


def _merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def latlng_to_px(lat, lon, b, W, H):
    x = (lon - b["west"]) / (b["east"] - b["west"]) * W
    myN, myS = _merc_y(b["north"]), _merc_y(b["south"])
    y = (_merc_y(lat) - myN) / (myS - myN) * H
    return x, y


def fetch_osm(b, timeout=60, tries=4, cache_dir="runs/osm_cache"):
    q = (f'[out:json][timeout:45];'
         f'(way["highway"]({b["south"]},{b["west"]},{b["north"]},{b["east"]}););'
         f'(._;>;);out body;')
    # disk cache: OSM for a fixed bbox is stable within a session; spares Overpass rate limits
    key = f"{b['south']:.6f}_{b['west']:.6f}_{b['north']:.6f}_{b['east']:.6f}.json"
    cpath = os.path.join(cache_dir, key)
    if os.path.exists(cpath):
        return json.load(open(cpath, encoding="utf-8"))
    data = urllib.parse.urlencode({"data": q}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(OVERPASS, data=data,
                                         headers={"User-Agent": "RouteResilience/1.0 (ISRO hackathon)"})
            out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            os.makedirs(cache_dir, exist_ok=True)
            json.dump(out, open(cpath, "w", encoding="utf-8"))
            return out
        except urllib.error.HTTPError as e:                 # 429 rate-limit / 504 gateway -> back off
            if e.code in (429, 504) and k < tries - 1:
                time.sleep(6 * (k + 1))
                continue
            raise


def build_osm_graph(osm_json, b, W, H):
    """OSM way network -> NetworkX graph in pixel space (weight = px length) + per-way pixel
    polylines (for rasterization) + downsampled lat/lon polylines (for the map overlay)."""
    coords = {e["id"]: (e["lat"], e["lon"]) for e in osm_json["elements"] if e["type"] == "node"}
    G = nx.Graph()
    px_lines, ll_lines = [], []
    for e in osm_json["elements"]:
        if e["type"] != "way" or e.get("tags", {}).get("highway") not in ROAD_TYPES:
            continue
        refs = [r for r in e.get("nodes", []) if r in coords]
        if len(refs) < 2:
            continue
        pxs = []
        for r in refs:
            lat, lon = coords[r]
            x, y = latlng_to_px(lat, lon, b, W, H)
            G.add_node(r, pos=(x, y))
            pxs.append((x, y))
        for u, v in zip(refs, refs[1:]):
            (x1, y1), (x2, y2) = G.nodes[u]["pos"], G.nodes[v]["pos"]
            d = math.hypot(x2 - x1, y2 - y1)
            if d > 0:
                G.add_edge(u, v, weight=d)
        px_lines.append(pxs)
        ll = [coords[r] for r in refs]
        step = max(1, len(ll) // 12)
        ll_lines.append([[round(la, 6), round(lo, 6)] for la, lo in ll[::step]])
    G.remove_nodes_from([n for n in list(G) if G.degree[n] == 0])
    return G, px_lines, ll_lines


def rasterize(px_lines, W, H):
    m = np.zeros((H, W), np.uint8)
    for ln in px_lines:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in ln], np.int32)
        if len(pts) >= 2:
            cv2.polylines(m, [pts], False, 1, 1)
    return m


def estimate_offset(osm_mask, model_mask, search=12, step=2):
    """Esri imagery and OSM are different sources and carry a few-pixel georegistration
    offset; find the (dx, dy) shift of the OSM raster that best overlaps the prediction, so
    coverage/snapping aren't penalized for a systematic shift rather than real road error."""
    md = cv2.dilate(model_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    om = osm_mask.astype(bool)
    best = (0, 0, -1)
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            ov = int((np.roll(np.roll(om, dy, 0), dx, 1) & md).sum())
            if ov > best[2]:
                best = (dx, dy, ov)
    return best[0], best[1]


def coverage(osm_mask, model_mask, buffer=5):
    """Centerline coverage vs OSM ground truth (Wiedemann completeness/correctness).

    Both legs compare CENTERLINES with a buffer tolerance, so road WIDTH doesn't bias the
    score: a model that paints the true full-width surface of a correctly-found road must not
    lose precision for it (the old full-mask precision capped ~0.5 once masks became
    width-realistic). recall = completeness (OSM centerline near predicted road);
    precision = correctness (predicted centerline near OSM road)."""
    from skimage.morphology import skeletonize
    k = 2 * buffer + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    o = osm_mask.astype(bool)
    sk = skeletonize(model_mask.astype(bool))
    o_near_m = o & cv2.dilate(model_mask, ker).astype(bool)     # OSM centerline within buffer of a prediction
    sk_near_o = sk & cv2.dilate(osm_mask, ker).astype(bool)     # predicted centerline within buffer of an OSM road
    recall = float(o_near_m.sum()) / max(1, int(o.sum()))
    precision = float(sk_near_o.sum()) / max(1, int(sk.sum()))
    f1 = 2 * recall * precision / max(1e-9, recall + precision)
    return dict(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                buffer_px=buffer, method="centerline")


def path_length_error(model_G, osm_G, W, H, res_m=None, n_pairs=200, max_snap_px=45, seed=42):
    """Average Path Length error: route random OSM point-pairs on OSM vs the model graph."""
    m_ids = [n for n in model_G if "pos" in model_G.nodes[n]]
    o_ids = [n for n in osm_G if 0 <= osm_G.nodes[n]["pos"][0] < W and 0 <= osm_G.nodes[n]["pos"][1] < H]
    if len(m_ids) < 2 or len(o_ids) < 2:
        return None
    tree = cKDTree(np.array([model_G.nodes[n]["pos"] for n in m_ids]))
    rng = np.random.default_rng(seed)
    errs, routable, well_snapped, sampled, snap_fail = [], 0, 0, 0, 0
    for _ in range(n_pairs * 6):
        if well_snapped >= n_pairs:
            break
        a, b = (int(x) for x in rng.choice(o_ids, 2, replace=False))
        try:
            Losm = nx.shortest_path_length(osm_G, a, b, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if not (Losm > 0):
            continue
        sampled += 1
        da, ia = tree.query(osm_G.nodes[a]["pos"]); db, ib = tree.query(osm_G.nodes[b]["pos"])
        if da > max_snap_px or db > max_snap_px:        # OSM endpoint has no nearby extracted road
            snap_fail += 1
            continue
        well_snapped += 1                               # model HAS roads at both ends -> test connectivity
        try:
            Lmod = nx.shortest_path_length(model_G, m_ids[ia], m_ids[ib], weight="weight")
        except nx.NetworkXNoPath:
            continue
        if Lmod > 0:
            routable += 1
            errs.append(abs(Lmod - Losm) / Losm)
    osm_km = sum(d["weight"] for *_, d in osm_G.edges(data=True)) * (res_m or 0) / 1000.0
    return dict(
        pairs_well_snapped=well_snapped, routable_pairs=routable,
        routing_success=round(routable / max(1, well_snapped), 4),       # connectivity fidelity
        coverage_gap_rate=round(snap_fail / max(1, sampled), 4),         # OSM pairs with no extracted road nearby
        avg_path_length_error_pct=round(float(np.mean(errs)) * 100, 2) if errs else None,
        median_path_length_error_pct=round(float(np.median(errs)) * 100, 2) if errs else None,
        osm_road_km=round(osm_km, 2))


def run(stem, out="runs/geo", n_pairs=200, buffer=5):
    geo = json.load(open(os.path.join(out, f"{stem}_geo.json")))
    b, W, H, res_m = geo["bounds"], geo["img_w"], geo["img_h"], geo.get("res_m_per_px")
    model_G = pickle.load(open(os.path.join(out, f"{stem}_graph.gpickle"), "rb"))
    model_mask = (cv2.imread(os.path.join(out, f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    osm_json = fetch_osm(b)
    osm_G, px_lines, ll_lines = build_osm_graph(osm_json, b, W, H)
    osm_mask = rasterize(px_lines, W, H)
    dx, dy = estimate_offset(osm_mask, model_mask)              # correct cross-source registration
    if dx or dy:
        osm_mask = np.roll(np.roll(osm_mask, dy, 0), dx, 1)
        for n in osm_G:                                        # shift graph too (for fair snapping)
            px, py = osm_G.nodes[n]["pos"]; osm_G.nodes[n]["pos"] = (px + dx, py + dy)
    cov = coverage(osm_mask, model_mask, buffer)
    ple = path_length_error(model_G, osm_G, W, H, res_m, n_pairs)
    result = dict(osm_nodes=osm_G.number_of_nodes(), osm_edges=osm_G.number_of_edges(),
                  registration_offset_px=[dx, dy], coverage=cov, path_length=ple)
    return result, ll_lines    # ll_lines stay in true lat/lon for the map overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default="runs/geo")
    ap.add_argument("--n-pairs", type=int, default=200)
    args = ap.parse_args()
    result, ll_lines = run(args.stem, args.out, args.n_pairs)
    # patch the tile report so export_web_geo can surface it
    rp = os.path.join(args.out, f"{args.stem}_report.json")
    report = json.load(open(rp))
    report["osm"] = result
    report["osm_lines"] = ll_lines
    json.dump(report, open(rp, "w"), indent=2)
    cov, ple = result["coverage"], result["path_length"] or {}
    print(f"[osm] {args.stem}: recall {cov['recall']*100:.1f}% precision {cov['precision']*100:.1f}% "
          f"| path-len err {ple.get('avg_path_length_error_pct')}% | "
          f"routing success {(ple.get('routing_success') or 0)*100:.0f}% | {result['osm_edges']} OSM edges")


if __name__ == "__main__":
    main()
