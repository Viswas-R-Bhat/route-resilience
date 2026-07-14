"""Build an India-wide OSM-supervised training set on Esri z18 imagery (the deployment domain).

Why: the shipped model was trained on DeepGlobe (rural/suburban, non-Indian fabric, different
sensor) and deployed on Esri z18 Indian cities — that domain gap is the measured root cause of
low OSM recall / routing success. This builder creates real training pairs in the deployment
domain itself: Esri World Imagery mosaics + masks rasterized from OpenStreetMap ways (the
Massachusetts-benchmark method — real imagery, real map ground truth, no fabrication).

Design:
  - ~22 TRAIN sites across diverse Indian geography (canopy Kerala, desert Rajasthan, plains
    Punjab/UP, dense metros, planned grids, rural/highway Karnataka). Each site = one 2048px
    z18 mosaic sliced into four 1024px tiles -> {stem}_sat.jpg + {stem}_mask.png (DeepGlobe
    file format, so dataset.py / train.py work unchanged).
  - VAL sites are whole cities never seen in training (generalization is measured, not assumed).
  - The 5 benchmark sector bboxes (runs/geo/*_geo.json) are ALWAYS excluded from training tiles.
  - Masks: per-highway-class stroke widths (px at ~0.6 m/px); tunnels skipped (invisible);
    same drivable ROAD_TYPES as the OSM benchmark, so training target == evaluation target.
  - Tile QC: reject under-mapped/empty tiles (road fraction < 0.4%) and blank imagery.
  - QC gallery per site (sat + red mask overlay) -> eyeball label alignment before training.

Usage:
  python src/build_osm_dataset.py --out C:/Users/VISWAS/route_data/osm_esri [--sites blr_whitefield ...]
"""
import os, sys, io, json, time, argparse
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_geo_tile import ESRI, TILE, deg2tile, tile2deg, fetch
from osm_benchmark import fetch_osm, latlng_to_px, ROAD_TYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# stroke width (px at z18, ~0.55-0.6 m/px) per OSM highway class — eyeballed against imagery
WIDTH_PX = {"motorway": 30, "trunk": 26, "motorway_link": 16, "trunk_link": 16,
            "primary": 20, "primary_link": 14, "secondary": 16, "secondary_link": 12,
            "tertiary": 12, "tertiary_link": 10, "unclassified": 9, "residential": 9,
            "living_street": 8}

# (site_stem, lat, lon) — TRAIN: diverse Indian fabric. VAL: cities never trained on.
SITES_TRAIN = [
    # Bengaluru (test city, non-benchmark areas)
    ("blr_whitefield",   12.9698, 77.7500),
    ("blr_yelahanka",    13.1007, 77.5963),
    ("blr_hebbal",       13.0358, 77.5970),
    ("blr_banashankari", 12.9255, 77.5468),
    # Kerala — dense tropical canopy (the hard occlusion regime)
    ("koc_kochi",        9.9816, 76.2999),
    ("tvm_trivandrum",   8.5241, 76.9366),
    ("clt_kozhikode",    11.2588, 75.7804),
    # Metros
    ("del_karolbagh",    28.6519, 77.1909),
    ("del_rohini",       28.7360, 77.1170),
    ("bom_andheri",      19.1197, 72.8468),
    ("maa_annanagar",    13.0850, 80.2101),
    ("hyd_secbad",       17.4399, 78.4983),
    ("pnq_pune",         18.5074, 73.8077),
    # Desert / arid
    ("jai_jaipur",       26.9124, 75.7873),
    ("jod_jodhpur",      26.2389, 73.0243),
    # Plains
    ("ldh_ludhiana",     30.9010, 75.8573),
    ("lko_lucknow",      26.8467, 80.9462),
    # NE / river
    ("ghy_guwahati",     26.1445, 91.7362),
    # Planned grid
    ("chd_chandigarh",   30.7333, 76.7794),
    # Rural / highway Karnataka (small-town + NH fabric near the test city)
    ("rur_devanahalli",  13.2437, 77.7130),
    ("rur_ramanagara",   12.7217, 77.2812),
    ("rur_hoskote",      13.0707, 77.798),
]
SITES_VAL = [
    ("val_coimbatore",   11.0168, 76.9558),   # mixed canopy/plains
    ("val_ahmedabad",    23.0225, 72.5714),   # arid grid
    ("val_bhubaneswar",  20.2961, 85.8245),   # planned, plains
    ("val_thrissur",     10.5276, 76.2144),   # Kerala canopy (unseen)
]

GRID = 8          # 8x8 Esri tiles = 2048px mosaic -> four 1024px training tiles
ZOOM = 18
TILE_OUT = 1024
MIN_ROAD_FRAC, MAX_ROAD_FRAC = 0.004, 0.30
MIN_BRIGHTNESS = 18


def benchmark_bboxes():
    import glob
    out = []
    for p in glob.glob(os.path.join(ROOT, "runs", "geo", "*_geo.json")):
        if os.path.basename(p).startswith("live_"):
            continue
        b = json.load(open(p))["bounds"]
        out.append((b["south"], b["west"], b["north"], b["east"]))
    return out


def bbox_intersects(b1, b2):
    s1, w1, n1, e1 = b1; s2, w2, n2, e2 = b2
    return not (e1 <= w2 or e2 <= w1 or n1 <= s2 or n2 <= s1)


def fetch_mosaic(lat, lon, grid=GRID, zoom=ZOOM):
    from PIL import Image
    cx, cy = deg2tile(lat, lon, zoom)
    x0, y0 = int(cx) - grid // 2, int(cy) - grid // 2
    canvas = Image.new("RGB", (grid * TILE, grid * TILE))
    fails = 0
    for j in range(grid):
        for i in range(grid):
            try:
                data = fetch(ESRI.format(z=zoom, x=x0 + i, y=y0 + j))
                canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (i * TILE, j * TILE))
            except Exception:
                fails += 1
            time.sleep(0.03)
    north, west = tile2deg(x0, y0, zoom)
    south, east = tile2deg(x0 + grid, y0 + grid, zoom)
    bounds = dict(north=north, south=south, east=east, west=west)
    return np.array(canvas), bounds, fails


def rasterize_osm_mask(osm_json, bounds, W, H):
    """OSM drivable ways -> binary mask with per-class stroke widths. Tunnels skipped."""
    coords = {e["id"]: (e["lat"], e["lon"]) for e in osm_json["elements"] if e["type"] == "node"}
    m = np.zeros((H, W), np.uint8)
    n_ways = 0
    for e in osm_json["elements"]:
        if e["type"] != "way":
            continue
        tags = e.get("tags", {})
        hw = tags.get("highway")
        if hw not in ROAD_TYPES or tags.get("tunnel") in ("yes", "building_passage"):
            continue
        pts = [latlng_to_px(*coords[r], bounds, W, H) for r in e.get("nodes", []) if r in coords]
        if len(pts) < 2:
            continue
        arr = np.array([[int(round(x)), int(round(y))] for x, y in pts], np.int32)
        cv2.polylines(m, [arr], False, 255, WIDTH_PX[hw], lineType=cv2.LINE_8)
        n_ways += 1
    return m, n_ways


def process_site(stem, lat, lon, out_dir, bench_boxes, qc_dir):
    print(f"=== {stem} @ {lat},{lon}")
    img, bounds, fails = fetch_mosaic(lat, lon)
    if fails > GRID * GRID * 0.15:
        print(f"  [skip site] {fails} imagery tiles failed"); return []
    osm = fetch_osm(bounds, cache_dir=os.path.join(ROOT, "runs", "osm_cache"))
    H, W = img.shape[:2]
    mask, n_ways = rasterize_osm_mask(osm, bounds, W, H)
    print(f"  imagery {W}x{H} ({fails} fails) | {n_ways} OSM ways | road frac {(mask > 0).mean()*100:.1f}%")

    kept = []
    k = 0
    for j in range(H // TILE_OUT):
        for i in range(W // TILE_OUT):
            y, x = j * TILE_OUT, i * TILE_OUT
            im = img[y:y + TILE_OUT, x:x + TILE_OUT]
            mk = mask[y:y + TILE_OUT, x:x + TILE_OUT]
            # tile bounds (linear in Mercator-y is fine at this scale for the exclusion test)
            tb = (bounds["south"] + (1 - (y + TILE_OUT) / H) * (bounds["north"] - bounds["south"]),
                  bounds["west"] + x / W * (bounds["east"] - bounds["west"]),
                  bounds["south"] + (1 - y / H) * (bounds["north"] - bounds["south"]),
                  bounds["west"] + (x + TILE_OUT) / W * (bounds["east"] - bounds["west"]))
            if any(bbox_intersects(tb, bb) for bb in bench_boxes):
                print(f"  [exclude] {stem}_{k}: overlaps a benchmark sector"); k += 1; continue
            frac = (mk > 0).mean()
            if not (MIN_ROAD_FRAC <= frac <= MAX_ROAD_FRAC):
                print(f"  [skip] {stem}_{k}: road frac {frac*100:.2f}% outside range"); k += 1; continue
            if im.mean() < MIN_BRIGHTNESS:
                print(f"  [skip] {stem}_{k}: blank imagery"); k += 1; continue
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{k}_sat.jpg"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{k}_mask.png"), mk)
            kept.append(f"{stem}_{k}")
            k += 1

    # QC overlay (quarter-res): red = OSM label over imagery — eyeball alignment/width
    qc = img.copy(); qc[mask > 0] = (0.35 * qc[mask > 0] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    qc = cv2.resize(qc, (W // 4, H // 4))
    cv2.imwrite(os.path.join(qc_dir, f"qc_{stem}.jpg"), cv2.cvtColor(qc, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  kept {len(kept)}/{k} tiles")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="C:/Users/VISWAS/route_data/osm_esri")
    ap.add_argument("--sites", nargs="*", help="subset of site stems (default: all)")
    args = ap.parse_args()

    bench = benchmark_bboxes()
    print(f"{len(bench)} benchmark sector bboxes will be excluded from training tiles")
    qc_dir = os.path.join(ROOT, "runs", "osm_dataset")
    os.makedirs(qc_dir, exist_ok=True)
    meta = dict(zoom=ZOOM, grid=GRID, widths=WIDTH_PX, sites={})

    for split, sites in (("train", SITES_TRAIN), ("val", SITES_VAL)):
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        for stem, lat, lon in sites:
            if args.sites and stem not in args.sites:
                continue
            try:
                kept = process_site(stem, lat, lon, out_dir, bench, qc_dir)
                meta["sites"][stem] = dict(split=split, lat=lat, lon=lon, tiles=kept)
            except Exception as e:
                print(f"  [site FAILED] {stem}: {type(e).__name__}: {e}")
                meta["sites"][stem] = dict(split=split, lat=lat, lon=lon, error=str(e))
            time.sleep(3)          # be polite to Overpass between sites
            json.dump(meta, open(os.path.join(qc_dir, "dataset_meta.json"), "w"), indent=2)

    n_tr = len([t for s in meta["sites"].values() if s.get("split") == "train" for t in s.get("tiles", [])])
    n_va = len([t for s in meta["sites"].values() if s.get("split") == "val" for t in s.get("tiles", [])])
    print(f"\nDONE: {n_tr} train tiles, {n_va} val tiles -> {args.out}")
    print(f"QC gallery: {qc_dir}/qc_*.jpg — CHECK label alignment before training.")


if __name__ == "__main__":
    main()
