"""Batch-bake several REAL 16x16 Bangalore sector tiles for the map dropdown.

Geocodes the requested residential area (OSM Nominatim) and picks planned/arterial
zones (higher connectivity than dense tree-covered cores). For each: fetch imagery ->
run model (recall config) -> export geo tile. Finally removes the abstract DeepGlobe
3D tiles and rebuilds the manifest so only geolocated map tiles remain.
"""
import subprocess, sys, os, json, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(__file__))
PY = sys.executable
CKPT = "C:/Users/VISWAS/route_data/runs/phase1_full/best.pt"
os.chdir(ROOT)


def geocode(q, fallback):
    try:
        url = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + urllib.parse.quote(q)
        req = urllib.request.Request(url, headers={"User-Agent": "RouteResilience/1.0 (geo demo)"})
        d = json.load(urllib.request.urlopen(req, timeout=20))
        if d:
            print(f"  geocoded '{q}' -> {d[0]['lat']},{d[0]['lon']}")
            return float(d[0]["lat"]), float(d[0]["lon"])
    except Exception as e:
        print(f"  geocode failed ({e}); using fallback")
    return fallback


def run(stem, lat, lon, label, grid=16):
    print(f"\n=== {stem} ({label}) @ {lat},{lon} ===")
    subprocess.run([PY, "src/fetch_geo_tile.py", "--lat", str(lat), "--lon", str(lon),
                    "--zoom", "18", "--grid", str(grid), "--stem", stem, "--out", "runs/geo"], check=True)
    subprocess.run([PY, "src/run_pipeline.py", "--image", f"runs/geo/{stem}_sat.png", "--ckpt", CKPT,
                    "--device", "cuda", "--out", "runs/geo", "--thr", "0.35", "--max-gap", "95",
                    "--ang-tol", "50", "--multiscale", "--no-tta"], check=True)  # multiscale (recovers roads missed at 1x) replaces TTA's diversity
    subprocess.run([PY, "src/export_web_geo.py", "--stem", stem, "--label", label, "--no-manifest"], check=True)


def main():
    time.sleep(1)  # be polite to Nominatim
    ap_lat, ap_lon = geocode("Aryahamsa Grande Apartments, JP Nagar, Bengaluru", (12.8845, 77.5855))
    # planned/arterial Bangalore zones -> better connectivity than dense cores
    sectors = [
        (ap_lat, ap_lon, "blr_res", "Residential sector"),                 # apartment area (unlabeled)
        (12.9250, 77.5838, "blr_jayanagar", "Planned grid sector"),        # Jayanagar (textbook grid)
        (12.9920, 77.5550, "blr_rajajinagar", "Planned grid sector II"),   # Rajajinagar
        (12.8450, 77.6600, "blr_ecity", "IT corridor sector"),             # Electronic City arterials
    ]
    for lat, lon, stem, label in sectors:
        try:
            run(stem, lat, lon, label)
        except Exception as e:
            print(f"  [warn] {stem} failed: {e}")

    # remove the abstract DeepGlobe 3D tiles (the "outline" road-skeletons)
    for s in ["100841", "100892", "106676"]:
        for ext in (".json", ".jpg"):
            p = os.path.join(ROOT, "web", "data", f"{s}{ext}")
            if os.path.exists(p):
                os.remove(p); print(f"  removed {s}{ext}")
    # rebuild manifest from remaining (geo-only) tiles
    subprocess.run([PY, "src/export_web_geo.py", "--stem", "blr_sector", "--label", "HSR sector · 2.4 km"], check=True)
    print("\nBAKE_SECTORS_DONE")


if __name__ == "__main__":
    main()
