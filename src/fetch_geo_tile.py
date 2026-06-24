"""Fetch a REAL georeferenced high-res satellite mosaic for a city area.

Stitches Esri World Imagery XYZ tiles (z=18 -> ~0.58 m/px at Bengaluru latitude,
matching DeepGlobe's resolution so the trained model stays in-distribution) into one
RGB image, and records its true lat/lon bounds (Web Mercator). No rasterio needed.

Output: <out>/<stem>_sat.png  +  <out>/<stem>_geo.json  (bbox + zoom + size)

Usage:
  python src/fetch_geo_tile.py --lat 12.9719 --lon 77.6412 --zoom 18 --grid 5 --stem blr_indiranagar --out runs/geo
"""
import os, sys, json, math, time, argparse, io, urllib.request
from PIL import Image

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE = 256


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile2deg(x, y, z):
    """NW corner lon/lat of tile (x, y)."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RouteResilience/1.0"})
            return urllib.request.urlopen(req, timeout=25).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.6 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--grid", type=int, default=5, help="grid x grid tiles (256px each)")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default="runs/geo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = args.zoom
    cx, cy = deg2tile(args.lat, args.lon, z)
    half = args.grid // 2
    x0, y0 = int(cx) - half, int(cy) - half          # top-left tile of the grid
    gx = gy = args.grid
    canvas = Image.new("RGB", (gx * TILE, gy * TILE))
    print(f"Fetching {gx}x{gy} Esri tiles @z{z} around ({args.lat},{args.lon}) ...")
    fails = 0
    for j in range(gy):
        for i in range(gx):
            tx, ty = x0 + i, y0 + j
            try:
                data = fetch(ESRI.format(z=z, x=tx, y=ty))
                canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (i * TILE, j * TILE))
            except Exception as e:
                fails += 1
                print(f"  [warn] tile {tx},{ty} failed ({e}); leaving blank")
            time.sleep(0.04)
        print(f"  row {j+1}/{gy}")
    if fails:
        print(f"  {fails} tile(s) failed and left blank")
    sat_path = os.path.join(args.out, f"{args.stem}_sat.png")
    canvas.save(sat_path)

    # geographic bounds of the stitched mosaic (NW of first tile -> NW of tile just past the last)
    north, west = tile2deg(x0, y0, z)
    south, east = tile2deg(x0 + gx, y0 + gy, z)
    res_m = 156543.03392 * math.cos(math.radians(args.lat)) / (2 ** z)
    geo = dict(stem=args.stem, zoom=z, grid=args.grid, tile_origin=[x0, y0],
               img_w=canvas.width, img_h=canvas.height,
               bounds=dict(north=north, south=south, east=east, west=west),
               res_m_per_px=round(res_m, 3))
    json.dump(geo, open(os.path.join(args.out, f"{args.stem}_geo.json"), "w"), indent=2)
    print(f"saved {sat_path}  ({canvas.width}x{canvas.height}px, ~{res_m:.2f} m/px)")
    print(f"bounds N{north:.5f} S{south:.5f} E{east:.5f} W{west:.5f}")


if __name__ == "__main__":
    main()
