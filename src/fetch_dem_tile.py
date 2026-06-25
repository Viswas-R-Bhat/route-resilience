"""Fetch a DEM (elevation) raster aligned to an already-fetched satellite tile.

Uses the public AWS 'terrarium' terrain tiles (no API key): elevation is RGB-encoded as
    elev_m = R*256 + G + B/256 - 32768
Same XYZ Web-Mercator scheme as the Esri imagery, so we stitch the tiles covering the
tile's geographic bounds, crop to those exact bounds (fractional-tile precise), and resize
to the satellite grid — giving a per-pixel elevation map registered to <stem>_sat.png.

This feeds the flood-resilience stress test (phase4_flood): rising water submerges the
lowest-elevation roads first, and we watch the network's connectivity collapse.

Output: <out>/<stem>_dem.npy  (float32, shape = img_h x img_w, metres)

Usage:
  python src/fetch_dem_tile.py --stem blr_hsr --out runs/geo
"""
import os, sys, io, json, argparse, urllib.request, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from fetch_geo_tile import deg2tile, fetch                      # reuse XYZ math + retrying fetch

TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE = 256
DEM_ZOOM = 14                                                   # terrarium has global coverage to ~z15


def _decode(rgb):
    a = rgb.astype(np.float64)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0) - 32768.0


def fetch_dem(bounds, img_w, img_h, zoom=DEM_ZOOM):
    """Return a (img_h, img_w) float32 elevation map registered to the given bounds."""
    n, s = bounds["north"], bounds["south"]
    w, e = bounds["west"], bounds["east"]
    fx0, fy0 = deg2tile(n, w, zoom)                             # NW corner (fractional tile)
    fx1, fy1 = deg2tile(s, e, zoom)                             # SE corner
    tx0, ty0 = int(np.floor(fx0)), int(np.floor(fy0))
    tx1, ty1 = int(np.floor(fx1)), int(np.floor(fy1))
    gx, gy = (tx1 - tx0 + 1), (ty1 - ty0 + 1)
    mosaic = np.zeros((gy * TILE, gx * TILE, 3), np.uint8)
    for j in range(gy):
        for i in range(gx):
            try:
                data = fetch(TERRARIUM.format(z=zoom, x=tx0 + i, y=ty0 + j))
                mosaic[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = \
                    np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception as ex:
                print(f"  [warn] DEM tile {tx0+i},{ty0+j} failed ({ex})")
            time.sleep(0.03)
    elev = _decode(mosaic)
    # crop the mosaic to the exact bounds (fractional-tile pixels), then resize to sat grid
    import cv2
    px0 = (fx0 - tx0) * TILE; px1 = (fx1 - tx0) * TILE
    py0 = (fy0 - ty0) * TILE; py1 = (fy1 - ty0) * TILE
    crop = elev[int(round(py0)):max(int(round(py1)), int(round(py0)) + 1),
                int(round(px0)):max(int(round(px1)), int(round(px0)) + 1)]
    return cv2.resize(crop.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default="runs/geo")
    ap.add_argument("--zoom", type=int, default=DEM_ZOOM)
    args = ap.parse_args()
    geo = json.load(open(os.path.join(args.out, f"{args.stem}_geo.json")))
    dem = fetch_dem(geo["bounds"], geo["img_w"], geo["img_h"], args.zoom)
    path = os.path.join(args.out, f"{args.stem}_dem.npy")
    np.save(path, dem)
    print(f"saved {path}  ({dem.shape[1]}x{dem.shape[0]}px) | elev {dem.min():.0f}–{dem.max():.0f} m "
          f"(range {dem.max()-dem.min():.0f} m)")


if __name__ == "__main__":
    main()
