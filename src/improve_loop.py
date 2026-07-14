"""Benchmark loop — the measuring stick for every improvement.

Re-runs the full pipeline + OSM Topological-Accuracy benchmark on all baked benchmark
sectors (runs/geo/*_geo.json, excluding live_*) with the canonical recall settings
(thr 0.35, max-gap 95, ang-tol 50 — same as bake_sectors.py), then prints a scoreboard
and appends it to runs/loop/history.jsonl so every change is measured against baseline.

Imagery is re-fetched deterministically from the stored tile_origin when the sat mosaic
isn't on disk (the PNGs are too big for the repo). OSM responses are disk-cached.

Usage:
  python src/improve_loop.py --ckpt models/best.pt --tag baseline
  python src/improve_loop.py --ckpt route_data/runs/ft/best.pt --tag ft_v1 --device cuda
"""
import os, sys, io, json, glob, time, argparse, subprocess, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_geo_tile import ESRI, TILE, fetch

COLS = ["recall", "precision", "f1", "routing_success", "median_plerr_pct", "lcc_after", "nodes"]


def refetch_mosaic(geo, out_png):
    """Re-fetch the exact Esri mosaic recorded in a _geo.json (deterministic: tile_origin)."""
    from PIL import Image
    x0, y0 = geo["tile_origin"]; g = geo["grid"]; z = geo["zoom"]
    canvas = Image.new("RGB", (g * TILE, g * TILE))
    for j in range(g):
        for i in range(g):
            data = fetch(ESRI.format(z=z, x=x0 + i, y=y0 + j))
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (i * TILE, j * TILE))
            time.sleep(0.03)
    canvas.save(out_png)


def sector_row(report):
    osm = report.get("osm") or {}
    cov = osm.get("coverage") or {}
    pl = osm.get("path_length") or {}
    conn = report.get("connectivity") or {}
    return dict(recall=cov.get("recall"), precision=cov.get("precision"), f1=cov.get("f1"),
                routing_success=pl.get("routing_success"),
                median_plerr_pct=pl.get("median_path_length_error_pct"),
                lcc_after=conn.get("lcc_frac_after"),
                nodes=(report.get("raw_graph") or {}).get("nodes"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--tag", required=True, help="run label, e.g. baseline / ft_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--max-gap", type=float, default=95)
    ap.add_argument("--ang-tol", type=float, default=50)
    ap.add_argument("--stems", nargs="*", help="subset of sectors (default: all baked)")
    args = ap.parse_args()
    os.chdir(ROOT)

    stems = args.stems or sorted(
        os.path.basename(p)[:-len("_geo.json")]
        for p in glob.glob("runs/geo/*_geo.json")
        if not os.path.basename(p).startswith("live_"))
    out_dir = os.path.join("runs", "loop", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    rows = {}
    for stem in stems:
        geo = json.load(open(f"runs/geo/{stem}_geo.json"))
        sat = f"runs/geo/{stem}_sat.png"
        if not os.path.exists(sat):
            print(f"[{stem}] sat mosaic missing -> re-fetching {geo['grid']}x{geo['grid']} tiles @z{geo['zoom']}")
            refetch_mosaic(geo, sat)
        # pipeline + OSM benchmark write into the tag dir (baseline artifacts stay untouched)
        json.dump(geo, open(os.path.join(out_dir, f"{stem}_geo.json"), "w"))
        t0 = time.time()
        subprocess.run([sys.executable, "src/run_pipeline.py", "--image", sat, "--ckpt", args.ckpt,
                        "--device", args.device, "--out", out_dir, "--thr", str(args.thr),
                        "--max-gap", str(args.max_gap), "--ang-tol", str(args.ang_tol), "--osm"],
                       check=True)
        report = json.load(open(os.path.join(out_dir, f"{stem}_report.json")))
        rows[stem] = sector_row(report)
        print(f"[{stem}] done in {time.time()-t0:.0f}s -> {rows[stem]}")

    # averages over sectors (None-safe)
    avg = {c: round(sum(r[c] for r in rows.values() if r[c] is not None)
                    / max(1, sum(1 for r in rows.values() if r[c] is not None)), 4)
           for c in COLS}
    board = dict(tag=args.tag, ckpt=args.ckpt, thr=args.thr, max_gap=args.max_gap,
                 ang_tol=args.ang_tol, time=time.strftime("%Y-%m-%d %H:%M"),
                 sectors=rows, average=avg)
    json.dump(board, open(os.path.join(out_dir, "scoreboard.json"), "w"), indent=2)
    with open("runs/loop/history.jsonl", "a") as f:
        f.write(json.dumps(board) + "\n")

    w = max(len(s) for s in list(rows) + ["AVERAGE"])
    print(f"\n=== SCOREBOARD [{args.tag}] ===")
    print(f"{'sector':<{w}}  " + "  ".join(f"{c:>16}" for c in COLS))
    for s, r in list(rows.items()) + [("AVERAGE", avg)]:
        print(f"{s:<{w}}  " + "  ".join(f"{(r[c] if r[c] is not None else '-'):>16}" for c in COLS))


if __name__ == "__main__":
    main()
