"""Re-score saved loop runs with the current OSM benchmark (e.g. after a metric fix).

Uses each tag's saved masks/graphs + the disk-cached OSM responses, so old and new models
are re-measured under the IDENTICAL protocol — no re-inference, no cherry-picking.

Usage:  python src/rescore_loops.py --tags v1model_newheal ft_v1 ft_v1_thr50
"""
import os, sys, json, glob, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import osm_benchmark
from improve_loop import sector_row, COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)

    for tag in args.tags:
        out = os.path.join("runs", "loop", tag)
        stems = sorted(os.path.basename(p)[:-len("_geo.json")]
                       for p in glob.glob(os.path.join(out, "*_geo.json")))
        rows = {}
        for stem in stems:
            result, _ = osm_benchmark.run(stem, out)
            rp = os.path.join(out, f"{stem}_report.json")
            report = json.load(open(rp))
            report["osm"] = result
            json.dump(report, open(rp, "w"), indent=2)
            rows[stem] = sector_row(report)
            c, p = result["coverage"], result["path_length"] or {}
            print(f"[{tag}/{stem}] recall {c['recall']:.3f} precision {c['precision']:.3f} "
                  f"f1 {c['f1']:.3f} | routing {p.get('routing_success')}")
        avg = {c: round(sum(r[c] for r in rows.values() if r[c] is not None)
                        / max(1, sum(1 for r in rows.values() if r[c] is not None)), 4)
               for c in COLS}
        sb_path = os.path.join(out, "scoreboard.json")
        board = json.load(open(sb_path)) if os.path.exists(sb_path) else dict(tag=tag)
        board["sectors"] = rows; board["average"] = avg; board["coverage_method"] = "centerline"
        json.dump(board, open(sb_path, "w"), indent=2)
        print(f"=== {tag} AVERAGE: " + "  ".join(f"{c}={avg[c]}" for c in COLS))


if __name__ == "__main__":
    main()
