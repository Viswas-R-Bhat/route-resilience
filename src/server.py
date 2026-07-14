"""Route Resilience server — serves the web app AND a live analysis API on one port.

  GET  /                      -> static web/ files
  GET  /api/analyze?lat&lon&grid
       -> fetch real Esri imagery for that location, run the model pipeline
          (segmentation -> graph -> healing -> criticality/resilience), export a geo
          tile, and return {stem}. The page then loads data/<stem>.json on the map.

Pure stdlib (no Flask). Model runs in a subprocess (route env). Analysis is serialized
with a lock so concurrent requests don't fight over the GPU.

Run:  python src/server.py 8766
"""
import http.server, socketserver, json, os, sys, subprocess, time, urllib.parse, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")

# Detect virtual environment python dynamically
venv_py = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(venv_py):
    venv_py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(venv_py):
    venv_py = r"C:/Users/VISWAS/route_env/Scripts/python.exe"
if not os.path.exists(venv_py):
    venv_py = sys.executable

ROUTE_PY = venv_py
CKPT = os.path.join(ROOT, "models", "best.pt")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

_LOCK = threading.Lock()   # serialize GPU/CPU pipeline runs


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            return self._analyze(parsed.query)
        return super().do_GET()

    def _analyze(self, query):
        q = urllib.parse.parse_qs(query)
        try:
            lat = float(q["lat"][0]); lon = float(q["lon"][0])
            grid = max(2, min(16, int(q.get("grid", ["8"])[0])))
        except Exception as e:
            return self._json({"error": f"bad params ({e})"}, 400)
        print(f"\n[API Analyze] Search coordinates received -> Lat: {lat}, Lon: {lon}, Grid size: {grid}", flush=True)
        stem = "live_%d" % int(time.time() * 1000)
        t0 = time.time()
        with _LOCK:
            try:
                self._pipeline(lat, lon, grid, stem)
            except subprocess.CalledProcessError as e:
                tail = (e.stderr or e.stdout or "")[-400:]
                return self._json({"error": "pipeline failed", "detail": tail}, 500)
        self._json({"stem": stem, "lat": lat, "lon": lon, "grid": grid,
                    "secs": round(time.time() - t0, 1)})

    def _pipeline(self, lat, lon, grid, stem):
        def sub(args):
            subprocess.run([ROUTE_PY] + args, cwd=ROOT, check=True,
                           capture_output=True, text=True, timeout=600)
        sub(["src/fetch_geo_tile.py", "--lat", str(lat), "--lon", str(lon),
             "--zoom", "18", "--grid", str(grid), "--stem", stem, "--out", "runs/geo"])
        try:                                                  # DEM is best-effort: flood overlay is optional
            sub(["src/fetch_dem_tile.py", "--stem", stem, "--out", "runs/geo"])
            dem = ["--dem", "runs/geo/%s_dem.npy" % stem]
        except Exception:
            dem = []
        tta = ["--no-tta"] if DEVICE == "cpu" else []
        sub(["src/run_pipeline.py", "--image", "runs/geo/%s_sat.png" % stem, "--ckpt", CKPT,
             "--device", DEVICE, "--out", "runs/geo", "--thr", "0.35", "--max-gap", "95", "--ang-tol", "50",
             "--osm"] + dem + tta)
        sub(["src/export_web_geo.py", "--stem", stem, "--label", "Live · %.4f,%.4f" % (lat, lon), "--no-manifest"])
        # tidy heavy intermediates for ephemeral live tiles (page only needs web/data/<stem>.{json,jpg})
        if stem.startswith("live_"):
            for suf in ("_sat.png", "_mask.png", "_graph.gpickle", "_report.json", "_pipeline.png", "_geo.json", "_dem.npy"):
                try:
                    os.remove(os.path.join(ROOT, "runs", "geo", stem + suf))
                except OSError:
                    pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", port), Handler) as httpd:
        print("Route Resilience server: web/ + /api/analyze on http://localhost:%d" % port)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
