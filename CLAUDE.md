# CLAUDE.md

Working guidance for this repo. For the project pitch see [README.md](README.md). The big
`CLAUDE_CODE_CONTEXT.md` is the *original pre-build spec* — it describes a layout
(`phase1_segmentation/`, Streamlit, folium) that was **not** the one built. Trust this file
and `README.md` over it.

## What this is
**Route Resilience** (ISRO BAH 2026, PS4). Occlusion-robust road extraction from satellite
imagery → topologically-healed routable graph → criticality + resilience stress-test → web
dashboard. Pipeline is staged P1→P4; each phase writes a clean artifact the next consumes.

## Python environment — read first
Two different interpreters; the bare `python` on PATH is a Windows stub with none of the deps.

- **Everything** (training, pipeline, server, scripts) uses the `route` conda env:
  `C:/Users/VISWAS/anaconda3/envs/route/python.exe` (torch, smp, sknw, scipy, skimage,
  opencv, networkx).
- **Exception — `build_deck.py`** uses **base** anaconda python
  `C:/Users/VISWAS/anaconda3/python.exe` (needs `python-pptx`, not in the route env).

## Run
```bash
# Web app + live analysis API (stdlib http; model runs in a route-env subprocess)
python src/server.py 8766            # -> http://localhost:8766   (.claude launch: "route-web")

# Full pipeline on one tile: image -> mask -> graph -> healed -> criticality/resilience
python src/run_pipeline.py --image <x_sat.jpg> --ckpt <best.pt> --device cuda --out runs/pipeline
python src/run_pipeline.py --mask  <x_mask.png> --out runs/pipeline      # skip P1
#   note: default --device is cpu so a run won't disturb a GPU training job.

# Train (config-driven) / fast smoke loop
python src/train.py
python src/train.py --epochs 3 --limit 600

# Full-tile inference only (overlapping 512 tiles + Hann blending)
python src/predict.py --ckpt <best.pt> --image <x_sat.jpg> --out runs/phase1/pred

# Rebuild the idempotent 10-slide pitch deck (BASE python, run from repo root)
python build_deck.py
```

## Layout
- `src/config.yaml` — single source of truth for P1 hyperparameters / paths / eval thresholds.
- `src/` pipeline:
  - **P1 segmentation** — `model.py` (U-Net/ResNet34 via smp), `dataset.py`, `augment.py`
    (synthetic-occlusion aug), `losses.py` (Dice+BCE+connectivity), `train.py`, `predict.py`, `metrics.py`
  - **P2 skeleton→graph** — `phase2_skeleton.py`, `phase2_graph.py` (sknw → NetworkX)
  - **P3 healing** — `phase3_heal.py` (Union-Find + MST + angular/soft-prob/T-junction bridging)
  - **P4 analysis** — `phase4_analysis.py` (betweenness gatekeepers, node-ablation Resilience Index, flood/DEM)
  - orchestration `run_pipeline.py`; geo fetch `fetch_geo_tile.py` / `fetch_dem_tile.py`;
    web export `export_web.py` / `export_web_geo.py`; live server `server.py`
- `web/` — standalone dashboard: `index.html`, `app.js`, `mapview.js` (Leaflet), `scene.js`
  (three.js), `styles.css`; exported tiles in `web/data/`.
- `runs/` — pipeline outputs by phase; `runs/ensemble/FINAL_MODEL.md` is the production-model
  + metrics + ensemble study write-up.
- `models/best.pt` — committed checkpoint copy.

## Data & checkpoints (outside the repo)
Datasets and run outputs live under `C:/Users/VISWAS/route_data/` — deliberately **outside**
the OneDrive-synced project dir so ~GBs of imagery/checkpoints aren't cloud-synced.
- Default pipeline/server checkpoint: `C:/Users/VISWAS/route_data/runs/phase1_full/best.pt`
  (also committed at `models/best.pt`).
- DeepGlobe train tiles: `C:/Users/VISWAS/route_data/deepglobe/train` (`{id}_sat.jpg` + `{id}_mask.png`).

## Conventions & gotchas
- Real data only — every reported metric is on a real held-out split; no fabricated numbers.
  Current: Road IoU 0.605 · Dice 0.754 · Relaxed-IoU 0.770 · Occlusion-recall 0.63 (U-Net/ResNet34
  + D4 TTA, threshold-tuned). Multi-arch ensemble honestly reported as **no gain**.
- Inference uses 8-way D4 TTA + tuned threshold (`eval.threshold` in config); `run_pipeline.py`
  supports `--multiscale` to recover roads missed at native scale.
- gitignored: checkpoints (`*.pt`), graphs (`*.gpickle`), source/intermediate tile PNGs,
  ephemeral `web/data/live_*` (regenerated on demand), `.qa/`, the blank deck template.
- `server.py` serializes pipeline runs with a lock so concurrent `/api/analyze` calls don't
  fight over the GPU; `live_*` tiles are cleaned up after export.
