# Route Resilience

**Geospatial mobility intelligence** — occlusion-robust road extraction from satellite imagery, turned into a routable graph for criticality and disaster-resilience analysis.

Standard satellite road extraction breaks under tree canopy, shadows, and buildings, producing fragmented masks that are useless for routing. Route Resilience extracts roads with a context-aware model, **heals** the topology into a connected weighted graph, finds the **gatekeeper intersections** (single points of failure), and lets you **stress-test** the network — disable nodes (flood / accident / closure) and watch resilience, rerouting, and travel-time degrade live.

## Pipeline
1. **Segmentation** — U-Net / ResNet34 (segmentation-models-pytorch), Dice + BCE + connectivity loss, synthetic-occlusion augmentation, 8-way D4 test-time augmentation.
2. **Skeletonization → graph** — morphological thinning + `sknw` → NetworkX weighted graph.
3. **Topological healing** — Union-Find + MST + angular alignment bridge occlusion gaps.
4. **Criticality + resilience** — betweenness-centrality gatekeepers; node-ablation Resilience Index, rerouting, and travel-time impact.

## Validation (real held-out DeepGlobe, 934 tiles)
Road **IoU 0.605** · Dice 0.754 · Relaxed-IoU 0.770 · Occlusion-recall 0.63 (U-Net/ResNet34 + D4 TTA, threshold-tuned). A multi-architecture soft-voting ensemble was evaluated and honestly reported as **no gain** (correlated same-encoder members) — see `runs/ensemble/FINAL_MODEL.md`.

## Web app
Premium dark dashboard (`web/`): a Leaflet map of the real road network at true lat/lon over satellite imagery, criticality heatmap, gatekeeper list, interactive stress-test slider, rerouting, and a **live "Analyze any location"** search that fetches imagery and runs the full pipeline on demand.

```bash
# requires the project env (PyTorch + smp, NetworkX, sknw, Pillow)
python src/server.py 8766        # serves the app + /api/analyze
# open http://localhost:8766
```

## Layout
- `src/` — model, training, pipeline (P1–P4), evaluation, web export, live server
- `web/` — standalone dashboard (Leaflet + ECharts + three.js), real exported tiles
- `runs/ensemble/FINAL_MODEL.md` — production model + metrics + ensemble study

Built on real satellite imagery (DeepGlobe for the model; Esri World Imagery for geolocated map tiles). No fabricated data.
