# Route Resilience

**Geospatial mobility intelligence** — occlusion-robust road extraction from satellite imagery, turned into a routable graph for criticality and disaster-resilience analysis.

Standard satellite road extraction breaks under tree canopy, shadows, and buildings, producing fragmented masks that are useless for routing. Route Resilience extracts roads with a context-aware model, **heals** the topology into a connected weighted graph, finds the **gatekeeper intersections** (single points of failure), and lets you **stress-test** the network — disable nodes (flood / accident / closure) and watch resilience, rerouting, and travel-time degrade live.

## Pipeline
1. **Segmentation** — U-Net / ResNet34 (segmentation-models-pytorch), pre-trained on DeepGlobe and **domain-fine-tuned on India-wide Esri z18 imagery with OSM-supervised masks** (22 diverse sites: Kerala canopy, Rajasthan desert, plains, metros, planned grids, rural Karnataka; Dice + BCE + clDice topology loss, synthetic-occlusion augmentation, 8-way D4 TTA).
2. **Skeletonization → graph** — morphological thinning + `sknw` → NetworkX weighted graph.
3. **Probability-guided topological healing** — bridges follow least-cost paths through the model's soft output (faint road evidence under canopy), gated by Union-Find + angular alignment; open-ground bridges with zero image evidence are vetoed.
4. **Criticality + resilience** — betweenness-centrality gatekeepers; node-ablation Resilience Index, rerouting, travel-time impact, flood-DEM stress test.

## Validation — deployment domain (5 held-out Bengaluru sectors vs OpenStreetMap)
Measured by the OSM Topological-Accuracy benchmark (`src/osm_benchmark.py`, centerline
completeness/correctness + routing fidelity). Training tiles exclude these sectors;
model selection used only unseen-city validation tiles.

| | v1 (DeepGlobe model) | **v2 (domain fine-tune + prob-healing)** |
|---|---|---|
| Road recall (completeness) | 0.62 | **0.88** |
| Precision (correctness) | 0.78 | 0.72 |
| F1 | 0.69 | **0.79** |
| Routing success | 0.53 | **0.99** |
| Median path-length error | 26.5 % | **6.9 %** |
| Largest-connected-component | 0.70 | **0.995** |

Reproduce: `python src/improve_loop.py --ckpt models/best.pt --tag mytest` (full scoreboards
in `runs/loop/`). Prior DeepGlobe-domain metrics + the honest no-gain ensemble study:
`runs/ensemble/FINAL_MODEL.md`. v1 checkpoint kept at `models/best_v1_deepglobe.pt`.

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
