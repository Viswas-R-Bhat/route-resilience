# Route Resilience v2 — Domain Generalization Design (2026-07-14)

## Goal
Huge, measured improvement of the deployment-domain metrics (OSM Topological-Accuracy benchmark,
the PS-4 grading rubric) — not another ~5% inference tweak. Secondary goal: error-free pipeline
(tests + reproducible benchmark loop).

## Baseline (5 Bengaluru sectors, Esri z18, measured in runs/geo/*_report.json)
| Sector | OSM recall | F1 | routing success | LCC after heal |
|---|---|---|---|---|
| blr_hsr | 0.65 | 0.65 | 0.67 | 0.90 |
| blr_jayanagar | 0.54 | 0.60 | 0.38 | 0.60 |
| blr_rajajinagar | 0.65 | 0.66 | 0.68 | 0.81 |
| blr_res | 0.52 | 0.57 | 0.33 | 0.46 |
| blr_sector | 0.58 | 0.60 | 0.58 | 0.74 |

Average F1 ≈ 0.62, routing success ≈ 0.53. DeepGlobe val IoU 0.605 is fine — wrong domain.

## Root cause
Model trained on DeepGlobe (rural/suburban, non-Indian fabric, different sensor); deployed on
Esri z18 Indian cities. All prior gains (hysteresis, canopy healing, thr 0.35) were inference-side
compensation. The fix is training on the deployment domain.

## Decision (user-approved)
1. **OSM-supervised fine-tuning** on Esri z18 imagery of diverse Indian geography — canopy
   (Kerala), desert (Jaipur/Jodhpur), plains (Punjab/Lucknow), dense metro (Mumbai/Delhi),
   planned grids (Chandigarh/Chennai), mixed (Pune/Hyderabad/Guwahati), rural Karnataka.
   Masks rasterized from OSM ways with per-class widths (Massachusetts-benchmark method:
   real imagery, real map ground truth). **The 5 benchmark sector bboxes are excluded from
   training. Validation = tiles from cities never trained on. The 5 sectors are the untouched
   final test.**
2. **Architecture: fine-tune existing U-Net/ResNet34 from models/best.pt.** No new architecture:
   converges in few epochs, known-stable, drops into pipeline unchanged, no wasted GPU
   (user constraint). Loss: Dice + BCE + clDice (topology-aware; already in losses.py).
3. **Probability-guided healing**: replace straight-line MST bridges with least-cost paths
   through the (1 − prob) cost field; accept only if evidence along path clears a floor.
4. **Benchmark loop** (`src/improve_loop.py`): one command re-runs pipeline + OSM benchmark on
   all 5 sectors → scoreboard. Every change measured (no-fake-data rule).
5. **Tests**: pytest over P2–P4 invariants + benchmark math; CPU-only.

## Rejected alternatives (considered, with reasons)
- New encoder / SegFormer / SAM / foundation models: 6 GB VRAM, 5–10× GPU time to match a
  fine-tuned baseline on modest data; prior EfficientNet run collapsed. Escalate only if loop plateaus.
- Direct graph predictors (RNGDet++/Sat2Graph/RoadTracer): research-grade rewrite. No.
- Same-encoder ensembles: already measured negative (runs/ensemble/FINAL_MODEL.md).
- OSM conflation into the output graph: copying the answer key — display-only at most.
- Persistent-homology topo loss: too slow. clDice gives the topology signal cheaply.

## Environment (rebuilt 2026-07-14 — old conda env + route_data were deleted)
- venv `C:/Users/VISWAS/route_env` (Python 3.13, torch cu126, smp, albumentations, sknw…).
- Data/checkpoints outside OneDrive: `C:/Users/VISWAS/route_data/` (recreated).
- Checkpoint source of truth: `models/best.pt` in-repo (Git LFS).

## Success criteria
- OSM buffered F1 and routing success improve by ≳ +0.10 absolute on the 5-sector average
  (target 0.75+ each), measured by the same benchmark protocol as baseline.
- No regression on blr_hsr (the current best sector).
- All tests green; pipeline runs end-to-end error-free on a fresh clone.

## Risks
- OSM label noise / registration offset: per-class width masks tolerate ±few px; benchmark
  already estimates registration offset; QC gallery before training.
- Overpass rate limits: on-disk cache + polite backoff.
- Under-mapped OSM areas would teach "miss roads": skip tiles with near-zero OSM road density.
