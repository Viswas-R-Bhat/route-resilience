# Route Resilience — Project Deep-Dive Report

**Project:** Route Resilience (ISRO BAH 2026, Problem Statement 4)
**Scope:** Occlusion-robust road extraction from satellite imagery → topologically-healed routable graph → criticality & resilience stress-test → web dashboard.
**Audience:** Project manager / technical reviewer.
**Source of truth:** This report is generated from the actual code in `src/`, not from the original spec. Where the shipped build differs from the pre-build spec (`CLAUDE_CODE_CONTEXT.md`), the code wins.

---

## 1. Executive summary

Indian metro road networks, seen from space, are **fragmented** (tree canopy, shadows, clouds break roads into disconnected pieces) and analysts have no easy way to know **which intersections are load-bearing**. Route Resilience is a 4-stage pipeline that:

1. Extracts roads from a satellite tile with a CNN trained to *see through* occlusion (P1).
2. Converts the road mask into a routable graph of intersections and edges (P2).
3. **Heals** the breaks the occlusion left behind, using three independent evidence channels to avoid inventing fake roads (P3).
4. Ranks the critical "gatekeeper" intersections and runs disaster stress-tests (node failure, flood) to produce a **Resilience Index** (P4).

Each stage writes a clean artifact the next consumes, so any stage can be run, inspected, or swapped independently.

**Headline performance (real held-out DeepGlobe, 934 tiles never seen in training):**

| Metric | Value | What it means |
|---|---|---|
| Road IoU | **0.605** | Standard pixel overlap vs ground truth |
| Dice | **0.754** | Overlap, imbalance-robust |
| Relaxed-IoU (3 px buffer) | **0.770** | Overlap allowing minor alignment shift |
| **Occlusion-recall** | **0.63** | Roads recovered *inside deliberately hidden regions* — the core claim |

---

## 2. System architecture (stage-by-stage)

```
 REAL satellite tile (RGB .jpg)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ P1  SEGMENTATION   model.py · predict.py · losses.py         │
│     U-Net/ResNet34 → soft prob map → hysteresis mask         │
│     IN : HxWx3 image          OUT: binary mask + soft prob   │
└─────────────────────────────────────────────────────────────┘
        │  {stem}_mask.png  (+ in-memory soft prob)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ P2  SKELETON → GRAPH   phase2_skeleton.py · phase2_graph.py  │
│     close→denoise→skeletonize→sknw→prune→merge junctions     │
│     IN : binary mask          OUT: NetworkX graph (nodes/edges)│
└─────────────────────────────────────────────────────────────┘
        │  G (nodes=intersections, edges=road segments, weight=px length)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ P3  HEALING   phase3_heal.py                                 │
│     Union-Find + shortest-gap-first bridging,                │
│     3 evidence channels (prob / canopy / geom) + T-junctions │
│     IN : graph + RGB + soft prob  OUT: healed graph H        │
└─────────────────────────────────────────────────────────────┘
        │  H (+ heal_kind tags, connectivity report)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ P4  ANALYSIS   phase4_analysis.py                            │
│     betweenness gatekeepers · node-ablation Resilience Index │
│     · demand-weighting · flood/DEM stress test              │
│     IN : healed graph H       OUT: report.json + figures     │
└─────────────────────────────────────────────────────────────┘
        │  {stem}_report.json · {stem}_graph.gpickle · {stem}_pipeline.png
        ▼
   web/ dashboard (Leaflet map + three.js 3D scene)
```

Orchestrated end-to-end by `run_pipeline.py`; served live (model in a locked subprocess) by `server.py`.

---

## 3. Phase 1 — Occlusion-robust segmentation

**Goal:** turn an RGB satellite tile into a binary road mask, robust to canopy/shadow/cloud occlusion.

### 3.1 Model architecture & logic
- **Architecture:** U-Net with a **ResNet34 encoder**, ImageNet-pretrained, via `segmentation-models-pytorch` (`model.py`). Single binary road class; outputs raw **logits** (sigmoid applied inside loss/metrics for `BCEWithLogits` numerical stability).
- **Why this design:**
  - ResNet34 ImageNet weights give strong low-level edge/texture features that transfer to thin road structures.
  - U-Net **skip connections** fuse shallow (texture) and deep (context) features — the multi-scale fusion that lets the model *infer road continuity under occlusion* rather than just trace visible asphalt.
- **Configurable input channels** (`augment.py build_channels`): base is RGB; can append HSV (`hue/sat/val`), Excess-Green vegetation index (`exg`), or NIR (`nir`, zero-filled for RGB-only sources like DeepGlobe/Esri). Default ships RGB.

### 3.2 Loss — why three terms (`losses.py`)
Roads are only ~2–5% of pixels, so plain BCE under-segments. The shipped `CombinedLoss` = **0.5·Dice + 0.3·BCE + 0.2·Connectivity**:
| Term | Role |
|---|---|
| Dice | Overlap-based, robust to the heavy class imbalance |
| BCE | Per-pixel calibration (stable `BCEWithLogits`) |
| Connectivity | Soft-Dice between *dilated* prediction and *dilated* target — rewards continuous road structure (a lightweight topological proxy that directly helps occlusion recovery) |

Two stronger losses are available but not shipped by default: **clDice** (centerline-Dice, penalizes a single broken pixel far more than area Dice) via `--loss cldice`, and a **Lovász+Dice** loss for the heavyweight model.

### 3.3 The signature trick — synthetic-occlusion augmentation (`augment.py`)
`CoarseDropout` blacks out 2–8 random patches of the **real** image while leaving the **mask untouched** (`fill_mask=None`). This teaches the model: *the road still exists under the black patch*. It augments real data; it does not fabricate it. Standard geometric/photometric augs (flips, affine, brightness/CLAHE, noise, blur) round out the pipeline.

### 3.4 Inference recipe (the "config" that ships, `predict.py`)
1. **Overlapping 512-px tiling + Hann-window blending** — large images are tiled with 64-px overlap; Hann weighting removes tile seams.
2. **8-way D4 Test-Time Augmentation** — average sigmoid probability over 4 rotations × {no-flip, h-flip}, each mapped back to input space.
3. **Hysteresis (double-threshold) extraction** — keep confident road *seeds* (prob ≥ `thr_hi` = 0.475), grow into faint pixels (prob ≥ `thr_lo`) **only where connected to a seed**. Under the adaptive canopy mask the low threshold drops further — roads under trees are the faintest signal of all. Recovers occluded continuations while dropping isolated low-confidence blobs.
4. **Light morphological close (5×5 ellipse)** to seal pinholes.
5. *(Optional)* `--multiscale`: run inference at scales {0.6…1.6}, fuse prob maps (max = recall-first). Recovers roads whose width sits outside the 1× receptive field. Roads fully under canopy stay invisible at all scales (an RGB limit, not a scale issue) — those are P3's job.

**P1 input/output**
| | |
|---|---|
| Input | RGB satellite tile (any size), trained checkpoint |
| Output | `{stem}_mask.png` (binary) + in-memory soft probability map (handed to P3) |

### 3.5 Training setup
DeepGlobe only, 0.85/0.15 split, seed 42 (5292 train / 934 held-out val). AdamW, lr 3e-4, cosine schedule, AMP mixed precision (fits 6 GB VRAM), batch 8, early stop patience 8. Best epoch 22/25.

---

## 4. Phase 2 — Skeleton → routable graph

**Goal:** convert the binary road mask into a clean weighted graph where nodes = intersections/endpoints and edges = road segments.

### 4.1 Skeletonization (`phase2_skeleton.py`)
Order matters (skeletonizing a noisy/thick mask creates spurious branches):
1. **Morphological close** (5×5) — bridge tiny gaps.
2. **Remove small objects** (< 80 px) — drop noise blobs.
3. **Zhang-Suen skeletonize** → 1-pixel-wide centerline.

### 4.2 Graph construction & cleanup (`phase2_graph.py`)
- `sknw` traces the skeleton into a network: nodes at degree≠2 points (intersections/endpoints), straight runs become single edges. **Edge weight = geometric pixel length** of the traced centerline (so later shortest paths are by distance, not hop count).
- **Two artifact cleanups:**
  - **Iterative spur pruning** (`min_spur_px=12`): drop short dangling branches; repeat until stable, because removing one spur can expose another. A short edge *between two real junctions* is protected (degree-1 test).
  - **Junction de-duplication** (`merge_radius_px=6`): sknw often splits one real X/T/Y intersection into a tight cluster of nodes; left alone they inflate node counts and **skew betweenness** (and therefore the whole gatekeeper ranking). Union-Find contracts them to the cluster centroid, keeping the shortest of any parallel edges.

**P2 input/output**
| | |
|---|---|
| Input | Binary road mask |
| Output | NetworkX `Graph` G — `nodes` (pos, yx), `edges` (weight, length_px, pts). `graph_stats()` reports nodes/edges/components/LCC fraction/total length |

---

## 5. Phase 3 — Topological healing (the differentiator)

**Goal:** reconnect road fragments that occlusion broke apart — **without inventing roads that don't exist.**

### 5.1 Core strategy (`phase3_heal.py`)
- **MST-style, shortest-gap-first** bridging of dead-ends (degree-1 nodes).
- **Union-Find (Disjoint-Set)** guarantees we only ever connect *different* components → no cycles, no redundant bridges.
- **Geometric alignment is ALWAYS required**: a bridge must continue the road's local direction (PCA over nearby nodes) at *both* ends — accepted via straight collinearity **or** a trajectory-corridor test (handles roads gently curving into canopy). We never invent illogical sharp turns.

### 5.2 Three evidence channels — is this gap occlusion or a real dead-end?
The evidence only **widens the gap/angle budget**; it never bridges an unaligned pair.

| Channel | Trigger | Logic | Confidence |
|---|---|---|---|
| **prob** (strongest) | Span over the segmenter's soft map averages ≥ 0.20 | The model still believes there's road *below the mask threshold* (thin canopy/shadow). Its own evidence → full relaxed gates even in saturated scenes | high / med |
| **canopy** | Span lies mostly over the dense Excess-Green canopy mask | Likely occlusion → relaxed gates | low / vlow |
| **geom** | Neither signal | Visible bare ground; a road that just stops is probably a real end → strict base gates | med |

**Anti-fabrication safeguard (the Perumbavoor problem):** the canopy mask threshold is `max(absolute floor, in-tile percentile)`. In a rain-forest tile where the *whole frame* reads as green, the percentile rises so only the **densest** canopy qualifies — preventing the "everything is canopy, relax everywhere" failure. When a scene is **canopy-saturated** (`green_frac ≥ 0.6`), the canopy path is throttled and bridges are tagged low-confidence rather than fabricated.

### 5.3 Two healing topologies (run in sequence)
1. **Endpoint-pair** — bridge two dead-ends that continue each other's trajectory.
2. **T-junction** — connect a remaining dead-end to the **nearest point on another component's edge**, splitting that edge to create a real node. This recovers *side-street-meets-through-road* gaps that pair-healing structurally cannot (the meeting point is mid-edge, with no node to pair).

Every synthetic edge is tagged `healed=True`, `heal_kind` ∈ {prob, canopy, geom}, a `conf`, plus `canopy_frac` and `road_prob` — so the dashboard can color-code exactly what was recovered and why.

**P3 input/output**
| | |
|---|---|
| Input | Graph G + RGB tile + soft prob map |
| Output | Healed graph H (tagged synthetic edges) + connectivity report |

### 5.4 Metric — Connectivity Ratio (`connectivity_report`)
Reports `components_before/after`, `components_merged`, and **LCC fraction before→after** (share of nodes in the largest connected component). This is the direct measure of healing impact: more nodes routable from one another.

---

## 6. Phase 4 — Criticality & resilience analysis

**Goal:** find the intersections whose failure hurts the network most, and quantify how the network degrades under disaster.

### 6.1 Gatekeeper identification (`phase4_analysis.py`)
- **Betweenness centrality** (weighted by road length, exact Brandes; k-sampled approximation above 450 nodes for ~4× speed — only the *ranking* is approximate, never the Resilience Index). A high-betweenness node lies on many shortest paths; its loss strands traffic.
- **Node classification** into critical / important / normal by the 90th / 75th percentile of *non-zero* betweenness (a zero-betweenness node is never critical).
- **Service-criticality** (`demand_weighted_criticality`) = betweenness × local road density (a demand proxy — busy built-up areas have denser road tangles). A node ranks high only if it is *both* structurally central *and* serves a busy area, so its failure strands **real travel**, not just graph-theoretic paths.

### 6.2 Resilience Index — node-ablation stress test
- Repeatedly remove the **highest-betweenness node**, recompute **weighted global efficiency** (mean of 1/shortest-path-distance over all node pairs), and report **R = eff(perturbed) / eff(baseline)**. R=1 → no degradation; R→0 → collapse.
- **Subtlety that makes R honest:** the efficiency denominator is fixed to the *baseline* node count (`n_ref`). Otherwise removing nodes would shrink the denominator and could *inflate* efficiency. Fixed `n_ref` → R is bounded in [0,1] and monotonic.

### 6.3 Flood stress test (optional, needs a DEM)
`flood_simulation` raises water from the ~2nd to ~88th elevation percentile; at each level all nodes below the waterline submerge, and R + LCC fraction are recomputed against the dry baseline. The real-world analogue of betweenness ablation: *which routes survive a flood.*

**P4 input/output**
| | |
|---|---|
| Input | Healed graph H (+ optional DEM `.npy`) |
| Output | `{stem}_report.json` (gatekeepers, service-gatekeepers, resilience curve, flood curve), `{stem}_graph.gpickle` (with centrality attrs), `{stem}_pipeline.png` composite |

---

## 7. Metrics reference (and which one checks "ground reality")

All metrics are computed on a **real held-out split**; none are fabricated (`metrics.py`).

| Metric | Definition | Purpose |
|---|---|---|
| **IoU** | Intersection / Union of road pixels vs ground-truth mask | Standard overlap vs **ground truth** |
| **Dice** | 2·overlap / (pred + target) | Imbalance-robust overlap |
| **Relaxed-IoU** | Buffered IoU — a predicted pixel within 3 px of GT counts as a hit | Honest overlap when labels are slightly misaligned (the metric to lead with for noisy/OSM ground truth) |
| **Occlusion-recall** | TP/(TP+FN) over road pixels **inside a controlled occluded region** | **Directly measures the occlusion-robustness claim** |

**Ground-reality metric — direct answer:** ground truth is the held-out label masks. Overall fidelity to ground truth is measured by **IoU / Dice**, with **Relaxed-IoU** the fairer variant when labels don't pixel-align (e.g. OSM-derived masks). The metric that validates the *project's core promise* is **occlusion-recall**: an occlusion mask is generated by us at eval time, so we know *exactly* which real pixels were hidden, and we measure how many roads the model recovered *there*. Current value: **0.63**.

Network-level metrics: **Connectivity Ratio** (P3, components merged + LCC growth) and **Resilience Index** (P4, efficiency retained under node/flood ablation).

---

## 8. Honest limitations & decisions on record

- **No ensemble gain (recorded negative result).** A 3-model soft-vote (U-Net 0.605, DeepLabV3+ 0.598, U-Net++ 0.587) scored 0.604 — *below* the best single model; the greedy selector kept U-Net alone. Cause: shared encoder + training data → correlated errors. Honestly reported as no gain. To make an ensemble pay off later: add a genuinely diverse encoder (EfficientNet/transformer) and fully converge U-Net++.
- **RGB ceiling.** Roads fully under dense canopy are invisible at every scale and below every probability threshold — P1/multiscale cannot recover them; P3's canopy channel infers them geometrically but tags them low-confidence rather than asserting them.
- **Domain shift to ISRO imagery.** Numbers above are on DeepGlobe (high-res aerial). The PS4 evaluation uses Sentinel-2 (10 m) / LISS-IV (5.8 m) / Cartosat-3 with OSM-derived ground truth; expect raw IoU to move and **Relaxed-IoU + occlusion-recall** to be the fairer headline there.
- **Demand proxy.** Service-criticality uses road density as a demand stand-in; swap in WorldPop/OSM-POI for a true demand surface.

---

## 9. How to run (reference)

```bash
# Full pipeline on one tile (image → mask → graph → healed → criticality/resilience)
python src/run_pipeline.py --image <x_sat.jpg> --ckpt <best.pt> --device cuda --out runs/pipeline
python src/run_pipeline.py --mask  <x_mask.png> --out runs/pipeline          # skip P1
python src/run_pipeline.py --image <x_sat.jpg> --ckpt <best.pt> --multiscale # recover small roads

# P1 inference only · web app · training
python src/predict.py --ckpt <best.pt> --image <x_sat.jpg> --out runs/phase1/pred
python src/server.py 8766            # → http://localhost:8766
python src/train.py                  # config-driven; --epochs 3 --limit 600 for a smoke loop
```

> Environment note: everything runs on the `route` conda env python (`C:/Users/VISWAS/anaconda3/envs/route/python.exe`); the bare `python` on PATH is a Windows stub with none of the deps. `build_deck.py` is the one exception (base anaconda python, needs `python-pptx`).

---

*Generated from the live `src/` codebase. Config single-source-of-truth: `src/config.yaml`. Production model + ensemble write-up: `runs/ensemble/FINAL_MODEL.md`.*
