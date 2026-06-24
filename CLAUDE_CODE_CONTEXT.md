# ISRO BAH 2026 — Problem Statement 4
## Route Resilience: Full Technical Context for Claude Code

> Feed this entire document into Claude Code (Opus 4.8) as the initial context before starting any implementation. Every architectural decision, parameter, library choice, and data flow is specified here. Do not deviate unless a specific technical blocker forces it — in that case, document the deviation.

---

## Project Overview

An end-to-end pipeline that takes raw satellite imagery (Sentinel-2 / LISS-4 / Cartosat-3) and produces:
1. A clean, occlusion-robust binary road segmentation mask
2. A skeletonized road centerline
3. A topologically-healed, routable weighted graph
4. Network criticality analysis (Gatekeeper Nodes + Edge vulnerabilities)
5. A stress-test simulation with a Resilience Index
6. A premium dark-themed interactive Streamlit + Leaflet.js dashboard

The pipeline must be modular — each phase outputs a clean artifact that the next phase consumes. This allows parallel development and independent testing.

---

## Repository Structure

```
route-resilience/
├── data/
│   ├── raw/                  # Raw satellite tiles (GeoTIFF)
│   ├── processed/            # Normalized, tiled patches
│   ├── masks/                # Ground truth binary masks
│   └── graphs/               # Serialized NetworkX graphs (.gpickle)
├── phase1_segmentation/
│   ├── dataset.py            # PyTorch Dataset class
│   ├── augmentations.py      # Albumentations pipeline
│   ├── model.py              # U-Net + ResNet34 via SMP
│   ├── losses.py             # Combined loss function
│   ├── train.py              # Training loop
│   ├── predict.py            # Inference on full tiles
│   └── metrics.py            # IoU, Dice, Occlusion-Recall
├── phase2_skeletonization/
│   ├── skeletonize.py        # Binary mask → centerline
│   └── graph_builder.py      # Centerline → raw NetworkX graph
├── phase3_healing/
│   ├── union_find.py         # Disjoint Set implementation
│   └── mst_healer.py         # MST + angular alignment gap bridging
├── phase4_analysis/
│   ├── centrality.py         # Betweenness + Edge Betweenness
│   ├── ablation.py           # Node removal simulation
│   └── resilience.py         # Resilience Index calculation
├── phase5_dashboard/
│   ├── app.py                # Main Streamlit app
│   ├── map_component.py      # Leaflet.js via folium / st_folium
│   ├── style.css             # Custom dark theme CSS
│   └── components/           # Individual UI panel components
├── utils/
│   ├── geo_utils.py          # Rasterio / GDAL helpers
│   ├── tile_utils.py         # Image tiling / reassembly
│   └── viz_utils.py          # Matplotlib visualization helpers
├── configs/
│   └── config.yaml           # All hyperparameters in one place
├── requirements.txt
└── run_pipeline.py           # Full end-to-end runner
```

---

## config.yaml — Single Source of Truth

```yaml
data:
  tile_size: 512              # px — larger tiles preserve more context
  tile_overlap: 64            # px — overlap for seamless reassembly
  num_channels: 3             # RGB or false-color composite
  train_val_split: 0.85

model:
  architecture: unet
  encoder: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1                  # Binary: road / not-road
  activation: sigmoid

training:
  epochs: 50
  batch_size: 8
  learning_rate: 3.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine_annealing
  early_stopping_patience: 10
  mixed_precision: true       # Use torch.cuda.amp for 3050 VRAM efficiency

loss:
  dice_weight: 0.5
  bce_weight: 0.3
  connectivity_weight: 0.2

healing:
  max_gap_distance_px: 50     # Max Euclidean gap to bridge
  angular_tolerance_deg: 30   # Max deviation from road trajectory
  min_road_segment_px: 20     # Remove noise below this length

analysis:
  top_k_nodes: 20             # Top gatekeeper nodes to highlight
  ablation_steps: 10          # How many nodes to progressively remove

dashboard:
  map_center: [12.9716, 77.5946]   # Bengaluru
  default_zoom: 13
  tile_layer: "CartoDB.DarkMatter"  # Dark base map
```

---

## Phase 1: Occlusion-Robust Segmentation

### 1.1 Dataset Strategy

**Primary training data (open, zero-cost):**
- SpaceNet Roads Dataset (AOI 2 Vegas, AOI 3 Paris, AOI 5 Khartoum) — highest quality road labels
- DeepGlobe Road Extraction Dataset — diverse terrains
- OpenStreetMap road vector layers clipped to Bengaluru — used as pseudo-labels for LISS-4 tiles

**Data loading:**
- All images are GeoTIFF. Use `rasterio` to read, always read as `float32`, normalize to `[0, 1]` using per-dataset band statistics (not per-image, to maintain consistency).
- Tile large images into `512×512` patches with `64px` overlap using a sliding window. Store as `.npy` arrays for fast loading.
- Ground truth masks are binary `uint8` arrays: `1 = road`, `0 = background`.

```python
# dataset.py key structure
class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        # image_paths: list of .npy tile paths
        # mask_paths: corresponding binary mask .npy paths
        # transform: Albumentations Compose pipeline

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])  # (H, W, C) float32
        mask = np.load(self.mask_paths[idx])    # (H, W) uint8
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        # Convert to tensors: image (C, H, W), mask (1, H, W)
        return image.transpose(2,0,1), mask[None]
```

### 1.2 Augmentation Pipeline

The key differentiator is **synthetic occlusion simulation** — we teach the model to reconstruct roads it cannot see.

```python
# augmentations.py
import albumentations as A

def get_train_transforms():
    return A.Compose([
        # Geometric
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),

        # Photometric — simulates illumination/seasonal variation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30),
            A.CLAHE(clip_limit=4.0),
        ], p=0.7),

        # Noise — simulates sensor noise
        A.GaussNoise(var_limit=(10, 50), p=0.3),

        # SYNTHETIC OCCLUSION — critical for our claim
        # Randomly drops rectangular patches to simulate tree canopy / cloud cover
        A.CoarseDropout(
            max_holes=8,
            max_height=64,    # up to 64px × 64px occlusion patches
            max_width=64,
            min_holes=2,
            min_height=16,
            min_width=16,
            fill_value=0,     # fill with black (simulates shadow)
            mask_fill_value=None,  # DO NOT modify the mask — road still exists under occlusion
            p=0.5
        ),

        # Blur — simulates atmospheric haze
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

**Why `mask_fill_value=None` in CoarseDropout is critical:** The image gets blacked out (occluded), but the mask retains the road label. This forces the model to learn that roads exist even when pixels are missing — exactly the occlusion-robustness we claim.

### 1.3 Model Architecture

```python
# model.py
import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    return model
```

**Why ResNet34 + U-Net:**
- ResNet34 encoder pretrained on ImageNet gives strong low-level feature detection (edges, textures) that transfers well to road detection
- U-Net skip connections preserve spatial detail lost in downsampling — critical for thin road structures (often only 2–5px wide in satellite imagery)
- Fits comfortably in 4GB VRAM at batch_size=8 with 512×512 tiles using mixed precision

**The U-Net skip connections serve as our "multi-scale feature fusion"** — features from multiple encoder depths (shallow=texture, deep=context) are fused in the decoder. This is how the model uses surrounding context to infer occluded road continuity.

### 1.4 Loss Function

Standard BCE alone is insufficient — roads are ~2–5% of image pixels (severe class imbalance). We use a combined loss:

```python
# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class ConnectivityLoss(nn.Module):
    """
    Penalizes broken road predictions by comparing connectivity of
    predicted mask vs ground truth mask using morphological operations.
    Lightweight proxy for true topological loss.
    """
    def forward(self, pred, target):
        # Dilate both pred and target, compare overlap
        # Encourages predicted roads to form continuous structures
        pred_dilated = F.max_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_dilated = F.max_pool2d(target, kernel_size=5, stride=1, padding=2)
        return F.binary_cross_entropy(pred_dilated, target_dilated)

class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.5, bce_w=0.3, conn_w=0.2):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.conn = ConnectivityLoss()
        self.w = (dice_w, bce_w, conn_w)

    def forward(self, pred, target):
        return (self.w[0] * self.dice(pred, target) +
                self.w[1] * self.bce(pred, target.float()) +
                self.w[2] * self.conn(pred, target.float()))
```

### 1.5 Training Loop

```python
# train.py key structure
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = torch.cuda.amp.GradScaler()  # Mixed precision for VRAM efficiency

for epoch in range(config.epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(images)
            loss = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    # Validate, save best checkpoint by IoU
```

### 1.6 Inference on Full Tiles

After training, run inference on full satellite images using sliding window with overlap. Merge overlapping predictions using Gaussian weight blending (center pixels weighted higher than edges).

```python
# predict.py
def predict_full_image(model, image_array, tile_size=512, overlap=64):
    """
    Tiles a full image, runs model on each tile, reassembles with
    Gaussian blending to eliminate seam artifacts at tile boundaries.
    Returns: binary mask (H, W) uint8
    """
    # Threshold predictions at 0.5 for binary mask
    # Apply morphological closing (kernel=5) to fill small holes in predicted roads
```

### 1.7 Metrics

```python
# metrics.py
# Standard:
#   - IoU (Intersection over Union)
#   - Dice Score (F1 for segmentation)
# Custom:
#   - Occlusion-Recall: IoU computed ONLY on pixels that were occluded during augmentation
#     (requires storing the occlusion mask from augmentation)
#   - Relaxed IoU (buffer=3px): predicted road pixel counts as TP if within 3px of GT road
```

---

## Phase 2: Skeletonization & Graph Construction

### 2.1 Skeletonization

The binary road mask contains thick blobs. We reduce them to 1-pixel-wide centerlines.

```python
# skeletonize.py
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import gaussian

def mask_to_skeleton(binary_mask):
    # Step 1: Clean the mask
    # Remove isolated noise pixels smaller than min_road_segment_px
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=20)

    # Step 2: Morphological closing to fill small gaps before skeletonizing
    from scipy.ndimage import binary_closing
    closed = binary_closing(cleaned, structure=np.ones((5,5)))

    # Step 3: Zhang-Suen thinning (scikit-image skeletonize uses this)
    skeleton = skeletonize(closed)

    return skeleton.astype(np.uint8)
```

### 2.2 Graph Construction from Skeleton

```python
# graph_builder.py
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

def skeleton_to_graph(skeleton, geo_transform=None):
    """
    Converts a skeleton image to a NetworkX graph.
    
    Node placement rules:
    - A pixel is a NODE if it has ≠ 2 skeleton neighbors (i.e., intersection or endpoint)
    - A pixel is an EDGE if it has exactly 2 skeleton neighbors (straight road segment)
    
    Edge weights:
    - Euclidean length of the road segment (in pixels, or meters if geo_transform provided)
    - Optional: curvature penalty (longer/curvier = higher weight)
    """
    G = nx.Graph()
    
    # Find all skeleton pixels
    ys, xs = np.where(skeleton)
    
    # 8-connectivity neighbor check
    def get_neighbors(y, x, skel):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                    if skel[ny, nx_]:
                        neighbors.append((ny, nx_))
        return neighbors
    
    # Nodes at intersections (3+ neighbors) and endpoints (1 neighbor)
    node_pixels = set()
    for y, x in zip(ys, xs):
        n = len(get_neighbors(y, x, skeleton))
        if n != 2:
            node_pixels.add((y, x))
            G.add_node((y, x), pos=(x, y))  # pos in (col, row) = (x, y) order
    
    # Trace edges between nodes by following skeleton pixels
    # Each edge gets weight = pixel length of the path
    # (Implementation: BFS/DFS from each node pixel along skeleton until next node)
    
    return G
```

---

## Phase 3: Topological Healing

This is the most novel phase. The raw graph from Phase 2 has gaps caused by heavy occlusions the model couldn't fully recover. We bridge these gaps intelligently.

### 3.1 Disjoint Set (Union-Find)

```python
# union_find.py
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_components(self):
        components = {}
        for e in self.parent:
            root = self.find(e)
            components.setdefault(root, []).append(e)
        return list(components.values())
```

### 3.2 MST Healer with Angular Alignment

```python
# mst_healer.py
import numpy as np
import networkx as nx
from itertools import combinations
from union_find import UnionFind

def compute_road_angle(G, node, depth=3):
    """
    Estimates the local road direction at `node` by looking at
    neighbors up to `depth` hops away and fitting a direction vector.
    Returns angle in degrees [0, 180).
    """
    neighbors = list(nx.single_source_shortest_path_length(G, node, cutoff=depth).keys())
    if len(neighbors) < 2:
        return None
    # PCA on neighbor positions to find principal direction
    positions = np.array([G.nodes[n]['pos'] for n in neighbors])
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]
    angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180
    return angle

def heal_graph(G, max_gap_px=50, angular_tolerance=30):
    """
    Bridges disconnected components using MST-inspired gap closing.
    
    Algorithm:
    1. Find all connected components
    2. For each pair of components, find the closest endpoint pair
    3. Check if the bridging edge is angularly consistent with both roads
    4. If yes AND distance < max_gap_px: add the healing edge
    5. Continue until all components are connected (or no valid bridges remain)
    
    Returns: healed graph G with added 'healed' edge attribute on synthetic edges
    """
    uf = UnionFind(list(G.nodes()))
    
    # Get endpoint nodes (degree 1 in graph = road terminus)
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    
    # Build candidate healing edges sorted by distance
    candidates = []
    for n1, n2 in combinations(endpoints, 2):
        if not uf.connected(n1, n2):
            p1 = np.array(G.nodes[n1]['pos'])
            p2 = np.array(G.nodes[n2]['pos'])
            dist = np.linalg.norm(p1 - p2)
            if dist <= max_gap_px:
                candidates.append((dist, n1, n2))
    candidates.sort()  # Process shortest gaps first
    
    healed_edges = 0
    for dist, n1, n2 in candidates:
        if uf.connected(n1, n2):
            continue
        
        # Angular consistency check
        angle1 = compute_road_angle(G, n1)
        angle2 = compute_road_angle(G, n2)
        
        if angle1 is not None and angle2 is not None:
            # Compute angle of the proposed healing edge
            p1 = np.array(G.nodes[n1]['pos'])
            p2 = np.array(G.nodes[n2]['pos'])
            bridge_angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
            
            # Check if bridge angle is consistent with both road directions
            dev1 = min(abs(bridge_angle - angle1), 180 - abs(bridge_angle - angle1))
            dev2 = min(abs(bridge_angle - angle2), 180 - abs(bridge_angle - angle2))
            
            if dev1 > angular_tolerance or dev2 > angular_tolerance:
                continue  # Bridge would create an illogical turn — skip
        
        # Add healing edge
        G.add_edge(n1, n2, weight=dist, healed=True)
        uf.union(n1, n2)
        healed_edges += 1
    
    print(f"Healed {healed_edges} gaps. Components before: {nx.number_connected_components(G)}")
    return G
```

**Why angular alignment matters:** Without it, the MST might bridge two nearby road endpoints that are going in completely different directions — creating a nonsensical sharp turn. The angular check ensures every synthetic edge is plausible as a real road continuation.

---

## Phase 4: Network Analysis & Stress Testing

### 4.1 Centrality Analysis

```python
# centrality.py
import networkx as nx

def compute_centrality(G):
    """
    Betweenness Centrality: fraction of all shortest paths that pass through a node.
    High score = "Gatekeeper Node" — traffic collapses if this node fails.
    
    Weight parameter: uses road LENGTH as edge weight, so shortest paths
    are by distance (meters), not hop count. More realistic for urban routing.
    """
    # Node betweenness — identifies critical intersections
    node_bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # Edge betweenness — identifies critical road SEGMENTS (our differentiator)
    edge_bc = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    
    # Store centrality as node/edge attributes
    nx.set_node_attributes(G, node_bc, 'betweenness')
    nx.set_edge_attributes(G, edge_bc, 'edge_betweenness')
    
    # Rank nodes
    ranked_nodes = sorted(node_bc.items(), key=lambda x: x[1], reverse=True)
    
    return G, ranked_nodes

def classify_nodes(G, top_k=20):
    """
    Classifies nodes into tiers based on betweenness centrality percentile.
    Returns color mapping for dashboard visualization.
    """
    scores = [G.nodes[n].get('betweenness', 0) for n in G.nodes()]
    p75 = np.percentile(scores, 75)
    p90 = np.percentile(scores, 90)
    
    classification = {}
    for node in G.nodes():
        bc = G.nodes[node].get('betweenness', 0)
        if bc >= p90:
            classification[node] = 'critical'    # Red — top 10%
        elif bc >= p75:
            classification[node] = 'important'   # Amber — top 25%
        else:
            classification[node] = 'normal'      # Teal — rest
    
    return classification
```

### 4.2 Node Ablation Simulation

```python
# ablation.py
import networkx as nx
import copy

def ablation_simulation(G, top_k=10):
    """
    Progressively removes the highest-betweenness nodes and measures
    network degradation. Returns a list of (removed_node, resilience_index, 
    largest_component_size, avg_path_length) tuples.
    
    This is the "Stress Test" — simulates flood/accident/construction.
    """
    results = []
    G_sim = copy.deepcopy(G)
    
    # Get initial baseline metrics
    baseline_efficiency = nx.global_efficiency(G_sim)
    baseline_lcc = len(max(nx.connected_components(G_sim), key=len))
    
    # Sort nodes by betweenness (highest first)
    ranked = sorted(
        nx.betweenness_centrality(G_sim, weight='weight').items(),
        key=lambda x: x[1], reverse=True
    )
    
    for i, (node, bc_score) in enumerate(ranked[:top_k]):
        G_sim.remove_node(node)
        
        if nx.number_of_nodes(G_sim) == 0:
            break
        
        current_efficiency = nx.global_efficiency(G_sim)
        current_lcc = len(max(nx.connected_components(G_sim), key=len)) if nx.number_connected_components(G_sim) > 0 else 0
        
        # Resilience Index: ratio of current to baseline efficiency
        # R=1.0 means no degradation, R=0.0 means complete collapse
        resilience_index = current_efficiency / baseline_efficiency if baseline_efficiency > 0 else 0
        
        results.append({
            'step': i + 1,
            'removed_node': node,
            'betweenness_score': bc_score,
            'resilience_index': round(resilience_index, 4),
            'lcc_size': current_lcc,
            'lcc_fraction': current_lcc / baseline_lcc,
            'global_efficiency': current_efficiency,
        })
    
    return results
```

### 4.3 Resilience Index

```
R = global_efficiency(G_perturbed) / global_efficiency(G_baseline)

Where:
  global_efficiency(G) = (1 / n(n-1)) * Σ (1 / d(u,v))  for all u≠v
  
  d(u,v) = shortest weighted path length between u and v
  If u and v are disconnected: d(u,v) = ∞ → contributes 0 to efficiency

R = 1.0   → No degradation (network is fully resilient)
R = 0.5   → 50% efficiency loss (severe degradation)
R → 0.0   → Network near collapse
```

---

## Phase 5: Dashboard (Streamlit + Leaflet.js)

### 5.1 Premium Dark Theme via Custom CSS

Inject into Streamlit using `st.markdown` with `unsafe_allow_html=True`:

```css
/* style.css */
:root {
    --bg-primary: #080A0F;
    --bg-secondary: #0D1117;
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.08);
    --text-primary: #F5F0E8;
    --text-secondary: #8B949E;
    --accent-teal: #00FFD1;
    --accent-gold: #C9A84C;
    --accent-red: #FF3B3B;
    --accent-amber: #FFB347;
    --glow-teal: 0 0 20px rgba(0, 255, 209, 0.3);
    --glow-gold: 0 0 20px rgba(201, 168, 76, 0.3);
}

/* Override Streamlit defaults */
.stApp { background-color: var(--bg-primary) !important; }
.main .block-container { padding: 1rem 2rem; max-width: 100%; }

/* Glass morphism panels */
.metric-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0, 255, 209, 0.3);
    box-shadow: var(--glow-teal);
}

/* Resilience Index number */
.resilience-score {
    font-size: 3.5rem;
    font-weight: 700;
    color: var(--accent-teal);
    text-shadow: var(--glow-teal);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -2px;
}

/* Node classification badges */
.badge-critical { color: var(--accent-red); background: rgba(255,59,59,0.15); }
.badge-important { color: var(--accent-amber); background: rgba(255,179,71,0.15); }
.badge-normal { color: var(--accent-teal); background: rgba(0,255,209,0.1); }

/* Sidebar */
.css-1d391kg { background-color: var(--bg-secondary) !important; }

/* Headings */
h1, h2, h3 { color: var(--text-primary) !important; font-weight: 600; }
h1 { 
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 1.6rem;
}
```

### 5.2 Map Component (Folium + Leaflet.js)

```python
# map_component.py
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

def build_map(G, node_classification, ablated_nodes=set(), healed_edges=False):
    """
    Builds a Folium map with:
    - CartoDB DarkMatter base tiles (dark background)
    - Road network edges colored by criticality (edge betweenness)
    - Node markers colored by classification tier
    - Healed edges shown as dashed lines
    - Ablated nodes shown as X markers
    """
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=14,
        tiles='CartoDB.DarkMatter',
        prefer_canvas=True,  # WebGL rendering — much faster
    )
    
    # Draw edges
    for u, v, data in G.edges(data=True):
        p1 = G.nodes[u].get('latlon')
        p2 = G.nodes[v].get('latlon')
        if not p1 or not p2:
            continue
        
        eb = data.get('edge_betweenness', 0)
        color = edge_color(eb)  # gradient: teal → amber → red
        weight = 1 + eb * 8    # thicker = more critical
        dash = '5 5' if data.get('healed') else None
        
        folium.PolyLine(
            locations=[p1, p2],
            color=color,
            weight=weight,
            opacity=0.85,
            dash_array=dash,
            tooltip=f"Criticality: {eb:.4f}" + (" [HEALED]" if data.get('healed') else ""),
        ).add_to(m)
    
    # Draw nodes
    for node, data in G.nodes(data=True):
        latlon = data.get('latlon')
        if not latlon:
            continue
        
        bc = data.get('betweenness', 0)
        tier = node_classification.get(node, 'normal')
        
        if node in ablated_nodes:
            # Show ablated node as X
            folium.Marker(
                latlon,
                icon=folium.DivIcon(html='<div style="color:#FF3B3B;font-size:18px;font-weight:bold;">✕</div>'),
                tooltip=f"DISABLED (was BC={bc:.4f})"
            ).add_to(m)
        else:
            color_map = {'critical': '#FF3B3B', 'important': '#FFB347', 'normal': '#00FFD1'}
            radius_map = {'critical': 8, 'important': 5, 'normal': 3}
            folium.CircleMarker(
                latlon,
                radius=radius_map[tier],
                color=color_map[tier],
                fill=True,
                fill_color=color_map[tier],
                fill_opacity=0.9,
                tooltip=f"BC={bc:.4f} | {tier.upper()}",
                popup=folium.Popup(f"Node ID: {node}<br>Betweenness: {bc:.4f}<br>Tier: {tier}", max_width=200),
            ).add_to(m)
    
    return m

def edge_color(eb_score, max_eb=0.1):
    """Maps edge betweenness [0, max_eb] to color gradient teal → amber → red"""
    t = min(eb_score / max_eb, 1.0)
    if t < 0.5:
        # teal to amber
        r = int(0 + t * 2 * 255)
        g = int(255 - t * 2 * 80)
        b = int(209 - t * 2 * 209)
    else:
        # amber to red
        t2 = (t - 0.5) * 2
        r = 255
        g = int(179 - t2 * 179)
        b = int(71 - t2 * 71)
    return f'#{r:02x}{g:02x}{b:02x}'
```

### 5.3 Main App Structure

```python
# app.py
import streamlit as st
import networkx as nx
import json

st.set_page_config(
    page_title="Route Resilience | ISRO BAH 2026",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 2rem 0;">
  <div style="letter-spacing: 0.15em; color: #C9A84C; font-size: 0.75rem; text-transform: uppercase;">
    ISRO BAH 2026 · Problem Statement 4
  </div>
  <h1 style="margin: 0.3rem 0; font-size: 2rem; letter-spacing: 0.08em;">
    ROUTE RESILIENCE
  </h1>
  <div style="color: #8B949E; font-size: 0.9rem;">
    Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR CONTROLS ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    stress_test_mode = st.toggle("🔴 Stress Test Mode", value=False)
    show_healed = st.toggle("Show Healed Edges", value=True)
    show_heatmap = st.toggle("Show Criticality Heatmap", value=True)
    
    if stress_test_mode:
        st.markdown("---")
        st.markdown("**Node Ablation**")
        nodes_to_remove = st.slider("Nodes to disable", 1, 20, 1)
        st.caption("Removes top-N Gatekeeper Nodes to simulate disaster scenarios")

# ── METRICS ROW ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">
        Resilience Index
      </div>
      <div class="resilience-score">{resilience_index:.2f}</div>
      <div style="color: {'#00FFD1' if resilience_index > 0.7 else '#FFB347' if resilience_index > 0.4 else '#FF3B3B'}; font-size: 0.8rem;">
        {'STABLE' if resilience_index > 0.7 else 'DEGRADED' if resilience_index > 0.4 else 'CRITICAL'}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Road Network Stats
    # ...

with col3:
    # Top Gatekeeper Node
    # ...

with col4:
    # Connectivity Ratio (after healing)
    # ...

# ── MAP ─────────────────────────────────────────────────────────────
map_col, panel_col = st.columns([3, 1])

with map_col:
    folium_map = build_map(G, node_classification, ablated_nodes, show_healed)
    map_data = st_folium(folium_map, width=None, height=600, returned_objects=["last_object_clicked"])
    
    # Handle node click → disable it
    if map_data and map_data.get("last_object_clicked") and stress_test_mode:
        clicked = map_data["last_object_clicked"]
        # Find nearest node to clicked lat/lon and add to ablated_nodes
        # Re-run ablation simulation, update resilience index

with panel_col:
    # Gatekeeper Node Rankings
    st.markdown("### 🔴 Gatekeeper Nodes")
    for i, (node, bc) in enumerate(top_nodes[:10]):
        tier = node_classification[node]
        badge_class = f"badge-{tier}"
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.5rem; padding: 0.8rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #F5F0E8; font-size: 0.85rem;">#{i+1} Node {node}</span>
            <span class="{badge_class}" style="border-radius: 4px; padding: 2px 8px; font-size: 0.7rem;">
              {tier.upper()}
            </span>
          </div>
          <div style="color: #8B949E; font-size: 0.75rem; margin-top: 0.3rem;">
            BC Score: {bc:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── ABLATION CHART ──────────────────────────────────────────────────
if stress_test_mode and ablation_results:
    st.markdown("---")
    st.markdown("### 📉 Network Degradation Under Stress")
    # Line chart: Resilience Index vs Nodes Removed
    # Also show: LCC fraction, Global Efficiency
    # Use st.line_chart or plotly with dark theme
```

### 5.4 Making Streamlit Look Premium

Key tricks to make Streamlit match the Cascade & Coal aesthetic:

1. **Hide Streamlit branding:** Add `[theme]` config + `#MainMenu {visibility: hidden;} footer {visibility: hidden;}` in CSS
2. **Custom fonts:** Load JetBrains Mono + Inter via Google Fonts `@import` in the CSS injection
3. **Full-width layout:** `st.set_page_config(layout="wide")` + override `.main .block-container { max-width: 100%; }`
4. **Animated metrics:** Use `st.empty()` + loop for counting animations on the Resilience Index
5. **Dark map tiles:** CartoDB DarkMatter is free, no API key needed, matches the color scheme perfectly
6. **Smooth sidebar:** Override sidebar background to `#0D1117` with a subtle border
7. **Glassmorphism cards:** Pure HTML/CSS injected via `st.markdown(unsafe_allow_html=True)` — Streamlit renders raw HTML in markdown blocks

---

## run_pipeline.py — End-to-End Orchestration

```python
"""
Full pipeline runner. Can be run in stages or end-to-end.
Usage: python run_pipeline.py --stage all
       python run_pipeline.py --stage segmentation
       python run_pipeline.py --stage graph
       python run_pipeline.py --stage analysis
"""
import argparse
import yaml

def run_segmentation(config): ...
def run_skeletonization(config): ...
def run_healing(config): ...
def run_analysis(config): ...
def launch_dashboard(config): ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    stages = {
        'segmentation': run_segmentation,
        'graph': run_skeletonization,
        'healing': run_healing,
        'analysis': run_analysis,
        'dashboard': launch_dashboard,
    }
    
    if args.stage == 'all':
        for fn in stages.values():
            fn(config)
    else:
        stages[args.stage](config)
```

---

## requirements.txt

```
# Core ML
torch>=2.1.0
torchvision>=0.16.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0

# Geospatial
rasterio>=1.3.0
gdal>=3.6.0
pyproj>=3.5.0
shapely>=2.0.0
geopandas>=0.13.0
osmnx>=1.6.0

# Image Processing
scikit-image>=0.21.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0

# Graph
networkx>=3.1
# PyTorch Geometric (optional, advanced GNN extension)
# torch-geometric>=2.3.0

# Visualization
matplotlib>=3.7.0
folium>=0.14.0
streamlit>=1.28.0
streamlit-folium>=0.15.0
plotly>=5.17.0

# Utilities
PyYAML>=6.0
tqdm>=4.66.0
pandas>=2.0.0
```

---

## Key Pitfalls to Avoid

1. **Coordinate system confusion:** Satellite images use geographic CRS (lat/lon). Pixel operations need projected CRS (meters). Always `reproject` with rasterio before computing pixel distances. Never mix the two.

2. **Skeletonization on thick masks:** If the road mask is 20px wide and you skeletonize directly, you get noise. Always apply morphological closing first, then remove_small_objects, THEN skeletonize.

3. **Graph node explosion:** A 512×512 skeleton image can have 10,000+ potential nodes. Keep only true intersections (degree ≠ 2) as nodes; compress straight segments into single weighted edges. Otherwise NetworkX operations become slow.

4. **Betweenness Centrality on large graphs:** BC is O(VE) for unweighted, O(VE + V²logV) for weighted. For a city-scale graph with >5000 nodes, use `betweenness_centrality(G, k=500)` (sampling approximation) — fast enough and accurate for ranking.

5. **Folium performance with large graphs:** Don't render all edges as PolyLine objects — for city-scale graphs with >10,000 edges, use `folium.GeoJson()` with a GeoJSON FeatureCollection instead. Order of magnitude faster rendering.

6. **Streamlit reruns:** Every widget interaction triggers a full Python rerun. Cache the graph, model, and analysis results with `@st.cache_resource` and `@st.cache_data` to avoid re-running expensive computations on every click.

```python
@st.cache_resource
def load_graph():
    return nx.read_gpickle('data/graphs/healed_graph.gpickle')

@st.cache_data
def run_centrality(_G):  # underscore prefix = not hashed by streamlit
    return compute_centrality(_G)
```

---

## Evaluation Metric Targets

| Metric | Target |
|---|---|
| IoU (overall) | > 0.65 |
| Dice Score | > 0.75 |
| Occlusion-Recall | > 0.55 (hardest metric) |
| Connectivity Ratio (post-healing) | > 85% of components merged |
| Resilience Index (baseline) | 1.00 (by definition) |
| Relaxed IoU (3px buffer) | > 0.80 |

---

*Context document prepared for ISRO BAH 2026 — Problem Statement 4. Feed this entire file to Claude Code Opus 4.8 before starting any implementation phase.*
