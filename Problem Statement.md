PROBLEM STATEMENT 4
Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
Overview
Modern urban centres, particularly rapidly expanding Indian metropolises (e.g., Bengaluru), face a dual challenge in spatial modelling: fragmentation and stagnation. This challenge sits squarely within the mandate of ISRO's National Natural Resources Management System (NNRMS), which seeks to maximise the downstream utility of indigenous remote sensing EO satellites such as Cartosat and Resourcesat LISS-4.

Standard satellite-based road extraction often fails due to "spectral blindness" caused by tree canopies, building shadows, and cloud cover. These "broken" masks are useless for real-world applications like disaster response or traffic simulation because they lack topological connectivity. The need for robust, automated road asset mapping has been specifically highlighted in recent National Meets - by the Ministry of Electronics and Information Technology (MeitY) for GIS-based urban planning and e-governance decision support, and by the Ministry of Consumer Affairs, Food and Public Distribution for infrastructure mapping, facility layout, and route verification.

This solution bridges that gap with an end-to-end pipeline: first, using context-aware Deep Learning to "see through" occlusions, and second, transforming those masks into a mathematically continuous, weighted graph to identify systemic bottlenecks and simulate urban collapse scenarios. By transitioning from legacy analytical methods to modern Deep Learning and graph-theoretic modelling, the framework provides automated, occlusion-robust road asset mapping for urban planning, while its predictive "what-if" simulations offer actionable decision support for disaster response, structural resilience, and facility distribution across dynamic Indian metropolises.

The proposed problem statement is well-suited for a 30-hour hackathon through a parallel team workflow: while one sub-team builds and trains the context-aware segmentation model, the other can simultaneously develop the topological healing and network ablation scripts using mock or open-source vector baselines. By eliminating complex manual preprocessing bottlenecks, the team can focus entirely on Deep Learning optimisation, graph healing, and the UI dashboard.

Objective
Occlusion-Aware Extraction
Develop a Transformer-based deep learning architecture to infer road continuity under heavy tree cover, vehicles, shadows, and urban clutter, across varying illumination and seasonal conditions.
Topological Reconstruction
Convert fragmented pixel masks into a unified, routable weighted vector graph using graph-theoretic "healing" (MST, Disjoint Sets).
Structural Intelligence
Quantify urban vulnerability by identifying "Gatekeeper Nodes"/bottlenecks through centrality metrics.
Simulated Stress Testing
Build a framework to predict the systemic impact of localised infrastructure failure (e.g., flooding or accidents).
Expected Outcomes
High-Fidelity Routable Topology
A mathematically closed, connected vector network derived from high-resolution imagery, far surpassing standard pixel-based segmentation, a robust, generalizable model that suits different terrains (urban, rural, forested).
Quantitative Criticality Map (Bottleneck Identification)
A spatial heatmap identifying high-betweenness intersections that act as single points of failure, identifying "Gatekeeper Nodes".
Predictive Impact Assessment for Disaster Scenarios
A simulation framework that quantifies the systemic "cost" of losing specific network nodes. By systematically disabling high-centrality nodes (e.g., simulating floods, accidents, or construction), the project will produce a Resilience Index - a tool for planners to disable nodes and instantly see rerouting effects and travel time increases.
Dataset Required
Primary

Source	Resolution / Notes
Sentinel-2	10 m spatial resolution - openly available
Resourcesat LISS-IV	5.8 m spatial resolution - openly available
Cartosat-3	High resolution - provided to participants during the 30-hour hackathon for challenge-specific experimentation and evaluation
Ground Truth & Open Datasets

Data readiness is fully secured through a zero-manual-effort automation pipeline that pairs open-source ground truth with multi-resolution satellite feeds. Participants may utilise:

SpaceNet Roads Dataset - for model development and pre-training
DeepGlobe Road Extraction Dataset - for model development and pre-training
OpenSatMap - for model development and pre-training
OpenStreetMap (OSM) road vector layers - ground-truth road masks generated automatically and used as reference annotations for training, validation, and performance assessment
Commended Stack
Source	Commended Stack
Libraries	Albumentations (data augmentation), Rasterio, GDAL (geospatial processing), OpenCV, NumPy
Segmentation Model	U-Net with ResNet Backbone, UNet++, DeepLabV3+, PyTorch (Python); Transformer-based models, Attention mechanisms (spatial + channel), Generative models (optional for reconstruction)
Skeletonization	Scikit-Image / FilFinder / OSMnx
Graph Logic	NetworkX; PyTorch Geometric (PyG) for advanced GNN approach
Centrality Analysis	Betweenness Centrality (NetworkX)
Visualization	QGIS / Matplotlib / Streamlit / Leaflet.js
Compute Requirements
Graph-theoretic post-processing and UI execution are lightweight and run on standard CPU architecture. Training state-of-the-art Deep Learning models within the 30-hour limit will require access to high-performance GPU instances, which can be sourced via local workstations.

Expected Solution / Steps to be followed to achieve the objectives
Phase I: Occlusion-Robust Segmentation (The Foundation)

Data Preprocessing
Tile images, normalize/enhance contrast, simulate occlusions, and balance dataset with occluded roads.
Baseline Model Development
Train a U-Net/DeepLabV3+ model, evaluate on occluded regions, and identify failure cases.
Advanced Model Design
Implement context-aware architectures (Transformer, attention), focusing on long-range dependencies.
Loss Function Engineering
Use combined Dice, IoU, and boundary-aware losses, with optional connectivity loss.
Occlusion Handling Strategy
Use context-based inference and multi-scale feature fusion, with optional inpainting.
Phase II: Graph Skeletonization & Healing

Thinning
Apply morphological skeletonization to reduce binary masks into 1-pixel wide centerlines. Nodes are generated at intersections and line endpoints, while road segments are represented as edges.
Topological Healing
Use a Minimum Spanning Tree (MST) and Disjoint Set algorithm to bridge gaps caused by extreme occlusions. The algorithm evaluates logical gaps based on Euclidean distance and angular alignment to ensure the "healed" road follows a natural trajectory.
Phase III: Network Analysis & Stress Testing

Centrality Calculation
Apply Betweenness Centrality to identify nodes that lie on the shortest paths across the city. A high centrality score indicates a critical bottleneck.
Node Ablation Simulation
Once the critical nodes are identified, the system performs a series of "Network Stress Tests" to quantify vulnerability. The algorithm systematically "removes" nodes with the highest Betweenness Centrality scores from the graph to simulate real-world closures (e.g., severe flooding or structural failure).
Resilience Index Calculation
The system recalculates the global network efficiency after each removal. The Resilience Index is defined as the ratio of the average shortest path length in the baseline network to that in the perturbed network. A lower R indicates a highly vulnerable network.
Phase IV (Advanced): Interactive Dashboard

The final stage is the development of a web-based visualisation tool (using Streamlit and Leaflet.js) to make the graph data actionable for non-technical planners:

Heatmap Overlay
A dynamic map layer that colors road segments based on their "Criticality Worth," allowing planners to see the city's "weakest links" at a glance.
Simulation Toggle
An interactive feature where a user can manually click a node to "disable" it. The dashboard instantly updates to show the rerouted paths and the estimated increase in travel time across the affected sector.
Evaluation Parameters
Core Technical Metrics

Metric	Description
IoU & Dice Score	Segmentation accuracy with specific focus on Occlusion-Recall (recovery of roads under shadows).
Generalisation	Success rate across diverse terrains - dense urban, forested suburban, and rural landscapes.
Connectivity Ratio	Percentage increase in the largest connected component after the MST healing phase.
Topological Accuracy	Comparison of the final graph against OpenStreetMap (OSM) benchmarks using Average Path Length error. Run shortest-path between random point pairs on ground-truth OSM vs. model graph and calculate error.
Length-Complete / Relaxed IoU	Introduces a tolerance buffer (3–5 pixels). If the predicted road pixel falls within the buffer zone of the ground truth road, it counts as a true positive. Prevents penalising minor alignment shifts.
Image representing problem statement
Flow diagram from raw Sentinel-2 imagery through ML extraction, skeletonization, topological healing, network analysis, stress testing, to an interactive visualization dashboard