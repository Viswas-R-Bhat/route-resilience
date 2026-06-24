"""Phase 3 — topological healing: bridge gaps between disconnected road fragments.

Real occlusion (canopy/shadow/cloud) leaves the extracted graph broken into components.
We reconnect them with an MST-style, shortest-gap-first strategy gated by:
  - max_gap_px        : only bridge endpoints closer than this (Euclidean)
  - angular_tolerance : the bridge must continue the road's local direction (PCA on
                        nearby nodes) at BOTH ends, so we never invent illogical sharp turns
Disjoint-Set (Union-Find) guarantees we only ever connect *different* components (no cycles).
Metric: "Connectivity Ratio" — components merged + largest-connected-component growth.
"""
import numpy as np
import networkx as nx
from itertools import combinations


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


def compute_road_angle(G, node, depth=3):
    """Local road direction at `node` (degrees [0,180)) via PCA over nodes within `depth` hops."""
    nbrs = list(nx.single_source_shortest_path_length(G, node, cutoff=depth).keys())
    if len(nbrs) >= 2:
        pos = np.array([G.nodes[n]["pos"] for n in nbrs], dtype=np.float64)
        c = pos - pos.mean(axis=0)
        if not np.allclose(c, 0):
            _, _, vt = np.linalg.svd(c)
            d = vt[0]
            return np.degrees(np.arctan2(d[1], d[0])) % 180
    nb = list(G.neighbors(node))
    if not nb:
        return None
    v = np.array(G.nodes[nb[0]]["pos"]) - np.array(G.nodes[node]["pos"])
    return None if np.allclose(v, 0) else np.degrees(np.arctan2(v[1], v[0])) % 180


def _ang_dev(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def heal_graph(G, max_gap_px=60, angular_tolerance_deg=35, depth=3):
    """Return (healed_graph, n_healed). Synthetic edges carry attr healed=True."""
    H = G.copy()
    uf = UnionFind(list(H.nodes()))
    for u, v in H.edges():
        uf.union(u, v)
    endpoints = [n for n in H.nodes() if H.degree[n] == 1]

    candidates = []
    for n1, n2 in combinations(endpoints, 2):
        if uf.connected(n1, n2):
            continue
        p1 = np.array(H.nodes[n1]["pos"]); p2 = np.array(H.nodes[n2]["pos"])
        dist = float(np.linalg.norm(p1 - p2))
        if dist <= max_gap_px:
            candidates.append((dist, n1, n2))
    candidates.sort()                                    # shortest gaps first (MST-like)

    healed = 0
    for dist, n1, n2 in candidates:
        if uf.connected(n1, n2):
            continue
        a1 = compute_road_angle(H, n1, depth); a2 = compute_road_angle(H, n2, depth)
        p1 = np.array(H.nodes[n1]["pos"]); p2 = np.array(H.nodes[n2]["pos"])
        bridge = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
        if a1 is not None and _ang_dev(bridge, a1) > angular_tolerance_deg:
            continue
        if a2 is not None and _ang_dev(bridge, a2) > angular_tolerance_deg:
            continue
        H.add_edge(n1, n2, weight=dist, length_px=dist, healed=True)
        uf.union(n1, n2)
        healed += 1
    return H, healed


def connectivity_report(G, H):
    def lcc_frac(g):
        if g.number_of_nodes() == 0:
            return 0.0
        return max((len(c) for c in nx.connected_components(g)), default=0) / g.number_of_nodes()
    cb = nx.number_connected_components(G)
    ca = nx.number_connected_components(H)
    return dict(
        components_before=cb, components_after=ca, components_merged=cb - ca,
        lcc_frac_before=round(lcc_frac(G), 4), lcc_frac_after=round(lcc_frac(H), 4),
        healed_edges=H.number_of_edges() - G.number_of_edges(),
    )
