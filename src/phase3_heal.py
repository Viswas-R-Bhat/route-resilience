"""Phase 3 — topological healing: bridge gaps between disconnected road fragments.

Real occlusion (canopy/shadow/cloud) leaves the extracted graph broken into components.
We reconnect them with an MST-style, shortest-gap-first strategy gated by:
  - max_gap_px        : only bridge endpoints closer than this (Euclidean)
  - angular_tolerance : the bridge must continue the road's local direction (PCA on
                        nearby nodes) at BOTH ends, so we never invent illogical sharp turns
Disjoint-Set (Union-Find) guarantees we only ever connect *different* components (no cycles).

CANOPY-AWARE EVIDENCE (occlusion robustness): given the source RGB tile, we build a
tree-canopy mask via the Excess-Green index (ExG = 2G - R - B). A candidate bridge whose
straight span lies mostly OVER canopy is positive evidence the gap is occlusion — not a
true dead-end — so we RELAX the gates there (longer gap + wider angle). Over bare ground we
stay strict (a road that just stops in the open is probably a real end). Each synthetic edge
is tagged heal_kind="canopy" | "geom" so the dashboard can show what was recovered under trees.

Acceptance also allows a trajectory-corridor test (each endpoint lies near the OTHER road's
projected centerline), which catches roads that gently curve into the canopy — cases the
strict straight-line collinearity test rejects.

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


def canopy_evidence(rgb, exg_floor=12, dense_pct=78):
    """Adaptive tree-canopy analysis from the Excess-Green index ExG = 2G - R - B (no NIR).

    Returns dict(dense=HxW bool mask, green_frac=float), or None for a non-RGB input.

      - `dense`      : the *discriminative* canopy mask. Its threshold is max(absolute floor,
                       an in-tile percentile), so a fixed cutoff can't degenerate: in an arid
                       scene the floor dominates (almost nothing flagged); in a rain-forest the
                       percentile rises so only the DENSEST canopy qualifies — preventing the
                       "the whole frame is canopy, relax everywhere" failure (e.g. Perumbavoor).
      - `green_frac` : the *raw* vegetated fraction (ExG > floor). This is the "how occluded is
                       this whole scene" signal the healer uses to throttle itself + lower the
                       confidence of inferred bridges when a region is saturated with canopy.
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        return None
    R = a[..., 0].astype(np.int32); Gc = a[..., 1].astype(np.int32); B = a[..., 2].astype(np.int32)
    exg = 2 * Gc - R - B
    green_frac = float((exg > exg_floor).mean())
    thr = max(float(exg_floor), float(np.percentile(exg, dense_pct)))
    return dict(dense=exg > thr, green_frac=green_frac)


def _canopy_overlap(p1, p2, veg, samples=32):
    """Fraction of the straight p1->p2 bridge that passes over canopy pixels.

    pos is (x, y) = (col, row); the vegetation mask is indexed [row, col] = [y, x].
    """
    Hh, Ww = veg.shape
    xs = np.clip(np.linspace(p1[0], p2[0], samples).astype(int), 0, Ww - 1)
    ys = np.clip(np.linspace(p1[1], p2[1], samples).astype(int), 0, Hh - 1)
    return float(veg[ys, xs].mean())


def _line_offset(p0, ang_deg, p):
    """Perpendicular distance from point `p` to the infinite line through `p0` at `ang_deg`."""
    a = np.radians(ang_deg)
    dx, dy = np.cos(a), np.sin(a)
    wx, wy = p[0] - p0[0], p[1] - p0[1]
    return abs(wx * dy - wy * dx)                        # |cross(w, unit_dir)|


def _aligned(p1, p2, a1, a2, ang_tol, corridor_px):
    """True if the bridge continues both roads' trajectories.

    Accept on EITHER of two signals (a missing local angle is treated as 'no objection'):
      - straight collinearity : the p1->p2 bridge bearing is within `ang_tol` of both ends, OR
      - trajectory corridor   : each endpoint lies within `corridor_px` of the OTHER road's
                                projected centerline (handles roads curving into the canopy).
    """
    bridge = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
    dev1 = _ang_dev(bridge, a1) if a1 is not None else 0.0
    dev2 = _ang_dev(bridge, a2) if a2 is not None else 0.0
    straight_ok = dev1 <= ang_tol and dev2 <= ang_tol
    off1 = _line_offset(p1, a1, p2) if a1 is not None else 0.0
    off2 = _line_offset(p2, a2, p1) if a2 is not None else 0.0
    corridor_ok = off1 <= corridor_px and off2 <= corridor_px
    return straight_ok or corridor_ok


def heal_graph(G, max_gap_px=60, angular_tolerance_deg=35, depth=3, rgb=None,
               canopy_max_gap_px=None, canopy_angular_tolerance_deg=None,
               exg_floor=12, dense_pct=78, canopy_frac=0.5, corridor_px=16,
               saturation_frac=0.6):
    """Return (healed_graph, n_healed). Synthetic edges carry healed=True,
    heal_kind in {"canopy", "geom"}, a confidence `conf` in {"med", "low", "vlow"}, and
    canopy_frac (span fraction over canopy). Scene-level stats land on H.graph
    (canopy_frac_scene, canopy_saturated).

    If `rgb` (the source tile, same pixel space as the graph) is given, gaps whose bridge lies
    mostly over canopy use the relaxed `canopy_*` gates; everything else uses the strict base
    gates. In a CANOPY-SATURATED scene (raw vegetated fraction >= saturation_frac, e.g. dense
    rain-forest) the canopy signal is non-discriminative, so the healer throttles itself —
    smaller relaxation, stricter overlap demand — and flags those bridges very-low confidence,
    rather than fabricating phantom roads. With rgb=None it's purely geometric (back-compatible).
    """
    H = G.copy()
    # canopy gates default to a relaxation of the base gates (longer gap, wider angle)
    cg = canopy_max_gap_px if canopy_max_gap_px is not None else max_gap_px * 1.6
    ca = canopy_angular_tolerance_deg if canopy_angular_tolerance_deg is not None \
        else min(angular_tolerance_deg + 20, 70)
    ev = canopy_evidence(rgb, exg_floor, dense_pct) if rgb is not None else None
    veg = ev["dense"] if ev else None
    green_frac = ev["green_frac"] if ev else 0.0
    saturated = green_frac >= saturation_frac
    H.graph["canopy_frac_scene"] = round(green_frac, 3)
    H.graph["canopy_saturated"] = bool(saturated)
    if saturated:                                        # unreliable signal -> be conservative
        cg = min(cg, max_gap_px * 1.25)
        ca = min(ca, angular_tolerance_deg + 10)
    req_frac = max(canopy_frac, 0.65) if saturated else canopy_frac
    search_r = max(max_gap_px, cg) if veg is not None else max_gap_px

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
        if dist <= search_r:                             # widen search when canopy evidence is in play
            candidates.append((dist, n1, n2))
    candidates.sort()                                    # shortest gaps first (MST-like)

    healed = 0
    for dist, n1, n2 in candidates:
        if uf.connected(n1, n2):
            continue
        p1 = np.array(H.nodes[n1]["pos"]); p2 = np.array(H.nodes[n2]["pos"])
        over = _canopy_overlap(p1, p2, veg) if veg is not None else 0.0
        is_canopy = over >= req_frac                     # bridge mostly over dense canopy -> occlusion
        eff_gap = cg if is_canopy else max_gap_px
        eff_ang = ca if is_canopy else angular_tolerance_deg
        if dist > eff_gap:
            continue
        a1 = compute_road_angle(H, n1, depth); a2 = compute_road_angle(H, n2, depth)
        if not _aligned(p1, p2, a1, a2, eff_ang, corridor_px):
            continue
        kind = "canopy" if is_canopy else "geom"
        conf = ("vlow" if saturated else "low") if is_canopy else "med"   # visible roads = "high"
        H.add_edge(n1, n2, weight=dist, length_px=dist, healed=True,
                   heal_kind=kind, conf=conf, canopy_frac=round(over, 3))
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
