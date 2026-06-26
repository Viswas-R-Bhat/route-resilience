"""Phase 2b — skeleton -> routable weighted graph (NetworkX).

Nodes are placed at intersections (degree != 2) and endpoints; straight runs become
single weighted edges (edge weight = geometric pixel length of the traced centerline).
Uses `sknw` to trace the skeleton network, then cleans up two skeleton artifacts:
  - iterative spur pruning : a dropped dangling branch can expose a fresh stub behind it,
                             so we repeat until stable rather than a single pass, and
  - junction de-duplication: sknw often splits one real X/T/Y intersection into a tight
                             cluster of nodes a few px apart, which inflates node counts
                             and skews betweenness; we contract nodes within a small radius.
The resulting graph is what Phase 3 (healing) and Phase 4 (centrality/resilience) operate on.
"""
import numpy as np
import networkx as nx
import sknw
from scipy.spatial import cKDTree


def _polyline_len(pts):
    if pts is None or len(pts) < 2:
        return 1.0
    d = np.diff(pts.astype(np.float64), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def skeleton_to_graph(skeleton, min_spur_px=12, merge_radius_px=6):
    raw = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)
    G = nx.Graph()
    for n, d in raw.nodes(data=True):
        y, x = d["o"]                       # sknw centroid is (row, col)
        G.add_node(int(n), pos=(float(x), float(y)), yx=(float(y), float(x)))
    for u, v, d in raw.edges(data=True):
        pts = d.get("pts")
        length = _polyline_len(pts)
        G.add_edge(int(u), int(v), weight=length, length_px=length, pts=pts)
    _prune_spurs(G, min_spur_px)            # iterative: a pruned spur can expose another
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree[n] == 0])
    if merge_radius_px:                     # collapse a split intersection into one node
        G = _merge_close_nodes(G, merge_radius_px)
    return G


def _prune_spurs(G, min_spur_px, max_iter=8):
    """Iteratively drop short dangling branches (skeleton thinning artifacts).

    Removing a spur can demote its junction to a fresh degree-1 stub, so a single pass
    leaves second-order spurs behind; we repeat until no short leaf edge remains (capped
    by max_iter as a safety bound). A short edge between two real junctions is never a
    spur — the degree-1 test protects through-connectors.
    """
    for _ in range(max_iter):
        spurs = [(u, v) for u, v, d in G.edges(data=True)
                 if d["length_px"] < min_spur_px and (G.degree[u] == 1 or G.degree[v] == 1)]
        if not spurs:
            break
        G.remove_edges_from(spurs)
        G.remove_nodes_from([n for n in list(G.nodes()) if G.degree[n] == 0])


def _merge_close_nodes(G, radius_px):
    """Contract nodes within `radius_px` of each other into one (junction de-duplication).

    sknw frequently splits a single X/T/Y intersection into 2-3 nodes a few pixels apart;
    left alone they inflate node counts and skew betweenness centrality (and therefore the
    gatekeeper ranking + Resilience Index). We union nodes inside a small radius, place the
    survivor at the cluster centroid, drop the self-loops that collapse out, and keep the
    shortest of any parallel edges. Edge geometry (`pts`, weight, length_px) is preserved.
    """
    if G.number_of_nodes() < 2:
        return G
    nodes = list(G.nodes())
    coords = np.array([G.nodes[n]["pos"] for n in nodes], dtype=float)
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]               # path halving
            x = parent[x]
        return x

    for i, j in cKDTree(coords).query_pairs(radius_px):
        a, b = find(nodes[i]), find(nodes[j])
        if a != b:
            parent[b] = a

    groups = {}
    for idx, n in enumerate(nodes):
        groups.setdefault(find(n), []).append(idx)

    H = nx.Graph()
    for root, idxs in groups.items():
        cx, cy = coords[idxs].mean(axis=0)              # survivor sits at the cluster centroid
        H.add_node(int(root), pos=(float(cx), float(cy)), yx=(float(cy), float(cx)))
    for u, v, d in G.edges(data=True):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue                                    # self-loop from a collapsed cluster
        if H.has_edge(ru, rv):
            if d.get("weight", 1.0) < H[ru][rv].get("weight", 1.0):
                H[ru][rv].clear(); H[ru][rv].update(d)  # keep the shorter parallel edge
        else:
            H.add_edge(int(ru), int(rv), **d)
    H.remove_nodes_from([n for n in list(H.nodes()) if H.degree[n] == 0])
    return H


def graph_stats(G):
    comps = list(nx.connected_components(G))
    lcc = max((len(c) for c in comps), default=0)
    total_len = sum(d["weight"] for _, _, d in G.edges(data=True))
    return dict(
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        components=len(comps),
        lcc_nodes=lcc,
        lcc_fraction=round(lcc / max(1, G.number_of_nodes()), 4),
        total_length_px=round(total_len, 1),
    )
