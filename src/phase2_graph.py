"""Phase 2b — skeleton -> routable weighted graph (NetworkX).

Nodes are placed at intersections (degree != 2) and endpoints; straight runs become
single weighted edges (edge weight = geometric pixel length of the traced centerline).
Uses `sknw` to trace the skeleton network, then prunes tiny dangling spurs (skeleton noise).
The resulting graph is what Phase 3 (healing) and Phase 4 (centrality/resilience) operate on.
"""
import numpy as np
import networkx as nx
import sknw


def _polyline_len(pts):
    if pts is None or len(pts) < 2:
        return 1.0
    d = np.diff(pts.astype(np.float64), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def skeleton_to_graph(skeleton, min_spur_px=12):
    raw = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)
    G = nx.Graph()
    for n, d in raw.nodes(data=True):
        y, x = d["o"]                       # sknw centroid is (row, col)
        G.add_node(int(n), pos=(float(x), float(y)), yx=(float(y), float(x)))
    for u, v, d in raw.edges(data=True):
        pts = d.get("pts")
        length = _polyline_len(pts)
        # prune short dangling spurs (degree-1 endpoint + tiny length = thinning artifact)
        if length < min_spur_px and (raw.degree[u] == 1 or raw.degree[v] == 1):
            continue
        G.add_edge(int(u), int(v), weight=length, length_px=length, pts=pts)
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree[n] == 0])
    return G


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
