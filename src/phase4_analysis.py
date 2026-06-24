"""Phase 4 — network criticality & resilience on the routable road graph.

  - Betweenness centrality -> "Gatekeeper Nodes": intersections that lie on the most
    shortest paths; their failure hurts the whole network. (weight = road length, so paths
    are by distance, not hop count.)
  - Node-ablation stress test: progressively remove the highest-betweenness node, recompute
    network global efficiency, and report the Resilience Index R = eff(perturbed)/eff(baseline).
    R=1 means no degradation; R->0 means collapse.
"""
import numpy as np
import networkx as nx


def global_efficiency_weighted(G, n_ref=None):
    """Mean over node pairs of 1/d(u,v) (d = shortest *weighted* path length in px).

    n_ref fixes the normalization denominator to a reference node count (the baseline
    network size). This is essential for the Resilience Index: when nodes are ablated we
    must keep the SAME denominator, otherwise shrinking n can spuriously inflate efficiency
    (a removed node's lost pairs would raise the average). With a fixed n_ref, removing
    nodes can only lower efficiency -> R is bounded in [0, 1] and monotonic.
    """
    n = G.number_of_nodes()
    denom = n_ref if n_ref is not None else n
    if n < 2 or denom < 2:
        return 0.0
    total = 0.0
    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
        for tgt, d in lengths.items():
            if tgt != src and d > 0:
                total += 1.0 / d
    return total / (denom * (denom - 1))


def compute_centrality(G, k=None):
    n = G.number_of_nodes()
    kk = k if (k and k < n) else None      # k-sampling approximation for big graphs
    node_bc = nx.betweenness_centrality(G, weight="weight", normalized=True, k=kk, seed=42)
    edge_bc = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
    nx.set_node_attributes(G, node_bc, "betweenness")
    nx.set_edge_attributes(G, edge_bc, "edge_betweenness")
    return sorted(node_bc.items(), key=lambda x: x[1], reverse=True)


def classify_nodes(G):
    sc = np.array([G.nodes[n].get("betweenness", 0.0) for n in G.nodes()])
    nz = sc[sc > 1e-9]                          # rank over non-zero betweenness only
    if len(nz) == 0:
        return {n: "normal" for n in G.nodes()}
    p75, p90 = np.percentile(nz, 75), np.percentile(nz, 90)
    out = {}
    for n in G.nodes():
        b = G.nodes[n].get("betweenness", 0.0)
        if b <= 1e-9:
            out[n] = "normal"                   # a zero-betweenness node is never critical
        elif b >= p90:
            out[n] = "critical"
        elif b >= p75:
            out[n] = "important"
        else:
            out[n] = "normal"
    return out


def ablation_simulation(G, top_k=8):
    """Remove highest-betweenness node repeatedly; track Resilience Index + LCC fraction."""
    H = G.copy()
    n0 = H.number_of_nodes()                      # fixed reference for the Resilience Index
    base_eff = global_efficiency_weighted(H, n_ref=n0)
    base_lcc = max((len(c) for c in nx.connected_components(H)), default=1)
    results = [dict(step=0, removed=None, betweenness=None, resilience_index=1.0,
                    global_efficiency=round(base_eff, 6), lcc_fraction=1.0)]
    for i in range(top_k):
        if H.number_of_nodes() < 3:
            break
        bc = nx.betweenness_centrality(H, weight="weight", normalized=True, seed=42)
        node = max(bc, key=bc.get)
        H.remove_node(node)
        eff = global_efficiency_weighted(H, n_ref=n0)
        lcc = max((len(c) for c in nx.connected_components(H)), default=0)
        results.append(dict(step=i + 1, removed=int(node), betweenness=round(bc[node], 4),
                            resilience_index=round(eff / base_eff, 4) if base_eff > 0 else 0.0,
                            global_efficiency=round(eff, 6),
                            lcc_fraction=round(lcc / base_lcc, 4)))
    return results
