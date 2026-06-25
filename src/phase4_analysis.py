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

# Betweenness via exact Brandes is O(N*E) and dominates Phase 4 on large tiles. Above this
# node count we switch to k-sampled Brandes (NetworkX `k=` argument), the standard fast
# approximation: only the node *ranking* is approximate (which gatekeeper / which node to
# ablate); every Resilience Index is still computed from EXACT global efficiency, so the
# stress-test curves are unchanged. ~4x faster on the 1500-node tiles.
_BC_EXACT_MAX = 450
_BC_SAMPLES = 350


def _auto_k(n):
    return None if n <= _BC_EXACT_MAX else min(n, _BC_SAMPLES)


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
    kk = (k if (k and k < n) else None) or _auto_k(n)   # k-sampled Brandes on big graphs
    node_bc = nx.betweenness_centrality(G, weight="weight", normalized=True, k=kk, seed=42)
    edge_bc = nx.edge_betweenness_centrality(G, weight="weight", normalized=True, k=kk, seed=42)
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


def local_road_density(G, radius_px=150):
    """Demand proxy (self-contained): total road length within `radius_px` of each node.

    A dense tangle of roads marks a busy, built-up area — more travel demand — whereas a
    lone road through open land carries little. (Swap in WorldPop/OSM POI density for a true
    demand surface; road density is a defensible no-extra-data stand-in.)
    """
    nodes = list(G.nodes())
    if G.number_of_edges() == 0:
        return {n: 0.0 for n in nodes}
    mids, lens = [], []
    for u, v, d in G.edges(data=True):
        pu = np.array(G.nodes[u]["pos"]); pv = np.array(G.nodes[v]["pos"])
        mids.append((pu + pv) / 2.0); lens.append(float(d.get("length_px", d.get("weight", 1.0))))
    mids = np.array(mids); lens = np.array(lens); r2 = radius_px * radius_px
    dens = {}
    for n in nodes:
        p = np.array(G.nodes[n]["pos"])
        dens[n] = float(lens[((mids - p) ** 2).sum(axis=1) <= r2].sum())
    return dens


def demand_weighted_criticality(G, radius_px=150):
    """Service-criticality = topological betweenness x local demand, normalized to [0,1].

    A node ranks high only if it is BOTH structurally central AND serves a busy area, so its
    failure strands real travel rather than just graph-theoretic paths. Writes node attrs
    'demand' and 'service_crit'; returns nodes ranked by service-criticality (desc).
    Requires betweenness to already be on the graph (run compute_centrality first).
    """
    dens = local_road_density(G, radius_px)
    dmax = max(dens.values()) or 1.0
    raw = {n: G.nodes[n].get("betweenness", 0.0) * (dens[n] / dmax) for n in G.nodes()}
    smax = max(raw.values()) or 1.0
    score = {n: round(raw[n] / smax, 4) for n in G.nodes()}
    nx.set_node_attributes(G, {n: round(dens[n], 1) for n in G.nodes()}, "demand")
    nx.set_node_attributes(G, score, "service_crit")
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def sample_node_elevations(G, dem):
    """Per-node elevation (metres) sampled from a DEM registered to the graph's pixel space."""
    Hh, Ww = dem.shape
    out = {}
    for n, d in G.nodes(data=True):
        x, y = d["pos"]
        out[n] = float(dem[int(np.clip(y, 0, Hh - 1)), int(np.clip(x, 0, Ww - 1))])
    return out


def flood_simulation(G, node_elev, n_steps=12, lo_pct=2, hi_pct=88):
    """Rising-water stress test: the lowest-elevation roads submerge first.

    Water rises from the ~lo_pct to ~hi_pct elevation percentile; at each level every node
    below the waterline is removed, and we recompute the Resilience Index (global efficiency
    vs the dry baseline) and the largest-connected-component fraction. This is the real-world
    'which routes survive a flood' analogue of the betweenness ablation test.
    """
    if not node_elev:
        return []
    H = G.copy()
    n0 = H.number_of_nodes()
    base_eff = global_efficiency_weighted(H, n_ref=n0)
    base_lcc = max((len(c) for c in nx.connected_components(H)), default=1)
    elevs = np.array([node_elev[n] for n in H.nodes()], dtype=np.float64)
    levels = np.linspace(np.percentile(elevs, lo_pct), np.percentile(elevs, hi_pct), n_steps)
    out = [dict(step=0, water_level=round(float(levels[0]), 1), submerged_frac=0.0,
                resilience_index=1.0, lcc_fraction=1.0)]
    for i, lvl in enumerate(levels[1:], start=1):
        Hs = H.copy()
        submerged = [n for n in H.nodes() if node_elev[n] <= lvl]
        Hs.remove_nodes_from(submerged)
        eff = global_efficiency_weighted(Hs, n_ref=n0)
        lcc = max((len(c) for c in nx.connected_components(Hs)), default=0)
        out.append(dict(step=i, water_level=round(float(lvl), 1),
                        submerged_frac=round(len(submerged) / n0, 4),
                        resilience_index=round(eff / base_eff, 4) if base_eff > 0 else 0.0,
                        lcc_fraction=round(lcc / base_lcc, 4)))
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
        bc = nx.betweenness_centrality(H, weight="weight", normalized=True,
                                       k=_auto_k(H.number_of_nodes()), seed=42)
        node = max(bc, key=bc.get)
        H.remove_node(node)
        eff = global_efficiency_weighted(H, n_ref=n0)
        lcc = max((len(c) for c in nx.connected_components(H)), default=0)
        results.append(dict(step=i + 1, removed=int(node), betweenness=round(bc[node], 4),
                            resilience_index=round(eff / base_eff, 4) if base_eff > 0 else 0.0,
                            global_efficiency=round(eff, 6),
                            lcc_fraction=round(lcc / base_lcc, 4)))
    return results
