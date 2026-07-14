"""Invariant tests for the graph pipeline (P2-P4) and the OSM benchmark math.

CPU-only, no model, no network — tiny synthetic masks/graphs. These are the checks that
fail if the resilience/benchmark logic breaks (the 'error-free' contract).

Run:  python -m pytest tests/ -q
"""
import os, sys
import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph
from phase3_heal import heal_graph, connectivity_report
from phase4_analysis import global_efficiency_weighted, compute_centrality, ablation_simulation
import osm_benchmark as ob


# ---------- helpers ----------
def line_graph(points):
    """Path graph with 'pos' (x, y) node attrs and length weights."""
    G = nx.Graph()
    for i, p in enumerate(points):
        G.add_node(i, pos=tuple(map(float, p)))
    for i in range(len(points) - 1):
        d = float(np.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]))
        G.add_edge(i, i + 1, weight=d, length_px=d)
    return G


def gap_mask(gap_px=20):
    """Horizontal road with a gap in the middle, 3px thick, on 128x128."""
    m = np.zeros((128, 128), np.uint8)
    m[63:66, 4:64 - gap_px // 2] = 1
    m[63:66, 64 + gap_px // 2:124] = 1
    return m


# ---------- Phase 2: mask -> skeleton -> graph ----------
def test_mask_to_graph_basic():
    G = skeleton_to_graph(mask_to_skeleton(gap_mask()))
    assert G.number_of_nodes() >= 2
    assert nx.number_connected_components(G) == 2          # the gap really splits it
    for _, _, d in G.edges(data=True):
        assert d["weight"] > 0


# ---------- Phase 3: healing ----------
def test_heal_bridges_collinear_gap():
    G = skeleton_to_graph(mask_to_skeleton(gap_mask(20)))
    H, healed = heal_graph(G, max_gap_px=40, angular_tolerance_deg=35)
    assert healed >= 1
    assert nx.number_connected_components(H) < nx.number_connected_components(G)
    for _, _, d in H.edges(data=True):
        if d.get("healed"):
            assert d["weight"] > 0 and d["heal_kind"] in ("canopy", "geom")


def test_heal_respects_max_gap():
    G = skeleton_to_graph(mask_to_skeleton(gap_mask(60)))
    H, healed = heal_graph(G, max_gap_px=30, angular_tolerance_deg=35)
    assert healed == 0                                      # gap 2x the limit stays open


def test_heal_never_connects_same_component():
    G = line_graph([(0, 0), (50, 0), (100, 0)])
    H, healed = heal_graph(G, max_gap_px=500, angular_tolerance_deg=89)
    assert healed == 0                                      # single component: nothing to do


def test_prob_guided_heal_traces_evidence():
    G = skeleton_to_graph(mask_to_skeleton(gap_mask(30)))
    prob = np.zeros((128, 128), np.float32)
    prob[62:67, :] = 0.4                                    # faint road evidence across the gap
    H, healed = heal_graph(G, max_gap_px=50, angular_tolerance_deg=35, prob=prob)
    assert healed >= 1
    traced = [d for *_, d in H.edges(data=True) if d.get("heal_kind") == "trace"]
    assert traced and all(d.get("pts") is not None and d["weight"] > 0 for d in traced)
    assert all(0.0 <= d["evidence"] <= 1.0 for d in traced)


def test_prob_veto_blocks_empty_open_ground():
    G = skeleton_to_graph(mask_to_skeleton(gap_mask(20)))
    prob = np.zeros((128, 128), np.float32)                 # model: definitely nothing anywhere
    H, healed = heal_graph(G, max_gap_px=40, angular_tolerance_deg=35, prob=prob)
    assert healed == 0                                      # same gap heals geometrically w/o prob


def test_connectivity_report_math():
    G = nx.Graph(); G.add_nodes_from([(0, {"pos": (0, 0)}), (1, {"pos": (10, 0)})])
    H = G.copy(); H.add_edge(0, 1, weight=10.0)
    rep = connectivity_report(G, H)
    assert rep["components_before"] == 2 and rep["components_after"] == 1
    assert rep["components_merged"] == 1 and rep["healed_edges"] == 1
    assert rep["lcc_frac_before"] == 0.5 and rep["lcc_frac_after"] == 1.0


# ---------- Phase 4: efficiency + resilience ----------
def test_efficiency_fixed_denominator_monotonic():
    G = line_graph([(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)])
    n_ref = G.number_of_nodes()
    e0 = global_efficiency_weighted(G, n_ref=n_ref)
    assert e0 > 0
    prev = e0
    for n in [2, 1]:                                        # remove interior nodes
        G.remove_node(n)
        e = global_efficiency_weighted(G, n_ref=n_ref)
        assert 0 <= e <= prev + 1e-12                       # can only degrade with fixed n_ref
        prev = e


def test_resilience_index_in_unit_range():
    G = nx.grid_2d_graph(4, 4)
    G = nx.convert_node_labels_to_integers(G)
    for i, n in enumerate(G.nodes()):
        G.nodes[n]["pos"] = (float(n % 4) * 10, float(n // 4) * 10)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = 10.0; G.edges[u, v]["length_px"] = 10.0
    compute_centrality(G)
    abl = ablation_simulation(G, top_k=5)
    rs = [r["resilience_index"] for r in abl]
    assert all(0.0 <= r <= 1.0 + 1e-9 for r in rs)
    assert abs(rs[0] - 1.0) < 1e-6                          # step 0 = baseline
    assert all(rs[i + 1] <= rs[i] + 1e-9 for i in range(len(rs) - 1))   # monotone decline


# ---------- OSM benchmark math ----------
def test_coverage_perfect_and_disjoint():
    a = np.zeros((64, 64), np.uint8); a[30:33, :] = 1
    perfect = ob.coverage(a, a, buffer=2)
    assert perfect["recall"] == 1.0 and perfect["precision"] == 1.0 and perfect["f1"] == 1.0
    b = np.zeros_like(a); b[50:53, :] = 1                   # far from a (> buffer)
    disjoint = ob.coverage(a, b, buffer=2)
    assert disjoint["recall"] == 0.0 and disjoint["precision"] == 0.0


def test_path_length_error_identical_graph():
    pts = [(float(x), float(x) * 0.5) for x in range(0, 200, 10)]
    G = line_graph(pts)
    res = ob.path_length_error(G, G.copy(), W=256, H=256, res_m=0.6, n_pairs=30)
    assert res["routing_success"] == 1.0
    assert res["avg_path_length_error_pct"] is not None and res["avg_path_length_error_pct"] < 1.0


def test_rasterize_and_offset():
    lines = [[(10.0, 10.0), (50.0, 10.0)]]
    m = ob.rasterize(lines, 64, 64)
    assert m.sum() >= 40                                    # the polyline landed
    dx, dy = ob.estimate_offset(m, m, search=4, step=2)
    assert abs(dx) <= 2 and abs(dy) <= 2                    # self-alignment within dilation radius
