"""Route Resilience — interactive planner dashboard (Phase 5).

Loads real pipeline artifacts (runs/pipeline/<stem>_{graph.gpickle, report.json, sat.png, mask.png}),
shows the criticality map + Gatekeeper rankings, and lets the planner run a live stress test:
disable top-N gatekeepers (or specific nodes) and watch the Resilience Index recompute in real time.
Clean, readable theme (no neon). Run:  streamlit run src/dashboard.py
"""
import os, sys, glob, json, pickle, io
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from phase4_analysis import global_efficiency_weighted, classify_nodes

PIPE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "pipeline")

st.set_page_config(page_title="Route Resilience", page_icon="🛰️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1300px}
[data-testid="stMetricValue"]{font-size:1.7rem}
h1{font-weight:700;letter-spacing:.01em}
.cap{color:#5a6472;font-size:.85rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_tile(stem):
    G = pickle.load(open(os.path.join(PIPE, f"{stem}_graph.gpickle"), "rb"))
    report = json.load(open(os.path.join(PIPE, f"{stem}_report.json")))
    sat = plt.imread(os.path.join(PIPE, f"{stem}_sat.png"))
    base_eff = global_efficiency_weighted(G)
    ranked = sorted(((n, d.get("betweenness", 0.0)) for n, d in G.nodes(data=True)),
                    key=lambda x: x[1], reverse=True)
    return G, report, sat, base_eff, ranked


def render_map(G, sat, cls, disabled):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sat)
    ebs = [d.get("edge_betweenness", 0) for _, _, d in G.edges(data=True)] or [0]; emax = max(ebs) or 1
    for u, v, d in G.edges(data=True):
        eb = d.get("edge_betweenness", 0) / emax
        if d.get("healed"):
            p1, p2 = G.nodes[u]["pos"], G.nodes[v]["pos"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="#e0a51a", lw=1.8)
        else:
            p = d["pts"]; ax.plot(p[:, 1], p[:, 0], "-",
                                  color=(min(1, .15 + eb), max(0, .55 - eb * .4), max(0, .55 - eb)), lw=1 + 4 * eb)
    cmap = {"critical": "#d63b3b", "important": "#e0941a", "normal": "#1f9bd1"}
    rmap = {"critical": 7, "important": 5, "normal": 3}
    for n, d in G.nodes(data=True):
        x, y = d["pos"]
        if n in disabled:
            ax.plot(x, y, "x", color="#d63b3b", ms=11, mew=3)
        else:
            t = cls[n]; ax.plot(x, y, "o", color=cmap[t], ms=rmap[t], mec="white", mew=.5)
    ax.axis("off"); fig.tight_layout(pad=0)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf


# ---------------- UI ----------------
st.title("🛰️ Route Resilience")
st.markdown('<div class="cap">Occlusion-robust road extraction → routable graph → criticality & resilience · ISRO BAH 2026 · PS-4</div>', unsafe_allow_html=True)

tiles = sorted(os.path.splitext(os.path.basename(p))[0].replace("_report", "")
               for p in glob.glob(os.path.join(PIPE, "*_report.json")))
if not tiles:
    st.warning("No processed tiles found. Run: python src/run_pipeline.py --image <sat.jpg> --device cpu --out runs/pipeline")
    st.stop()

with st.sidebar:
    st.header("Controls")
    stem = st.selectbox("Region (processed tile)", tiles)
    G, report, sat, base_eff, ranked = load_tile(stem)
    cls = classify_nodes(G)
    n_nodes = G.number_of_nodes()
    abl_order = [r["removed"] for r in report.get("resilience", []) if r.get("removed") is not None]
    max_n = max(1, min(len(abl_order), 12))
    st.markdown("**Stress test — simulate failures**")
    topn = st.slider("Disable top-N Gatekeeper Nodes", 0, max_n, 0,
                     help="Sequentially removes the most critical intersection (betweenness recomputed each step) — models flood / accident / closure.")
    manual = st.multiselect("…or disable specific nodes", [n for n, _ in ranked],
                            help="Pick intersections to close manually.")

disabled = set(abl_order[:topn]) | set(manual)

# live recompute — fixed baseline-N normalization so R stays in [0,1]
H = G.copy(); H.remove_nodes_from(disabled)
eff = global_efficiency_weighted(H, n_ref=n_nodes)
R = round(eff / base_eff, 3) if base_eff > 0 else 0.0
comps = list(nx.connected_components(H))
lcc = max((len(c) for c in comps), default=0)
lcc_frac = lcc / max(1, n_nodes)
status = "STABLE" if R > 0.7 else ("DEGRADED" if R > 0.4 else "CRITICAL")
status_color = "#1f9bd1" if R > 0.7 else ("#e0941a" if R > 0.4 else "#d63b3b")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Resilience Index", f"{R:.2f}", f"{status}")
c2.metric("Road nodes", f"{n_nodes}", f"{G.number_of_edges()} edges")
c3.metric("Connectivity (LCC)", f"{lcc_frac*100:.0f}%", f"{report['connectivity']['components_after']} components")
c4.metric("Gatekeepers disabled", f"{len(disabled)}", f"of {max_n} ranked")
st.markdown(f"<div style='height:4px;background:{status_color};border-radius:3px;margin:.2rem 0 1rem'></div>", unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.caption("Criticality map — red = Gatekeeper (high betweenness), gold dashed = healed gap, ✕ = disabled")
    st.image(render_map(G, sat, cls, disabled), width="stretch")

with right:
    st.caption("Top Gatekeeper Nodes")
    rows = [{"node": int(n), "betweenness": round(b, 4), "tier": cls[n], "disabled": "✕" if n in disabled else ""}
            for n, b in ranked[:10]]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Resilience under progressive gatekeeper removal (baseline stress test)")
    abl = report.get("resilience", [])
    if abl:
        st.line_chart({"Resilience Index": [r["resilience_index"] for r in abl],
                       "LCC fraction": [r["lcc_fraction"] for r in abl]})
