import { startBackground, RoadScene } from "./scene.js?v=6";
import { MapView } from "./mapview.js?v=6";

const $ = s => document.querySelector(s);
const TIERCOL = { critical: "#ff6b5e", important: "#f0b75e", normal: "#5ec8a0" };

let DATA, ADJ, IDS, NTOTAL, BASE_EFF, ABL_ORDER, topN = 0, floodStep = 0, gkMode = "bc";
let OD = null, BASE_OD_PATH = [], SECTOR = [];   // representative route + sector pairs for travel-time
const manual = new Set();           // nodes disabled by click (on top of slider)
let scene, mapView = null, activeView, chart;

startBackground($("#bg-canvas"));
scene = new RoadScene($("#scene-canvas"), id => toggleManual(id));
activeView = scene;
chart = echarts.init($("#resilience-chart"), null, { renderer: "canvas" });
addEventListener("resize", () => chart.resize());

/* ---------- graph + resilience math (mirrors phase4_analysis) ---------- */
function buildGraph(data) {
  ADJ = {}; IDS = data.nodes.map(n => n.id); NTOTAL = IDS.length;
  IDS.forEach(i => ADJ[i] = []);
  for (const e of data.edges) { ADJ[e.u].push([e.v, e.w]); ADJ[e.v].push([e.u, e.w]); }
}
function efficiency(disabled) {
  const live = IDS.filter(i => !disabled.has(i));
  if (live.length < 2) return 0;
  let total = 0;
  for (const s of live) {
    const { dist } = dijkstra(s, disabled);          // heap Dijkstra (scales to large graphs)
    for (const t of live) if (t !== s && dist[t] < Infinity && dist[t] > 0) total += 1 / dist[t];
  }
  return total / (NTOTAL * (NTOTAL - 1));
}
function lccFraction(disabled) {
  const live = new Set(IDS.filter(i => !disabled.has(i)));
  let best = 0; const seen = new Set();
  for (const s of live) {
    if (seen.has(s)) continue;
    let sz = 0; const stack = [s];
    while (stack.length) { const u = stack.pop(); if (seen.has(u)) continue; seen.add(u); sz++; for (const [v] of ADJ[u]) if (live.has(v) && !seen.has(v)) stack.push(v); }
    best = Math.max(best, sz);
  }
  return best / NTOTAL;
}
function floodSubmerged() {                          // road nodes below the current waterline
  if (floodStep <= 0 || !DATA.flood || !DATA.flood[floodStep]) return [];
  const lvl = DATA.flood[floodStep].water_level;
  return DATA.nodes.filter(n => n.elev != null && n.elev <= lvl).map(n => n.id);
}
const disabledSet = () => new Set([...ABL_ORDER.slice(0, topN), ...manual, ...floodSubmerged()]);

/* ---------- shortest paths (rerouting + travel-time, PS-4 Phase IV) ---------- */
function dijkstra(src, disabled) {
  const dist = {}, prev = {};
  IDS.forEach(i => dist[i] = Infinity);
  if (disabled.has(src)) return { dist, prev };
  dist[src] = 0;
  const heap = [[0, src]];                          // binary min-heap of [dist, node]
  const swap = (a, b) => { const t = heap[a]; heap[a] = heap[b]; heap[b] = t; };
  const up = i => { while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; swap(p, i); i = p; } };
  const down = () => { let i = 0; const n = heap.length; for (;;) { let s = i, l = 2 * i + 1, r = 2 * i + 2; if (l < n && heap[l][0] < heap[s][0]) s = l; if (r < n && heap[r][0] < heap[s][0]) s = r; if (s === i) break; swap(s, i); i = s; } };
  while (heap.length) {
    const [d, u] = heap[0]; const last = heap.pop(); if (heap.length) { heap[0] = last; down(); }
    if (d > dist[u]) continue;
    for (const [v, w] of ADJ[u]) {
      if (disabled.has(v)) continue;
      const nd = d + w;
      if (nd < dist[v]) { dist[v] = nd; prev[v] = u; heap.push([nd, v]); up(heap.length - 1); }
    }
  }
  return { dist, prev };
}
function pathOf(prev, dst) {
  const p = [dst]; let u = dst;
  while (prev[u] !== undefined) { u = prev[u]; p.push(u); }
  return p.reverse();
}
// Pick a representative O->D route whose baseline shortest path runs THROUGH the top
// gatekeeper (so disabling it forces a visible reroute), + sample sector pairs for travel-time.
function computeRoutes() {
  const empty = new Set();
  OD = null; BASE_OD_PATH = []; SECTOR = [];
  if (NTOTAL < 2) return;
  const g = (DATA.gatekeepers && DATA.gatekeepers.length) ? DATA.gatekeepers[0].id : null;
  const DIST = {}, PREV = {};
  for (const s of IDS) { const r = dijkstra(s, empty); DIST[s] = r.dist; PREV[s] = r.prev; }
  let best = -1, O = null, D = null;
  for (const s of IDS) for (const t of IDS) {
    if (t === s || DIST[s][t] === Infinity || (g != null && (s === g || t === g))) continue;
    if (g == null || pathOf(PREV[s], t).includes(g)) {
      if (DIST[s][t] > best) { best = DIST[s][t]; O = s; D = t; }
    }
  }
  if (O == null) {  // fallback: plain weighted diameter
    for (const s of IDS) for (const t of IDS)
      if (t !== s && DIST[s][t] < Infinity && DIST[s][t] > best) { best = DIST[s][t]; O = s; D = t; }
  }
  if (O == null) return;
  OD = [O, D]; BASE_OD_PATH = pathOf(PREV[O], D);
  const all = [];
  for (const s of IDS) for (const t of IDS) if (t > s && DIST[s][t] < Infinity) all.push([s, t, DIST[s][t]]);
  const stepp = Math.max(1, Math.floor(all.length / 40));
  for (let i = 0; i < all.length; i += stepp) SECTOR.push(all[i]);
}

/* ---------- UI ---------- */
function kpi(label, val, sub, accent) {
  return `<div class="kpi" style="--accent:${accent}"><div class="k-label">${label}</div>
    <div class="k-val">${val}</div><div class="k-sub">${sub}</div></div>`;
}
function render() {
  const fset = new Set(floodSubmerged());
  const dis = disabledSet();
  // when ONE stressor is active, use the matching precomputed Python curve (instant at any size)
  let pre = null;
  if (floodStep > 0 && topN === 0 && manual.size === 0 && DATA.flood && DATA.flood[floodStep]) pre = DATA.flood[floodStep];
  else if (floodStep === 0 && manual.size === 0 && DATA.resilience && DATA.resilience[topN]) pre = DATA.resilience[topN];
  const R = pre ? pre.resilience_index : (BASE_EFF > 0 ? efficiency(dis) / BASE_EFF : 0);
  const lcc = lccFraction(dis);   // absolute connectivity (cheap BFS) — NOT the curve's baseline-ratio
  const status = R > 0.7 ? "STABLE" : R > 0.4 ? "DEGRADED" : "CRITICAL";
  const col = R > 0.7 ? "#5ec8a0" : R > 0.4 ? "#f0b75e" : "#ff6b5e";
  const gk = DATA.gatekeepers[0];

  $("#kpis").innerHTML = [
    kpi("Resilience Index", R.toFixed(2), status, col),
    kpi("Road nodes", NTOTAL, `${DATA.edges.length} edges`, "#c9ccd3"),
    kpi("Connectivity (LCC)", (lcc * 100).toFixed(0) + "%", `${DATA.healed_bridges} healed bridges`, "#5ec8a0"),
    kpi("Top gatekeeper", "N-" + gk.id, "betweenness " + gk.bc.toFixed(3), "#ff6b5e"),
    kpi("Disabled", dis.size, `of ${NTOTAL} nodes`, "#f5c06b"),
  ].join("");

  $("#ri-num").textContent = R.toFixed(2); $("#ri-num").style.color = col; $("#ri-num").style.textShadow = `0 0 34px ${col}40`;
  $("#ri-status").textContent = status; $("#ri-status").style.color = col;
  $("#status-text").textContent = status;
  const dot = $("#status-pill .dot"); dot.style.background = col; dot.style.boxShadow = `0 0 12px ${col}`;
  $("#ablate-n").textContent = topN;

  // flood readout (water level + submerged share)
  const fm = $("#flood-m"), fsub = $("#flood-sub");
  if (fm && DATA.flood && DATA.flood.length) {
    if (floodStep > 0) {
      const f = DATA.flood[floodStep];
      fm.textContent = `${f.water_level.toFixed(0)} m`;
      fsub.textContent = `${Math.round(f.submerged_frac * 100)}% of roads submerged · resilience ${f.resilience_index.toFixed(2)}`;
    } else { fm.textContent = "dry"; fsub.textContent = ""; }
  }

  activeView.applyState(dis, fset);

  // ---- rerouting + travel-time increase (PS-4 Phase IV) ----
  const distCache = {};
  const dfrom = s => (distCache[s] || (distCache[s] = dijkstra(s, dis).dist));
  if (OD) {
    const dj = dijkstra(OD[0], dis);
    const reach = !dis.has(OD[0]) && !dis.has(OD[1]) && dj.dist[OD[1]] < Infinity;
    activeView.setRoute(BASE_OD_PATH, reach ? pathOf(dj.prev, OD[1]) : [], !reach);
  }
  let ttTxt = "+0%", ttSub = `${SECTOR.length} sector routes`, ttCol = col;
  if (SECTOR.length) {
    let sev = 0; const inc = [];
    for (const [s, t, d0] of SECTOR) {
      const d1 = dfrom(s)[t];
      if (!(d1 < Infinity)) sev++; else inc.push(d1 / d0 - 1);
    }
    const mean = inc.length ? inc.reduce((a, b) => a + b, 0) / inc.length : 0;
    const cutPct = Math.round(100 * sev / SECTOR.length);
    if (dis.size === 0) { ttTxt = "+0%"; ttSub = `${SECTOR.length} sector routes`; ttCol = "#5ec8a0"; }
    else if (inc.length === 0) { ttTxt = "cut off"; ttSub = `all ${sev} routes severed`; ttCol = "#ff6b5e"; }
    else if (sev > inc.length) {                       // severing dominates -> lead with it (honest)
      ttTxt = `${cutPct}% cut`; ttSub = `+${Math.round(mean * 100)}% on ${inc.length} open routes`; ttCol = "#ff6b5e";
    } else {                                           // mostly detours -> lead with travel-time
      ttTxt = `+${Math.round(mean * 100)}%`;
      ttSub = sev ? `${sev} severed · ${inc.length} open` : `${inc.length} sector routes`;
      ttCol = sev ? "#f0b75e" : (mean > 0.25 ? "#f0b75e" : "#5ec8a0");
    }
  }
  const ttn = $("#tt-num"); if (ttn) { ttn.textContent = ttTxt; ttn.style.color = ttCol; ttn.style.textShadow = `0 0 30px ${ttCol}40`; }
  const tts = $("#tt-sub"); if (tts) { tts.textContent = ttSub; tts.style.color = ttCol; }

  // gatekeeper rows off-state
  document.querySelectorAll(".gk-row").forEach(r => r.classList.toggle("off", dis.has(+r.dataset.id)));
  // chart current marker
  chart.setOption({ series: [{ markLine: { silent: true, symbol: "none", lineStyle: { color: col, type: "dashed" }, data: [{ xAxis: topN }] } }] });
}

function renderGatekeepers() {
  const svc = gkMode === "svc" && DATA.service_gatekeepers && DATA.service_gatekeepers.length;
  const list = svc ? DATA.service_gatekeepers : DATA.gatekeepers;
  $("#gatekeepers").innerHTML = list.map((g, i) => {
    const id = svc ? g.node : g.id;
    const metric = svc ? `SC ${(+g.service_crit).toFixed(2)}` : `BC ${(+g.bc).toFixed(3)}`;
    return `<div class="gk-row" role="listitem" data-id="${id}" tabindex="0">
       <span class="gk-rank">#${i + 1}</span>
       <span class="gk-dot" style="background:${TIERCOL[g.tier]};box-shadow:0 0 10px ${TIERCOL[g.tier]}"></span>
       <span class="gk-id">N-${id}</span><span class="gk-bc">${metric}</span></div>`;
  }).join("");
  document.querySelectorAll(".gk-row").forEach(r => {
    const id = +r.dataset.id;
    r.onclick = () => toggleManual(id);
    r.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleManual(id); } };
  });
}

function renderChart() {
  const steps = DATA.resilience.map(r => r.step);
  const ri = DATA.resilience.map(r => r.resilience_index);
  const lcc = DATA.resilience.map(r => r.lcc_fraction);
  chart.setOption({
    grid: { left: 40, right: 16, top: 30, bottom: 26 },
    legend: { top: 0, textStyle: { color: "#a1a1a6", fontSize: 10 }, itemWidth: 14, itemHeight: 8 },
    tooltip: { trigger: "axis", backgroundColor: "rgba(12,14,20,.92)", borderColor: "rgba(255,255,255,.1)", textStyle: { color: "#f5f5f7" } },
    xAxis: { type: "category", data: steps, name: "nodes removed", nameTextStyle: { color: "#6e6e73", fontSize: 9 }, axisLine: { lineStyle: { color: "rgba(255,255,255,.12)" } }, axisLabel: { color: "#a1a1a6", fontSize: 10 } },
    yAxis: { type: "value", min: 0, max: 1, splitLine: { lineStyle: { color: "rgba(255,255,255,.05)" } }, axisLabel: { color: "#a1a1a6", fontSize: 10 } },
    series: [
      { name: "Resilience", type: "line", data: ri, smooth: true, symbol: "circle", symbolSize: 6, lineStyle: { color: "#5ec8a0", width: 3 }, itemStyle: { color: "#5ec8a0" }, areaStyle: { color: "rgba(94,200,160,.14)" } },
      { name: "LCC", type: "line", data: lcc, smooth: true, symbol: "none", lineStyle: { color: "#f0b75e", width: 2, type: "dashed" }, itemStyle: { color: "#f0b75e" } },
    ],
  }, true);
}

function toggleManual(id) { manual.has(id) ? manual.delete(id) : manual.add(id); render(); }

/* ---------- load a region ---------- */
async function loadTile(stem) {
  $("#loader").classList.remove("hidden");
  DATA = await (await fetch(`data/${stem}.json?t=${Date.now()}`)).json();
  buildGraph(DATA);
  BASE_EFF = efficiency(new Set());
  ABL_ORDER = DATA.resilience.map(r => r.removed).filter(r => r !== null && r !== undefined);
  topN = 0; manual.clear(); floodStep = 0;
  const slider = $("#ablate"); slider.max = ABL_ORDER.length; slider.value = 0;
  // flood control: only when this tile has a DEM-derived flood curve
  const hasFlood = !!(DATA.flood && DATA.flood.length > 1);
  ["#flood-wrap", "#flood", "#flood-sub"].forEach(s => { const el = $(s); if (el) el.hidden = !hasFlood; });
  if (hasFlood) { const f = $("#flood"); f.max = DATA.flood.length - 1; f.value = 0; }
  const gkb = $("#gk-mode"); if (gkb) gkb.hidden = !(DATA.service_gatekeepers && DATA.service_gatekeepers.length);
  // pick the view: Leaflet map for georeferenced tiles, 3D scene otherwise
  const geo = !!DATA.geo;
  const sc = $("#scene-canvas"), mp = $("#map");
  if (geo) {
    sc.style.display = "none"; mp.style.display = "block";
    if (!mapView) mapView = new MapView("map", id => toggleManual(id));
    activeView = mapView;
  } else {
    mp.style.display = "none"; sc.style.display = "block";
    activeView = scene;
  }
  const tools = document.querySelector(".scene-tools"); if (tools) tools.style.display = geo ? "none" : "flex";
  const hint = document.querySelector(".scene-hint");
  if (hint) hint.textContent = geo ? "drag to pan · scroll to zoom · click a node to disable it"
                                   : "Drag to orbit · scroll to zoom · click a node to disable it";

  activeView.setData(DATA);
  computeRoutes();
  renderGatekeepers(); renderChart(); render();
  setTimeout(() => $("#loader").classList.add("hidden"), 350);
}

/* ---------- dynamic analysis: any location -> live stress graph ---------- */
function setLoader(msg, show = true) {
  const l = $("#loader"); if (!l) return;
  const s = l.querySelector("span"); if (s && msg) s.textContent = msg;
  l.classList.toggle("hidden", !show);
}
async function geocode(q) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`;
  const d = await (await fetch(url, { headers: { Accept: "application/json" } })).json();
  return d.length ? { lat: +d[0].lat, lon: +d[0].lon } : null;
}
async function analyzeLocation() {
  const raw = $("#loc-input").value.trim();
  if (!raw) return;
  const grid = $("#loc-size").value;
  let lat, lon;
  const m = raw.match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/);
  setLoader("Locating…", true);
  if (m) { lat = +m[1]; lon = +m[2]; }
  else {
    let g = null;
    try { g = (await geocode(raw + ", Bengaluru, Karnataka, India")) || (await geocode(raw)); } catch (e) {}
    if (!g) { setLoader("Place not found — try “lat, lon”", true); setTimeout(() => setLoader("", false), 2400); return; }
    lat = g.lat; lon = g.lon;
  }
  const est = grid >= 16 ? "~70s" : grid >= 12 ? "~55s" : "~40s";
  setLoader(`Fetching satellite imagery & running the model… (${est})`, true);
  try {
    const j = await (await fetch(`/api/analyze?lat=${lat}&lon=${lon}&grid=${grid}`)).json();
    if (j.error) { setLoader(`Analysis failed: ${j.error}`, true); setTimeout(() => setLoader("", false), 3200); return; }
    const sel = $("#region");
    if (![...sel.options].some(o => o.value === j.stem)) {
      const opt = document.createElement("option");
      opt.value = j.stem; opt.textContent = `Live · ${(+lat).toFixed(3)}, ${(+lon).toFixed(3)} · map`;
      sel.insertBefore(opt, sel.firstChild);
    }
    sel.value = j.stem;
    await loadTile(j.stem);                       // hides the loader at the end
  } catch (e) {
    setLoader("Server error — is the API running? (python src/server.py)", true);
    setTimeout(() => setLoader("", false), 3500);
  }
}

/* ---------- wire controls ---------- */
async function init() {
  const man = await (await fetch(`data/manifest.json?t=${Date.now()}`)).json();
  const sel = $("#region");
  sel.innerHTML = man.tiles.map(t => `<option value="${t.stem}">${t.label || t.stem} · ${t.nodes} nodes${t.geo ? " · map" : ""}</option>`).join("");
  sel.onchange = () => loadTile(sel.value);
  $("#loc-go").onclick = analyzeLocation;
  $("#loc-input").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); analyzeLocation(); } });
  $("#ablate").oninput = e => { topN = +e.target.value; render(); };
  $("#flood").oninput = e => { floodStep = +e.target.value; render(); };
  $("#reset-stress").onclick = () => { topN = 0; manual.clear(); floodStep = 0;
    $("#ablate").value = 0; const f = $("#flood"); if (f) f.value = 0; render(); };
  const gkb = $("#gk-mode");
  if (gkb) gkb.onclick = () => { gkMode = gkMode === "bc" ? "svc" : "bc";
    gkb.setAttribute("aria-pressed", gkMode === "svc"); gkb.textContent = gkMode === "svc" ? "Service" : "Betweenness";
    renderGatekeepers(); render(); };
  $("#reset-view").onclick = () => activeView.resetView();
  const tr = $("#toggle-rotate"); tr.onclick = () => { const on = tr.getAttribute("aria-pressed") !== "true"; tr.setAttribute("aria-pressed", on); tr.textContent = `Auto-orbit: ${on ? "On" : "Off"}`; activeView.setAutoRotate(on); };
  const th = $("#toggle-heat"); th.onclick = () => { const on = th.getAttribute("aria-pressed") !== "true"; th.setAttribute("aria-pressed", on); th.textContent = `Criticality heat: ${on ? "On" : "Off"}`; activeView.setHeat(on); render(); };
  await loadTile(man.default || man.tiles[0].stem);
}
init().catch(e => { console.error(e); $("#loader").innerHTML = "<span>Failed to load data. Run src/export_web.py</span>"; });
