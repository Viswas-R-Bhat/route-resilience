/* Leaflet map view for GEOREFERENCED tiles — same interface as RoadScene
 * (setData / applyState / setRoute / resetView), so app.js can swap views per tile.
 * Renders the real criticality graph at true lat/lon over a dark basemap + the
 * satellite imagery the model actually ran on. Requires global L (Leaflet). */
const TIER = { critical: "#ff6b5e", important: "#f0b75e", normal: "#5ec8a0" };
const HEAL = "#f5c06b", CANOPY = "#7ed957", PROB = "#b388ff", OFF = "#5a6172", FLOOD = "#3b9eff";
const healColor = e => e.healed ? (e.hk === "canopy" ? CANOPY : e.hk === "prob" ? PROB : HEAL) : null;
// confidence -> opacity: inferred bridges look fainter (faint-road=high, geom=med, canopy=low, saturated-canopy=vlow)
const CONF_OP = { high: .9, med: .85, low: .66, vlow: .48 };
const healOpacity = e => e.healed ? (CONF_OP[e.conf] ?? .85) : .9;

export class MapView {
  constructor(elId, onPick) {
    this.onPick = onPick;
    this.map = L.map(elId, { zoomControl: true, attributionControl: true, preferCanvas: true });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      { maxZoom: 20, attribution: '&copy; OpenStreetMap &copy; CARTO' }).addTo(this.map);
    this.overlay = null;
    this.edgeLayer = L.layerGroup().addTo(this.map);
    this.nodeLayer = L.layerGroup().addTo(this.map);
    this.routeLayer = L.layerGroup().addTo(this.map);
    this.nodeMarkers = {}; this.edgeObjs = []; this.heat = true; this.pos = {}; this.osmLayer = null;
  }

  setOsm(lines) {                                  // overlay OSM ground-truth roads (lat/lng polylines)
    if (this.osmLayer) { this.map.removeLayer(this.osmLayer); this.osmLayer = null; }
    if (!lines || !lines.length) return;
    const renderer = L.svg({ padding: 0.5 });      // own SVG renderer: paints immediately (the
    this.osmLayer = L.layerGroup();                // canvas renderer defers post-hoc layers)
    for (const ln of lines)
      if (ln.length >= 2) L.polyline(ln, { renderer, color: "#ffffff", weight: 2, opacity: .9, dashArray: "5 4", interactive: false }).addTo(this.osmLayer);
    this.osmLayer.addTo(this.map);
  }

  _heat(t) {                                   // green (0.33) -> coral (0) in HSL, refined
    const h = Math.round((1 - Math.min(1, t)) * 0.33 * 360);
    return `hsl(${h},62%,58%)`;
  }

  setData(data) {
    this.data = data;
    if (this.overlay) this.map.removeLayer(this.overlay);
    if (this.osmLayer) { this.map.removeLayer(this.osmLayer); this.osmLayer = null; }
    this.edgeLayer.clearLayers(); this.nodeLayer.clearLayers(); this.routeLayer.clearLayers();
    this.nodeMarkers = {}; this.edgeObjs = []; this.pos = {};
    const b = data.bounds, bounds = [[b.south, b.west], [b.north, b.east]];
    this.overlay = L.imageOverlay(`data/${data.tex}`, bounds, { opacity: 0.5, interactive: false }).addTo(this.map);
    this.map.fitBounds(bounds);
    data.nodes.forEach(n => this.pos[n.id] = [n.lat, n.lng]);

    const emax = Math.max(1e-6, ...data.edges.map(e => e.bc));
    for (const e of data.edges) {
      const t = e.bc / emax;
      const latlngs = (e.ll && e.ll.length >= 2) ? e.ll : [this.pos[e.u], this.pos[e.v]];
      const base = healColor(e) || this._heat(t);
      const pl = L.polyline(latlngs, { color: base, weight: 2 + 4 * t, opacity: healOpacity(e) });
      pl.userData = { u: e.u, v: e.v, t, healed: e.healed, hk: e.hk, conf: e.conf };
      pl.addTo(this.edgeLayer); this.edgeObjs.push(pl);
    }
    const bcmax = Math.max(1e-6, ...data.nodes.map(n => n.bc));
    for (const n of data.nodes) {
      const r = 3 + (n.bc / bcmax) * 9;
      const m = L.circleMarker([n.lat, n.lng], { radius: r, color: "#0b0d12", weight: 1,
        fillColor: TIER[n.tier], fillOpacity: .95 });
      m.userData = { id: n.id, tier: n.tier, base: TIER[n.tier], r };
      m.on("click", () => this.onPick && this.onPick(n.id));
      m.addTo(this.nodeLayer); this.nodeMarkers[n.id] = m;
    }
    setTimeout(() => this.map.invalidateSize(), 60);
  }

  applyState(disabled, flooded = new Set()) {
    this.disabled = disabled; this.flooded = flooded;
    for (const id in this.nodeMarkers) {
      const m = this.nodeMarkers[id], i = +id, fl = flooded.has(i), off = disabled.has(i);
      const fill = fl ? FLOOD : (off ? OFF : m.userData.base);
      m.setStyle({ fillColor: fill, fillOpacity: fl ? .9 : (off ? .4 : .95) });
      m.setRadius(off && !fl ? m.userData.r * .7 : m.userData.r);
    }
    for (const e of this.edgeObjs) {
      const fl = flooded.has(e.userData.u) || flooded.has(e.userData.v);
      const dead = disabled.has(e.userData.u) || disabled.has(e.userData.v);
      const base = healColor(e.userData) || (this.heat ? this._heat(e.userData.t) : "#5ec8a0");
      e.setStyle({ color: fl ? FLOOD : (dead ? OFF : base), opacity: fl ? .85 : (dead ? .4 : healOpacity(e.userData)) });
    }
  }

  setRoute(baseIds, curIds, severed) {
    this.routeLayer.clearLayers();
    if (!baseIds || baseIds.length < 2 || !this.pos) return;
    const toLL = ids => ids.map(i => this.pos[i]).filter(Boolean);
    L.polyline(toLL(baseIds), { color: "#aab2bf", weight: 3, opacity: .5, dashArray: "4 6" }).addTo(this.routeLayer);
    L.circleMarker(this.pos[baseIds[0]], { radius: 6, color: "#fff", weight: 2, fillColor: "#fff", fillOpacity: 1 }).addTo(this.routeLayer);
    L.circleMarker(this.pos[baseIds[baseIds.length - 1]], { radius: 6, color: "#fff", weight: 2, fillColor: "#ff8a5b", fillOpacity: 1 }).addTo(this.routeLayer);
    if (severed || !curIds || curIds.length < 2) {
      L.polyline(toLL(baseIds), { color: "#ff6b5e", weight: 4, opacity: .85 }).addTo(this.routeLayer);
    } else {
      L.polyline(toLL(curIds), { color: "#ffffff", weight: 4, opacity: .95 }).addTo(this.routeLayer);
    }
  }

  resetView() {
    if (this.data) { const b = this.data.bounds; this.map.fitBounds([[b.south, b.west], [b.north, b.east]]); }
  }
  setAutoRotate() { /* n/a for map */ }
  setHeat(on) { this.heat = on; if (this.disabled) this.applyState(this.disabled, this.flooded); }
  invalidate() { this.map.invalidateSize(); }
}
