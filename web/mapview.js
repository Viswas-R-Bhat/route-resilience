/* Leaflet map view for GEOREFERENCED tiles — same interface as RoadScene
 * (setData / applyState / setRoute / resetView), so app.js can swap views per tile.
 * Renders the real criticality graph at true lat/lon over a dark basemap + the
 * satellite imagery the model actually ran on. Requires global L (Leaflet). */
const TIER = { critical: "#ff6b5e", important: "#f0b75e", normal: "#5ec8a0" };
const HEAL = "#f5c06b", OFF = "#5a6172";

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
    this.nodeMarkers = {}; this.edgeObjs = []; this.heat = true; this.pos = {};
  }

  _heat(t) {                                   // green (0.33) -> coral (0) in HSL, refined
    const h = Math.round((1 - Math.min(1, t)) * 0.33 * 360);
    return `hsl(${h},62%,58%)`;
  }

  setData(data) {
    this.data = data;
    if (this.overlay) this.map.removeLayer(this.overlay);
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
      const base = e.healed ? HEAL : this._heat(t);
      const pl = L.polyline(latlngs, { color: base, weight: 2 + 4 * t, opacity: .9 });
      pl.userData = { u: e.u, v: e.v, t, healed: e.healed };
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

  applyState(disabled) {
    this.disabled = disabled;
    for (const id in this.nodeMarkers) {
      const m = this.nodeMarkers[id], off = disabled.has(+id);
      m.setStyle({ fillColor: off ? OFF : m.userData.base, fillOpacity: off ? .4 : .95 });
      m.setRadius(off ? m.userData.r * .7 : m.userData.r);
    }
    for (const e of this.edgeObjs) {
      const dead = disabled.has(e.userData.u) || disabled.has(e.userData.v);
      const base = e.userData.healed ? HEAL : (this.heat ? this._heat(e.userData.t) : "#5ec8a0");
      e.setStyle({ color: dead ? OFF : base, opacity: dead ? .4 : .9 });
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
  setHeat(on) { this.heat = on; if (this.disabled) this.applyState(this.disabled); }
  invalidate() { this.map.invalidateSize(); }
}
