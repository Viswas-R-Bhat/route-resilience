import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SIZE = 10;                 // world size of the satellite plane
const TIER = { critical: 0xff6b5e, important: 0xf0b75e, normal: 0x5ec8a0 };
const HEAL = 0xf5c06b, CANOPY = 0x7ed957, OFF = 0x5a6172, FLOOD = 0x3b9eff;
const healColor = e => e.hk === "canopy" ? CANOPY : HEAL;
// confidence -> glow: inferred bridges glow dimmer (geom=med, canopy=low, saturated-canopy=vlow)
const CONF_GLOW = { high: .55, med: .5, low: .32, vlow: .18 };
const healGlow = e => CONF_GLOW[e.conf] ?? .5;

/* ============================================================
 *  Animated space + rocket-launch background (decorative)
 * ============================================================ */
export function startBackground(canvas) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  r.setPixelRatio(Math.min(devicePixelRatio, 2));
  r.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 600);
  cam.position.set(0, 0, 60);

  // a single warm "sun" from upper-left -> soft terminator on the planet
  scene.add(new THREE.AmbientLight(0x1c2636, 0.55));
  const sun = new THREE.DirectionalLight(0xfff0dd, 2.4);
  sun.position.set(-1, 0.45, 0.7); scene.add(sun);

  // --- procedural planet texture (no external assets) ---
  function makeCanvas(w, h, draw){ const c = document.createElement("canvas"); c.width = w; c.height = h; draw(c.getContext("2d")); return c; }
  const planetTex = new THREE.CanvasTexture(makeCanvas(512, 256, x => {
    x.fillStyle = "#0a111d"; x.fillRect(0, 0, 512, 256);
    for (let i = 0; i < 1000; i++) {
      const px = Math.random()*512, py = Math.random()*256, rad = 2 + Math.random()*28;
      const g = x.createRadialGradient(px, py, 0, px, py, rad);
      const tone = Math.random() < .55 ? "34,52,66" : "16,26,38";
      g.addColorStop(0, `rgba(${tone},${.05 + Math.random()*.13})`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g; x.beginPath(); x.arc(px, py, rad, 0, 7); x.fill();
    }
  }));
  planetTex.colorSpace = THREE.SRGBColorSpace;

  // --- the planet: large, lower-right, partly below the fold ---
  const planet = new THREE.Group();
  planet.add(new THREE.Mesh(
    new THREE.SphereGeometry(26, 64, 64),
    new THREE.MeshStandardMaterial({ map: planetTex, roughness: 1, metalness: 0, emissive: 0x04070d, emissiveIntensity: 1 })
  ));
  // atmosphere rim glow (backside additive sphere)
  planet.add(new THREE.Mesh(
    new THREE.SphereGeometry(27.7, 48, 48),
    new THREE.MeshBasicMaterial({ color: 0xffb478, transparent: true, opacity: .34, side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false })
  ));
  planet.position.set(25, -23, -8); planet.rotation.z = 0.2; scene.add(planet);

  // --- soft sun flare upper-left ---
  const flareTex = new THREE.CanvasTexture(makeCanvas(256, 256, x => {
    const g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    g.addColorStop(0, "rgba(255,238,205,.9)"); g.addColorStop(.25, "rgba(255,182,120,.4)"); g.addColorStop(1, "rgba(255,150,90,0)");
    x.fillStyle = g; x.fillRect(0, 0, 256, 256);
  }));
  const flare = new THREE.Sprite(new THREE.SpriteMaterial({ map: flareTex, transparent: true, opacity: .75, blending: THREE.AdditiveBlending, depthWrite: false }));
  flare.scale.set(64, 64, 1); flare.position.set(-44, 27, -34); scene.add(flare);

  // --- faint orbit rings centred on the planet ---
  const orbits = new THREE.Group(); orbits.position.copy(planet.position);
  [[40, .08], [53, .05]].forEach(([rad, op]) => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(rad, .07, 8, 220),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: op }));
    ring.rotation.set(1.18, .35, 0); orbits.add(ring);
  });
  scene.add(orbits);

  // --- a small satellite orbiting along the inner ring ---
  const satOrbit = new THREE.Group(); satOrbit.position.copy(planet.position); satOrbit.rotation.set(1.18, .35, 0);
  const satPivot = new THREE.Group(); satOrbit.add(satPivot);
  const sat = new THREE.Group();
  sat.add(new THREE.Mesh(new THREE.BoxGeometry(.9, .9, 1.5),
    new THREE.MeshStandardMaterial({ color: 0xd9dde6, metalness: .6, roughness: .35, emissive: 0xff8a5b, emissiveIntensity: .25 })));
  const panelMat = new THREE.MeshStandardMaterial({ color: 0x2a4a6b, metalness: .3, roughness: .5, emissive: 0x2b4a6b, emissiveIntensity: .35, side: THREE.DoubleSide });
  [-1.7, 1.7].forEach(dx => { const p = new THREE.Mesh(new THREE.PlaneGeometry(2.3, .95), panelMat); p.position.x = dx; sat.add(p); });
  sat.position.set(40, 0, 0); satPivot.add(sat); scene.add(satOrbit);

  // --- minimal starfield ---
  const N = 700, sp = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) { const rr = 80 + Math.random()*200, th = Math.random()*6.283, ph = Math.acos(2*Math.random()-1);
    sp[i*3] = rr*Math.sin(ph)*Math.cos(th); sp[i*3+1] = rr*Math.sin(ph)*Math.sin(th); sp[i*3+2] = rr*Math.cos(ph)-40; }
  const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  const stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: .34, transparent: true, opacity: .5, depthWrite: false }));
  scene.add(stars);

  function resize(){ r.setSize(innerWidth, innerHeight, false); cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix(); r.render(scene, cam); }
  resize(); addEventListener("resize", resize);

  const reduce = matchMedia("(prefers-reduced-motion:reduce)").matches;
  if (reduce) { satPivot.rotation.z = 0.7; r.render(scene, cam); return; }
  let t = 0;
  (function loop(){
    requestAnimationFrame(loop);
    t += .004;
    planet.rotation.y += .0007;
    stars.rotation.y = t * .015;
    satPivot.rotation.z = t * .35;
    sat.rotation.y += .01;
    cam.position.x = Math.sin(t * .12) * 1.6; cam.lookAt(0, 0, -8);
    r.render(scene, cam);
  })();
}

/* ============================================================
 *  Live 3D road network (the data centerpiece)
 * ============================================================ */
export class RoadScene {
  constructor(canvas, onPick) {
    this.canvas = canvas; this.onPick = onPick;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    this.cam.position.set(0, 9, 9);
    this.controls = new OrbitControls(this.cam, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = .08;
    this.controls.maxPolarAngle = Math.PI * 0.49; this.controls.minDistance = 4; this.controls.maxDistance = 26;
    this.controls.autoRotate = true; this.controls.autoRotateSpeed = .6;
    this.scene.add(new THREE.AmbientLight(0xfff4e8, .85));
    const d = new THREE.DirectionalLight(0xfff1e0, 1.1); d.position.set(5, 12, 6); this.scene.add(d);
    this.scene.add(new THREE.PointLight(0xff8a5b, .5, 40).translateY(6));
    this.nodes = new THREE.Group(); this.edges = new THREE.Group(); this.routes = new THREE.Group();
    this.scene.add(this.nodes); this.scene.add(this.edges); this.scene.add(this.routes);
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2();
    this.heat = true; this.disabled = new Set();
    canvas.addEventListener("pointerdown", e => this._click(e));
    this._loop = this._loop.bind(this); this._loop();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }
  _to3D(x, y) { return new THREE.Vector3((x / this.data.img_w - .5) * SIZE, .14, (y / this.data.img_h - .5) * SIZE); }

  setData(data) {
    this.data = data; this.disabled = new Set();
    this.nodes.clear(); this.edges.clear(); this._clearRoutes();
    this.nodeMap = {};
    // satellite plane
    if (this.plane) this.scene.remove(this.plane);
    new THREE.TextureLoader().load(data.tex, tex => {
      tex.colorSpace = THREE.SRGBColorSpace;
      const g = new THREE.PlaneGeometry(SIZE, SIZE);
      const m = new THREE.MeshStandardMaterial({ map: tex, roughness: .95, metalness: 0 });
      this.plane = new THREE.Mesh(g, m); this.plane.rotation.x = -Math.PI/2; this.plane.position.y = -0.02;
      this.scene.add(this.plane);
    });
    // edges
    this.edgeObjs = [];
    const emax = Math.max(1e-6, ...data.edges.map(e => e.bc));
    for (const e of data.edges) {
      const pts = e.pts.map(p => this._to3D(p[0], p[1]));
      if (pts.length < 2) continue;
      const curve = new THREE.CatmullRomCurve3(pts);
      const t = e.bc / emax;
      const rad = 0.012 + t * 0.05;
      const tube = new THREE.TubeGeometry(curve, Math.max(2, pts.length * 2), rad, 6, false);
      const color = e.healed ? healColor(e) : this._heatColor(t);
      const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: e.healed ? healGlow(e) : .35 + t * .6, roughness: .4 });
      const mesh = new THREE.Mesh(tube, mat); mesh.userData = { u: e.u, v: e.v, t, healed: e.healed, hk: e.hk, conf: e.conf };
      this.edges.add(mesh); this.edgeObjs.push(mesh);
    }
    // nodes
    const bcmax = Math.max(1e-6, ...data.nodes.map(n => n.bc));
    for (const n of data.nodes) {
      const r = 0.06 + (n.bc / bcmax) * 0.16;
      const geo = new THREE.SphereGeometry(r, 18, 18);
      const col = TIER[n.tier];
      const mat = new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: n.tier === "normal" ? .35 : .9, roughness: .3 });
      const mesh = new THREE.Mesh(geo, mat); mesh.position.copy(this._to3D(n.x, n.y));
      mesh.userData = { id: n.id, tier: n.tier, baseColor: col, r }; this.nodes.add(mesh); this.nodeMap[n.id] = mesh;
    }
    this.resize(); this.resetView();
  }

  _heatColor(t) {
    const c = new THREE.Color();
    return c.setHSL((1 - Math.min(1, t)) * 0.42, 0.62, 0.58).getHex(); // green(0.42)->coral(0), refined
  }
  setHeat(on) { this.heat = on; this.applyState(this.disabled || new Set(), this.flooded); }
  setAutoRotate(on) { this.controls.autoRotate = on; }
  setOsm() { /* OSM overlay is map-only */ }
  resetView() { this.cam.position.set(0, 9, 9); this.controls.target.set(0, 0, 0); this.controls.update(); }

  applyState(disabled, flooded = new Set()) {
    this.disabled = disabled; this.flooded = flooded;
    for (const id in this.nodeMap) {
      const m = this.nodeMap[id], i = +id, fl = flooded.has(i), off = disabled.has(i);
      const col = fl ? FLOOD : (off ? OFF : m.userData.baseColor);
      m.material.color.set(col);
      m.material.emissive.set(fl ? FLOOD : (off ? 0x111620 : m.userData.baseColor));
      m.material.emissiveIntensity = fl ? .8 : (off ? .1 : (m.userData.tier === "normal" ? .35 : .9));
      m.scale.setScalar(off && !fl ? .7 : 1);
    }
    for (const e of this.edgeObjs) {
      const fl = flooded.has(e.userData.u) || flooded.has(e.userData.v);
      const dead = disabled.has(e.userData.u) || disabled.has(e.userData.v);
      const base = e.userData.healed ? healColor(e.userData) : this._heatColor(this.heat ? e.userData.t : 0.0);
      e.material.color.set(fl ? FLOOD : (dead ? OFF : base));
      e.material.emissive.set(fl ? FLOOD : (dead ? 0x0b0f18 : base));
      e.material.emissiveIntensity = fl ? .7 : (dead ? .05
        : (e.userData.healed ? healGlow(e.userData) : (this.heat ? .35 + e.userData.t * .6 : .3)));
      e.material.opacity = dead && !fl ? .5 : 1; e.material.transparent = dead && !fl;
    }
  }

  _clearRoutes() {
    if (!this.routes) return;
    while (this.routes.children.length) {
      const c = this.routes.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
      this.routes.remove(c);
    }
  }

  _routeTube(ids, color, rad, yOff, opacity = 1) {
    const pts = ids.filter(id => this.nodeMap[id]).map(id => {
      const p = this.nodeMap[id].position.clone(); p.y = yOff; return p;
    });
    if (pts.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, Math.max(2, pts.length * 4), rad, 8, false);
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .85,
      roughness: .3, transparent: opacity < 1, opacity });
    this.routes.add(new THREE.Mesh(geo, mat));
  }

  _routeMarker(id, color) {
    if (!this.nodeMap[id]) return;
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 16),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .9, roughness: .3 }));
    const p = this.nodeMap[id].position.clone(); p.y = 0.5; m.position.copy(p);
    this.routes.add(m);
  }

  /** Draw the baseline O->D route (dim reference) + the live rerouted path (bright),
   *  or flag the route as severed (red) when the destination is unreachable. */
  setRoute(baseIds, curIds, severed) {
    this._clearRoutes();
    if (!baseIds || baseIds.length < 2) return;
    this._routeTube(baseIds, 0xaab2bf, 0.028, 0.42, 0.5);            // baseline reference (dim)
    this._routeMarker(baseIds[0], 0xffffff);                         // origin
    this._routeMarker(baseIds[baseIds.length - 1], 0xff8a5b);        // destination
    if (severed || !curIds || curIds.length < 2) {
      this._routeTube(baseIds, 0xff6b5e, 0.05, 0.52, 0.9);           // severed -> red
    } else {
      this._routeTube(curIds, 0xffffff, 0.06, 0.54);                 // live reroute (bright path)
    }
  }

  _click(ev) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    this.ray.setFromCamera(this.mouse, this.cam);
    const hit = this.ray.intersectObjects(this.nodes.children, false)[0];
    if (hit && this.onPick) this.onPick(hit.object.userData.id);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.cam.aspect = w / h; this.cam.updateProjectionMatrix();
  }
  _loop() { requestAnimationFrame(this._loop); this.controls.update(); this.renderer.render(this.scene, this.cam); }
}
