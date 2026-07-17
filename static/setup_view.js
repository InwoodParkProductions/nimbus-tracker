// Nimbus Tracker — shot-setup viewport
import * as THREE from 'three';
import {OrbitControls} from '/static/addons/OrbitControls.js';
import {TransformControls} from '/static/addons/TransformControls.js';
import {GLTFLoader} from '/static/addons/GLTFLoader.js';

const CFG = JSON.parse(document.getElementById('cfg').textContent);
const Q = CFG.q;
const PRE = CFG.mode === 'pre';

const data = await (await fetch('/setup_data?' + Q + '&file=campath.json')).json();

const vp = document.getElementById('vp');
const canvas = document.getElementById('c');
const plate = document.getElementById('plate');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
renderer.setPixelRatio(window.devicePixelRatio);
const scene3 = new THREE.Scene();

scene3.add(new THREE.HemisphereLight(0xdfefff, 0x8a9aa8, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(4, 8, 5);
scene3.add(sun);
const grid = new THREE.GridHelper(20, 20, 0x77aacc, 0xbcd8e8);
grid.material.transparent = true;
grid.material.opacity = .5;
scene3.add(grid);

// Blender is Z-up; three.js is Y-up — everything Blender-space lives here
const bWorld = new THREE.Group();
bWorld.rotation.x = -Math.PI / 2;
scene3.add(bWorld);

const root = new THREE.Group();
root.position.fromArray(data.root.loc);
root.quaternion.set(data.root.quat[1], data.root.quat[2],
                    data.root.quat[3], data.root.quat[0]);
root.scale.setScalar(data.root.scale);
bWorld.add(root);

if (data.path.length > 1) {
  const pts = data.path.map(p => new THREE.Vector3(...p.loc));
  const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color: 0x1181cc}));
  root.add(line);
}

const aspect = data.res_x / data.res_y;
const fovx = 2 * Math.atan(data.sensor_mm / 2 / data.focal_mm);
const fovy = 2 * Math.atan(Math.tan(fovx / 2) / aspect);
const holder = new THREE.Group();
root.add(holder);
{
  const d = 1.2, hw = Math.tan(fovx / 2) * d, hh = Math.tan(fovy / 2) * d;
  const c4 = [[-hw, -hh, -d], [hw, -hh, -d], [hw, hh, -d], [-hw, hh, -d]];
  const segs = [];
  c4.forEach((p, i) => {
    segs.push(0, 0, 0, ...p);
    segs.push(...p, ...c4[(i + 1) % 4]);
  });
  segs.push(-hw * .35, hh, -d, 0, hh * 1.3, -d,
            0, hh * 1.3, -d, hw * .35, hh, -d);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3));
  holder.add(new THREE.LineSegments(
      g, new THREE.LineBasicMaterial({color: 0x0d76c0})));
}

function setFrame(i) {
  if (!data.path.length) return;
  const p = data.path[Math.min(i, data.path.length - 1)];
  holder.position.fromArray(p.loc);
  holder.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
}
setFrame(0);

let sceneGeo = null;
new GLTFLoader().load('/setup_data?' + Q + '&file=setup.glb',
    g => { sceneGeo = g.scene; scene3.add(g.scene); frameAll();
           setPosRanges(); syncSliders(); }, undefined,
    e => console.warn('glb load failed', e));

const orbitCam = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
orbitCam.position.set(6, 5, 8);
const orbit = new OrbitControls(orbitCam, canvas);

// Auto-frame: point the orbit camera so the scene geometry AND the camera
// path both fit in view, wherever they sit in world space.
function frameAll() {
  const box = new THREE.Box3();
  if (sceneGeo) box.expandByObject(sceneGeo);
  root.updateWorldMatrix(true, true);
  data.path.forEach(p => box.expandByPoint(
      root.localToWorld(new THREE.Vector3(...p.loc))));
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.5);
  const dist = r / Math.sin(THREE.MathUtils.degToRad(orbitCam.fov / 2)) * 1.15;
  orbit.target.copy(c);
  const dir = new THREE.Vector3(0.7, 0.55, 0.9).normalize();
  orbitCam.position.copy(c).addScaledVector(dir, dist);
  orbitCam.near = Math.max(dist / 500, 0.01);
  orbitCam.far = dist * 20;
  orbitCam.updateProjectionMatrix();
  orbit.update();
}
const gizmo = new TransformControls(orbitCam, canvas);
gizmo.attach(root);
gizmo.addEventListener('dragging-changed', e => orbit.enabled = !e.value);
gizmo.addEventListener('objectChange', () => {
  const s = (root.scale.x + root.scale.y + root.scale.z) / 3;
  root.scale.setScalar(s);  // uniform only — non-uniform breaks tracks
  syncSliders();
});
scene3.add(gizmo);

// ---- transform sliders (position / rotation° / scale) ----
// These drive the same root the gizmo does, and work in Camera view too —
// nudge them while looking through the shot to line CG up with the plate.
const S = id => document.getElementById(id);
const sl = {
  px: [S('pxr'), S('pxn')], py: [S('pyr'), S('pyn')], pz: [S('pzr'), S('pzn')],
  rx: [S('rxr'), S('rxn')], ry: [S('ryr'), S('ryn')], rz: [S('rzr'), S('rzn')],
  sc: [S('scr'), S('scn')],
};
let sliderInit = false;
function setPosRanges() {
  const box = new THREE.Box3();
  if (sceneGeo) box.expandByObject(sceneGeo);
  root.updateWorldMatrix(true, true);
  data.path.forEach(p => box.expandByPoint(
      root.localToWorld(new THREE.Vector3(...p.loc))));
  const size = box.isEmpty() ? 10 : box.getSize(new THREE.Vector3()).length();
  const span = Math.max(size * 2, 20);
  for (const k of ['px', 'py', 'pz']) {
    const cur = root.position[k[1]];
    sl[k][0].min = (cur - span).toFixed(2);
    sl[k][0].max = (cur + span).toFixed(2);
  }
}
function syncSliders() {
  const set = (k, v) => { sl[k][0].value = v; sl[k][1].value = (+v).toFixed(2); };
  set('px', root.position.x); set('py', root.position.y);
  set('pz', root.position.z);
  const e = root.rotation;
  set('rx', THREE.MathUtils.radToDeg(e.x));
  set('ry', THREE.MathUtils.radToDeg(e.y));
  set('rz', THREE.MathUtils.radToDeg(e.z));
  set('sc', root.scale.x);
}
function applySliders() {
  root.position.set(+sl.px[0].value, +sl.py[0].value, +sl.pz[0].value);
  root.rotation.set(THREE.MathUtils.degToRad(+sl.rx[0].value),
                    THREE.MathUtils.degToRad(+sl.ry[0].value),
                    THREE.MathUtils.degToRad(+sl.rz[0].value));
  root.scale.setScalar(Math.max(+sl.sc[0].value, 0.01));
}
for (const [rng, num] of Object.values(sl)) {
  rng.addEventListener('input', () => {
    num.value = (+rng.value).toFixed(2); applySliders();
  });
  num.addEventListener('input', () => {
    rng.value = num.value; applySliders();
  });
}

const povCam = new THREE.PerspectiveCamera(
    THREE.MathUtils.radToDeg(fovy), aspect, 0.01, 5000);
let pov = false;

const frameEl = document.getElementById('frame');
frameEl.max = Math.max(data.path.length, 1);
if (PRE || data.path.length < 2)
  document.getElementById('frameWrap').style.display = 'none';
function plateSrc() {
  plate.src = '/plate_frame?' + Q + '&frame=' + frameEl.value;
}
frameEl.oninput = () => {
  document.getElementById('frameNo').textContent = frameEl.value;
  setFrame(frameEl.value - 1);
  plateSrc();
};
plateSrc();
document.getElementById('plateOp').oninput =
    e => plate.style.opacity = e.target.value / 100;
plate.style.opacity = .55;

const on = (id, fn) => { const el = document.getElementById(id);
  if (el) el.onclick = fn; };
const modes = {btnMove: 'translate', btnRot: 'rotate', btnScale: 'scale'};
for (const [id, m] of Object.entries(modes))
  on(id, () => gizmo.setMode(m));
on('btnFrame', frameAll);
document.getElementById('btnPOV').onclick = function() {
  pov = !pov;
  this.classList.toggle('vp-on', pov);
  gizmo.visible = !pov;
  gizmo.enabled = !pov;
  orbit.enabled = !pov;
  resize();
};

function resize() {
  const W = vp.clientWidth, H = vp.clientHeight;
  let w = W, h = H, l = 0, t = 0;
  if (pov) {  // letterbox to the plate's aspect so the overlay lines up
    if (W / H > aspect) { w = H * aspect; l = (W - w) / 2; }
    else { h = W / aspect; t = (H - h) / 2; }
  }
  for (const el of [canvas, plate]) {
    el.style.width = w + 'px';
    el.style.height = h + 'px';
    el.style.left = l + 'px';
    el.style.top = t + 'px';
  }
  renderer.setSize(w, h, false);
  orbitCam.aspect = w / h;
  orbitCam.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(vp);
resize();
frameAll();  // frame the camera path immediately; refined when GLB loads
setPosRanges();
syncSliders();

renderer.setAnimationLoop(() => {
  plate.style.visibility = pov ? 'visible' : 'hidden';
  if (pov) {
    holder.updateWorldMatrix(true, false);
    holder.matrixWorld.decompose(povCam.position, povCam.quaternion,
                                 new THREE.Vector3());
    renderer.render(scene3, povCam);
  } else {
    orbit.update();
    renderer.render(scene3, orbitCam);
  }
});

document.getElementById('btnApply').onclick = async () => {
  const r = await fetch('/setup_apply', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      footage: CFG.footage, shot: CFG.shot, mode: CFG.mode, scene: CFG.scene,
      loc: root.position.toArray(),
      quat_wxyz: [root.quaternion.w, root.quaternion.x,
                  root.quaternion.y, root.quaternion.z],
      scale: root.scale.x,
    }),
  });
  const d = await r.json();
  if (d.redirect) window.location = d.redirect;
};
