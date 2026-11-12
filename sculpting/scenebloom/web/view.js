import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export function createView(host, selected) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0xe9ece4);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x89977c, 2.5));
  const sun = new THREE.DirectionalLight(0xfff5e6, 3);
  sun.position.set(5, 9, 3);
  scene.add(sun);
  const grid = new THREE.GridHelper(20, 40, 0xaab89c, 0xd3d9ca);
  grid.position.y = -0.03;
  scene.add(grid);
  const zones = new THREE.Group();
  scene.add(zones);
  let model,
    highlight,
    layout,
    epoch = 0,
    frame = 0,
    mode = "orbit",
    spinning = false,
    walls = true;
  let center = new THREE.Vector3(0, 1, 0),
    extent = 6,
    previous = 0;
  const keys = new Set();
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");

  function dispose(group) {
    group?.traverse((o) => {
      o.geometry?.dispose();
      for (const m of [].concat(o.material || [])) {
        for (const value of Object.values(m))
          if (value?.isTexture) value.dispose();
        m.dispose();
      }
    });
    group?.removeFromParent();
  }
  function draw(time = 0) {
    frame = 0;
    const dt = Math.min((time - previous) / 1000 || 0, 0.05);
    previous = time;
    if (document.hidden) return;
    controls.autoRotate = spinning && !reduced.matches && mode === "orbit";
    if (mode === "walk" && keys.size) {
      const forward = camera.getWorldDirection(new THREE.Vector3());
      forward.y = 0;
      forward.normalize();
      const side = new THREE.Vector3()
        .crossVectors(forward, camera.up)
        .normalize();
      const shift = new THREE.Vector3();
      if (keys.has("w") || keys.has("arrowup")) shift.add(forward);
      if (keys.has("s") || keys.has("arrowdown")) shift.sub(forward);
      if (keys.has("d") || keys.has("arrowright")) shift.add(side);
      if (keys.has("a") || keys.has("arrowleft")) shift.sub(side);
      shift.normalize().multiplyScalar(dt * 2);
      camera.position.add(shift);
      controls.target.add(shift);
    }
    controls.update();
    renderer.render(scene, camera);
    if (controls.autoRotate || keys.size) invalidate();
  }
  function invalidate() {
    if (!frame && !document.hidden) frame = requestAnimationFrame(draw);
  }
  controls.addEventListener("change", invalidate);
  new ResizeObserver(() => {
    renderer.setSize(host.clientWidth, host.clientHeight);
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    invalidate();
  }).observe(host);
  function setMode(value) {
    mode = value;
    keys.clear();
    controls.enableRotate = value !== "plan";
    controls.enablePan = value !== "walk";
    controls.enableZoom = value !== "walk";
    controls.minDistance = value === "walk" ? 0.01 : 0.1;
    controls.target.copy(center);
    camera.up.set(0, 1, 0);
    if (value === "plan") {
      camera.position
        .copy(center)
        .add(new THREE.Vector3(0, extent * 1.5, 0.001));
    } else if (value === "walk") {
      camera.position.copy(center).add(new THREE.Vector3(0, 0, extent * 0.35));
      camera.position.y = 1.6;
      controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -0.01));
    } else
      camera.position
        .copy(center)
        .add(new THREE.Vector3(extent, extent * 0.8, extent));
    controls.update();
    invalidate();
  }
  function clear() {
    epoch++;
    dispose(model);
    model = null;
    dispose(highlight);
    highlight = null;
    layout = null;
    while (zones.children.length) dispose(zones.children[0]);
    center.set(0, 1, 0);
    extent = 6;
    setMode("orbit");
  }
  function setLayout(data) {
    layout = data;
    while (zones.children.length) dispose(zones.children[0]);
    const plan = data.zoning_plan || data;
    const bounds = new THREE.Box3();
    for (const room of plan.rooms || [])
      for (const [i, zone] of room.functional_zones.entries()) {
        const vertices = zone.polygon_2d.map(
          ([x, z]) => new THREE.Vector3(x, 0.025, -z),
        );
        for (const v of vertices) bounds.expandByPoint(v);
        if (vertices.length) vertices.push(vertices[0].clone());
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(vertices),
          new THREE.LineBasicMaterial({
            color: new THREE.Color().setHSL((i * 0.17 + 0.22) % 1, 0.45, 0.42),
            depthTest: false,
          }),
        );
        line.renderOrder = 10;
        zones.add(line);
      }
    if (!bounds.isEmpty() && !model) {
      center = bounds.getCenter(new THREE.Vector3());
      extent = Math.max(...bounds.getSize(new THREE.Vector3()).toArray(), 3);
      setMode(mode);
    }
    invalidate();
  }
  function wallVisibility() {
    model?.traverse((o) => {
      if (/^wall_\d/.test(o.name)) o.visible = walls;
    });
    invalidate();
  }
  async function load(url) {
    const token = ++epoch;
    const result = await new GLTFLoader().loadAsync(url);
    if (token !== epoch) {
      dispose(result.scene);
      return false;
    }
    dispose(model);
    model = result.scene;
    scene.add(model);
    const bounds = new THREE.Box3().setFromObject(model);
    center = bounds.getCenter(new THREE.Vector3());
    extent = Math.max(...bounds.getSize(new THREE.Vector3()).toArray(), 3);
    wallVisibility();
    setMode(mode);
    return true;
  }
  function select(id) {
    dispose(highlight);
    highlight = null;
    const object = model?.getObjectByName("object_" + id);
    if (object) {
      highlight = new THREE.BoxHelper(object, 0x98b85a);
      scene.add(highlight);
    }
    selected(layout?.placed_objects?.find((o) => o.instance_id === id) || null);
    invalidate();
  }
  let down;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    down = [e.clientX, e.clientY];
    host.focus({ preventScroll: true });
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (
      !model ||
      !down ||
      Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4
    )
      return;
    const rect = host.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(
      new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        1 - ((e.clientY - rect.top) / rect.height) * 2,
      ),
      camera,
    );
    for (const hit of ray.intersectObject(model, true)) {
      let o = hit.object;
      let visible = true;
      for (let p = o; p; p = p.parent) if (!p.visible) visible = false;
      if (!visible) continue;
      while (o && !o.name.startsWith("object_")) o = o.parent;
      if (o) {
        select(o.name.slice(7));
        break;
      }
    }
  });
  host.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (
      mode === "walk" &&
      [
        "w",
        "a",
        "s",
        "d",
        "arrowup",
        "arrowdown",
        "arrowleft",
        "arrowright",
      ].includes(k)
    ) {
      e.preventDefault();
      keys.add(k);
      invalidate();
    }
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  host.addEventListener("blur", () => keys.clear());
  window.addEventListener("blur", () => keys.clear());
  document.addEventListener("visibilitychange", () => {
    keys.clear();
    invalidate();
  });
  setMode("orbit");
  return {
    clear,
    setLayout,
    load,
    select,
    setMode,
    walls(value) {
      walls = value;
      wallVisibility();
    },
    zones(value) {
      zones.visible = value;
      invalidate();
    },
    spin(value) {
      spinning = value;
      invalidate();
    },
    capture() {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL("image/png");
    },
  };
}
