import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export class AssetView {
  constructor(host, callbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
    this.epoch = 0;
    this.frame = 0;
    this.playing = false;
    this.loop = true;
    this.speed = 1;
    this.dirty = true;
    this.joints = [];
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.001, 10000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    host.append(this.renderer.domElement);
    const room = new RoomEnvironment(),
      pmrem = new THREE.PMREMGenerator(this.renderer);
    this.environment = pmrem.fromScene(room, 0.04);
    this.scene.environment = this.environment.texture;
    room.dispose();
    pmrem.dispose();
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.addEventListener("change", () => this.request());
    this.controls.autoRotateSpeed = 0.65;
    this.wrapper = new THREE.Group();
    this.scene.add(this.wrapper);
    this.light = new THREE.HemisphereLight(0xe1f0ff, 0x344942, 2);
    this.scene.add(this.light);
    this.ray = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.abort = new AbortController();
    const opts = { signal: this.abort.signal };
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (e) => this.down(e), {
      ...opts,
      capture: true,
    });
    canvas.addEventListener("pointermove", (e) => this.move(e), opts);
    canvas.addEventListener("pointerup", (e) => this.up(e), opts);
    canvas.addEventListener("pointercancel", (e) => this.up(e), opts);
    document.addEventListener("visibilitychange", () => this.request(), opts);
    this.resize = new ResizeObserver(() => {
      const w = host.clientWidth,
        h = host.clientHeight;
      if (!w || !h) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.request();
    });
    this.resize.observe(host);
    this.home();
  }
  request() {
    this.dirty = true;
    if (!this.frame && !document.hidden)
      this.frame = requestAnimationFrame((t) => this.tick(t));
  }
  tick(now) {
    this.frame = 0;
    const dt = Math.min((now - (this.last || now)) / 1000, 0.05);
    this.last = now;
    if (document.hidden) return;
    if (this.mixer && this.playing) {
      this.mixer.update(dt * this.speed);
      this.callbacks.time?.(this.action?.time || 0, this.duration || 0);
    }
    const changed = this.controls.update();
    if (this.dirty || changed || this.playing || this.controls.autoRotate) {
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
    if (!this.frame && (this.playing || this.controls.autoRotate || changed))
      this.frame = requestAnimationFrame((t) => this.tick(t));
  }
  disposeTree(root) {
    const geometries = new Set(),
      materials = new Set(),
      textures = new Set();
    root.traverse((o) => {
      if (o.geometry) geometries.add(o.geometry);
      for (const m of Array.isArray(o.material)
        ? o.material
        : o.material
          ? [o.material]
          : []) {
        materials.add(m);
        for (const v of Object.values(m)) if (v?.isTexture) textures.add(v);
      }
    });
    for (const g of geometries) g.dispose();
    for (const m of materials) m.dispose();
    for (const t of textures) {
      t.source?.data?.close?.();
      t.dispose();
    }
  }
  clear() {
    ++this.epoch;
    this.playing = false;
    this.drag = null;
    this.controls.enabled = true;
    this.controls.autoRotate = false;
    this.mixer?.stopAllAction();
    if (this.root) this.mixer?.uncacheRoot(this.root);
    this.mixer = null;
    this.action = null;
    this.root = null;
    this.clips = [];
    this.joints = [];
    this.rest = [];
    this.ring = null;
    while (this.wrapper.children.length) {
      const o = this.wrapper.children[0];
      this.wrapper.remove(o);
      this.disposeTree(o);
    }
    this.callbacks.clips?.([]);
    this.callbacks.joints?.([]);
    this.callbacks.time?.(0, 0);
    this.callbacks.play?.(false);
    this.request();
  }
  async load(url, bindings = { joints: [] }) {
    this.clear();
    const epoch = this.epoch;
    this.callbacks.status?.("Loading mesh…");
    try {
      const gltf = await new GLTFLoader().loadAsync(url);
      if (epoch !== this.epoch) {
        this.disposeTree(gltf.scene);
        return;
      }
      this.root = gltf.scene;
      this.wrapper.add(this.root);
      this.wrapper.position.set(0, 0, 0);
      const box = new THREE.Box3().setFromObject(this.root);
      if (box.isEmpty()) throw Error("Mesh has no visible geometry");
      this.center = box.getCenter(new THREE.Vector3());
      this.size = box.getSize(new THREE.Vector3()).length();
      this.wrapper.position.copy(this.center).negate();
      this.rest = [];
      this.root.traverse((o) =>
        this.rest.push([
          o,
          o.position.clone(),
          o.quaternion.clone(),
          o.scale.clone(),
          o.morphTargetInfluences?.slice(),
        ]),
      );
      this.clips = gltf.animations;
      this.mixer = new THREE.AnimationMixer(this.root);
      this.mixer.addEventListener("finished", () => {
        if (!this.loop) {
          this.playing = false;
          this.callbacks.play?.(false);
        }
      });
      const nodes = new Map();
      for (const [object, association] of gltf.parser.associations)
        if (Number.isInteger(association.nodes))
          nodes.set(gltf.parser.json.nodes[association.nodes].name, object);
      this.joints = (bindings.joints || []).map((j) => ({
        ...j,
        object: nodes.get(j.node),
        value: j.initial,
        vector: new THREE.Vector3(...j.axis).normalize(),
      }));
      if (this.joints.some((j) => !j.object))
        throw Error("A part binding is missing from this GLB");
      const material = new THREE.MeshBasicMaterial({
        color: 0xe6cd8c,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
      });
      this.ring = new THREE.Mesh(
        new THREE.TorusGeometry(this.size * 0.18, this.size * 0.0015, 6, 90),
        material,
      );
      this.ring.visible = false;
      this.ring.renderOrder = 99;
      this.wrapper.add(this.ring);
      this.callbacks.clips?.(this.clips);
      this.callbacks.joints?.(this.joints);
      this.callbacks.status?.(
        this.clips.length
          ? "Drag to orbit · select a part to manipulate"
          : "Static native candidate · drag to orbit",
      );
      this.home();
      if (this.clips.length) this.selectClip(0);
      this.request();
    } catch (error) {
      if (epoch === this.epoch) this.callbacks.status?.(error.message);
    }
  }
  home() {
    const radius = this.size || 2;
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(radius * 0.9, radius * 0.5, radius * 1.1);
    this.camera.near = Math.max(radius / 10000, 0.0001);
    this.camera.far = Math.max(radius * 100, 100);
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = radius * 0.05;
    this.controls.maxDistance = radius * 12;
    this.controls.update();
    this.request();
  }
  resetRest() {
    this.mixer?.stopAllAction();
    for (const [o, p, q, s, m] of this.rest || []) {
      o.position.copy(p);
      o.quaternion.copy(q);
      o.scale.copy(s);
      if (m) o.morphTargetInfluences.splice(0, m.length, ...m);
    }
    this.root?.updateMatrixWorld(true);
  }
  selectClip(index) {
    if (!this.clips?.[index]) return;
    this.resetRest();
    this.action = this.mixer.clipAction(this.clips[index]);
    this.duration = this.clips[index].duration;
    this.action.reset();
    this.action.setLoop(
      this.loop ? THREE.LoopRepeat : THREE.LoopOnce,
      Infinity,
    );
    this.action.clampWhenFinished = true;
    this.action.play();
    this.mixer.setTime(0);
    this.playing = false;
    if (this.ring) this.ring.visible = false;
    this.callbacks.play?.(false);
    this.callbacks.time?.(0, this.duration);
    this.request();
  }
  play() {
    if (!this.action) return;
    this.playing = !this.playing;
    if (this.playing) {
      const time = this.action.time >= this.duration ? 0 : this.action.time;
      this.resetRest();
      this.action.enabled = true;
      this.action.paused = false;
      this.action.play();
      this.action.time = time;
      this.mixer.update(0);
    }
    if (this.ring) this.ring.visible = false;
    this.callbacks.play?.(this.playing);
    this.request();
  }
  seek(time) {
    if (!this.action) return;
    this.playing = false;
    this.resetRest();
    this.action.reset();
    this.action.paused = false;
    this.action.play();
    this.action.time = Math.max(0, Math.min(time, this.duration - 0.000001));
    this.mixer.update(0);
    if (this.ring) this.ring.visible = false;
    this.callbacks.play?.(false);
    this.callbacks.time?.(this.action.time, this.duration);
    this.request();
  }
  setLoop(value) {
    this.loop = value;
    this.action?.setLoop(value ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
  }
  wire(value) {
    this.root?.traverse((o) => {
      for (const m of Array.isArray(o.material)
        ? o.material
        : o.material
          ? [o.material]
          : [])
        m.wireframe = value;
    });
    this.request();
  }
  orbit(value) {
    this.controls.autoRotate = value;
    this.request();
  }
  selectJoint(index) {
    this.selected = index;
    this.callbacks.selected?.(index);
    this.applyJoints();
  }
  setJoint(index, value) {
    const j = this.joints[index];
    if (!j) return;
    j.value = Math.max(j.min, Math.min(value, j.max));
    this.selected = index;
    this.applyJoints();
    this.callbacks.joint?.(index, j.value);
    this.callbacks.selected?.(index);
  }
  resetJoints() {
    for (const j of this.joints) j.value = j.initial;
    this.applyJoints();
    this.joints.forEach((j, i) => this.callbacks.joint?.(i, j.value));
  }
  applyJoints() {
    this.playing = false;
    this.callbacks.play?.(false);
    this.resetRest();
    for (const j of this.joints) {
      const delta = j.value - j.initial;
      if (j.kind === "rotate")
        j.object.quaternion.multiply(
          new THREE.Quaternion().setFromAxisAngle(
            j.vector,
            THREE.MathUtils.degToRad(delta),
          ),
        );
      else
        j.object.position.add(
          j.vector
            .clone()
            .applyQuaternion(j.object.quaternion)
            .multiplyScalar(delta),
        );
    }
    this.root?.updateMatrixWorld(true);
    const j = this.joints[this.selected];
    if (this.ring) {
      this.ring.visible = !!j;
      if (j) {
        this.ring.position.copy(
          this.wrapper.worldToLocal(
            j.object.getWorldPosition(new THREE.Vector3()),
          ),
        );
        this.ring.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          j.vector
            .clone()
            .applyQuaternion(
              j.object.getWorldQuaternion(new THREE.Quaternion()),
            ),
        );
      }
    }
    this.request();
  }
  down(event) {
    if (event.button !== 0 || !this.joints.length || !this.root) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      (-(event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.ray.setFromCamera(this.pointer, this.camera);
    const hits = this.ray.intersectObject(this.root, true);
    if (!hits.length) return;
    let obj = hits[0].object,
      index = -1;
    while (obj && index < 0) {
      index = this.joints.findIndex((j) => j.object === obj);
      obj = obj.parent;
    }
    if (index < 0) return;
    event.stopImmediatePropagation();
    this.controls.enabled = false;
    this.drag = {
      id: event.pointerId,
      index,
      x: event.clientX,
      value: this.joints[index].value,
    };
    this.renderer.domElement.setPointerCapture(event.pointerId);
    this.selectJoint(index);
  }
  move(event) {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    const j = this.joints[this.drag.index];
    this.setJoint(
      this.drag.index,
      this.drag.value + ((event.clientX - this.drag.x) / 240) * (j.max - j.min),
    );
  }
  up(event) {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    this.drag = null;
    this.controls.enabled = true;
    if (this.renderer.domElement.hasPointerCapture(event.pointerId))
      this.renderer.domElement.releasePointerCapture(event.pointerId);
  }
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }
  dispose() {
    cancelAnimationFrame(this.frame);
    this.clear();
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.abort.abort();
    this.controls.dispose();
    this.environment.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
