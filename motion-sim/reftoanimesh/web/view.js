import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const Y = new THREE.Vector3(0, 1, 0);
const toY = ([x, y, z]) => new THREE.Vector3(x, z, -y);

export class MotionView {
  constructor(host, color) {
    this.host = host;
    this.color = color;
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 2000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new THREE.HemisphereLight(0xecedff, 0x535966, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 3.5);
    light.position.set(4, 8, 6);
    this.scene.add(light);
    const rim = new THREE.DirectionalLight(0xa4c7e3, 2);
    rim.position.set(-4, 3, -5);
    this.scene.add(rim);
    this.grid = new THREE.GridHelper(200, 200, 0x536171, 0x39434e);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.25;
    this.scene.add(this.grid);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.cache = new Map();
    this.abort = new AbortController();
    this.options = { trail: true, wire: false, skeleton: false, follow: false };
    this.root = new THREE.Vector3();
    this.origin = new THREE.Vector3();
    this.resize = new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / Math.max(height, 1);
      this.camera.updateProjectionMatrix();
      this.draw();
    });
    this.resize.observe(host);
    this.reset();
  }

  clear() {
    this.abort.abort();
    this.abort = new AbortController();
    const materials = new Set(),
      geometries = new Set();
    this.content.traverse((obj) => {
      if (obj.geometry) geometries.add(obj.geometry);
      if (obj.material)
        (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(
          (m) => materials.add(m),
        );
    });
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
    this.content.clear();
    this.cache.clear();
    this.data = null;
    this.mesh = null;
    this.rig = null;
    this.joints = null;
    this.trail = null;
    this.frame = -1;
    this.draw();
  }

  async get(url, type = "json", signal = this.abort.signal) {
    const result = await fetch(url, { signal });
    if (!result.ok) throw new Error(`Viewer asset ${result.status}: ${url}`);
    return type === "json" ? result.json() : result.arrayBuffer();
  }

  makeTrail(points) {
    this.trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color: this.color,
        transparent: true,
        opacity: 0.7,
      }),
    );
    this.trail.visible = this.options.trail;
    this.content.add(this.trail);
  }

  async loadMesh(base, data) {
    this.clear();
    const signal = this.abort.signal;
    const [faces, skeleton] = await Promise.all([
      this.get(base + "faces.bin", "binary", signal),
      this.get(base + "skeleton.json", "json", signal),
    ]);
    if (signal.aborted) return;
    this.base = base;
    this.data = data;
    this.skeleton = skeleton;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(data.vertices * 3),
        3,
      ).setUsage(THREE.DynamicDrawUsage),
    );
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
    this.mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: this.color,
        roughness: 0.72,
        metalness: 0.04,
        side: THREE.DoubleSide,
        wireframe: this.options.wire,
      }),
    );
    this.mesh.frustumCulled = false;
    this.content.add(this.mesh);
    this.edges = skeleton.parents.map((p, i) => [p, i]).filter(([p]) => p >= 0);
    const lines = new THREE.BufferGeometry();
    lines.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(this.edges.length * 6), 3),
    );
    this.joints = new THREE.LineSegments(
      lines,
      new THREE.LineBasicMaterial({
        color: 0xffffff,
        depthTest: false,
        transparent: true,
        opacity: 0.8,
      }),
    );
    this.joints.frustumCulled = false;
    this.joints.visible = this.options.skeleton;
    this.content.add(this.joints);
    this.makeTrail(
      data.trajectory.map((p) => new THREE.Vector3(p[0], 0.015, p[2])),
    );
    this.origin.fromArray(data.trajectory[0]);
    this.root.copy(this.origin);
    this.reset();
    await this.setFrame(0);
  }

  loadPhysics(data) {
    this.clear();
    this.data = data;
    this.rig = new THREE.Group();
    this.rig.rotation.x = -Math.PI / 2;
    this.content.add(this.rig);
    this.bodies = data.names.map(() => {
      const body = new THREE.Group();
      this.rig.add(body);
      return body;
    });
    const material = new THREE.MeshStandardMaterial({
      color: this.color,
      metalness: 0.08,
      roughness: 0.63,
      wireframe: this.options.wire,
    });
    for (const shape of data.geometry) {
      let geometry;
      const size = shape.size || [0.04];
      const from = shape.fromto
        ? new THREE.Vector3(...shape.fromto.slice(0, 3))
        : null;
      const to = shape.fromto
        ? new THREE.Vector3(...shape.fromto.slice(3))
        : null;
      const length = from ? from.distanceTo(to) : (size[1] || 0.04) * 2;
      if (shape.type === "capsule")
        geometry = new THREE.CapsuleGeometry(size[0], length, 5, 12);
      else if (shape.type === "cylinder")
        geometry = new THREE.CylinderGeometry(size[0], size[0], length, 12);
      else if (shape.type === "box")
        geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
      else geometry = new THREE.SphereGeometry(size[0], 12, 8);
      const mesh = new THREE.Mesh(geometry, material);
      if (shape.type === "ellipsoid")
        mesh.scale.set(1, size[1] / size[0], size[2] / size[0]);
      if (from) {
        mesh.position.copy(from).add(to).multiplyScalar(0.5);
        mesh.quaternion.setFromUnitVectors(Y, to.clone().sub(from).normalize());
      } else {
        mesh.position.fromArray(shape.pos || [0, 0, 0]);
        const q = shape.quat || [1, 0, 0, 0];
        mesh.quaternion.set(q[1], q[2], q[3], q[0]);
        if (["capsule", "cylinder"].includes(shape.type))
          mesh.rotateX(Math.PI / 2);
      }
      this.bodies[shape.body].add(mesh);
    }
    this.edges = data.parents.map((p, i) => [p, i]).filter(([p]) => p >= 0);
    const lines = new THREE.BufferGeometry();
    lines.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(this.edges.length * 6), 3),
    );
    this.joints = new THREE.LineSegments(
      lines,
      new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
      }),
    );
    this.joints.frustumCulled = false;
    this.joints.visible = this.options.skeleton;
    this.rig.add(this.joints);
    this.makeTrail(
      data.positions.map((p) => {
        const point = toY(p[0]);
        point.y = 0.015;
        return point;
      }),
    );
    this.origin.copy(toY(data.positions[0][0]));
    this.root.copy(this.origin);
    this.reset();
    this.setPhysicsTime(0);
  }

  updateRoot(point) {
    if (this.options.follow) {
      const shift = point.clone().sub(this.root);
      this.camera.position.add(shift);
      this.controls.target.add(shift);
    }
    this.root.copy(point);
  }

  updateJoints(positions) {
    const array = this.joints.geometry.attributes.position.array;
    this.edges.forEach(([parent, child], i) => {
      array.set(positions[parent], i * 6);
      array.set(positions[child], i * 6 + 3);
    });
    this.joints.geometry.attributes.position.needsUpdate = true;
  }

  async setFrame(frame) {
    if (!this.mesh || this.frame === frame) return;
    const signal = this.abort.signal;
    const chunk = this.data.chunks.find(
      (c) => frame >= c.start && frame < c.start + c.frames,
    );
    if (!chunk) return;
    let buffer = this.cache.get(chunk.file);
    if (!buffer) {
      buffer = new Float32Array(
        await this.get(this.base + chunk.file, "binary", signal),
      );
      if (signal.aborted) return;
      this.cache.set(chunk.file, buffer);
      while (this.cache.size > 2)
        this.cache.delete(this.cache.keys().next().value);
    }
    const count = this.data.vertices * 3;
    const offset = (frame - chunk.start) * count;
    this.mesh.geometry.attributes.position.array.set(
      buffer.subarray(offset, offset + count),
    );
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.updateJoints(this.skeleton.positions[frame]);
    this.updateRoot(new THREE.Vector3(...this.data.trajectory[frame]));
    this.trail.geometry.setDrawRange(0, frame + 1);
    this.frame = frame;
    this.draw();
  }

  setPhysicsTime(time) {
    if (!this.rig) return;
    const frame = Math.min(
      this.data.positions.length - 1,
      Math.max(0, Math.round((time - this.data.first_time) * this.data.fps)),
    );
    if (frame === this.frame) return;
    this.bodies.forEach((body, i) => {
      body.position.fromArray(this.data.positions[frame][i]);
      body.quaternion.fromArray(this.data.rotations[frame][i]);
    });
    this.updateJoints(this.data.positions[frame]);
    this.updateRoot(toY(this.data.positions[frame][0]));
    this.trail.geometry.setDrawRange(0, frame + 1);
    this.frame = frame;
    this.draw();
  }

  configure(options) {
    Object.assign(this.options, options);
    if (this.trail) this.trail.visible = this.options.trail;
    if (this.joints) this.joints.visible = this.options.skeleton;
    this.content.traverse((object) => {
      if (object.isMesh) object.material.wireframe = this.options.wire;
    });
    this.draw();
  }

  reset() {
    const target = this.root?.clone() || new THREE.Vector3();
    target.y = Math.max(0.8, target.y);
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(new THREE.Vector3(3.4, 1.6, 4.3));
    this.controls.update();
    this.draw();
  }

  draw() {
    this.renderer.render(this.scene, this.camera);
  }
}
