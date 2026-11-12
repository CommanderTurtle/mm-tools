import * as THREE from "three";

export class CameraRecorder {
  constructor(element, onChange, onTime) {
    this.element = element;
    this.onChange = onChange;
    this.onTime = onTime;
    this.keys = new Set();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#11170d");
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.05, 1000);
    this.camera.rotation.order = "YXZ";
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    element.append(this.renderer.domElement);
    this.scene.add(new THREE.GridHelper(100, 100, 0x63784d, 0x293321));
    this.scene.add(new THREE.HemisphereLight(0xe8f6ce, 0x233414, 2));
    const marker = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.8, 1),
      new THREE.MeshStandardMaterial({ color: 0xb7d787, wireframe: true }),
    );
    marker.position.y = 1;
    this.scene.add(marker);
    this.reset();
    this.resize = new ResizeObserver(() => {
      const r = element.getBoundingClientRect();
      this.renderer.setSize(r.width, r.height);
      this.camera.aspect = r.width / Math.max(1, r.height);
      this.camera.updateProjectionMatrix();
      this.render();
    });
    this.resize.observe(element);
    this.codes = new Set([
      "KeyW",
      "KeyA",
      "KeyS",
      "KeyD",
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "ShiftLeft",
      "ShiftRight",
    ]);
    element.addEventListener("keydown", (e) => {
      if (this.codes.has(e.code)) {
        e.preventDefault();
        if (!this.keys.has(e.code)) {
          this.keys.add(e.code);
          this.keyEvent(e.code, true);
        }
        this.animate();
      }
    });
    element.addEventListener("keyup", (e) => {
      if (this.codes.has(e.code)) {
        e.preventDefault();
        this.keys.delete(e.code);
        this.keyEvent(e.code, false);
      }
    });
    this.release = () => {
      for (const key of this.keys) this.keyEvent(key, false);
      this.keys.clear();
      this.dragging = false;
    };
    element.addEventListener("blur", this.release);
    window.addEventListener("blur", this.release);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.release();
        if (this.recording) this.stop();
      }
    });
    element.addEventListener("contextmenu", (e) => e.preventDefault());
    element.addEventListener("pointerdown", (e) => {
      element.focus();
      if (e.button === 2) {
        this.dragging = true;
        element.setPointerCapture(e.pointerId);
      }
    });
    element.addEventListener("pointermove", (e) => {
      if (this.dragging) {
        this.camera.rotation.y -= e.movementX * 0.0024;
        this.camera.rotation.x = THREE.MathUtils.clamp(
          this.camera.rotation.x - e.movementY * 0.0024,
          -Math.PI / 2 + 0.01,
          Math.PI / 2 - 0.01,
        );
        this.render();
      }
    });
    element.addEventListener("pointerup", () => {
      this.dragging = false;
    });
    element.addEventListener("lostpointercapture", () => {
      this.dragging = false;
    });
  }
  render() {
    this.renderer.render(this.scene, this.camera);
  }
  reset() {
    if (this.recording) this.stop();
    this.camera.position.set(0, 1.5, 10);
    this.camera.rotation.set(0, 0, 0);
    this.frames = [];
    this.events = [];
    this.path = null;
    this.keys.clear();
    this.onChange(null);
    this.render();
  }
  keyEvent(code, pressed) {
    if (this.recording && !code.startsWith("Shift"))
      this.events.push({
        t: Math.round(performance.now() - this.started),
        code,
        pressed,
      });
  }
  start() {
    this.frames = [];
    this.events = [];
    this.started = performance.now();
    this.nextSample = 0;
    this.recording = true;
    this.path = null;
    this.onChange(null);
    this.sample(0);
    this.nextSample = 1000 / 24;
    this.element.focus();
    this.animate();
  }
  sample(t) {
    const p = this.camera.position,
      q = this.camera.quaternion;
    this.frames.push({
      frame: this.frames.length,
      t: Math.round(t),
      position: { x: p.x, y: p.y, z: p.z },
      quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
      fov: this.camera.fov,
    });
  }
  animate() {
    if (this.raf) return;
    let previous = performance.now();
    const tick = (now) => {
      this.raf = 0;
      const dt = Math.min((now - previous) / 1000, 0.08);
      previous = now;
      const speed =
        this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 7 : 2.8;
      const local = new THREE.Vector3(
        Number(this.keys.has("KeyD")) - Number(this.keys.has("KeyA")),
        0,
        Number(this.keys.has("KeyS")) - Number(this.keys.has("KeyW")),
      );
      if (local.lengthSq())
        this.camera.position.add(
          local
            .normalize()
            .applyQuaternion(this.camera.quaternion)
            .multiplyScalar(speed * dt),
        );
      this.camera.rotation.y +=
        (((Number(this.keys.has("ArrowLeft")) -
          Number(this.keys.has("ArrowRight"))) *
          Math.PI) /
          9) *
        dt;
      this.camera.rotation.x = THREE.MathUtils.clamp(
        this.camera.rotation.x +
          (((Number(this.keys.has("ArrowUp")) -
            Number(this.keys.has("ArrowDown"))) *
            Math.PI) /
            9) *
            dt,
        -Math.PI / 2 + 0.01,
        Math.PI / 2 - 0.01,
      );
      if (this.recording) {
        const elapsed = now - this.started;
        while (this.nextSample <= Math.min(elapsed, 60000)) {
          this.sample(this.nextSample);
          this.nextSample += 1000 / 24;
        }
        this.onTime(elapsed);
        if (elapsed >= 60000) this.stop();
      }
      this.render();
      if (this.recording || this.keys.size)
        this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  stop() {
    if (!this.recording) return;
    this.release();
    this.recording = false;
    // Native camera conditioning consumes poses, including intentional pauses.
    // Remove only the redundant static suffix after the final movement.
    const same = (a, b) =>
      ["x", "y", "z"].every(
        (k) => Math.abs(a.position[k] - b.position[k]) < 1e-6,
      ) &&
      ["x", "y", "z", "w"].every(
        (k) => Math.abs(a.quaternion[k] - b.quaternion[k]) < 1e-6,
      );
    while (
      this.frames.length > 2 &&
      same(this.frames.at(-1), this.frames.at(-2))
    )
      this.frames.pop();
    const durationMs = Math.min(
      60000,
      (this.frames.at(-1)?.t || 0) + 1000 / 24,
    );
    let distance = 0;
    for (let i = 1; i < this.frames.length; i++) {
      const a = this.frames[i - 1].position,
        b = this.frames[i].position;
      distance += Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    this.events = this.events.map((e) => ({
      ...e,
      t: Math.min(e.t, durationMs),
    }));
    this.path = {
      version: "1.0",
      type: "camera-trajectory",
      name: "Recorded path",
      recordedAt: new Date().toISOString(),
      sampleRate: 24,
      durationMs,
      totalDistance: distance,
      coordinateSystem: {
        handedness: "right",
        up: "+Y",
        forward: "-Z",
        unit: "meter",
      },
      frames: this.frames,
      keyEvents: this.events,
    };
    this.onChange(this.path);
    this.onTime(durationMs);
    return this.path;
  }
  import(value) {
    if (this.recording) throw Error("Stop recording before importing a path");
    if (
      !value ||
      !Array.isArray(value.frames) ||
      value.frames.length < 2 ||
      value.frames.length > 20000
    )
      throw Error("A trajectory needs 2–20,000 camera frames");
    let time = -1;
    for (const f of value.frames) {
      if (
        !Number.isFinite(f.t) ||
        f.t < 0 ||
        f.t < time ||
        !f.position ||
        !f.quaternion ||
        !["x", "y", "z"].every((k) => Number.isFinite(f.position[k])) ||
        !["x", "y", "z", "w"].every((k) => Number.isFinite(f.quaternion[k])) ||
        !Number.isFinite(f.fov) ||
        f.fov <= 0 ||
        f.fov >= 180
      )
        throw Error("Invalid trajectory pose");
      if (
        Math.abs(
          Math.hypot(...["x", "y", "z", "w"].map((k) => f.quaternion[k])) - 1,
        ) > 0.01
      )
        throw Error("Camera quaternions must have unit length");
      time = f.t;
    }
    if (time <= 0 || time > 60000)
      throw Error("Camera duration must be between 0 and 60 seconds");
    if (
      value.coordinateSystem &&
      (value.coordinateSystem.up !== "+Y" ||
        value.coordinateSystem.forward !== "-Z" ||
        value.coordinateSystem.handedness !== "right")
    )
      throw Error("Use right-handed +Y-up, -Z-forward coordinates");
    const rate = value.sampleRate ?? 24;
    if (!Number.isFinite(rate) || rate <= 0 || rate > 240)
      throw Error("Invalid camera sample rate");
    const duration =
      value.durationMs ??
      Math.max(
        (value.frames.length * 1000) / rate,
        time - value.frames[0].t + 1000 / rate,
      );
    if (!Number.isFinite(duration) || duration <= 0 || duration > 60000)
      throw Error("Invalid camera duration");
    const events = value.keyEvents || [];
    if (
      !Array.isArray(events) ||
      events.some(
        (e) =>
          !Number.isFinite(e.t) ||
          e.t < 0 ||
          e.t > duration ||
          typeof e.code !== "string" ||
          typeof e.pressed !== "boolean",
      )
    )
      throw Error("Invalid camera key events");
    this.path = {
      ...value,
      durationMs: duration,
      sampleRate: rate,
      keyEvents: events,
    };
    this.frames = value.frames;
    const last = this.frames.at(-1);
    this.camera.position.copy(last.position);
    this.camera.quaternion.copy(last.quaternion);
    this.render();
    this.onChange(this.path);
    this.onTime(time);
  }
}
