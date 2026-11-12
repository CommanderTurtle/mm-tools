import * as T from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const identity = () => new T.Matrix4();
const matrix = (p) =>
  new T.Matrix4().compose(
    new T.Vector3(...p.slice(0, 3)),
    new T.Quaternion(...p.slice(3, 7)),
    new T.Vector3(1, 1, 1),
  );

class Character {
  constructor(
    data,
    color = 0xc4d8cf,
    ghost = false,
    maxTextureSize = 4096,
    meshVisible = true,
  ) {
    this.data = data;
    this.group = new T.Group();
    this.meshes = new T.Group();
    this.group.add(this.meshes);
    this.world = [];
    this.lines = new T.LineSegments(
      new T.BufferGeometry(),
      new T.LineBasicMaterial({
        color: ghost ? color : 0xc9eb9e,
        transparent: true,
        opacity: ghost ? 0.6 : 1,
      }),
    );
    this.group.add(this.lines);
    this.ghost = ghost;
    if (data.kind === "skeleton") {
      this.bones = new Float32Array(data.names.length * 16);
      this.texture = new T.DataTexture(
        this.bones,
        4,
        data.names.length,
        T.RGBAFormat,
        T.FloatType,
      );
      this.inverse = data.inverse_bind.map(matrix);
      (meshVisible ? data.meshes : []).forEach((part) => {
        if (!part.positions.length) return;
        const geometry = new T.BufferGeometry();
        geometry.setAttribute(
          "position",
          new T.Float32BufferAttribute(part.positions, 3),
        );
        geometry.setIndex(part.indices);
        geometry.computeVertexNormals();
        geometry.setAttribute(
          "skinVertex",
          new T.Float32BufferAttribute(
            Array.from({ length: part.positions.length / 3 }, (_, i) => i),
            1,
          ),
        );
        const width = Math.min(maxTextureSize, Math.max(1, part.joints.length)),
          height = Math.ceil(part.joints.length / width);
        if (height > maxTextureSize)
          throw Error("Rig influences exceed this browser’s texture capacity.");
        const influences = new Float32Array(width * height * 4);
        part.joints.forEach((joint, i) => {
          influences[i * 4] = joint;
          influences[i * 4 + 1] = part.weights[i];
        });
        const weights = new T.DataTexture(
          influences,
          width,
          height,
          T.RGBAFormat,
          T.FloatType,
        );
        weights.needsUpdate = true;
        const material = new T.MeshStandardMaterial({
          color,
          roughness: 0.66,
          metalness: 0.06,
          transparent: ghost,
          opacity: ghost ? 0.22 : 1,
          depthWrite: !ghost,
          side: T.DoubleSide,
        });
        material.onBeforeCompile = (shader) => {
          shader.uniforms.jointTexture = { value: this.texture };
          shader.uniforms.influenceTexture = { value: weights };
          shader.vertexShader =
            "attribute float skinVertex;\nuniform sampler2D jointTexture;\nuniform sampler2D influenceTexture;\n" +
            shader.vertexShader;
          const skin = `mat4 skin=mat4(0.0); for(int i=0;i<${part.influences};++i){int at=int(skinVertex)*${part.influences}+i;vec2 iw=texelFetch(influenceTexture,ivec2(at%${width},at/${width}),0).rg;int j=int(iw.x);skin+=iw.y*mat4(texelFetch(jointTexture,ivec2(0,j),0),texelFetch(jointTexture,ivec2(1,j),0),texelFetch(jointTexture,ivec2(2,j),0),texelFetch(jointTexture,ivec2(3,j),0));}`;
          shader.vertexShader = shader.vertexShader
            .replace(
              "#include <beginnormal_vertex>",
              skin + "\nvec3 objectNormal=mat3(skin)*normal;",
            )
            .replace(
              "#include <begin_vertex>",
              "vec3 transformed=(skin*vec4(position,1.0)).xyz;",
            );
        };
        material.customProgramCacheKey = () =>
          `motion-skin-${part.influences}-${width}`;
        const mesh = new T.Mesh(geometry, material);
        mesh.frustumCulled = false;
        mesh.userData.weights = weights;
        this.meshes.add(mesh);
      });
    } else
      (meshVisible ? data.meshes : []).forEach((part) => {
        let geometry;
        if (part.positions) {
          geometry = new T.BufferGeometry();
          geometry.setAttribute(
            "position",
            new T.Float32BufferAttribute(part.positions, 3),
          );
          geometry.setIndex(part.indices);
          geometry.computeVertexNormals();
        } else if (part.type === 7)
          geometry = new T.BoxGeometry(...part.scale.map((n) => n * 2));
        else if (part.type === 3 || part.type === 5)
          geometry = new T.SphereGeometry(1, 20, 12);
        else if (part.type === 4) {
          geometry = new T.CapsuleGeometry(
            part.scale[0],
            part.scale[1] * 2,
            6,
            16,
          );
          geometry.rotateX(Math.PI / 2);
        } else {
          geometry = new T.CylinderGeometry(
            part.type === 9 ? 0 : part.scale[0],
            part.scale[0],
            part.scale[1] * 2,
            20,
          );
          geometry.rotateX(Math.PI / 2);
        }
        const mesh = new T.Mesh(
          geometry,
          new T.MeshStandardMaterial({
            color: new T.Color(...part.color),
            roughness: 0.55,
            metalness: 0.2,
            transparent: ghost,
            opacity: ghost ? 0.25 : 1,
          }),
        );
        mesh.userData.part = part;
        if (part.positions || part.type === 5) mesh.scale.set(...part.scale);
        else if (part.type === 3) mesh.scale.setScalar(part.scale[0]);
        this.meshes.add(mesh);
      });
  }
  pose(values) {
    if (!values?.length) return;
    this.world = values.map(matrix);
    if (this.data.kind === "skeleton") {
      const drawn = values.map((value, i) => {
        const weight = Math.max(
          0,
          Math.min(1, this.data.override_mask?.[i] || 0),
        );
        if (!weight) return matrix(value);
        const override = this.data.override_pose[i],
          p = new T.Vector3(...value.slice(0, 3)).lerp(
            new T.Vector3(...override.slice(0, 3)),
            weight,
          ),
          q = new T.Quaternion(...value.slice(3, 7)).slerp(
            new T.Quaternion(...override.slice(3, 7)),
            weight,
          );
        return new T.Matrix4().compose(p, q, new T.Vector3(1, 1, 1));
      });
      this.world.forEach((m, i) => {
        const p = this.data.parents[i];
        if (p >= 0) {
          m.premultiply(this.world[p]);
          drawn[i].premultiply(drawn[p]);
        }
        identity()
          .multiplyMatrices(drawn[i], this.inverse[i])
          .toArray(this.bones, i * 16);
      });
      this.texture.needsUpdate = true;
    } else
      this.meshes.children.forEach((mesh) => {
        const p = mesh.userData.part;
        const m = matrix(p.transform);
        if (p.body >= 0) m.premultiply(this.world[p.body]);
        m.decompose(mesh.position, mesh.quaternion, new T.Vector3());
      });
    const lines = [];
    this.world.forEach((m, i) => {
      const p = this.data.parents?.[i];
      if (p >= 0)
        lines.push(
          ...new T.Vector3().setFromMatrixPosition(m).toArray(),
          ...new T.Vector3().setFromMatrixPosition(this.world[p]).toArray(),
        );
    });
    if (this.data.kind === "robot")
      this.data.bones.forEach(([parent, child]) => {
        lines.push(
          ...new T.Vector3()
            .setFromMatrixPosition(this.world[parent])
            .toArray(),
          ...new T.Vector3().setFromMatrixPosition(this.world[child]).toArray(),
        );
      });
    this.lines.geometry.dispose();
    this.lines.geometry = new T.BufferGeometry();
    this.lines.geometry.setAttribute(
      "position",
      new T.Float32BufferAttribute(lines, 3),
    );
  }
  dispose() {
    this.group.traverse((o) => {
      o.geometry?.dispose();
      o.material?.dispose();
      o.userData.weights?.dispose();
    });
    this.texture?.dispose();
  }
}

export class View {
  constructor(host, onSelect, onTransform, onDrag) {
    this.host = host;
    this.onSelect = onSelect;
    this.contactThreshold = 0.5;
    this.options = {
      mesh: true,
      skeleton: false,
      targets: true,
      context: false,
      characterTargets: false,
      trajectory: true,
      boxes: true,
      contacts: true,
      probe: false,
      actualVelocity: false,
      desiredVelocity: false,
      inputVelocity: false,
      blendspace: false,
    };
    this.scene = new T.Scene();
    this.scene.background = new T.Color(0x121a22);
    this.scene.fog = new T.Fog(0x121a22, 35, 110);
    this.camera = new T.PerspectiveCamera(45, 1, 0.01, 1000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-7, -7, 5);
    this.renderer = new T.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    host.prepend(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 1);
    this.controls.enableDamping = true;
    this.scene.add(new T.HemisphereLight(0xe6f1ff, 0x50604a, 2));
    const light = new T.DirectionalLight(0xffffff, 3);
    light.position.set(3, -4, 8);
    this.scene.add(light);
    const grid = new T.GridHelper(60, 120, 0x556b66, 0x25323c);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);
    const floor = new T.Mesh(
      new T.PlaneGeometry(200, 200),
      new T.MeshStandardMaterial({ color: 0x182129, roughness: 1 }),
    );
    floor.position.z = -0.012;
    this.scene.add(floor);
    this.items = new T.Group();
    this.ghosts = new T.Group();
    this.debug = new T.Group();
    this.traces = new T.Group();
    this.targetTraces = new T.Group();
    this.debugGhosts = new T.Group();
    this.scene.add(
      this.items,
      this.ghosts,
      this.debug,
      this.traces,
      this.targetTraces,
      this.debugGhosts,
    );
    this.ghostCharacters = [];
    this.debugCharacters = [];
    this.trailPoints = [];
    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmoProxy = new T.Object3D();
    this.dragRotation = new T.Quaternion();
    this.scene.add(this.gizmo.getHelper());
    this.gizmo.setSpace("local");
    this.gizmo.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      if (e.value && this.selectedObject)
        this.dragRotation.copy(this.selectedObject.quaternion);
      onDrag(e.value);
      if (!e.value) this.gizmoProxy.quaternion.identity();
    });
    this.gizmo.addEventListener("objectChange", () => {
      const o = this.selectedObject;
      if (o)
        onTransform(o.userData.id, {
          translation: this.gizmoProxy.position.toArray(),
          rotation_xyzw: this.gizmoProxy.quaternion
            .clone()
            .multiply(this.dragRotation)
            .toArray(),
        });
    });
    this.picking = new T.Raycaster();
    let down;
    host.addEventListener("pointerdown", (e) => {
      host.focus();
      down = [e.clientX, e.clientY];
    });
    host.addEventListener("pointerup", (e) => {
      if (
        !down ||
        Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 ||
        this.gizmo.dragging ||
        this.gizmo.axis
      )
        return;
      const r = host.getBoundingClientRect();
      this.picking.setFromCamera(
        new T.Vector2(
          (2 * (e.clientX - r.left)) / r.width - 1,
          1 - (2 * (e.clientY - r.top)) / r.height,
        ),
        this.camera,
      );
      const hit = this.picking.intersectObject(this.items, true)[0];
      if (hit) {
        let o = hit.object;
        while (o && o.userData.id === undefined) o = o.parent;
        if (o) onSelect(o.userData.id);
      }
    });
    new ResizeObserver(() => {
      const w = host.clientWidth,
        h = host.clientHeight;
      if (!w || !h) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.invalidate();
    }).observe(host);
    this.controls.addEventListener("change", () => this.invalidate());
    this.gizmo.addEventListener("change", () => this.invalidate());
    document.addEventListener("visibilitychange", () => this.invalidate());
    host.addEventListener(
      "wheel",
      (e) => {
        if (e.ctrlKey) {
          e.preventDefault();
          e.stopImmediatePropagation();
          this.camera.fov = T.MathUtils.clamp(
            this.camera.fov + Math.sign(e.deltaY) * 2,
            15,
            100,
          );
          this.camera.updateProjectionMatrix();
          this.invalidate();
        }
      },
      { capture: true, passive: false },
    );
    this.invalidate();
  }
  invalidate() {
    if (this.framePending) return;
    this.framePending = true;
    requestAnimationFrame(() => {
      this.framePending = false;
      if (document.hidden) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }
  load(data, pose) {
    this.clearCharacter();
    this.data = data;
    this.character = new Character(
      data,
      0xc4d8cf,
      false,
      this.renderer.capabilities.maxTextureSize,
    );
    this.scene.add(this.character.group);
    this.character.pose(pose);
    this.applyOptions();
  }
  clearCharacter() {
    this.debugData = null;
    this.overlayData = null;
    this.clearGhosts();
    this.debugCharacters.forEach((c) => c.dispose());
    this.debugCharacters = [];
    this.debugGhosts.clear();
    this.clearGroup(this.debug);
    this.clearGroup(this.traces);
    if (this.character) {
      this.scene.remove(this.character.group);
      this.character.dispose();
      this.character = null;
    }
    this.data = null;
    this.invalidate();
  }
  clearGhosts() {
    this.ghostCharacters.forEach((c) => c.dispose());
    this.ghostCharacters = [];
    this.ghosts.clear();
    this.clearGroup(this.targetTraces);
    this.invalidate();
  }
  ghost(poses, color = 0x96ca86, mesh = false, skeleton = true, id) {
    if (!this.data || (!mesh && !skeleton)) return;
    poses.forEach((p) => {
      const c = new Character(
        this.data,
        color,
        true,
        this.renderer.capabilities.maxTextureSize,
        mesh,
      );
      c.pose(p);
      c.group.userData.id = id;
      c.lines.visible = skeleton;
      this.ghosts.add(c.group);
      this.ghostCharacters.push(c);
    });
    this.invalidate();
  }
  liveTargets(targets) {
    for (const target of targets) {
      const existing = this.ghostCharacters.find(
        (c) => c.group.userData.id === target.id,
      );
      if (existing) existing.pose(target.pose);
      else
        this.ghost(
          [target.pose],
          new T.Color(...target.color),
          true,
          false,
          target.id,
        );
    }
    this.invalidate();
  }
  debugPoses(context = [], blendspace = [], targets = []) {
    if (!this.data) return;
    this.debugData = [context, blendspace, targets];
    const poses = [
      ...(this.options.context ? context.map((p) => [p, 0xe8b574]) : []),
      ...(this.options.blendspace
        ? blendspace.slice(0, 1).map((p) => [p, 0xb29be5])
        : []),
      ...(this.options.characterTargets
        ? targets.map((p) => [p, 0x79b7ff])
        : []),
    ];
    poses.forEach(([pose, color], i) => {
      let c = this.debugCharacters[i];
      if (!c) {
        c = new Character(
          this.data,
          color,
          true,
          this.renderer.capabilities.maxTextureSize,
          false,
        );
        this.debugCharacters.push(c);
        this.debugGhosts.add(c.group);
      }
      c.lines.material.color.set(color);
      c.group.visible = true;
      c.pose(pose);
    });
    this.debugCharacters.slice(poses.length).forEach((c) => {
      c.group.visible = false;
    });
    this.invalidate();
  }
  clearGroup(group) {
    group.traverse((o) => {
      o.geometry?.dispose();
      o.material?.dispose();
    });
    group.clear();
  }
  document(doc, kind) {
    this.gizmo.detach();
    this.gizmoProxy.removeFromParent();
    this.selectedObject = null;
    this.clearGroup(this.items);
    this.byId = new Map();
    const add = (item, parent, shape) => {
      const group = new T.Group();
      group.userData.id = item.id;
      const value = item.local_transform || item.transform;
      group.position.set(...value.translation);
      group.quaternion.set(...value.rotation_xyzw);
      group.visible = item.visible !== false;
      parent.add(group);
      this.byId.set(item.id, group);
      if (shape === "cube") {
        const d = item.dimensions;
        group.add(
          new T.Mesh(
            new T.BoxGeometry(d.width, d.depth, d.height),
            new T.MeshStandardMaterial({
              color: new T.Color(...item.color),
              roughness: 0.8,
            }),
          ),
        );
      } else {
        const marker = new T.Mesh(
          new T.OctahedronGeometry(0.12),
          new T.MeshBasicMaterial({
            color: shape === "start" ? 0xc5ee90 : 0x80bdde,
          }),
        );
        marker.position.z = 0.08;
        group.add(marker);
        const axes = new T.AxesHelper(0.45);
        group.add(axes);
      }
      for (const component of item.components || []) {
        if (component.type === "interaction_box") {
          const g = new T.BoxGeometry(
            component.width,
            component.depth,
            component.height,
          );
          const box = new T.LineSegments(
            new T.EdgesGeometry(g),
            new T.LineBasicMaterial({ color: new T.Color(...component.color) }),
          );
          box.userData.box = true;
          g.dispose();
          group.add(box);
        }
        if (component.type === "pivot") this.pivot(group, component);
      }
      return group;
    };
    if (kind === "inbetween")
      doc.entries.forEach((item, i) =>
        add(item, this.items, i ? "target" : "start"),
      );
    else if (kind === "primitive") {
      const walk = (n, p) => {
        const g = add(n, p, "node");
        n.children.forEach((c) => walk(c, g));
      };
      walk(doc.root, this.items);
    } else {
      add(
        { ...doc.character_start, name: "CharacterStart" },
        this.items,
        "start",
      );
      [...doc.smart_primitives, ...doc.solid_cubes].forEach((i) =>
        add(i, this.items, i.dimensions ? "cube" : "primitive"),
      );
      [...doc.smart_primitives, ...doc.solid_cubes].forEach((i) => {
        if (i.parent_id != null)
          this.byId.get(i.parent_id)?.add(this.byId.get(i.id));
      });
    }
    this.applyOptions();
    this.invalidate();
  }
  pivot(group, c) {
    const direction = new T.Vector3(...c.motion_direction),
      axis = (name) =>
        new T.Vector3(
          name === "x" ? 1 : 0,
          name === "y" ? 1 : 0,
          name === "z" ? 1 : 0,
        );
    const line = (points, color) =>
      group.add(
        new T.Line(
          new T.BufferGeometry().setFromPoints(points),
          new T.LineBasicMaterial({ color }),
        ),
      );
    if (c.show_motion_direction && direction.length())
      group.add(
        new T.ArrowHelper(
          direction.clone().normalize(),
          new T.Vector3(0, 0, 0.03),
          direction.length(),
          0xe3bb71,
        ),
      );
    if (
      c.show_translation_constraint &&
      c.translation_axis !== "none" &&
      c.translation_constraint === "line_projection"
    )
      line(
        [
          axis(c.translation_axis).multiplyScalar(c.line_min_offset),
          axis(c.translation_axis).multiplyScalar(c.line_max_offset),
        ],
        0x83cbb7,
      );
    if (
      c.show_rotation_constraint &&
      c.rotation_axis !== "none" &&
      c.rotation_constraint === "cone"
    ) {
      const half = (c.cone_half_angle_degrees * Math.PI) / 180,
        centers =
          c.cone_side === "both"
            ? [0, Math.PI]
            : [c.cone_side === "mirrored" ? Math.PI : 0];
      centers.forEach((center) => {
        const points = Array.from({ length: 49 }, (_, i) =>
          direction
            .clone()
            .multiplyScalar(0.6)
            .applyAxisAngle(
              axis(c.rotation_axis),
              center - half + (2 * half * i) / 48,
            ),
        );
        if (half < Math.PI) points.unshift(new T.Vector3());
        if (half < Math.PI) points.push(new T.Vector3());
        line(points, 0xe88d83);
      });
    }
  }
  select(id, locks = {}) {
    const object = this.byId?.get(id);
    this.gizmoProxy.removeFromParent();
    this.selectedObject = object;
    if (object) {
      // An identity proxy gives the native parent-axis handles, independent of object rotation.
      object.parent.add(this.gizmoProxy);
      this.gizmoProxy.position.copy(object.position);
      this.gizmoProxy.quaternion.identity();
      this.dragRotation.copy(object.quaternion);
      this.gizmo.attach(this.gizmoProxy);
      const flags =
        this.gizmo.mode === "rotate"
          ? locks.rotation_locks
          : locks.translation_locks;
      ["X", "Y", "Z"].forEach((axis, i) => {
        this.gizmo["show" + axis] = !flags?.[i];
      });
    } else this.gizmo.detach();
  }
  mode(mode) {
    this.gizmo.setMode(mode);
  }
  frame() {
    const box = new T.Box3().setFromObject(this.items);
    if (box.isEmpty())
      box.setFromCenterAndSize(new T.Vector3(0, 0, 1), new T.Vector3(4, 4, 2));
    const center = box.getCenter(new T.Vector3()),
      size = box.getSize(new T.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position
      .copy(center)
      .add(new T.Vector3(-1, -1, 0.8).multiplyScalar(Math.max(3, size * 0.7)));
    this.controls.update();
  }
  cameraState() {
    const p = this.camera.position,
      t = this.controls.target,
      d = p.clone().sub(t);
    return {
      pos: p.toArray(),
      pivot: t.toArray(),
      pitch: (-Math.asin(d.z / d.length()) * 180) / Math.PI,
      yaw: (Math.atan2(d.y, d.x) * 180) / Math.PI,
      fov: this.camera.fov,
      near: this.camera.near,
      far: this.camera.far,
    };
  }
  restoreCamera(c) {
    if (!c) return;
    this.camera.position.set(...c.pos);
    this.controls.target.set(...c.pivot);
    Object.assign(this.camera, { fov: c.fov, near: c.near, far: c.far });
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }
  applyOptions() {
    if (this.character) {
      this.character.meshes.visible = this.options.mesh;
      this.character.lines.visible = this.options.skeleton;
    }
    this.ghosts.visible = this.options.targets;
    this.debugGhosts.visible =
      this.options.context ||
      this.options.blendspace ||
      this.options.characterTargets;
    this.traces.visible = this.options.trajectory;
    this.targetTraces.visible = this.options.targets;
    this.items.traverse((o) => {
      if (o.userData.box) o.visible = this.options.boxes;
    });
    if (this.debugData) this.debugPoses(...this.debugData);
    if (this.overlayData) this.overlays(this.overlayData);
    this.invalidate();
  }
  pose(pose) {
    this.character?.pose(pose);
    this.invalidate();
  }
  trail(frames) {
    this.clearGroup(this.traces);
    this.trailPoints = [];
    if (!frames.length) return;
    const points = frames.map((f) => new T.Vector3(...f[0].slice(0, 3)));
    this.traces.add(
      new T.Line(
        new T.BufferGeometry().setFromPoints(points),
        new T.LineBasicMaterial({
          color: 0xbee896,
          transparent: true,
          opacity: 0.65,
        }),
      ),
    );
    this.invalidate();
  }
  liveTrail(position) {
    if (!position) return;
    this.trailPoints.push(new T.Vector3(...position));
    if (this.trailPoints.length > 1800) this.trailPoints.shift();
    if (!this.liveLine || !this.traces.children.includes(this.liveLine)) {
      this.clearGroup(this.traces);
      this.liveLine = new T.Line(
        new T.BufferGeometry(),
        new T.LineBasicMaterial({
          color: 0xbee896,
          transparent: true,
          opacity: 0.65,
        }),
      );
      this.traces.add(this.liveLine);
    }
    this.liveLine.geometry.dispose();
    this.liveLine.geometry = new T.BufferGeometry().setFromPoints(
      this.trailPoints,
    );
    this.invalidate();
  }
  targetTrace(poses, target) {
    if (!poses.length) return;
    const points = (
      target.trajectory || poses.map((p) => p[0].slice(0, 3))
    ).map((p) => new T.Vector3(...p));
    if (target.show_trajectory)
      this.targetTraces.add(
        new T.Line(
          new T.BufferGeometry().setFromPoints(points),
          new T.LineBasicMaterial({ color: 0x91b6c9 }),
        ),
      );
    if (
      target.show_velocity &&
      poses.length > 1 &&
      (target.start || target.num_poses !== 1)
    ) {
      const at = target.start ? poses.length - 2 : 0,
        start = new T.Vector3(...poses[at][0].slice(0, 3)),
        v = new T.Vector3(...poses[at + 1][0].slice(0, 3))
          .sub(start)
          .multiplyScalar(30);
      if (v.length() > 0.001)
        this.targetTraces.add(
          new T.ArrowHelper(v.clone().normalize(), start, v.length(), 0xe7be79),
        );
    }
    this.invalidate();
  }
  overlays(data) {
    this.overlayData = data;
    this.clearGroup(this.debug);
    const wire = (b, color) => {
      const g = new T.BoxGeometry(...b.half_extents.map((v) => v * 2));
      const o = new T.LineSegments(
        new T.EdgesGeometry(g),
        new T.LineBasicMaterial({ color }),
      );
      g.dispose();
      o.applyMatrix4(matrix(b.transform));
      this.debug.add(o);
    };
    if (this.options.boxes)
      (data.boxes || []).forEach((b) => wire(b, 0x8cc889));
    if (this.options.probe)
      (data.probe || []).forEach((b) => wire(b, 0xe67aa9));
    if (data.position)
      ["actual_velocity", "desired_velocity", "input_velocity"].forEach(
        (key, i) => {
          if (
            !data[key] ||
            !this.options[
              ["actualVelocity", "desiredVelocity", "inputVelocity"][i]
            ]
          )
            return;
          const v = new T.Vector3(...data[key]);
          if (v.length() > 1e-4)
            this.debug.add(
              new T.ArrowHelper(
                v.clone().normalize(),
                new T.Vector3(...data.position).add(new T.Vector3(0, 0, 0.03)),
                v.length(),
                [0xf09575, 0xbde980, 0x79b7ff][i],
              ),
            );
        },
      );
    if (this.options.contacts && data.foot_contacts && this.character) {
      (this.data.contacts || []).forEach((contact) => {
        const value = data.foot_contacts[contact.contact],
          world = this.character.world[contact.transform];
        if (!(value > this.contactThreshold) || !world) return;
        const points = contact.points.map((p) => {
          const v = new T.Vector3(...p).applyMatrix4(world);
          v.z = Math.max(v.z, 0.003);
          return v;
        });
        this.debug.add(
          new T.LineLoop(
            new T.BufferGeometry().setFromPoints(points),
            new T.LineBasicMaterial({ color: new T.Color(1, 1 - value, 0) }),
          ),
        );
      });
    }
    this.invalidate();
  }
}
