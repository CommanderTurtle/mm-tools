import { Euler, Quaternion, MathUtils } from "three";

export const clone = (value) => structuredClone(value);
export const transform = () => ({
  translation: [0, 0, 0],
  rotation_xyzw: [0, 0, 0, 1],
});
export const rotation = (value) =>
  new Euler()
    .setFromQuaternion(new Quaternion(...value).normalize(), "ZYX")
    .toArray()
    .slice(0, 3)
    .map(MathUtils.radToDeg);
export const quaternion = (value) =>
  new Quaternion()
    .setFromEuler(new Euler(...value.map(MathUtils.degToRad), "ZYX"))
    .toArray();
export const camera = () => ({
  pos: [-7, -7, 5],
  pivot: [0, 0, 1],
  pitch: -22,
  yaw: -135,
  fov: 45,
  near: 0.01,
  far: 1000,
});
export const node = (id, name = "Node") => ({
  id,
  name,
  local_transform: transform(),
  euler_degrees: null,
  visible: true,
  translation_locks: [false, false, false],
  rotation_locks: [false, false, false],
  components: [],
  children: [],
});
export const entry = (id, clip = null) => ({
  id,
  name: id === 1 ? "Start" : `Target ${id - 1}`,
  clip,
  time: 0,
  frame_index: 0,
  playback_mode: "time",
  transform: transform(),
  num_poses: 4,
  early: 0,
  min_tokens: 8,
  max_tokens: 14,
  translation_locks: [false, false, true],
  rotation_locks: [true, true, false],
  show_mesh: true,
  show_skeletons: false,
  show_velocity: true,
  show_trajectory: true,
  trajectory_scale: 1,
});
export function component(type, id, clip = null) {
  if (type === "target_pose")
    return {
      type,
      id,
      clip,
      sample_time_seconds: 0,
      frame_index: 0,
      playback_mode: "time",
      hips_constrained: false,
      num_poses_to_use: 4,
      transition_early_by_frames: 0,
      min_tokens: 6,
      max_tokens: 14,
      ik_pin_mask: 0,
      show_mesh: true,
      show_skeletons: false,
      show_velocity: true,
      show_trajectory: true,
      trajectory_scale: 1,
      color: [0.45, 0.72, 0.9],
    };
  if (type === "interaction_box")
    return { type, id, width: 1, depth: 1, height: 1, color: [0.5, 0.8, 0.55] };
  return {
    type: "pivot",
    id,
    motion_direction: [0, -1, 0],
    operation_order: "translation_then_rotation",
    translation_axis: "none",
    translation_constraint: "unconstrained",
    line_min_offset: -0.5,
    line_max_offset: 0.5,
    rotation_axis: "z",
    rotation_constraint: "unconstrained",
    cone_half_angle_degrees: 45,
    cone_side: "forward",
    show_motion_direction: true,
    show_translation_constraint: true,
    show_rotation_constraint: true,
  };
}
export function fresh(kind, rig) {
  if (kind === "inbetween")
    return {
      format: "mm-tools.motionbricks.inbetweener",
      schema_version: 1,
      name: "Motion",
      rig,
      entries: [entry(1), entry(2)],
    };
  const editor = { camera: camera() };
  if (kind === "primitive")
    return {
      format: "motionbricks.smart_primitive.authoring",
      schema_version: 1,
      name: "Primitive",
      rig,
      editor: { ...editor, selected_node_id: 1 },
      root: node(1, "Root"),
      sequences: [],
    };
  return {
    format: "motionbricks.scene",
    schema_version: 1,
    name: "Scene",
    rig,
    editor: { ...editor, selected_item_id: 1 },
    character_start: {
      id: 1,
      transform: transform(),
      euler_degrees: [0, 0, 0],
    },
    smart_primitives: [],
    solid_cubes: [],
  };
}
export function nodes(doc) {
  if (doc.entries)
    return doc.entries.map((item, index) => ({
      item,
      parent: null,
      depth: 0,
      index,
    }));
  if (doc.root) {
    const result = [];
    const walk = (item, parent, depth) => {
      result.push({ item, parent, depth });
      item.children.forEach((child) => walk(child, item, depth + 1));
    };
    walk(doc.root, null, 0);
    return result;
  }
  const items = [
    doc.character_start,
    ...doc.smart_primitives,
    ...doc.solid_cubes,
  ];
  const depth = (item) => {
    let count = 0,
      seen = new Set([item.id]);
    while (item.parent_id != null) {
      item = items.find((i) => i.id === item.parent_id);
      if (!item || seen.has(item.id)) throw Error("Invalid scene hierarchy.");
      seen.add(item.id);
      count++;
    }
    return count;
  };
  return items.map((item) => ({
    item,
    parent: items.find((i) => i.id === item.parent_id) || null,
    depth: depth(item),
  }));
}
export const nextId = (doc) =>
  1 +
  Math.max(
    0,
    ...nodes(doc).flatMap(({ item }) => [
      item.id,
      ...(item.components || []).map((c) => c.id),
    ]),
    ...(doc.sequences || []).map((s) => s.id),
  );
export function kindOf(doc) {
  if (doc.format === "mm-tools.motionbricks.inbetweener") return "inbetween";
  if (doc.format === "motionbricks.smart_primitive.authoring")
    return "primitive";
  if (doc.format === "motionbricks.scene") return "scene";
  throw Error("Choose an Inbetweener, Smart Primitive or Scene document.");
}
export function validate(doc) {
  const kind = kindOf(doc);
  if (doc.schema_version !== 1 || !["soma", "unitree_g1"].includes(doc.rig))
    throw Error("Unsupported document version or rig.");
  if (
    kind === "inbetween" &&
    (!Array.isArray(doc.entries) || doc.entries.length < 2)
  )
    throw Error("Keep a Start and at least one target.");
  const ids = new Set();
  for (const { item } of nodes(doc)) {
    if (!Number.isInteger(item.id) || item.id < 1 || ids.has(item.id))
      throw Error("Node IDs must be unique positive integers.");
    ids.add(item.id);
    const t = item.local_transform || item.transform;
    if (
      !t ||
      t.translation?.length !== 3 ||
      t.rotation_xyzw?.length !== 4 ||
      ![...t.translation, ...t.rotation_xyzw].every(Number.isFinite) ||
      Math.hypot(...t.rotation_xyzw) < 1e-8
    )
      throw Error("Transforms require finite XYZ and a nonzero XYZW rotation.");
  }
  return kind;
}
