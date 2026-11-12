import * as T from "three";
import { View } from "./view.js";
import {
  clone,
  transform,
  rotation,
  quaternion,
  node,
  entry,
  component,
  fresh,
  nodes,
  nextId,
  kindOf,
  validate,
} from "./documents.js";

const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
};
const status = (text, error = false) => {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
};
const safe =
  (fn) =>
  async (...args) => {
    try {
      return await fn(...args);
    } catch (e) {
      status(e.message, true);
    }
  };
const button = (text, fn, parent) => {
  const b = el("button", text);
  b.onclick = safe(fn);
  parent?.append(b);
  return b;
};
const http = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) throw Error(await response.text());
  return response.json();
};
const downloadURL = (path) => "/api/download?path=" + encodeURIComponent(path);
let assets = [],
  mode = "inbetween",
  selected = 1,
  doc = fresh("inbetween", "soma"),
  path = null,
  dirty = false;
let undo = [],
  redo = [],
  savedModes = new Map(),
  loaded = null,
  motion = null,
  frame = 0,
  playing = false,
  sceneReady = false,
  recording = false;
let replaying = false;
let socket,
  opening,
  requestID = 0,
  pending = new Map(),
  queue = Promise.resolve(),
  epoch = 0,
  busy = 0,
  revision = 0,
  previewTimer;
let samples = new Map(),
  keys = new Set(),
  interact = false,
  stepPending = false,
  lastStep = 0,
  lastFrame = 0,
  gameButtons = [];
let sceneInspection = null,
  sequenceSelections = new Map(),
  sceneLocks = new WeakMap();
const kind = () => (mode === "preview" ? "scene" : mode);
const items = () => nodes(doc);
const chosen = () => items().find(({ item }) => item.id === selected)?.item;
const rootId = () =>
  doc.root?.id ?? doc.entries?.[0].id ?? doc.character_start.id;
const locksFor = (item) => {
  if (!item) return {};
  if (item.translation_locks) return item;
  if (!sceneLocks.has(item))
    sceneLocks.set(item, {
      translation_locks: [false, false, false],
      rotation_locks: [false, false, false],
    });
  return sceneLocks.get(item);
};
const view = new View(
  $("viewport"),
  (id) => select(id),
  (id, value) => {
    const item = items().find((n) => n.item.id === id)?.item;
    if (item) {
      const locks = locksFor(item),
        prior = item.local_transform || item.transform,
        angles = rotation(value.rotation_xyzw),
        previous = rotation(prior.rotation_xyzw);
      value.translation = value.translation.map((v, i) =>
        locks.translation_locks?.[i] ? prior.translation[i] : v,
      );
      item.euler_degrees = angles.map((v, i) =>
        locks.rotation_locks?.[i] ? previous[i] : v,
      );
      value.rotation_xyzw = quaternion(item.euler_degrees);
      item[item.local_transform ? "local_transform" : "transform"] = value;
      const object = view.byId.get(id);
      object.position.set(...value.translation);
      object.quaternion.set(...value.rotation_xyzw);
      dirty = true;
      revision++;
      sceneReady = false;
    }
  },
  (dragging) => {
    if (dragging) {
      checkpoint();
      playing = false;
    } else {
      renderProperties();
      persist();
      schedulePreview();
    }
  },
);

function connect() {
  if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
  if (opening) return opening;
  opening = new Promise((resolve, reject) => {
    socket = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/session`,
    );
    socket.onopen = () => {
      opening = null;
      resolve();
    };
    socket.onmessage = (event) => {
      const result = JSON.parse(event.data),
        wait = pending.get(result.id);
      if (!wait) return;
      pending.delete(result.id);
      result.error ? wait.reject(Error(result.error)) : wait.resolve(result);
    };
    socket.onerror = () =>
      reject(Error("Could not connect to the local studio."));
    socket.onclose = () => {
      opening = null;
      epoch++;
      loaded = null;
      playing = false;
      recording = false;
      sceneReady = false;
      for (const p of pending.values())
        p.reject(Error("Native session disconnected."));
      pending.clear();
      reject(Error("Session closed."));
      updateControls();
    };
  });
  return opening;
}
async function send(message) {
  await connect();
  return new Promise((resolve, reject) => {
    const id = ++requestID;
    pending.set(id, { resolve, reject });
    try {
      socket.send(JSON.stringify({ ...message, id }));
    } catch (e) {
      pending.delete(id);
      reject(e);
    }
  });
}
function request(message) {
  const generation = epoch;
  const result = queue
    .catch(() => {})
    .then(() => {
      if (generation !== epoch) throw Error("Operation cancelled.");
      return send(message);
    });
  queue = result;
  return result;
}
async function working(label, fn) {
  busy++;
  status(label);
  updateControls();
  try {
    return await fn();
  } finally {
    busy--;
    updateControls();
  }
}
function requireRig() {
  if (!loaded) throw Error("Load a rig first.");
  if (loaded.rig !== doc.rig) throw Error("Load the document’s rig first.");
}
function updateControls() {
  document.body.classList.toggle("busy", busy > 0);
  $("engine-state").textContent = loaded
    ? `${loaded.rig === "soma" ? "SOMA" : "G1"} · ${loaded.precision}`
    : busy
      ? "Loading"
      : "Offline";
  $("load").disabled = busy > 0 || recording;
  $("rig").disabled = busy > 0 || !!loaded;
  $("precision").disabled = busy > 0 || !!loaded;
  $("empty").hidden = !!loaded;
  $("generate").hidden = mode !== "inbetween";
  $("generate").disabled = busy > 0 || !loaded;
  $("record").hidden = mode !== "preview";
  $("record").textContent = recording ? "Finish recording" : "Record";
  $("record").classList.toggle("recording", recording);
  $("record").disabled = busy > 0 || !loaded;
  $("play").textContent = playing ? "Pause" : "Play";
  $("play").disabled = busy > 0 || (!motion && mode !== "preview");
  $("live-hint").hidden = mode !== "preview";
  $("scrub").disabled =
    !motion || recording || (playing && mode === "preview" && !replaying);
  $("return-live").hidden = mode !== "preview" || !replaying;
  $("properties").inert = busy > 0 || mode === "preview";
  $("additions").inert = busy > 0;
  $("sequences").inert = busy > 0 || mode === "preview";
  $("modes").inert = busy > 0;
  view.gizmo.enabled = busy === 0 && !recording;
  $("translate").disabled = mode === "preview";
  $("rotate").disabled = mode === "preview";
  $("undo").disabled =
    busy > 0 || recording || !undo.length || mode === "preview";
  $("redo").disabled =
    busy > 0 || recording || !redo.length || mode === "preview";
}
function checkpoint() {
  undo.push(clone(doc));
  if (undo.length > 100) undo.shift();
  redo = [];
}
function persist() {
  try {
    localStorage.setItem(
      "motionbricks-draft",
      JSON.stringify({ doc, mode, path, selected }),
    );
  } catch {
    status("Draft exceeds browser storage; save it to the workspace.", true);
  }
}
function change(fn, { preview = true } = {}) {
  if (busy) throw Error("Wait for the current operation or cancel it.");
  if (recording) throw Error("Finish recording before editing.");
  const before = clone(doc);
  checkpoint();
  try {
    fn();
    validate(doc);
  } catch (e) {
    doc = before;
    undo.pop();
    render();
    throw e;
  }
  dirty = true;
  revision++;
  sceneReady = false;
  playing = false;
  render();
  persist();
  if (preview) schedulePreview();
}
function select(id) {
  selected = id;
  renderTree();
  renderProperties();
  view.select(id, locksFor(chosen()));
  if (mode === "preview") view.gizmo.detach();
}
function render() {
  $("modes")
    .querySelectorAll("button")
    .forEach((b) => b.setAttribute("aria-selected", b.dataset.mode === mode));
  $("tree-title").textContent =
    kind() === "inbetween" ? "TARGETS" : "HIERARCHY";
  view.document(doc, kind());
  renderTree();
  renderProperties();
  renderSequences();
  renderAdditions();
  $("json").value = JSON.stringify(doc, null, 2);
  updateControls();
}
function clips() {
  const suffix = doc.rig === "soma" ? ".bvh" : ".csv";
  return assets.filter((a) => a.path.endsWith(suffix));
}
function clipDefault() {
  const list = clips();
  return (
    list.find((a) => a.name.startsWith("Neutral_walk_forward_002__A057."))
      ?.path ||
    list.find((a) => /walk/i.test(a.name))?.path ||
    list[0]?.path ||
    null
  );
}
async function refresh() {
  const data = await http("/api/catalog");
  assets = data.files;
  $("pin").textContent = data.source.revision.slice(0, 12);
  renderLibrary();
}
function renderLibrary() {
  const list = $("library"),
    previous = list.value,
    query = $("search").value.toLowerCase();
  list.replaceChildren();
  for (const file of assets) {
    if (!file.path.toLowerCase().includes(query) || file.path.endsWith(".npz"))
      continue;
    const o = new Option(file.path, file.path);
    list.add(o);
  }
  list.value = previous;
}
function renderTree() {
  $("tree").replaceChildren();
  for (const { item, depth, index } of items()) {
    const b = button(
      item.name || (item.id === 1 ? "CharacterStart" : String(item.id)),
      () => select(item.id),
      $("tree"),
    );
    b.style.paddingLeft = 8 + depth * 14 + "px";
    b.classList.toggle("selected", item.id === selected);
    b.setAttribute("role", "treeitem");
    b.setAttribute("aria-selected", item.id === selected);
    b.draggable = mode !== "preview" && item.id !== rootId();
    b.ondragstart = (e) =>
      e.dataTransfer.setData("text/plain", String(item.id));
    b.ondragover = (e) => e.preventDefault();
    b.ondrop = safe((e) => {
      e.preventDefault();
      const id = Number(e.dataTransfer.getData("text/plain"));
      if (kind() === "inbetween") {
        if (index === 0 || id === rootId()) return;
        change(() => {
          const from = doc.entries.findIndex((i) => i.id === id);
          if (from < 1) return;
          const [moving] = doc.entries.splice(from, 1);
          doc.entries.splice(
            doc.entries.findIndex((i) => i.id === item.id),
            0,
            moving,
          );
        });
      } else reparent(id, item.id);
    });
  }
  view.select(selected, locksFor(chosen()));
  if (mode === "preview") view.gizmo.detach();
}
function reparent(id, parentId) {
  if (kind() === "primitive" && parentId === null) parentId = doc.root.id;
  const source = items().find((n) => n.item.id === id),
    parent = items().find((n) => n.item.id === parentId);
  if (
    !source ||
    id === rootId() ||
    id === parentId ||
    (parentId === rootId() && kind() === "scene")
  )
    return;
  for (let n = parent; n; n = items().find((i) => i.item === n.parent))
    if (n.item.id === id) throw Error("A node cannot contain its own parent.");
  view.items.updateMatrixWorld(true);
  const world = view.byId.get(id).matrixWorld.clone(),
    target = parent ? view.byId.get(parentId).matrixWorld : new T.Matrix4();
  const local = target.clone().invert().multiply(world),
    p = new T.Vector3(),
    q = new T.Quaternion();
  local.decompose(p, q, new T.Vector3());
  change(() => {
    if (kind() === "primitive") {
      source.parent.children = source.parent.children.filter(
        (n) => n.id !== id,
      );
      (parent?.item || doc.root).children.push(source.item);
    } else source.item.parent_id = parentId;
    source.item[source.item.local_transform ? "local_transform" : "transform"] =
      { translation: p.toArray(), rotation_xyzw: q.toArray() };
    source.item.euler_degrees = rotation(q.toArray());
  });
}
function field(
  parent,
  label,
  obj,
  key,
  { type = "text", min, max, step, options, onChange, disabled = false } = {},
) {
  const wrap = el("label", label),
    input = el(options ? "select" : "input");
  if (options) {
    for (const option of options) {
      const [value, text] = Array.isArray(option) ? option : [option, option];
      input.add(new Option(text, String(value)));
    }
    input.value = String(obj[key] ?? "");
  } else {
    input.type = type;
    if (type === "checkbox") {
      input.checked = !!obj[key];
      wrap.className = "check";
    } else input.value = obj[key] ?? "";
    if (min !== undefined) input.min = min;
    if (max !== undefined) input.max = max;
    if (step !== undefined) input.step = step;
  }
  input.disabled = disabled || mode === "preview";
  input.onchange = safe(() => {
    if (!input.checkValidity()) {
      input.reportValidity();
      return;
    }
    let value =
      type === "checkbox"
        ? input.checked
        : type === "number" || type === "range"
          ? input.valueAsNumber
          : input.value;
    if (typeof value === "number" && !Number.isFinite(value))
      throw Error("Enter a finite number.");
    change(() => {
      obj[key] = value;
      onChange?.(value);
    });
  });
  wrap.append(input);
  parent.append(wrap);
  return input;
}
function vector(parent, label, obj, key, { step = 0.05, locked = [] } = {}) {
  const wrap = el("label", label),
    row = el("div", undefined, "vectors");
  wrap.append(row);
  parent.append(wrap);
  obj[key].forEach((value, i) => {
    const input = el("input");
    input.type = "number";
    input.step = step;
    input.value = Number(value.toFixed(5));
    input.ariaLabel = label + " " + "XYZ"[i];
    input.disabled = mode === "preview" || !!locked[i];
    input.onchange = safe(() => {
      if (!Number.isFinite(input.valueAsNumber))
        throw Error("Enter a finite number.");
      change(() => {
        obj[key][i] = input.valueAsNumber;
      });
    });
    row.append(input);
  });
  return row;
}
function title(parent, text) {
  parent.append(el("div", text, "section-title"));
}
function color(parent, label, obj, key) {
  const v = { value: "#" + new T.Color(...obj[key]).getHexString() };
  field(parent, label, v, "value", {
    type: "color",
    onChange: (value) => {
      obj[key] = new T.Color(value).toArray();
    },
  });
}
function controlsForTarget(parent, target, isEntry = false, isStart = false) {
  field(parent, "Clip", target, "clip", {
    options: [["", "Choose a clip"], ...clips().map((c) => [c.path, c.name])],
    onChange: (value) => {
      target.clip = value || null;
      if (isEntry) target.time = 0;
      else {
        target.sample_time_seconds = 0;
        target.frame_index = 0;
      }
      samples.clear();
    },
  });
  const info = samples.get(target.clip),
    rate = info?.sample_rate || (doc.rig === "soma" ? 30 : 120),
    duration = info?.duration;
  field(parent, "Position units", target, "playback_mode", {
    options: [
      ["time", "Seconds"],
      ["frames", "Frames"],
    ],
    onChange: () => {
      target.frame_index = Math.round(
        (isEntry ? target.time : target.sample_time_seconds) * rate,
      );
    },
  });
  const frames = target.playback_mode === "frames",
    key = frames ? "frame_index" : isEntry ? "time" : "sample_time_seconds";
  const position = (v) => {
    target[isEntry ? "time" : "sample_time_seconds"] = frames ? v / rate : v;
    target.frame_index = Math.round(
      (isEntry ? target.time : target.sample_time_seconds) * rate,
    );
  };
  field(parent, frames ? "Frame" : "Seconds", target, key, {
    type: "number",
    min: 0,
    max:
      duration === undefined
        ? undefined
        : frames
          ? Math.round(duration * rate)
          : duration,
    step: frames ? 1 : 1 / rate,
    onChange: position,
  });
  if (duration !== undefined)
    field(parent, "Sample position", target, key, {
      type: "range",
      min: 0,
      max: frames ? Math.round(duration * rate) : duration,
      step: frames ? 1 : 1 / rate,
      onChange: position,
    });
  if (!isEntry)
    field(parent, "Constrain root", target, "hips_constrained", {
      type: "checkbox",
    });
  if (!isStart) {
    field(
      parent,
      "Poses to use",
      target,
      isEntry ? "num_poses" : "num_poses_to_use",
      { type: "number", min: 1, max: 4, step: 1 },
    );
    field(
      parent,
      "Transition early · frames",
      target,
      isEntry ? "early" : "transition_early_by_frames",
      { type: "number", min: 0, max: 20, step: 1 },
    );
    field(parent, "Minimum tokens", target, "min_tokens", {
      type: "number",
      min: 6,
      max: 16,
      step: 1,
      onChange: (v) => {
        target.max_tokens = Math.max(v, target.max_tokens);
      },
    });
    field(parent, "Maximum tokens", target, "max_tokens", {
      type: "number",
      min: target.min_tokens,
      max: 16,
      step: 1,
    });
    parent.append(
      el(
        "small",
        `${target.min_tokens * 4}–${target.max_tokens * 4} raw frames`,
      ),
    );
  }
  if (!isEntry && doc.rig === "soma") {
    title(parent, "IK pins");
    ["Left hand", "Right hand", "Left foot", "Right foot"].forEach(
      (name, i) => {
        const state = { value: !!(target.ik_pin_mask & (1 << i)) };
        field(parent, name, state, "value", {
          type: "checkbox",
          onChange: (v) => {
            target.ik_pin_mask = v
              ? target.ik_pin_mask | (1 << i)
              : target.ik_pin_mask & ~(1 << i);
          },
        });
      },
    );
  }
  for (const [key, label] of [
    ["show_mesh", "Mesh"],
    ["show_skeletons", "Pose window"],
    ["show_velocity", "Velocity"],
    ["show_trajectory", "Trajectory"],
  ])
    field(parent, label, target, key, { type: "checkbox" });
  field(parent, "Trajectory scale", target, "trajectory_scale", {
    type: "range",
    min: 1,
    max: 4,
    step: 0.1,
  });
  if (target.color) color(parent, "Pose color", target, "color");
}
function renderProperties() {
  const parent = $("properties");
  parent.replaceChildren();
  const item = chosen();
  if (!item) return;
  field(parent, "Document", doc, "name");
  if (item.name !== undefined) field(parent, "Name", item, "name");
  const value = item.local_transform || item.transform,
    locks = locksFor(item);
  vector(parent, "Translation · m", value, "translation", {
    locked: locks.translation_locks,
  });
  const angles = { value: rotation(value.rotation_xyzw) },
    row = vector(parent, "Rotation · °", angles, "value", {
      step: 1,
      locked: locks.rotation_locks,
    });
  [...row.children].forEach((input, i) => {
    input.onchange = safe(() => {
      if (!Number.isFinite(input.valueAsNumber))
        throw Error("Enter a finite angle.");
      change(() => {
        angles.value[i] = input.valueAsNumber;
        value.rotation_xyzw = quaternion(angles.value);
        item.euler_degrees = angles.value;
      });
    });
  });
  button(
    "Reset transform",
    () =>
      change(() => {
        value.translation = value.translation.map((v, i) =>
          locks.translation_locks?.[i] ? v : 0,
        );
        const r = rotation(value.rotation_xyzw).map((v, i) =>
          locks.rotation_locks?.[i] ? v : 0,
        );
        value.rotation_xyzw = quaternion(r);
        item.euler_degrees = r;
      }),
    parent,
  ).disabled = mode === "preview";
  for (const key of ["translation_locks", "rotation_locks"]) {
    const group = el("div", undefined, "row");
    parent.append(group);
    locks[key].forEach((_, i) =>
      field(
        group,
        `${key === "translation_locks" ? "T" : "R"}${"XYZ"[i]}`,
        locks[key],
        i,
        { type: "checkbox" },
      ),
    );
  }
  if (item.visible !== undefined)
    field(parent, "Visible", item, "visible", { type: "checkbox" });
  if (kind() === "inbetween")
    controlsForTarget(parent, item, true, doc.entries[0] === item);
  else if (kind() === "primitive") {
    for (const part of item.components) {
      title(parent, part.type.replaceAll("_", " "));
      button(
        "Remove component",
        () =>
          change(() => {
            item.components = item.components.filter((c) => c !== part);
            for (const sequence of doc.sequences) {
              if (part.type === "target_pose")
                sequence.node_ids = sequence.node_ids.filter(
                  (id) => id !== item.id,
                );
              if (part.type === "pivot" && sequence.pivot_node_id === item.id)
                sequence.pivot_node_id = null;
            }
          }),
        parent,
      );
      if (part.type === "target_pose") controlsForTarget(parent, part);
      else if (part.type === "interaction_box") {
        for (const key of ["width", "depth", "height"])
          field(parent, key, part, key, {
            type: "number",
            min: 0.001,
            step: 0.05,
          });
        color(parent, "Color", part, "color");
      } else {
        vector(parent, "Motion direction", part, "motion_direction");
        for (const [key, options] of [
          [
            "operation_order",
            ["translation_then_rotation", "rotation_then_translation"],
          ],
          ["translation_axis", ["none", "x", "y", "z"]],
          ["translation_constraint", ["unconstrained", "line_projection"]],
          ["rotation_axis", ["none", "x", "y", "z"]],
          ["rotation_constraint", ["unconstrained", "cone"]],
          ["cone_side", ["forward", "mirrored", "both"]],
        ])
          field(parent, key.replaceAll("_", " "), part, key, { options });
        for (const key of [
          "line_min_offset",
          "line_max_offset",
          "cone_half_angle_degrees",
        ])
          field(parent, key.replaceAll("_", " "), part, key, {
            type: "number",
            step: key.includes("angle") ? 1 : 0.05,
            min: key.includes("angle") ? 0 : undefined,
            max: key.includes("angle") ? 180 : undefined,
          });
        for (const key of [
          "show_motion_direction",
          "show_translation_constraint",
          "show_rotation_constraint",
        ])
          field(parent, key.replaceAll("_", " "), part, key, {
            type: "checkbox",
          });
      }
    }
    const row = el("div", undefined, "row wrap");
    parent.append(row);
    for (const type of ["target_pose", "interaction_box", "pivot"])
      if (!item.components.some((c) => c.type === type))
        button(
          "+ " + type.replaceAll("_", " "),
          () =>
            change(() =>
              item.components.push(component(type, nextId(doc), clipDefault())),
            ),
          row,
        );
    const used = doc.sequences
      .filter(
        (s) => s.node_ids.includes(item.id) || s.pivot_node_id === item.id,
      )
      .map((s) => s.name);
    if (used.length)
      parent.append(el("p", "Referenced by: " + used.join(", ")));
  } else if (item.dimensions) {
    for (const key of ["width", "depth", "height"])
      field(parent, key, item.dimensions, key, {
        type: "number",
        min: 0.001,
        step: 0.05,
      });
    color(parent, "Color", item, "color");
  } else if (item.id !== rootId()) {
    field(parent, "Primitive", item, "reference", {
      options: [
        ["", "Choose a primitive"],
        ...assets
          .filter((a) => a.path.endsWith(".smart_primitive.json"))
          .map((a) => [a.path, a.path]),
      ],
      onChange: (v) => {
        item.reference = v || null;
      },
    });
    field(parent, "Behavior", item, "behavior", {
      options: loaded?.behaviors || [
        "VelocityMatchingSequenceBehavior",
        "SittingBehavior",
      ],
    });
    field(parent, "Target poses", item.visibility, "target_poses", {
      type: "checkbox",
    });
    field(parent, "Interaction boxes", item.visibility, "interaction_boxes", {
      type: "checkbox",
    });
    color(parent, "Target color", item, "target_pose_color");
    const sequences = sceneInspection?.sequences?.[item.id] || [],
      state = { value: sequenceSelections.get(item.id) || "" };
    const picker = field(parent, "Preview sequence", state, "value", {
      options: [["", "All"], ...sequences.map((s) => [s.name, s.name])],
    });
    picker.onchange = () => {
      sequenceSelections.set(item.id, picker.value);
      drawSceneInspection();
    };
  }
  if (kind() !== "inbetween" && item.id !== rootId()) {
    const state = {
      parent: items().find((n) => n.item === item).parent?.id ?? "",
    };
    const picker = field(parent, "Parent", state, "parent", {
      options: [
        ["", "World"],
        ...items()
          .filter(
            (n) =>
              n.item.id !== item.id &&
              (kind() === "primitive" || n.item.id !== rootId()),
          )
          .map((n) => [n.item.id, n.item.name]),
      ],
    });
    picker.onchange = safe(() =>
      reparent(item.id, picker.value === "" ? null : Number(picker.value)),
    );
  }
  $("delete").disabled =
    item.id === rootId() ||
    mode === "preview" ||
    (kind() === "inbetween" && doc.entries.length <= 2);
}
function renderAdditions() {
  const parent = $("additions");
  parent.replaceChildren();
  if (mode === "preview") return;
  if (kind() === "inbetween")
    button(
      "+ Target",
      () =>
        change(() => {
          const id = nextId(doc),
            from = chosen() || doc.entries.at(-1),
            add = { ...clone(from), id, name: `Target ${id - 1}` };
          add.transform.translation[1] -= 1.5;
          doc.entries.push(add);
          selected = id;
        }),
      parent,
    );
  else if (kind() === "primitive")
    button(
      "+ Child node",
      () =>
        change(() => {
          let n = 1;
          while (items().some(({ item }) => item.name === `Node ${n}`)) n++;
          const add = node(nextId(doc), `Node ${n}`);
          chosen().children.push(add);
          selected = add.id;
        }),
      parent,
    );
  else {
    button(
      "+ Primitive",
      () =>
        change(() => {
          const id = nextId(doc);
          doc.smart_primitives.push({
            id,
            name: "Primitive",
            parent_id: null,
            transform: transform(),
            euler_degrees: [0, 0, 0],
            reference: null,
            visibility: { target_poses: true, interaction_boxes: true },
            target_pose_color: [0.5, 0.75, 0.9],
            behavior: "VelocityMatchingSequenceBehavior",
          });
          selected = id;
        }),
      parent,
    );
    button(
      "+ Cube",
      () =>
        change(() => {
          const id = nextId(doc);
          doc.solid_cubes.push({
            id,
            name: "Cube",
            parent_id: null,
            transform: transform(),
            euler_degrees: [0, 0, 0],
            dimensions: { width: 1, depth: 1, height: 1 },
            color: [0.35, 0.46, 0.51],
          });
          selected = id;
        }),
      parent,
    );
  }
}
function renderSequences() {
  const parent = $("sequences");
  parent.replaceChildren();
  if (kind() !== "primitive") return;
  title(parent, "SEQUENCES");
  button(
    "+ Sequence",
    () =>
      change(() => {
        let n = 1;
        while (doc.sequences.some((s) => s.name === `Sequence ${n}`)) n++;
        doc.sequences.push({
          id: nextId(doc),
          name: `Sequence ${n}`,
          node_ids: [],
          pivot_node_id: null,
        });
      }),
    parent,
  );
  const targets = items().filter((n) =>
      n.item.components.some((c) => c.type === "target_pose"),
    ),
    pivots = items().filter((n) =>
      n.item.components.some((c) => c.type === "pivot"),
    );
  for (const sequence of doc.sequences) {
    const box = el("div", undefined, "sequence");
    parent.append(box);
    field(box, "Name", sequence, "name");
    button(
      "Delete sequence",
      () =>
        change(() => {
          doc.sequences = doc.sequences.filter((s) => s !== sequence);
        }),
      box,
    );
    field(box, "Pivot", sequence, "pivot_node_id", {
      options: [["", "None"], ...pivots.map((n) => [n.item.id, n.item.name])],
      onChange: (v) => {
        sequence.pivot_node_id = v === "" ? null : Number(v);
      },
    });
    const list = el("ol");
    box.append(list);
    sequence.node_ids.forEach((id, index) => {
      const li = el(
        "li",
        items().find((n) => n.item.id === id)?.item.name || String(id),
      );
      list.append(li);
      button(
        "↑",
        () =>
          change(() => {
            if (index)
              [sequence.node_ids[index - 1], sequence.node_ids[index]] = [
                id,
                sequence.node_ids[index - 1],
              ];
          }),
        li,
      );
      button(
        "↓",
        () =>
          change(() => {
            if (index < sequence.node_ids.length - 1)
              [sequence.node_ids[index + 1], sequence.node_ids[index]] = [
                id,
                sequence.node_ids[index + 1],
              ];
          }),
        li,
      );
      button("×", () => change(() => sequence.node_ids.splice(index, 1)), li);
    });
    const picker = el("select");
    targets.forEach((n) => picker.add(new Option(n.item.name, n.item.id)));
    box.append(picker);
    button(
      "+ Step",
      () =>
        change(() => {
          if (picker.value) sequence.node_ids.push(Number(picker.value));
        }),
      box,
    );
    box.ondragover = (e) => e.preventDefault();
    box.ondrop = safe((e) => {
      e.preventDefault();
      const id = Number(e.dataTransfer.getData("text/plain"));
      if (targets.some((n) => n.item.id === id))
        change(() => sequence.node_ids.push(id));
    });
  }
}
function schedulePreview() {
  clearTimeout(previewTimer);
  const version = revision;
  previewTimer = setTimeout(() => {
    if (
      loaded &&
      loaded.rig === doc.rig &&
      version === revision &&
      !busy &&
      mode !== "preview"
    )
      safe(preview)();
  }, 220);
}
async function preview() {
  requireRig();
  const version = revision;
  view.clearGhosts();
  if (kind() === "inbetween") {
    const list = clone(doc.entries);
    let shown = false;
    for (let i = 0; i < list.length; i++) {
      const item = list[i];
      if (!item.clip) continue;
      const response = await request({
        op: "sample",
        entry: item,
        start: i === 0,
      });
      if (version !== revision) return;
      samples.set(item.clip, {
        duration: response.duration,
        sample_rate: response.sample_rate,
      });
      const pose = response.poses[i === 0 ? 3 : 0];
      if (!shown) {
        view.pose(pose);
        shown = true;
      }
      view.ghost(
        item.show_skeletons
          ? response.poses.slice(0, i ? item.num_poses : 4)
          : [pose],
        i ? 0x80bfe6 : 0xaee891,
        item.show_mesh,
        item.show_skeletons,
        item.id,
      );
      view.targetTrace(response.poses, {
        ...item,
        trajectory: response.trajectory,
        start: i === 0,
      });
    }
  } else if (kind() === "primitive") {
    const response = await request({ op: "preview", document: clone(doc) });
    if (version !== revision) return;
    for (const info of response.clips || []) samples.set(info.clip, info);
    for (const target of response.targets) {
      view.ghost(
        target.show_skeletons ? target.poses : [target.poses[0]],
        new T.Color(...target.color),
        target.show_mesh,
        target.show_skeletons,
        target.id,
      );
      view.targetTrace(target.poses, target);
    }
    view.overlays(response);
  } else if (mode === "scene") {
    const response = await request({
      op: "scene_preview",
      document: clone(doc),
    });
    if (version !== revision) return;
    sceneInspection = response;
    drawSceneInspection();
  } else await startScene(false);
  if (version === revision) {
    renderProperties();
    status("Native preview ready.");
  }
}
function drawSceneInspection() {
  if (!sceneInspection) return;
  view.clearGhosts();
  for (const target of sceneInspection.targets) {
    const sequence = sceneInspection.sequences[target.owner]?.find(
      (s) => s.name === sequenceSelections.get(target.owner),
    );
    if (sequence && !sequence.targets.includes(target.id)) continue;
    view.ghost(
      [target.poses[0]],
      new T.Color(...target.color),
      true,
      false,
      target.owner,
    );
  }
  view.overlays(sceneInspection);
}
async function startScene(play = true) {
  requireRig();
  if (!sceneReady) {
    const response = await request({ op: "scene", document: clone(doc) });
    sceneReady = true;
    view.pose(response.pose);
    view.clearGhosts();
    view.liveTargets(response.targets || []);
    view.overlays(response);
    view.trail([]);
  }
  playing = play;
  if (play) $("viewport").focus();
  lastStep = performance.now();
  updateControls();
}
async function resultMotion(result) {
  const incoming = await http(downloadURL(result.motion));
  if (incoming.rig !== loaded?.rig)
    throw Error("Motion and loaded rig differ.");
  motion = incoming;
  frame = 0;
  replaying = mode === "preview";
  playing = false;
  $("scrub").max = motion.frames.length - 1;
  $("scrub").value = 0;
  $("export").hidden = false;
  $("export").href = downloadURL(result.archive);
  $("export").download = "motion.npz";
  $("segments").replaceChildren();
  for (const [i, segment] of (motion.segments || []).entries()) {
    const span = el("span", String(i + 1));
    span.style.flex = segment.frames;
    span.title = `Target ${segment.target}: ${segment.frames} frames`;
    $("segments").append(span);
  }
  displayFrame();
  view.trail(motion.frames);
  updateControls();
  await refresh();
}
function clearPlayback() {
  motion = null;
  frame = 0;
  replaying = false;
  playing = false;
  $("scrub").max = 0;
  $("scrub").value = 0;
  $("export").hidden = true;
  $("export").removeAttribute("href");
  $("segments").replaceChildren();
  $("time").textContent = "0.00 s";
  view.trail([]);
}
function displayFrame() {
  if (!motion?.frames.length || motion.rig !== loaded?.rig) return;
  const index = Math.min(motion.frames.length - 1, Math.floor(frame));
  view.pose(motion.frames[index]);
  view.overlays({ foot_contacts: motion.foot_contacts[index] });
  $("scrub").value = index;
  $("time").textContent =
    `${(index / motion.sample_rate).toFixed(2)} / ${((motion.frames.length - 1) / motion.sample_rate).toFixed(2)} s`;
}
function switchMode(next) {
  if (busy) throw Error("Wait for the current operation or cancel it.");
  if (recording) throw Error("Finish recording before changing workspace.");
  playing = false;
  if ((next === "preview" ? "scene" : next) !== kind()) {
    savedModes.set(kind(), { doc, path, selected, undo, redo, dirty });
    const stored = savedModes.get(next === "preview" ? "scene" : next);
    doc =
      stored?.doc || fresh(next === "preview" ? "scene" : next, $("rig").value);
    path = stored?.path || null;
    selected = stored?.selected || rootId();
    undo = stored?.undo || [];
    redo = stored?.redo || [];
    dirty = stored?.dirty || false;
    clearPlayback();
  }
  mode = next;
  replaying = false;
  revision++;
  sceneReady = false;
  sceneInspection = null;
  view.clearGhosts();
  view.debugPoses();
  render();
  persist();
  if (next === "preview" && loaded)
    safe(() => working("Preparing scene…", () => startScene(false)))();
  else schedulePreview();
}
async function save() {
  const suffix =
    kind() === "primitive"
      ? ".smart_primitive.json"
      : kind() === "scene"
        ? ".scene.json"
        : ".motion.json";
  const proposed = path?.startsWith("workspace/")
    ? path
    : "workspace/projects/" + doc.name.replace(/[^\w -]/g, "_") + suffix;
  const answer = prompt("Save document", proposed);
  if (!answer) return null;
  if (!answer.endsWith(suffix)) throw Error(`Use a ${suffix} filename.`);
  if (doc.editor) {
    doc.editor.camera = view.cameraState();
    doc.editor[
      kind() === "primitive" ? "selected_node_id" : "selected_item_id"
    ] = selected;
  }
  const result = await http("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: answer, document: doc }),
  });
  path = result.path;
  dirty = false;
  persist();
  await refresh();
  status("Saved " + path);
  return path;
}
async function openAsset() {
  if (busy || recording)
    throw Error("Finish the current operation before opening a document.");
  const file = $("library").value;
  if (!file) return;
  if (/\.(bvh|csv)$/.test(file)) {
    const item = chosen(),
      target =
        kind() === "inbetween"
          ? item
          : item.components?.find((c) => c.type === "target_pose");
    if (!target) throw Error("Select a target pose first.");
    if (!clips().some((c) => c.path === file))
      throw Error("Choose a clip for the current rig.");
    change(() => {
      target.clip = file;
      if (kind() === "inbetween") target.time = 0;
      else {
        target.sample_time_seconds = 0;
        target.frame_index = 0;
      }
    });
    return;
  }
  const incoming = await http("/api/document?path=" + encodeURIComponent(file));
  if (incoming.frames) {
    requireRig();
    if (incoming.rig !== loaded.rig) throw Error("Load the recording’s rig.");
    await resultMotion({
      motion: file,
      archive: file.replace(/\.json$/, ".npz"),
    });
    return;
  }
  const type = validate(incoming);
  if (dirty && !confirm("Open this document and replace the unsaved draft?"))
    return;
  if (loaded && incoming.rig !== loaded.rig)
    throw Error(
      "Unload the current rig before opening a different rig’s document.",
    );
  mode = type;
  doc = incoming;
  selected =
    incoming.editor?.selected_node_id ||
    incoming.editor?.selected_item_id ||
    rootId();
  path = file;
  dirty = false;
  undo = [];
  redo = [];
  clearPlayback();
  sceneReady = false;
  sceneInspection = null;
  sequenceSelections.clear();
  revision++;
  $("rig").value = doc.rig;
  view.clearGhosts();
  render();
  view.restoreCamera(doc.editor?.camera);
  persist();
  schedulePreview();
}
async function upload(files) {
  const body = new FormData();
  for (const file of files)
    if (/\.(json|bvh|csv)$/i.test(file.name))
      body.append("files", file, file.webkitRelativePath || file.name);
  if (!body.has("files")) throw Error("Choose JSON, BVH or CSV files.");
  await http("/api/upload", { method: "POST", body });
  await refresh();
  status("Imported to the local workspace.");
}

$("load").onclick = safe(() =>
  working("Loading rig · first use builds TensorRT plans…", async () => {
    if (doc.rig !== $("rig").value) {
      if (dirty && !confirm("Create a fresh document for this rig?")) return;
      doc = fresh(kind(), $("rig").value);
      selected = 1;
      path = null;
      undo = [];
      redo = [];
      dirty = false;
      savedModes.clear();
    }
    clearPlayback();
    loaded = await request({
      op: "load",
      rig: $("rig").value,
      precision: $("precision").value,
    });
    sceneReady = false;
    playing = false;
    view.load(loaded.geometry, loaded.pose);
    $("locomotion").replaceChildren(
      ...loaded.modes.map((m) => new Option(m.label, m.value)),
    );
    if (kind() === "inbetween") {
      const clip = clipDefault();
      doc.entries.forEach((e, i) => {
        if (!e.clip) {
          e.clip = clip;
          e.time = i ? 7 : 1;
          e.transform.translation[1] = i ? -2 : 0;
        }
      });
    }
    render();
    status("Rig ready.");
    queueMicrotask(() => {
      schedulePreview();
    });
  }),
);
$("unload").onclick = safe(async () => {
  clearTimeout(previewTimer);
  playing = false;
  recording = false;
  epoch++;
  for (const p of pending.values()) p.reject(Error("Cancelled."));
  pending.clear();
  queue = Promise.resolve();
  await send({ op: "unload" });
  loaded = null;
  sceneReady = false;
  sceneInspection = null;
  clearPlayback();
  view.clearCharacter();
  updateControls();
  status("Rig unloaded.");
});
$("modes").onclick = safe((e) => {
  const b = e.target.closest("[data-mode]");
  if (b) switchMode(b.dataset.mode);
});
$("search").oninput = renderLibrary;
$("refresh").onclick = safe(refresh);
$("open").onclick = safe(openAsset);
$("library").ondblclick = safe(openAsset);
$("import").onclick = () => $("upload").click();
$("folder").onclick = () => $("directory").click();
for (const id of ["upload", "directory"])
  $(id).onchange = safe(async () => {
    await upload($(id).files);
    $(id).value = "";
  });
$("new").onclick = safe(() => {
  if (busy) throw Error("Wait for the current operation or cancel it.");
  if (recording) throw Error("Finish recording first.");
  if (dirty && !confirm("Replace the unsaved draft?")) return;
  doc = fresh(kind(), $("rig").value);
  if (doc.entries)
    doc.entries.forEach((e) => {
      e.clip = clipDefault();
    });
  path = null;
  selected = 1;
  undo = [];
  redo = [];
  dirty = false;
  sceneReady = false;
  sceneInspection = null;
  clearPlayback();
  revision++;
  render();
  persist();
});
$("delete").onclick = safe(() => {
  const item = chosen();
  if (!item || item.id === rootId() || mode === "preview") return;
  change(() => {
    if (kind() === "inbetween") {
      if (doc.entries.length <= 2) throw Error("Keep one target.");
      doc.entries = doc.entries.filter((i) => i !== item);
    } else {
      const ids = new Set([item.id]);
      let expanded = true;
      while (expanded) {
        expanded = false;
        for (const n of items())
          if (n.parent && ids.has(n.parent.id) && !ids.has(n.item.id)) {
            ids.add(n.item.id);
            expanded = true;
          }
      }
      if (kind() === "primitive") {
        const p = items().find((n) => n.item === item).parent;
        p.children = p.children.filter((n) => n !== item);
        doc.sequences.forEach((s) => {
          s.node_ids = s.node_ids.filter((id) => !ids.has(id));
          if (ids.has(s.pivot_node_id)) s.pivot_node_id = null;
        });
      } else {
        doc.smart_primitives = doc.smart_primitives.filter(
          (i) => !ids.has(i.id),
        );
        doc.solid_cubes = doc.solid_cubes.filter((i) => !ids.has(i.id));
      }
    }
    selected = rootId();
  });
});
$("undo").onclick = safe(() => {
  if (!undo.length || recording || busy) return;
  redo.push(clone(doc));
  doc = undo.pop();
  selected = rootId();
  dirty = true;
  revision++;
  sceneReady = false;
  playing = false;
  render();
  persist();
  schedulePreview();
});
$("redo").onclick = safe(() => {
  if (!redo.length || recording || busy) return;
  undo.push(clone(doc));
  doc = redo.pop();
  selected = rootId();
  dirty = true;
  revision++;
  sceneReady = false;
  playing = false;
  render();
  persist();
  schedulePreview();
});
for (const id of ["translate", "rotate"])
  $(id).onclick = () => {
    view.mode(id);
    view.select(selected, locksFor(chosen()));
    $("translate").setAttribute("aria-pressed", id === "translate");
    $("rotate").setAttribute("aria-pressed", id === "rotate");
  };
$("frame").onclick = () => view.frame();
$("save").onclick = safe(save);
$("package").onclick = safe(async () => {
  const saved = await save();
  if (saved) {
    const a = el("a");
    a.href = "/api/package?path=" + encodeURIComponent(saved);
    a.download = "project.zip";
    a.click();
  }
});
$("apply-json").onclick = safe(() => {
  const incoming = JSON.parse($("json").value),
    type = validate(incoming);
  if (type !== kind() || incoming.rig !== doc.rig)
    throw Error("JSON must match the workspace and rig.");
  change(() => {
    doc = incoming;
    selected = rootId();
  });
});
$("validate").onclick = safe(() =>
  working("Preparing native preview…", preview),
);
$("generate").onclick = safe(() =>
  working("Generating motion…", async () => {
    requireRig();
    playing = false;
    clearTimeout(previewTimer);
    const result = await request({
      op: "generate",
      entries: clone(doc.entries),
    });
    await resultMotion(result);
    status("Generated " + result.frames + " frames.");
  }),
);
$("play").onclick = safe(async () => {
  if (mode === "preview" && !replaying) {
    if (playing) {
      playing = false;
      updateControls();
    } else await working("Preparing scene…", () => startScene());
  } else {
    if (!motion) return;
    playing = !playing;
    if (frame >= motion.frames.length - 1) frame = 0;
    lastFrame = performance.now();
    updateControls();
  }
});
$("stop").onclick = () => {
  playing = false;
  frame = 0;
  if (mode === "preview" && !replaying) {
    if (!recording) sceneReady = false;
  } else displayFrame();
  updateControls();
};
$("scrub").oninput = () => {
  playing = false;
  frame = Number($("scrub").value);
  displayFrame();
  updateControls();
};
$("record").onclick = safe(() =>
  working(recording ? "Saving recording…" : "Starting recording…", async () => {
    requireRig();
    replaying = false;
    if (!sceneReady) await startScene(false);
    const result = await request({ op: "record", enabled: !recording });
    recording = result.recording;
    if (result.motion) {
      playing = false;
      await resultMotion(result);
    } else {
      playing = recording;
      if (playing) $("viewport").focus();
      lastStep = performance.now();
    }
    status(
      recording ? "Recording native motion at 30 fps." : "Recording saved.",
    );
  }),
);
$("interact").onclick = () => {
  interact = true;
  $("viewport").focus();
};
$("return-live").onclick = () => {
  playing = false;
  replaying = false;
  sceneReady = false;
  updateControls();
};
for (const [key, value] of Object.entries(view.options)) {
  const wrap = el(
      "label",
      key[0].toUpperCase() +
        key.slice(1).replace(/[A-Z]/g, (letter) => " " + letter.toLowerCase()),
      "check",
    ),
    input = el("input");
  input.type = "checkbox";
  input.checked = value;
  input.onchange = () => {
    view.options[key] = input.checked;
    view.applyOptions();
    if (motion) {
      view.trail(motion.frames);
      displayFrame();
    }
  };
  wrap.append(input);
  $("view-options").append(wrap);
}
const threshold = el("label", "Contact threshold"),
  thresholdInput = el("input");
thresholdInput.type = "range";
thresholdInput.min = 0;
thresholdInput.max = 1;
thresholdInput.step = 0.05;
thresholdInput.value = 0.5;
thresholdInput.oninput = () => {
  view.contactThreshold = Number(thresholdInput.value);
  if (view.overlayData) view.overlays(view.overlayData);
};
threshold.append(thresholdInput);
$("view-options").append(threshold);

let lastStyle;
function changeLocomotion(action, delta = 1) {
  const select = $("locomotion"),
    styles = loaded?.styles || [];
  if (!select.options.length) return;
  if (action === "style") {
    const index = styles.indexOf(select.value);
    if (styles.length) {
      select.value =
        index < 0
          ? styles.includes(lastStyle)
            ? lastStyle
            : styles[0]
          : styles[(index + delta + styles.length) % styles.length];
      lastStyle = select.value;
    }
  } else if (action === "straf")
    select.value =
      select.value === "linear_1d" ? "directional_2d" : "linear_1d";
  else if (action === "crouch")
    select.value =
      select.value === "crouch_directional_2d"
        ? "directional_2d"
        : "crouch_directional_2d";
}
window.addEventListener(
  "keydown",
  safe((e) => {
    if (e.target.closest("input,textarea,select,button") || e.altKey) return;
    if (e.ctrlKey || e.metaKey) {
      if (e.code === "KeyZ") {
        e.preventDefault();
        (e.shiftKey ? $("redo") : $("undo")).click();
      }
      if (e.code === "KeyS") {
        e.preventDefault();
        $("save").click();
      }
      return;
    }
    if (
      ["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
        e.code,
      )
    )
      e.preventDefault();
    keys.add(e.code);
    if (e.repeat) return;
    if (mode === "preview") {
      if (e.code === "KeyI") interact = true;
      if (e.code === "KeyO") changeLocomotion("straf");
      if (e.code === "KeyP") changeLocomotion("crouch");
      if (e.code === "BracketLeft") changeLocomotion("style");
      if (e.code === "BracketRight") changeLocomotion("style", -1);
    } else {
      if (e.code === "KeyW") $("translate").click();
      if (e.code === "KeyE") $("rotate").click();
      if (e.code === "KeyF") $("frame").click();
      if (e.code === "Delete") $("delete").click();
    }
  }),
);
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());
document.addEventListener("visibilitychange", () => {
  keys.clear();
  if (document.hidden) {
    playing = false;
    updateControls();
  }
});
window.addEventListener("beforeunload", (e) => {
  if (dirty || recording) {
    e.preventDefault();
    e.returnValue = "";
  }
});
window.addEventListener("pagehide", () => socket?.close());
function input() {
  let x =
      (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) -
      (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0),
    y =
      (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) -
      (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0),
    run = keys.has("Space");
  const pad = navigator
    .getGamepads?.()
    ?.find((p) => p?.connected && p.mapping === "standard");
  if (pad && document.activeElement === $("viewport")) {
    const radial = (a, b) => {
        const length = Math.hypot(a, b),
          strength = (Math.min(length, 1) - 0.2) / 0.8;
        return length <= 0.2
          ? [0, 0]
          : [(a / length) * strength, (b / length) * strength];
      },
      move = radial(pad.axes[0] || 0, -(pad.axes[1] || 0)),
      orbit = radial(pad.axes[2] || 0, pad.axes[3] || 0);
    let [px, py] = move;
    if (!px && !py)
      [px, py] = radial(
        (pad.buttons[15]?.pressed ? 1 : 0) - (pad.buttons[14]?.pressed ? 1 : 0),
        (pad.buttons[12]?.pressed ? 1 : 0) - (pad.buttons[13]?.pressed ? 1 : 0),
      );
    if (px || py) {
      x = px;
      y = py;
      run = true;
    }
    for (const [index, action] of [
      [2, "interact"],
      [3, "straf"],
      [1, "crouch"],
      [0, "style"],
    ])
      if (pad.buttons[index]?.pressed && !gameButtons[index]) {
        if (action === "interact") interact = true;
        else changeLocomotion(action);
      }
    gameButtons = pad.buttons.map((b) => b.pressed);
    const delta = view.camera.position.clone().sub(view.controls.target),
      offset = new T.Spherical().setFromVector3(
        new T.Vector3(delta.x, delta.z, delta.y),
      );
    offset.theta -= orbit[0] * 0.035;
    offset.phi = Math.max(
      0.05,
      Math.min(Math.PI - 0.05, offset.phi + orbit[1] * 0.025),
    );
    const v = new T.Vector3().setFromSpherical(offset);
    view.camera.position
      .copy(view.controls.target)
      .add(new T.Vector3(v.x, v.z, v.y));
    view.invalidate();
  } else gameButtons = [];
  const facing = view.controls.target.clone().sub(view.camera.position);
  facing.z = 0;
  facing.normalize();
  const right = new T.Vector3().crossVectors(facing, new T.Vector3(0, 0, 1)),
    movement = facing.clone().multiplyScalar(y).addScaledVector(right, x);
  if (movement.length() > 1) movement.normalize();
  return {
    facing: facing.toArray(),
    movement: movement.toArray(),
    run,
    mode: $("locomotion").value,
    interact,
    auto: $("auto").checked,
    scene_targets: view.options.targets,
    debug:
      view.options.context ||
      view.options.blendspace ||
      view.options.characterTargets,
  };
}
function animate(now) {
  requestAnimationFrame(animate);
  if (document.hidden) return;
  if (
    playing &&
    mode === "preview" &&
    !replaying &&
    sceneReady &&
    !stepPending &&
    !busy &&
    now - lastStep >= 1000 / 30
  ) {
    const dt = Math.min(0.1, (now - lastStep) / 1000),
      controls = input();
    interact = false;
    lastStep = now;
    stepPending = true;
    const generation = epoch;
    request({ op: "step", dt, ...controls })
      .then((result) => {
        if (generation !== epoch || mode !== "preview") return;
        view.pose(result.pose);
        if (result.scene_targets) view.liveTargets(result.scene_targets);
        view.overlays(result);
        if ($("follow").checked && result.position) {
          const target = new T.Vector3(...result.position).add(
              new T.Vector3(0, 0, 0.9),
            ),
            delta = target.clone().sub(view.controls.target);
          view.camera.position.add(delta);
          view.controls.target.copy(target);
        }
        view.liveTrail(result.position);
        view.debugPoses(result.context, result.blendspace, result.targets);
        $("timing").textContent = Object.entries(result.timing_ms || {})
          .map(([k, v]) => `${k} ${Number(v).toFixed(1)}ms`)
          .join(" · ");
        $("time").textContent =
          `${result.state} · ${Number(result.transition_seconds).toFixed(2)}s${result.interactable.length ? " · I to interact" : ""}`;
      })
      .catch((e) => {
        playing = false;
        status(e.message, true);
        updateControls();
      })
      .finally(() => {
        stepPending = false;
      });
  } else if (playing && motion && (mode !== "preview" || replaying)) {
    frame +=
      Math.min(0.1, (now - lastFrame) / 1000) *
      motion.sample_rate *
      Number($("speed").value);
    if (frame >= motion.frames.length - 1) {
      frame = motion.frames.length - 1;
      playing = false;
      updateControls();
    }
    displayFrame();
  }
  lastFrame = now;
}
try {
  const draft = JSON.parse(
    localStorage.getItem("motionbricks-draft") || "null",
  );
  if (draft) {
    validate(draft.doc);
    doc = draft.doc;
    mode = draft.mode === "preview" ? "scene" : kindOf(doc);
    path = draft.path;
    selected = draft.selected;
    dirty = true;
    $("rig").value = doc.rig;
  }
} catch {
  status("Saved draft could not be restored.", true);
}
render();
safe(refresh)();
requestAnimationFrame(animate);
