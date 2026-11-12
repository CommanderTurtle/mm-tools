import { createView } from "./view.js";

const $ = (id) => document.getElementById(id);
const form = $("form");
const labels = {
  index: "Index",
  retrieve: "Retrieve",
  zone: "Zone",
  proposals: "Propose",
  compile: "Fit",
  complete: "Furnish",
  materials: "Materials",
  render: "Render",
  export: "Export",
};
const fields = {
  retrieval_candidate_k: ["Retrieval candidates", 1, 100, 1],
  retrieval_context_n: ["Reference rooms", 1, 12, 1],
  retrieval_temperature: ["Retrieval temperature", 0.001, 10, 0.001],
  retrieval_max_score_drop: ["Maximum score drop", 0, 10, 0.01],
  small_object_max_candidates: ["Surface candidates", 1, 100, 1],
  small_object_retry_rounds: ["Surface fitting retries", 0, 10, 1],
  wall_object_max_per_zone: ["Wall objects per zone", 1, 30, 1],
  texture_top_k: ["Material candidates", 1, 5, 1],
  texture_sample_top_k: ["Material sampling pool", 1, 5, 1],
};
const presets = {
  bedroom:
    "A warm bedroom with a bed, two nightstands, a wardrobe and a quiet workspace with a desk and chair.",
  "living room":
    "A comfortable living room with a sofa, coffee table, TV stand, armchair and bookshelves.",
  "dining room":
    "A dining room with a dining table, six chairs, a cabinet and warm wood flooring.",
  office:
    "A bright office with two desks, office chairs, bookshelves and a sofa for a small lounge area.",
  bathroom:
    "A bathroom with a bathtub, toilet, sink, mirror and a storage cabinet.",
};
let configuration,
  run,
  timer,
  epoch = 0,
  submitting = false,
  polling = false;
let layoutKey = "",
  modelKey = "",
  galleryKey = "",
  indexKey = "";
let active = null;
let initialized = false;
let view;

function notice(text = "", error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
}
function stored(key, value) {
  try {
    if (arguments.length === 1) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return null;
  }
}
async function api(path, data) {
  const response = await fetch(
    path,
    data === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        },
  );
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
function file(name, download = false) {
  return `/files/${run.id}/${name.split("/").map(encodeURIComponent).join("/")}${download ? "?download=1" : ""}`;
}
function options() {
  return Object.fromEntries(
    Object.keys(configuration.defaults).map((name) => {
      const field = form.elements.namedItem(name);
      return [
        name,
        field.type === "checkbox"
          ? field.checked
          : field.type === "number"
            ? Number(field.value)
            : field.value,
      ];
    }),
  );
}
function fill(data) {
  $("prompt").value = data.prompt;
  for (const [name, value] of Object.entries(data.settings || {})) {
    const field = form.elements.namedItem(name);
    if (field) {
      if (field.type === "checkbox") field.checked = value;
      else field.value = value;
    }
  }
}
function saveDraft() {
  if (configuration)
    stored("scenebloom.draft", {
      prompt: $("prompt").value,
      settings: options(),
    });
}
function buttons() {
  $("generate").disabled = !initialized || submitting || !!active;
  $("resume").hidden =
    !run || !["failed", "cancelled", "interrupted"].includes(run.status);
  $("resume").disabled = submitting || !!active;
  $("cancel").hidden = !active;
  $("cancel").disabled = submitting;
  $("historyOpen").disabled = !initialized || submitting;
}
function setMode(mode) {
  view?.setMode(mode);
  document
    .querySelectorAll("[data-view]")
    .forEach((button) =>
      button.setAttribute("aria-pressed", button.dataset.view === mode),
    );
  $("viewHelp").textContent =
    mode === "walk"
      ? "Click view · WASD / arrows · drag to look"
      : mode === "plan"
        ? "Drag to pan · scroll to zoom"
        : "Drag to orbit · scroll to zoom";
}
function selected(object) {
  $("selection").textContent = object
    ? `${object.category} · ${object.pose.size.map((n) => Number(n).toFixed(2)).join(" × ")} m · ${object.placement_method}`
    : "Select an object in the scene to inspect its placement.";
  document
    .querySelectorAll("[data-object]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        button.dataset.object === object?.instance_id,
      ),
    );
}
function sceneIndex(layout) {
  const plan = layout.zoning_plan || layout;
  $("objects").replaceChildren();
  for (const room of plan.rooms || []) {
    for (const zone of room.functional_zones || []) {
      const group = document.createElement("details");
      group.open = true;
      const heading = document.createElement("summary");
      heading.textContent = zone.label;
      group.append(heading);
      $("objects").append(group);
      for (const object of (layout.placed_objects || []).filter(
        (o) => o.zone_id === zone.zone_id,
      )) {
        const button = document.createElement("button");
        button.textContent = object.category;
        button.dataset.object = object.instance_id;
        button.setAttribute("aria-pressed", "false");
        button.onclick = () => view?.select(object.instance_id);
        group.append(button);
      }
    }
  }
  $("stats").textContent =
    `${plan.rooms?.length || 0} ROOM · ${layout.placed_objects?.length || 0} OBJECTS · METRES`;
}
async function show(data, token) {
  if (token !== epoch) return;
  run = data;
  stored("scenebloom.run", data.id);
  if (data.status === "running") active = data.id;
  else if (active === data.id) active = null;
  $("roomTitle").textContent = data.title;
  $("state").textContent = data.status;
  $("stageLabel").textContent =
    data.status === "running" ? labels[data.stage] : data.status;
  for (const item of $("stages").children) {
    item.className = data.done.includes(item.dataset.stage)
      ? "done"
      : data.stage === item.dataset.stage && data.status === "running"
        ? "active"
        : "";
  }
  buttons();
  notice(
    data.error ||
      (data.status === "cancelled" || data.status === "interrupted"
        ? "Completed stages are saved. Resume when ready."
        : ""),
    data.status === "failed",
  );
  const name = [
    "scene_layout.json",
    "compiled.json",
    "lifted.json",
    "plan.json",
  ].find((n) => data.files.includes(n));
  if (name && layoutKey !== `${data.id}/${name}`) {
    const url = file(name);
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not read the saved scene layout");
    const layout = await response.json();
    if (token !== epoch) return;
    view?.setLayout(layout);
    sceneIndex(layout);
    layoutKey = `${data.id}/${name}`;
    $("empty").hidden = true;
  }
  if (data.files.includes("scene.glb") && modelKey !== data.id && view) {
    const loaded = await view.load(file("scene.glb"));
    if (token !== epoch) return;
    if (loaded) {
      modelKey = data.id;
      $("capture").disabled = false;
      $("empty").hidden = true;
    }
  }
  const images = data.files.filter((name) => name.endsWith(".png"));
  const nextGallery = `${data.id}/${images.join("|")}`;
  if (galleryKey !== nextGallery) {
    galleryKey = nextGallery;
    $("gallery").replaceChildren();
    if (!images.length) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent =
        "Zone images and native camera renders appear here as they complete.";
      $("gallery").append(p);
    }
    for (const name of images) {
      const url = file(name),
        download = file(name, true);
      const button = document.createElement("button");
      const image = document.createElement("img");
      const caption = document.createElement("span");
      image.src = url;
      image.loading = "lazy";
      image.alt = name;
      caption.textContent = name
        .replace("proposals/", "")
        .replace("renders/", "Camera / ");
      button.append(image, caption);
      button.onclick = () => {
        $("largeImage").src = url;
        $("largeImage").alt = name;
        $("imageDownload").href = download;
        $("imageDialog").showModal();
      };
      $("gallery").append(button);
    }
  }
  const nextIndex = `${data.id}/${data.files.join("|")}`;
  if (indexKey !== nextIndex) {
    indexKey = nextIndex;
    $("exports").replaceChildren();
    for (const [name, title] of [
      ["scene.glb", "Scene GLB"],
      ["scene_layout.json", "Scene JSON"],
      ["plan.json", "Zone plan"],
      ["material_assignment.json", "Materials"],
      ["run.json", "Settings"],
    ]) {
      if (data.files.includes(name)) {
        const a = document.createElement("a");
        a.textContent = `${title} ↓`;
        a.href = file(name, true);
        a.download = name;
        $("exports").append(a);
      }
    }
  }
  if ($("logDetails").open) await showLog(token);
}
async function showLog(token = epoch) {
  if (!run) return;
  const response = await fetch(`/api/runs/${run.id}/log`);
  if (!response.ok) throw new Error("Could not read the run log");
  const text = await response.text();
  if (token === epoch) $("log").textContent = text;
}
async function activate(data) {
  const token = ++epoch;
  clearTimeout(timer);
  layoutKey = modelKey = galleryKey = indexKey = "";
  view?.clear();
  setMode("orbit");
  selected(null);
  $("objects").replaceChildren();
  $("capture").disabled = true;
  $("empty").hidden = false;
  $("log").textContent = "";
  fill(data);
  await show(data, token);
  if (token === epoch) schedule();
}
function schedule() {
  clearTimeout(timer);
  if (active && !document.hidden) timer = setTimeout(poll, 1500);
}
async function poll() {
  if (!initialized || polling || submitting || document.hidden) {
    schedule();
    return;
  }
  polling = true;
  const token = epoch;
  try {
    const config = await api("/api/config");
    if (token !== epoch || submitting) return;
    active = config.active;
    if (run) {
      const data = await api(`/api/runs/${run.id}`);
      if (token === epoch && !submitting) await show(data, token);
    }
    buttons();
  } catch (error) {
    if (token === epoch) notice(error.message, true);
  } finally {
    polling = false;
    schedule();
  }
}
async function mutate(operation) {
  if (!initialized || submitting) return;
  submitting = true;
  ++epoch;
  clearTimeout(timer);
  buttons();
  notice();
  try {
    await operation();
  } catch (error) {
    notice(error.message, true);
  } finally {
    submitting = false;
    buttons();
    schedule();
  }
}
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!initialized || active) return;
  const body = { prompt: $("prompt").value, settings: options() };
  saveDraft();
  void mutate(async () => {
    const data = await api("/api/runs", body);
    active = data.id;
    await activate(data);
  });
});
$("resume").onclick = () =>
  void mutate(async () => {
    const data = await api(`/api/runs/${run.id}/resume`, {});
    active = data.id;
    await activate(data);
  });
$("cancel").onclick = () =>
  void mutate(async () => {
    if (active) {
      await api(`/api/runs/${active}/cancel`, {});
      notice("Stopping the active stage…");
    }
  });
$("historyOpen").onclick = async () => {
  if (submitting) return;
  const token = epoch;
  try {
    const rows = await api("/api/runs");
    if (token !== epoch || submitting) return;
    $("runs").replaceChildren();
    for (const row of rows) {
      const button = document.createElement("button");
      const title = document.createElement("strong");
      const meta = document.createElement("small");
      title.textContent = row.title;
      meta.textContent = `${row.status} · ${new Date(row.created * 1000).toLocaleString()}`;
      button.append(title, meta);
      button.onclick = () =>
        void mutate(async () => {
          const data = await api(`/api/runs/${row.id}`);
          $("history").close();
          await activate(data);
        });
      $("runs").append(button);
    }
    if (!rows.length) $("runs").textContent = "No saved rooms yet.";
    $("history").showModal();
  } catch (error) {
    notice(error.message, true);
  }
};
$("historyClose").onclick = () => $("history").close();
$("imageClose").onclick = () => $("imageDialog").close();
$("imageDialog").addEventListener("close", () =>
  $("largeImage").removeAttribute("src"),
);
for (const button of document.querySelectorAll("[data-room]"))
  button.onclick = () => {
    $("prompt").value = presets[button.dataset.room];
    saveDraft();
  };
for (const button of document.querySelectorAll("[data-view]"))
  button.onclick = () => setMode(button.dataset.view);
for (const name of ["walls", "zones", "spin"])
  $(name).onchange = () => view?.[name]($(name).checked);
$("capture").onclick = () => {
  if (view && run) {
    const a = document.createElement("a");
    a.href = view.capture();
    a.download = `room-${run.id.slice(0, 8)}.png`;
    a.click();
  }
};
$("logDetails").addEventListener("toggle", () => {
  if ($("logDetails").open)
    void showLog().catch((error) => notice(error.message, true));
});
form.addEventListener("input", saveDraft);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearTimeout(timer);
  else void poll();
});
window.addEventListener("pagehide", () => clearTimeout(timer));
window.addEventListener("pageshow", (event) => {
  if (event.persisted) void poll();
});

async function init() {
  buttons();
  try {
    view = createView($("viewport"), selected);
  } catch (error) {
    $("empty").textContent =
      `3D viewer unavailable: ${error.message}. Generated files remain downloadable.`;
  }
  configuration = await api("/api/config");
  active = configuration.active;
  $("revision").textContent =
    `scenebloom @ ${configuration.source.revision.slice(0, 7)}`;
  for (const [name, [label, min, max, step]] of Object.entries(fields)) {
    const row = document.createElement("label");
    row.textContent = label;
    const input = document.createElement("input");
    Object.assign(input, {
      name,
      type: "number",
      min,
      max,
      step,
      required: true,
    });
    row.append(input);
    $("settings").append(row);
  }
  fill({ prompt: $("prompt").value, settings: configuration.defaults });
  const draft = stored("scenebloom.draft");
  if (draft && typeof draft.prompt === "string") fill(draft);
  for (const stage of configuration.stages) {
    const item = document.createElement("li");
    item.dataset.stage = stage;
    item.textContent = labels[stage];
    $("stages").append(item);
  }
  buttons();
  const previous = active || stored("scenebloom.run");
  if (previous) {
    try {
      await activate(await api(`/api/runs/${previous}`));
    } catch (error) {
      notice(error.message, true);
    }
  }
  initialized = true;
  buttons();
  schedule();
}
void init().catch((error) => notice(error.message, true));
