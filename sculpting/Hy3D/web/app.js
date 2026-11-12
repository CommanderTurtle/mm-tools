import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const keys = [
  "route",
  "resolution",
  "steps",
  "seed",
  "randomize",
  "offload",
  "remove_background",
];
let current = null,
  active = null,
  connecting = false,
  selecting = false,
  submitting = false,
  sourceImage = null,
  sourceMesh = null;
let epoch = 0,
  imageURL = null,
  timer = null,
  requestController = null,
  imageView = "input.png",
  stage = "refined",
  loaded = "";
let rows = [],
  coarseAvailable = false,
  suspended = false;
const message = (value = "") => {
  $("message").textContent = value;
};
const fileURL = (name) => `/files/${current.id}/${name}`;
const stopped = (run) =>
  run && ["failed", "cancelled", "interrupted"].includes(run.status);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function readSettings() {
  return Object.fromEntries(
    keys.map((key) => [
      key,
      $(key).type === "checkbox"
        ? $(key).checked
        : $(key).type === "number" || key === "resolution"
          ? Number($(key).value)
          : $(key).value,
    ]),
  );
}

function applySettings(value) {
  for (const key of keys) {
    if (value[key] === undefined) continue;
    if ($(key).type === "checkbox") $(key).checked = value[key];
    else $(key).value = value[key];
  }
  routes();
}

function routes() {
  $("mesh-field").hidden = $("route").value !== "mesh";
  $("coarse-note").hidden = coarseAvailable || $("route").value !== "image";
  $("seed").disabled = $("randomize").checked;
}

function controls() {
  const busy = connecting || selecting || submitting || Boolean(active);
  $("settings").disabled = busy;
  $("cancel").hidden = !active;
  $("cancel").disabled = connecting || selecting || submitting;
  $("resume").hidden = !stopped(current) || Boolean(active);
  $("resume").disabled = connecting || selecting || submitting;
  $("refresh").disabled = connecting || selecting || submitting;
  for (const button of $("history").children) button.disabled = busy;
}

function showImage(url) {
  $("reference").hidden = !url;
  $("drop-hint").hidden = Boolean(url);
  if (url) $("reference").src = url;
  else $("reference").removeAttribute("src");
}

function replaceImage(file) {
  if (imageURL) URL.revokeObjectURL(imageURL);
  sourceImage = file;
  imageURL = file ? URL.createObjectURL(file) : null;
  imageView = "input.png";
  showImage(imageURL);
}

function draft() {
  ++epoch;
  current = null;
  loaded = "";
  viewer.clear();
  imageView = "input.png";
  showImage(imageURL);
  $("run-title").textContent = "New geometry";
  message();
  paint();
  history();
}

function fresh() {
  if (active || connecting || selecting || submitting) return;
  ++epoch;
  requestController?.abort();
  current = null;
  loaded = "";
  viewer.clear();
  $("run-title").textContent = "A form, in detail.";
  $("image").value = $("mesh").value = "";
  sourceMesh = null;
  $("mesh-name").textContent = "GLB, PLY, OBJ or STL · embedded geometry";
  replaceImage(null);
  message();
  paint();
  history();
}

function history() {
  $("history").replaceChildren();
  if (!rows.length) {
    const hint = document.createElement("p");
    hint.className = "note";
    hint.textContent = "Completed and interrupted runs stay here.";
    $("history").append(hint);
  }
  for (const row of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-current", String(current?.id === row.id));
    const title = document.createElement("strong");
    title.textContent = row.title;
    const detail = document.createElement("small");
    detail.textContent = `${row.settings.resolution} · ${row.status} · ${new Date(row.created * 1000).toLocaleString()}`;
    button.append(title, detail);
    button.onclick = () =>
      selectRun(row.id).catch((error) => message(error.message));
    $("history").append(button);
  }
  controls();
}

async function refreshHistory() {
  rows = await api("/api/runs");
  history();
}

async function selectRun(id) {
  const ticket = ++epoch;
  requestController?.abort();
  const controller = (requestController = new AbortController());
  selecting = true;
  current = null;
  loaded = "";
  replaceImage(null);
  sourceMesh = null;
  viewer.clear();
  $("image").value = $("mesh").value = "";
  $("mesh-name").textContent = "Loading saved inputs…";
  $("run-title").textContent = "Loading run…";
  message();
  paint();
  controls();
  try {
    const run = await api(`/api/runs/${id}`, { signal: controller.signal });
    if (ticket !== epoch) return;
    current = run;
    if (run.status === "running") active = id;
    else if (active === id) active = null;
    applySettings(run.settings);
    paint();
    const names = ["input.png", ...(run.mesh_input ? [run.mesh_input] : [])];
    const files = await Promise.all(
      names.map(async (name) => {
        const response = await fetch(`/files/${id}/${name}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Saved input unavailable: ${name}`);
        return new File([await response.blob()], name, {
          type: response.headers.get("Content-Type") || "",
        });
      }),
    );
    if (ticket !== epoch) return;
    replaceImage(files[0]);
    sourceMesh = files[1] || null;
    $("mesh-name").textContent = sourceMesh
      ? `${sourceMesh.name} · saved input`
      : "GLB, PLY, OBJ or STL · embedded geometry";
  } catch (error) {
    if (error.name !== "AbortError" && ticket === epoch) message(error.message);
  } finally {
    if (ticket === epoch) {
      selecting = false;
      paint();
      history();
      schedule();
    }
  }
}

function download(name, label) {
  const link = document.createElement("a");
  link.href = `${fileURL(name)}?download=1`;
  link.download = name;
  link.textContent = label;
  $("exports").append(link);
}

function paint() {
  controls();
  $("exports").replaceChildren();
  const completed = current?.status === "complete" && current.result;
  const available = {
    coarse: Boolean(current?.coarse),
    refined: Boolean(completed),
  };
  if (!available[stage]) stage = available.refined ? "refined" : "coarse";
  for (const button of $("stages").children) {
    button.disabled = !available[button.dataset.stage];
    button.setAttribute("aria-pressed", String(stage === button.dataset.stage));
  }
  for (const button of $("images").children) {
    button.disabled =
      button.dataset.image !== "input.png" && !current?.prepared;
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.image === imageView),
    );
  }
  const stats = stage === "refined" ? current?.result : current?.coarse;
  $("vertices").textContent = stats?.vertices?.toLocaleString() || "—";
  $("faces").textContent = stats?.faces?.toLocaleString() || "—";
  $("grid").textContent = completed
    ? current.result.resolution
    : current?.settings.resolution || "—";
  const progress = current?.progress || {};
  $("pulse").classList.toggle("running", Boolean(active));
  $("phase").textContent =
    current?.status === "running"
      ? progress.phase || "Starting stage"
      : current?.status || "Ready";
  $("counter").textContent =
    current?.status === "running" && progress.steps
      ? `${progress.passes ? `PASS ${progress.pass_index + 1}/${progress.passes} · ` : ""}${progress.step}/${progress.steps}`
      : "";
  $("progress").value = completed
    ? 1
    : current?.status === "running" && progress.steps
      ? progress.step / progress.steps
      : 0;
  $("progress-note").textContent =
    current?.error ||
    (current?.status === "running"
      ? "Sampling counts native scheduler updates. Mesh extraction follows each pass."
      : completed
        ? `Refinement ${Math.round(current.result.refinement_seconds)}s · seed ${current.settings.seed}`
        : "Each stage saves its output and releases its models before the next.");
  if (current) {
    $("run-title").textContent = current.title;
    for (const extension of ["glb", "ply", "obj"])
      if (available[stage] && current.files.includes(`${stage}.${extension}`))
        download(
          `${stage}.${extension}`,
          `${stage === "coarse" ? "COARSE " : ""}${extension.toUpperCase()}`,
        );
    download("run.json", "SETTINGS");
    download("original", "ORIGINAL");
  }
  const key = current && available[stage] ? `${current.id}/${stage}` : "";
  if (key !== loaded) {
    loaded = key;
    if (key)
      viewer.load(fileURL(`${stage}.glb`)).catch((error) => {
        if (loaded === key) viewer.error(error.message);
      });
    else viewer.clear();
  }
  if ($("log-panel").open) refreshLog();
}

async function refreshLog() {
  const id = current?.id;
  if (!id) {
    $("log").textContent = "Choose a saved run to inspect its log.";
    return;
  }
  try {
    const response = await fetch(`/api/runs/${id}/log`);
    if (!response.ok) throw new Error("Log unavailable");
    const value = await response.text();
    if (current?.id === id)
      $("log").textContent = value || "Waiting for native output…";
  } catch (error) {
    if (current?.id === id) $("log").textContent = error.message;
  }
}

function schedule() {
  clearTimeout(timer);
  if (active && !suspended) timer = setTimeout(poll, 1200);
}

async function poll() {
  const id = active,
    ticket = epoch;
  if (!id || suspended) return;
  try {
    const run = await api(`/api/runs/${id}`);
    if (ticket !== epoch) return;
    if (current?.id === id) {
      const becameComplete =
        current.status !== "complete" && run.status === "complete";
      current = run;
      if (becameComplete) stage = "refined";
    }
    if (run.status !== "running") active = null;
    $("connection").textContent = "READY";
    paint();
    if (!active) await refreshHistory();
  } catch (error) {
    $("connection").textContent = "RECONNECTING";
    message(error.message);
  } finally {
    schedule();
  }
}

async function connect() {
  if (connecting) return;
  connecting = true;
  controls();
  try {
    const config = await api("/api/config");
    coarseAvailable = config.coarse;
    active = config.active;
    $("connection").textContent = "READY";
    if (!current && !sourceImage) {
      let saved = {};
      try {
        saved = JSON.parse(localStorage.getItem("hy3d.settings") || "{}");
      } catch {}
      applySettings({
        ...config.defaults,
        ...saved,
        ...(!config.coarse ? { route: "mesh" } : {}),
      });
    }
    await refreshHistory();
    if (active) await selectRun(active);
    else if (current?.status === "running") await selectRun(current.id);
    routes();
    controls();
    schedule();
  } catch (error) {
    $("connection").textContent = "UNAVAILABLE";
    message(error.message);
  } finally {
    connecting = false;
    controls();
  }
}

$("form").onsubmit = async (event) => {
  event.preventDefault();
  if (active || connecting || selecting || submitting) return;
  if (!sourceImage) return message("Choose a reference image.");
  const settings = readSettings();
  if (settings.route === "mesh" && !sourceMesh)
    return message("Choose a coarse mesh.");
  const form = new FormData();
  form.append("image", sourceImage);
  if (sourceMesh) form.append("mesh", sourceMesh);
  form.append("settings", JSON.stringify(settings));
  submitting = true;
  controls();
  message();
  try {
    try {
      localStorage.setItem("hy3d.settings", JSON.stringify(settings));
    } catch {}
    const run = await api("/api/runs", { method: "POST", body: form });
    ++epoch;
    current = run;
    active = run.id;
    stage = "coarse";
    applySettings(run.settings);
    paint();
    await refreshHistory();
    schedule();
  } catch (error) {
    message(error.message);
    await connect();
  } finally {
    submitting = false;
    controls();
  }
};

$("cancel").onclick = async () => {
  if (!active) return;
  $("cancel").disabled = true;
  try {
    await api(`/api/runs/${active}/cancel`, { method: "POST" });
    message("Stopping after saving completed stages…");
  } catch (error) {
    message(error.message);
  } finally {
    controls();
  }
};
$("resume").onclick = async () => {
  if (!current || active || connecting || submitting || selecting) return;
  submitting = true;
  controls();
  message();
  try {
    current = await api(`/api/runs/${current.id}/resume`, { method: "POST" });
    active = current.id;
    loaded = "";
    paint();
    schedule();
  } catch (error) {
    message(error.message);
    await connect();
  } finally {
    submitting = false;
    controls();
  }
};
$("new").onclick = fresh;
$("refresh").onclick = () => connect();
$("route").onchange = routes;
$("randomize").onchange = routes;
$("image").onchange = () => {
  if ($("image").files[0]) {
    draft();
    replaceImage($("image").files[0]);
    message();
  }
};
$("mesh").onchange = () => {
  draft();
  sourceMesh = $("mesh").files[0] || null;
  $("mesh-name").textContent =
    sourceMesh?.name || "GLB, PLY, OBJ or STL · embedded geometry";
};
for (const type of ["dragenter", "dragover"])
  $("drop").addEventListener(type, (event) => {
    event.preventDefault();
    if (!$("settings").disabled) $("drop").classList.add("drag");
  });
for (const type of ["dragleave", "drop"])
  $("drop").addEventListener(type, () => $("drop").classList.remove("drag"));
$("drop").addEventListener("drop", (event) => {
  event.preventDefault();
  if (!$("settings").disabled && event.dataTransfer.files[0]) {
    draft();
    replaceImage(event.dataTransfer.files[0]);
  }
});
$("stages").onclick = (event) => {
  if (event.target.dataset.stage && !event.target.disabled) {
    stage = event.target.dataset.stage;
    paint();
  }
};
$("images").onclick = (event) => {
  const name = event.target.dataset.image;
  if (!name || event.target.disabled) return;
  imageView = name;
  showImage(name === "input.png" ? imageURL : current ? fileURL(name) : null);
  for (const button of $("images").children)
    button.setAttribute("aria-pressed", String(button === event.target));
};
$("log-panel").ontoggle = () => {
  if ($("log-panel").open) refreshLog();
};

function createViewer() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: $("canvas"),
      antialias: true,
      alpha: true,
    });
  } catch {
    $("viewer-error").hidden = false;
    $("viewer-error").textContent =
      "WebGL is unavailable. Geometry downloads remain available.";
    return { clear() {}, async load() {}, error() {} };
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.autoRotateSpeed = 1.1;
  const hemisphere = new THREE.HemisphereLight(0xd6eee1, 0x27382f, 2.1);
  const key = new THREE.DirectionalLight(0xffedcd, 3),
    fill = new THREE.DirectionalLight(0xa8cfe0, 1.5);
  key.position.set(3, 5, 4);
  fill.position.set(-4, 1, -2);
  scene.add(hemisphere, key, fill);
  const materials = {
    solid: new THREE.MeshStandardMaterial({
      color: 0xc7d2c5,
      roughness: 0.75,
      metalness: 0.05,
      side: THREE.DoubleSide,
    }),
    normal: new THREE.MeshNormalMaterial({ side: THREE.DoubleSide }),
    wire: new THREE.MeshBasicMaterial({ color: 0x9ee1c3, wireframe: true }),
  };
  let object = null,
    pending = null,
    generation = 0,
    frame = null,
    alive = true;
  const render = () => {
    frame = null;
    if (!alive || document.hidden) return;
    if (orbit.autoRotate) orbit.update();
    renderer.render(scene, camera);
    if (orbit.autoRotate) invalidate();
  };
  function invalidate() {
    if (frame === null && alive && !document.hidden)
      frame = requestAnimationFrame(render);
  }
  function dispose(value) {
    value?.traverse((child) => {
      child.geometry?.dispose();
      if (child.material && !Object.values(materials).includes(child.material))
        for (const material of Array.isArray(child.material)
          ? child.material
          : [child.material])
          material.dispose();
    });
  }
  function clear() {
    ++generation;
    pending?.abort();
    pending = null;
    if (object) {
      scene.remove(object);
      dispose(object);
      object = null;
    }
    $("empty").hidden = false;
    $("viewer-error").hidden = true;
    invalidate();
  }
  function fit() {
    orbit.target.set(0, 0, 0);
    camera.position.set(2.5, 1.7, 3.2);
    orbit.update();
    invalidate();
  }
  function surface() {
    object?.traverse((child) => {
      if (child.isMesh) child.material = materials[$("surface").value];
    });
    invalidate();
  }
  async function load(url) {
    clear();
    const ticket = generation;
    pending = new AbortController();
    $("empty").hidden = true;
    try {
      const response = await fetch(url, { signal: pending.signal });
      if (!response.ok)
        throw new Error("Geometry preview could not be loaded.");
      const gltf = await new GLTFLoader().parseAsync(
        await response.arrayBuffer(),
        "",
      );
      if (ticket !== generation) {
        dispose(gltf.scene);
        return;
      }
      object = gltf.scene;
      object.traverse((child) => {
        if (child.isMesh) {
          for (const old of Array.isArray(child.material)
            ? child.material
            : [child.material])
            old?.dispose();
          child.material = materials[$("surface").value];
          if (!child.geometry.attributes.normal)
            child.geometry.computeVertexNormals();
        }
      });
      const box = new THREE.Box3().setFromObject(object),
        center = box.getCenter(new THREE.Vector3());
      const extent = box.getSize(new THREE.Vector3()).length();
      const scale = 2.5 / Math.max(extent, 1e-6);
      object.scale.multiplyScalar(scale);
      object.position.sub(center.multiplyScalar(scale));
      scene.add(object);
      fit();
    } catch (error) {
      if (error.name !== "AbortError" && ticket === generation) throw error;
    }
  }
  orbit.addEventListener("change", invalidate);
  const observer = new ResizeObserver(() => {
    const { width, height } = $("viewer").getBoundingClientRect();
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
    invalidate();
  });
  observer.observe($("viewer"));
  $("fit").onclick = fit;
  $("surface").onchange = surface;
  $("light").oninput = () => {
    key.intensity = Number($("light").value) * 2.15;
    invalidate();
  };
  $("spin").onchange = () => {
    orbit.autoRotate = $("spin").checked;
    invalidate();
  };
  document.addEventListener("visibilitychange", invalidate);
  window.addEventListener("pagehide", (event) => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    if (!event.persisted) {
      alive = false;
      clear();
      observer.disconnect();
      orbit.dispose();
      Object.values(materials).forEach((value) => value.dispose());
      renderer.dispose();
    }
  });
  window.addEventListener("pageshow", invalidate);
  fit();
  return {
    clear,
    load,
    error(value) {
      $("viewer-error").hidden = false;
      $("viewer-error").textContent = value;
    },
  };
}

const viewer = createViewer();
window.addEventListener("pagehide", () => {
  suspended = true;
  clearTimeout(timer);
  requestController?.abort();
});
window.addEventListener("pageshow", (event) => {
  suspended = false;
  if (event.persisted) connect();
});
connect();
