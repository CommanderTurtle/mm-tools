import { SplatPanel } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const panel = new SplatPanel($("scene"), $("viewer-status"));
let source = null,
  previewURL = null,
  presets = [],
  running = null,
  currentRun = null;
let timer = null,
  selection = 0,
  pollBusy = false,
  renderedRun = null,
  locked = false;
const defaults = { steps: 50, cfg: 3, rescale_t: 3 };

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

async function request(url, options) {
  const response = await fetch(url, {
    signal: AbortSignal.timeout(30000),
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function safely(fn) {
  return (...args) =>
    Promise.resolve()
      .then(() => fn(...args))
      .catch((error) => status(error.message, true));
}

function setTheme(light) {
  document.documentElement.classList.toggle("light", light);
  $("theme").textContent = light ? "☾" : "☀";
  try {
    localStorage.setItem("idol-theme", light ? "light" : "dark");
  } catch {}
}
try {
  setTheme(localStorage.getItem("idol-theme") === "light");
} catch {}
$("theme").onclick = () =>
  setTheme(!document.documentElement.classList.contains("light"));

for (const key of Object.keys(defaults)) {
  $(key).oninput = () => {
    $(`${key}-number`).value = $(key).value;
  };
  $(`${key}-number`).oninput = () => {
    if ($(`${key}-number`).validity.valid)
      $(key).value = $(`${key}-number`).value;
  };
}
for (const button of document.querySelectorAll("[data-reset]")) {
  button.onclick = () => {
    const key = button.dataset.reset;
    $(key).value = $(`${key}-number`).value = defaults[key];
  };
}

function controls(config) {
  document.querySelector(`input[name="mode"][value="${config.mode}"]`).checked =
    true;
  for (const key of Object.keys(defaults))
    $(key).value = $(`${key}-number`).value = config[key];
  $("seed").value = config.seed;
  $("randomize").checked = config.randomize || false;
  $("num-gaussians").value = config.num_gaussians;
}

function config() {
  return {
    mode: document.querySelector('input[name="mode"]:checked').value,
    steps: Number($("steps-number").value),
    cfg: Number($("cfg-number").value),
    rescale_t: Number($("rescale_t-number").value),
    seed: Number($("seed").value),
    randomize: $("randomize").checked,
    num_gaussians: Number($("num-gaussians").value),
  };
}

function image(value) {
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  source = value;
  const url = value?.file
    ? (previewURL = URL.createObjectURL(value.file))
    : value?.url;
  $("input-preview").hidden = !url;
  $("drop-label").hidden = !!url;
  if (url) $("input-preview").src = url;
  else $("input-preview").removeAttribute("src");
  for (const button of $("examples").children)
    button.classList.toggle("selected", button.dataset.image === value?.preset);
}

function chooseFile(file) {
  if (locked) return;
  if (!file) return;
  if (file.size > 64 * 1024 * 1024)
    throw new Error("Image must be smaller than 64 MiB");
  ++selection;
  image({ file });
  $("history").value = "";
}

$("upload").onchange = safely(() => {
  chooseFile($("upload").files[0]);
  $("upload").value = "";
});
$("upload-button").onclick = () => $("upload").click();
$("clear-input").onclick = () => {
  ++selection;
  image(null);
};
$("paste-button").onclick = safely(async () => {
  const version = ++selection;
  if (!navigator.clipboard?.read)
    throw new Error("Paste an image with Ctrl+V, or use Upload");
  for (const item of await navigator.clipboard.read()) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const file = new File([await item.getType(type)], "clipboard.png", {
        type,
      });
      if (version === selection) chooseFile(file);
      return;
    }
  }
  throw new Error("Clipboard contains no image");
});
document.addEventListener("paste", (event) => {
  if (locked) return;
  const file = [...event.clipboardData.items]
    .find((item) => item.type.startsWith("image/"))
    ?.getAsFile();
  if (file) {
    event.preventDefault();
    safely(() => chooseFile(file))();
  }
});
for (const name of ["dragenter", "dragover"])
  $("input-panel").addEventListener(name, (event) => {
    event.preventDefault();
    if (!running) $("input-panel").classList.add("dragging");
  });
$("input-panel").ondragleave = () =>
  $("input-panel").classList.remove("dragging");
$("input-panel").ondrop = (event) => {
  event.preventDefault();
  $("input-panel").classList.remove("dragging");
  safely(() => chooseFile(event.dataTransfer.files[0]))();
};

async function fullscreen(element) {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await element.requestFullscreen();
}
$("expand-input").onclick = safely(() => fullscreen($("input-panel")));
$("expand-result").onclick = safely(() => fullscreen($("result")));
$("reset-camera").onclick = () => panel.reset();
document.addEventListener("visibilitychange", () => panel.visibility());
window.addEventListener("pagehide", (event) => {
  clearTimeout(timer);
  if (event.persisted) panel.visibility(true);
  else panel.clear();
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  panel.visibility();
  clearTimeout(timer);
  poll();
});

function busy(value) {
  locked = value;
  $("inputs").disabled = value;
  $("history").disabled = value;
  $("cancel").hidden = !value;
  for (const button of $("examples").children) button.disabled = value;
}

function selectPreset(preset) {
  if (locked) return;
  ++selection;
  image({ preset: preset.image, url: `/${preset.image}` });
  controls(preset);
  $("history").value = "";
  status("Ready");
}

function gallery() {
  const shuffled = [...presets];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  for (const preset of shuffled) {
    const button = document.createElement("button");
    button.type = "button";
    button.disabled = locked;
    button.title = preset.title;
    button.setAttribute("aria-label", `Load ${preset.title}`);
    button.dataset.image = preset.image;
    const thumbnail = document.createElement("img");
    thumbnail.src = `/${preset.image}`;
    thumbnail.alt = preset.title;
    thumbnail.loading = "lazy";
    button.append(thumbnail);
    button.onclick = () => selectPreset(preset);
    $("examples").append(button);
  }
}

async function history() {
  const rows = await request("/api/runs");
  $("history").replaceChildren(new Option("Saved runs", ""));
  for (const row of rows)
    $("history").append(new Option(`${row.title} · ${row.status}`, row.id));
  $("history").value = currentRun || "";
}

function metadata(config, result) {
  const label = {
    65536: "64K",
    131072: "128K",
    262144: "256K",
    524288: "512K",
    1048576: "1M",
  }[config.num_gaussians];
  $("metadata").textContent =
    `${label} Gaussians · ${config.steps} steps${result ? ` · ${result.seconds.toFixed(1)}s · seed ${config.seed}` : ""}`;
}

async function showRun(row) {
  const output = `${row.id}:${row.status}`;
  if (row.status === "complete") {
    const base = `/files/${row.id}/`;
    $("download-ply").href = `${base}object.ply?download=1`;
    $("download-ply").hidden = false;
    $("prepared").href = `${base}prepared.png`;
    $("camera").href = `${base}camera.json?download=1`;
    $("run-settings").href = `${base}run.json?download=1`;
    $("artifacts").hidden = false;
    metadata(row.settings, row.result);
    if (renderedRun !== output) {
      renderedRun = output;
      panel
        .load(`${base}object.ply`)
        .catch((error) => status(error.message, true));
    }
  } else if (renderedRun !== output) {
    renderedRun = output;
    panel.clear();
    $("download-ply").hidden = true;
    $("download-ply").removeAttribute("href");
    $("artifacts").hidden = true;
    $("metadata").textContent = "";
    $("viewer-status").textContent = "No completed output for this run";
    $("viewer-status").hidden = false;
  }
  status(
    row.error ||
      (row.status === "running"
        ? row.progress.phase || "Starting"
        : row.status),
    !!row.error,
  );
  $("log-panel").hidden = false;
  if ($("log-panel").open) await log(row.id);
}

async function log(key) {
  const response = await fetch(`/api/runs/${key}/log`);
  if (response.ok && currentRun === key)
    $("log").textContent = await response.text();
}
$("log-panel").ontoggle = safely(() =>
  currentRun && $("log-panel").open ? log(currentRun) : undefined,
);

async function poll() {
  if (!running || pollBusy) return;
  pollBusy = true;
  const key = running;
  try {
    const row = await request(`/api/runs/${key}`);
    if (currentRun === key && !source) {
      const version = selection;
      controls(row.settings);
      try {
        const response = await fetch(`/files/${key}/input.png`, {
          signal: AbortSignal.timeout(30000),
        });
        if (!response.ok) throw new Error("Saved input is unavailable");
        const blob = await response.blob();
        if (version === selection && currentRun === key)
          image({
            file: new File([blob], `${row.title}.png`, { type: "image/png" }),
          });
      } catch (error) {
        $("drop-label").textContent =
          `${error.message}. Select this saved run to retry.`;
      }
    }
    if (currentRun === key) await showRun(row);
    if (row.status !== "running") {
      running = null;
      busy(false);
      await history();
    }
  } catch (error) {
    status(`${error.message} · Reconnecting`, true);
  } finally {
    pollBusy = false;
    if (running) timer = setTimeout(poll, document.hidden ? 3000 : 800);
  }
}

$("generate-form").onsubmit = (event) => {
  event.preventDefault();
  safely(async () => {
    if (locked) return;
    if (!source) throw new Error("Upload an image or choose an example");
    ++selection;
    const form = new FormData();
    form.set("settings", JSON.stringify(config()));
    if (source.file) form.set("image", source.file);
    else if (source.preset) form.set("preset", source.preset);
    else throw new Error("Choose an input image");
    busy(true);
    status("Saving input");
    try {
      const row = await request("/api/runs", { method: "POST", body: form });
      running = currentRun = row.id;
      $("seed").value = row.settings.seed;
      $("artifacts").hidden = true;
      $("log").textContent = "";
      await history();
      await poll();
    } catch (error) {
      if (!running) busy(false);
      else timer = setTimeout(poll, 800);
      throw error;
    }
  })();
};

$("cancel").onclick = safely(async () => {
  if (!running) return;
  await request(`/api/runs/${running}/cancel`, { method: "POST" });
  status("Stopping generation");
});

$("history").onchange = safely(async () => {
  const key = $("history").value;
  if (!key || locked) return;
  const version = ++selection;
  const row = await request(`/api/runs/${key}`);
  const response = await fetch(`/files/${key}/input.png`);
  if (!response.ok) throw new Error("Saved input is unavailable");
  const blob = await response.blob();
  if (version !== selection) return;
  currentRun = key;
  image({ file: new File([blob], `${row.title}.png`, { type: "image/png" }) });
  controls(row.settings);
  $("artifacts").hidden = row.status !== "complete";
  await showRun(row);
});

busy(true);
await safely(async () => {
  try {
    const data = await request("/api/config");
    running = currentRun = data.active || null;
    presets = data.presets;
    gallery();
    const initial = presets.find(
      (preset) => preset.image === "examples/obj-backpack.webp",
    );
    if (initial && !data.active) {
      image({ preset: initial.image, url: `/${initial.image}` });
      controls(initial);
      metadata(initial);
      $("download-ply").href = "/examples/default-obj-backpack.ply";
      $("download-ply").hidden = false;
      panel
        .load("/examples/default-obj-backpack.ply")
        .catch((error) => status(error.message, true));
    } else if (!data.active) {
      $("viewer-status").textContent = "Choose idol in the model downloader";
    }
    await history();
  } finally {
    if (running) timer = setTimeout(poll, 0);
    else busy(false);
  }
})();
