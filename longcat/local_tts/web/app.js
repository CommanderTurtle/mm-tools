const $ = (selector) => document.querySelector(selector);
const form = $("#tts-form");
const statusDot = $("#status-dot");
const modelState = $("#model-state");
const modelDetail = $("#model-detail");
const modelToggle = $("#model-toggle");
const secondaryToggle = $("#secondary-toggle");
const synthesizeButton = $("#synthesize");
const message = $("#message");
const player = $("#player");
const result = $("#result");
const download = $("#download");
const hoist = $("#hoist");
let loaded = false;
let localLoaded = false;
let activeBase = "";
let uiConfig = { secondary_port: 8230, secondary_scheme: "http", router_url: "http://127.0.0.1:8182" };
let resultUrl = null;
let resultBlob = null;

async function apiAt(base, path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return response;
}

const api = (path, options = {}) => apiAt(activeBase, path, options);
const localApi = (path, options = {}) => apiAt("", path, options);

function secondaryBase() {
  const bareHost = location.hostname.replace(/^\[|\]$/g, "");
  const host = bareHost.includes(":") ? `[${bareHost}]` : bareHost;
  return `${uiConfig.secondary_scheme}://${host}:${uiConfig.secondary_port}`;
}

function renderStatus(state, source) {
  loaded = state.loaded;
  statusDot.className = state.busy ? "busy" : loaded ? "ready" : "";
  modelState.textContent = state.busy ? "Synthesizing" : loaded ? `${source} model resident` : `${source} model unloaded`;
  modelDetail.textContent = `${state.dtype} · ${state.cuda ?? state.device}${state.vram_allocated_gb ? ` · ${state.vram_allocated_gb} GB` : ""}`;
}

async function refreshStatus() {
  try {
    const localState = await (await localApi("/api/status")).json();
    localLoaded = localState.loaded;
    modelToggle.textContent = localLoaded ? "Unload local" : "Load";
    modelToggle.disabled = localState.busy;
    secondaryToggle.textContent = activeBase ? "Use local model" : "Check Secondary Load";

    if (activeBase) {
      try {
        const secondaryState = await (await apiAt(activeBase, "/api/status")).json();
        renderStatus(secondaryState, "Secondary HTTP");
        return;
      } catch (error) {
        activeBase = "";
        secondaryToggle.textContent = "Check Secondary Load";
        message.textContent = `Secondary service disconnected; using local UI. ${error.message}`;
      }
    }
    renderStatus(localState, "UI-local");
  } catch (error) {
    modelState.textContent = "UI service unavailable";
    modelDetail.textContent = error.message;
  }
}

modelToggle.addEventListener("click", async () => {
  modelToggle.disabled = true;
  activeBase = "";
  secondaryToggle.textContent = "Check Secondary Load";
  message.textContent = localLoaded ? "Releasing the UI-local model…" : "Loading the UI-local 3.5B checkpoint…";
  try {
    await localApi(localLoaded ? "/api/unload" : "/api/load", { method: "POST" });
    message.textContent = localLoaded ? "UI-local model unloaded." : "UI-local model ready.";
  } catch (error) { message.textContent = error.message; }
  await refreshStatus();
});

secondaryToggle.addEventListener("click", async () => {
  if (activeBase) {
    activeBase = "";
    message.textContent = "Using the UI-local model path.";
    await refreshStatus();
    return;
  }
  secondaryToggle.disabled = true;
  message.textContent = "Checking the secondary HTTP model…";
  try {
    const candidate = secondaryBase();
    const state = await (await apiAt(candidate, "/api/status")).json();
    if (!state.loaded) throw new Error("Secondary service is running, but its model is unloaded.");
    activeBase = candidate;
    message.textContent = "Attached to the already-loaded secondary HTTP model.";
  } catch (error) {
    message.textContent = `Secondary load unavailable: ${error.message}`;
  } finally {
    secondaryToggle.disabled = false;
    await refreshStatus();
  }
});

$("#text").addEventListener("input", (event) => {
  $("#characters").textContent = `${event.target.value.length} characters`;
});
$("#prompt-audio").addEventListener("change", (event) => {
  $("#file-name").textContent = event.target.files[0]?.name ?? "Choose WAV, M4A, or MP3";
});
const duration = form.elements.duration_scale;
duration.addEventListener("input", () => { $("#duration-value").textContent = `${Number(duration.value).toFixed(2)}×`; });

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = $("#prompt-audio").files[0];
  if (prompt && !$("#prompt-text").value.trim()) {
    message.textContent = "Add the exact transcript for the reference audio.";
    return;
  }
  synthesizeButton.disabled = true;
  synthesizeButton.classList.add("working");
  message.textContent = loaded ? "Diffusion synthesis is running…" : "Loading the UI-local model, then synthesizing…";
  result.hidden = true;
  try {
    const response = await api("/api/synthesize", { method: "POST", body: new FormData(form) });
    const blob = await response.blob();
    resultBlob = blob;
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = URL.createObjectURL(blob);
    player.src = resultUrl;
    download.href = resultUrl;
    const metadata = JSON.parse(response.headers.get("X-LongCat-Metadata") ?? "{}");
    $("#result-meta").textContent = `${metadata.audio_seconds ?? "?"}s audio · ${metadata.generation_seconds ?? "?"}s render`;
    result.hidden = false;
    message.textContent = "Generation complete.";
    loaded = true;
  } catch (error) { message.textContent = error.message; }
  synthesizeButton.disabled = false;
  synthesizeButton.classList.remove("working");
  await refreshStatus();
});

hoist.addEventListener("click", async () => {
  if (!resultBlob) {
    message.textContent = "Generate speech before hoisting it.";
    return;
  }
  hoist.disabled = true;
  message.textContent = "Hoisting generated audio to the Vox microphone router…";
  try {
    await apiAt(uiConfig.router_url, "/v1/forward", {
      method: "POST",
      headers: { "Content-Type": resultBlob.type || "audio/wav" },
      body: resultBlob,
    });
    message.textContent = "Audio queued on the virtual microphone.";
  } catch (error) {
    message.textContent = `Vox router unavailable: ${error.message}`;
  } finally {
    hoist.disabled = false;
  }
});

async function initialize() {
  try { uiConfig = await (await localApi("/api/ui-config")).json(); } catch {}
  await refreshStatus();
}

initialize();
