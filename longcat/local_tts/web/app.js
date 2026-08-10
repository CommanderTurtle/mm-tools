const $ = (selector) => document.querySelector(selector);
const form = $("#tts-form");
const statusDot = $("#status-dot");
const modelState = $("#model-state");
const modelDetail = $("#model-detail");
const modelToggle = $("#model-toggle");
const synthesizeButton = $("#synthesize");
const message = $("#message");
const player = $("#player");
const result = $("#result");
const download = $("#download");
let loaded = false;
let resultUrl = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return response;
}

async function refreshStatus() {
  try {
    const state = await (await api("/api/status")).json();
    loaded = state.loaded;
    statusDot.className = state.busy ? "busy" : loaded ? "ready" : "";
    modelState.textContent = state.busy ? "Synthesizing" : loaded ? "Model resident" : "Model unloaded";
    modelDetail.textContent = `${state.dtype} · ${state.cuda ?? state.device}${state.vram_allocated_gb ? ` · ${state.vram_allocated_gb} GB` : ""}`;
    modelToggle.textContent = loaded ? "Unload" : "Load";
    modelToggle.disabled = state.busy;
  } catch (error) {
    modelState.textContent = "Service unavailable";
    modelDetail.textContent = error.message;
  }
}

modelToggle.addEventListener("click", async () => {
  modelToggle.disabled = true;
  message.textContent = loaded ? "Releasing model memory…" : "Loading 3.5B checkpoint into VRAM…";
  try {
    await api(loaded ? "/api/unload" : "/api/load", { method: "POST" });
    message.textContent = loaded ? "Model unloaded." : "Model ready.";
  } catch (error) { message.textContent = error.message; }
  await refreshStatus();
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
  message.textContent = loaded ? "Diffusion synthesis is running…" : "Loading the model, then synthesizing…";
  result.hidden = true;
  try {
    const response = await api("/api/synthesize", { method: "POST", body: new FormData(form) });
    const blob = await response.blob();
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

refreshStatus();
