const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let currentAudio = null;
let audioContext = null;
let analyser = null;
let animationFrame = 0;
let lyricRows = [];
let activeLyricIndex = -1;
let guideModels = [];

function setStatus(kind, text) {
  $("statusDot").className = `status-dot ${kind || ""}`;
  $("statusText").textContent = text;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = {};
  try { body = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

async function refreshHealth() {
  try {
    const state = await api("/api/health");
    const complete = Object.values(state.models).every(Boolean);
    if (!state.engine) setStatus("error", "Inference engine unavailable");
    else if (!complete) setStatus("error", "Model files incomplete");
    else if (state.loaded) setStatus("loaded", "Models resident · local GPU");
    else setStatus("ready", "Engine ready · weights unloaded");
    $("loadModel").disabled = state.loaded || !complete;
    $("unloadModel").disabled = !state.loaded;
  } catch (error) {
    setStatus("error", error.message);
  }
}

function setBusy(button, busy, label) {
  if (!button.dataset.originalHtml) button.dataset.originalHtml = button.innerHTML;
  button.disabled = busy;
  button.innerHTML = busy ? label : button.dataset.originalHtml;
}

async function loadModels() {
  const button = $("loadModel");
  setBusy(button, true, "Loading…");
  setStatus("", "Loading FP16 DiT, INT8 encoder, and DAV…");
  try {
    await api("/api/load", { method: "POST" });
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(button, false, "");
    await refreshHealth();
  }
}

async function unloadModels() {
  const button = $("unloadModel");
  setBusy(button, true, "Unloading…");
  try {
    await api("/api/unload", { method: "POST" });
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(button, false, "");
    await refreshHealth();
  }
}

function requestBody() {
  return {
    global_metadata: $("globalMetadata").value,
    vocal_details: $("vocalDetails").value,
    arrangement: $("arrangement").value,
    lyrics: $("lyrics").value,
    duration: Number($("duration").value),
    seed: Number($("seed").value),
    steps: Number($("steps").value),
    cfg: Number($("cfg").value),
    top_k: Number($("topK").value),
    batch: Number($("batch").value),
    sampler: $("sampler").value,
    scheduler: $("scheduler").value,
    tiled_decode: $("tiledDecode").checked,
    tile_size: Number($("tileSize").value),
    tile_overlap: Number($("tileOverlap").value),
    output_format: $("outputFormat").value,
    quality: $("quality").value,
  };
}

function preparePerformance(request) {
  $("performanceMetadata").textContent = request.global_metadata || "—";
  $("performanceVocals").textContent = request.vocal_details || "Instrumental / unspecified";
  $("performanceArrangement").textContent = request.arrangement || "—";
  const container = $("performanceLyrics");
  container.replaceChildren();
  activeLyricIndex = -1;
  lyricRows = request.lyrics.split(/\r?\n/).filter((line) => line.trim()).map((line) => {
    const p = document.createElement("p");
    p.textContent = line;
    if (/^\s*\[.*\]\s*$/.test(line)) p.className = "tag";
    container.appendChild(p);
    return p;
  });
  if (lyricRows.length) {
    lyricRows[0].classList.add("active");
    activeLyricIndex = 0;
  }
}

async function generate(event) {
  event.preventDefault();
  const button = $("generate");
  const request = requestBody();
  preparePerformance(request);
  setBusy(button, true, "Queued…");
  $("renderStatus").textContent = "Submitting the official MiniMax graph…";
  try {
    let job = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    while (!["complete", "error", "cancelled"].includes(job.status)) {
      $("renderStatus").textContent = job.status === "waiting" ? "Waiting for the local generation lane…" : "Rendering locally. Long songs can take a while.";
      await sleep(1200);
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error || "Generation failed");
    if (job.status === "cancelled") throw new Error("Generation was cancelled");
    renderAudioOutputs(job.audio);
    $("renderStatus").textContent = `${job.audio.length} local audio file${job.audio.length === 1 ? "" : "s"} ready.`;
    await refreshHealth();
  } catch (error) {
    $("renderStatus").textContent = error.message;
  } finally {
    setBusy(button, false, "");
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function updateLyrics(audio) {
  if (!lyricRows.length || !audio.duration) return;
  const index = Math.min(lyricRows.length - 1, Math.floor((audio.currentTime / audio.duration) * lyricRows.length));
  if (index !== activeLyricIndex) {
    lyricRows.forEach((row, i) => row.classList.toggle("active", i === index));
    const container = $("performanceLyrics");
    const targetTop = lyricRows[index].offsetTop - (container.clientHeight / 2) + (lyricRows[index].clientHeight / 2);
    container.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
    activeLyricIndex = index;
  }
  $("playTime").textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
}

function connectVisualizer(audio) {
  if (!audioContext) audioContext = new AudioContext();
  if (audio.dataset.connected !== "true") {
    const source = audioContext.createMediaElementSource(audio);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = .82;
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    audio.dataset.connected = "true";
  }
  currentAudio = audio;
  audioContext.resume();
  cancelAnimationFrame(animationFrame);
  drawSpectrum();
}

function drawSpectrum() {
  const canvas = $("spectrum");
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const values = new Uint8Array(analyser ? analyser.frequencyBinCount : 64);
  if (analyser && currentAudio && !currentAudio.paused) analyser.getByteFrequencyData(values);
  const bars = 72;
  const gap = 3;
  const barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
  for (let i = 0; i < bars; i += 1) {
    const sample = values[Math.floor((i / bars) * values.length)] || (3 + 5 * Math.sin(i * .7));
    const barHeight = Math.max(2, (sample / 255) * (height - 6));
    const x = i * (barWidth + gap);
    context.fillStyle = `rgba(244, 242, 235, ${.38 + (sample / 255) * .62})`;
    context.fillRect(x, height - barHeight, barWidth, barHeight);
  }
  if (currentAudio) updateLyrics(currentAudio);
  animationFrame = requestAnimationFrame(drawSpectrum);
}

function renderAudioOutputs(urls) {
  const container = $("audioOutputs");
  container.replaceChildren();
  urls.forEach((url, index) => {
    const card = document.createElement("div");
    card.className = "audio-card";
    const label = document.createElement("span");
    label.textContent = `Take ${String(index + 1).padStart(2, "0")}`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = url;
    audio.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach((other) => { if (other !== audio) other.pause(); });
      $("nowPlaying").textContent = `TAKE ${String(index + 1).padStart(2, "0")} · MINIMAX MUSIC 3`;
      connectVisualizer(audio);
    });
    const download = document.createElement("a");
    download.href = url;
    download.download = "";
    download.textContent = "Download ↘";
    card.append(label, audio, download);
    container.appendChild(card);
  });
}

function randomSeed() {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  $("seed").value = String((values[0] * 0x100000 + (values[1] & 0xfffff)) % Number.MAX_SAFE_INTEGER);
}

async function interrupt() {
  try {
    await api("/api/interrupt", { method: "POST" });
    $("renderStatus").textContent = "Interrupt requested.";
  } catch (error) {
    $("renderStatus").textContent = error.message;
  }
}

function switchWorkspace(panelId) {
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === panelId);
  });
  document.querySelectorAll(".workspace-panel").forEach((panel) => {
    const active = panel.id === panelId;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "unknown size";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit > 2 ? 2 : 1)} ${units[unit]}`;
}

function renderGuideModels(filter = "") {
  const list = $("guideModelList");
  list.replaceChildren();
  const needle = filter.trim().toLowerCase();
  const matches = guideModels.filter((model) => model.path.toLowerCase().includes(needle));
  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "empty-model-list";
    empty.textContent = "No local .safetensors checkpoints match this filter.";
    list.appendChild(empty);
    return;
  }
  matches.forEach((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    const copy = document.createElement("span");
    const name = document.createElement("b");
    name.textContent = model.name;
    const path = document.createElement("small");
    path.textContent = model.path;
    copy.append(name, path);
    const size = document.createElement("em");
    size.textContent = formatBytes(model.bytes);
    button.append(copy, size);
    button.addEventListener("click", () => {
      $("guideModel").value = model.path;
      $("guideModelDialog").close();
    });
    list.appendChild(button);
  });
}

async function refreshGuideModels() {
  try {
    const state = await api("/api/guide/models");
    guideModels = state.models || [];
    $("guideModelRoot").textContent = state.model_root;
    const current = $("guideModel").value;
    if (!current || (!guideModels.some((model) => model.path === current) && state.default_model)) {
      $("guideModel").value = state.default_model;
    }
    renderGuideModels($("guideModelSearch").value);
  } catch (error) {
    guideModels = [];
    $("guideModelRoot").textContent = error.message;
    renderGuideModels();
  }
}

async function refreshGuideStatus() {
  try {
    const state = await api("/api/guide/status");
    const enabled = Boolean(state.loaded);
    $("guideEnabled").checked = enabled;
    $("guideFields").disabled = !enabled;
    $("browseGuideModels").disabled = enabled;
    const heading = document.querySelector(".guide-runtime-card h2");
    if (enabled) {
      heading.textContent = "Prompt Guide is resident";
      $("guideStatus").textContent = `${state.loaded_model} · isolated local GPU lane`;
    } else if (state.running) {
      heading.textContent = "Prompt Guide is starting";
      $("guideStatus").textContent = "The private process is active; weights are not yet confirmed resident.";
    } else if (!state.default_exists) {
      heading.textContent = "Prompt Guide is off";
      $("guideStatus").textContent = "Default checkpoint is missing. Browse another local Krea2-compatible .safetensors file.";
    } else {
      heading.textContent = "Prompt Guide is off";
      $("guideStatus").textContent = "No secondary process and no Qwen weights are resident.";
    }
  } catch (error) {
    $("guideEnabled").checked = false;
    $("guideFields").disabled = true;
    document.querySelector(".guide-runtime-card h2").textContent = "Prompt Guide unavailable";
    $("guideStatus").textContent = error.message;
  }
}

async function toggleGuide() {
  const toggle = $("guideEnabled");
  const requested = toggle.checked;
  toggle.disabled = true;
  $("guideStatus").textContent = requested
    ? "Starting the isolated Comfy lane and loading Qwen as Krea2…"
    : "Interrupting the guide, releasing weights, and stopping its private process…";
  try {
    if (requested) {
      await api("/api/guide/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: $("guideModel").value }),
      });
    } else {
      await api("/api/guide/unload", { method: "POST" });
    }
  } catch (error) {
    $("guideStatus").textContent = error.message;
  } finally {
    toggle.disabled = false;
    await refreshGuideStatus();
  }
}

function promptGuideBody() {
  return {
    model: $("guideModel").value,
    direction: $("guideDirection").value,
    lyrics: $("guideLyrics").value,
    constraints: $("guideConstraints").value,
    duration: Number($("guideDuration").value),
    steps: Number($("guideSteps").value),
    cfg: Number($("guideCfg").value),
    acoustic_top_k: Number($("guideAcousticTopK").value),
    sampler: $("guideSampler").value,
    scheduler: $("guideScheduler").value,
    tiled_decode: $("guideTiled").checked,
    max_length: Number($("guideMaxLength").value),
    temperature: Number($("guideTemperature").value),
    top_k: Number($("guideTopK").value),
    top_p: Number($("guideTopP").value),
    min_p: Number($("guideMinP").value),
    repetition_penalty: Number($("guideRepeatPenalty").value),
    seed: Number($("guideSeed").value),
    presence_penalty: Number($("guidePresencePenalty").value),
    sampling: $("guideSampling").checked,
    thinking: $("guideThinking").checked,
    use_default_template: $("guideDefaultTemplate").checked,
  };
}

function renderGuideResult(result) {
  const sections = result.sections || {};
  $("guideGlobalOutput").textContent = sections["Global Metadata"] || "Section was not isolated; inspect the raw response below.";
  $("guideVocalOutput").textContent = sections["Vocal Details"] || "Section was not isolated; inspect the raw response below.";
  $("guideArrangementOutput").textContent = sections.Arrangement || "Section was not isolated; inspect the raw response below.";
  $("guideTuningOutput").textContent = sections["Tuning Notes"] || "Section was not isolated; inspect the raw response below.";
  $("guideRawOutput").textContent = result.text || "";
  $("guideResults").hidden = false;
  $("guideResults").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPromptGuide(event) {
  event.preventDefault();
  const button = $("runPromptGuide");
  setBusy(button, true, "Rewriting…");
  $("guideRunStatus").textContent = "Qwen is building a MiniMax-specific structured caption locally…";
  try {
    const result = await api("/api/guide/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(promptGuideBody()),
    });
    renderGuideResult(result);
    $("guideRunStatus").textContent = "Finished locally. Copy only the sections you want.";
  } catch (error) {
    $("guideRunStatus").textContent = error.message;
  } finally {
    setBusy(button, false, "");
    await refreshGuideStatus();
  }
}

async function copyGuideText(elementId, button) {
  const text = $(elementId).textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const helper = document.createElement("textarea");
    helper.value = text;
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }
  const original = button.textContent;
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = original; }, 1200);
}

document.querySelectorAll("[data-port]").forEach((link) => {
  link.href = `${location.protocol}//${location.hostname}:${link.dataset.port}/`;
});
document.querySelectorAll(".workspace-tab").forEach((button) => {
  button.addEventListener("click", () => switchWorkspace(button.dataset.tab));
});
document.querySelectorAll(".copy-guide").forEach((button) => {
  button.addEventListener("click", () => copyGuideText(button.dataset.copy, button));
});
$("duration").addEventListener("input", () => { $("durationReadout").textContent = `${$("duration").value} s`; });
$("randomSeed").addEventListener("click", randomSeed);
$("refreshStatus").addEventListener("click", refreshHealth);
$("loadModel").addEventListener("click", loadModels);
$("unloadModel").addEventListener("click", unloadModels);
$("interrupt").addEventListener("click", interrupt);
$("composer").addEventListener("submit", generate);
$("guideEnabled").addEventListener("change", toggleGuide);
$("browseGuideModels").addEventListener("click", async () => {
  await refreshGuideModels();
  $("guideModelDialog").showModal();
  $("guideModelSearch").focus();
});
$("guideModelSearch").addEventListener("input", () => renderGuideModels($("guideModelSearch").value));
$("promptGuideForm").addEventListener("submit", runPromptGuide);

drawSpectrum();
refreshHealth();
refreshGuideModels();
refreshGuideStatus();
setInterval(() => { refreshHealth(); refreshGuideStatus(); }, 15000);
