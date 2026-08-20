const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let currentAudio = null;
let audioContext = null;
let analyser = null;
let animationFrame = 0;
let lyricRows = [];
let activeLyricIndex = -1;

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

document.querySelectorAll("[data-port]").forEach((link) => {
  link.href = `${location.protocol}//${location.hostname}:${link.dataset.port}/`;
});
$("duration").addEventListener("input", () => { $("durationReadout").textContent = `${$("duration").value} s`; });
$("randomSeed").addEventListener("click", randomSeed);
$("refreshStatus").addEventListener("click", refreshHealth);
$("loadModel").addEventListener("click", loadModels);
$("unloadModel").addEventListener("click", unloadModels);
$("interrupt").addEventListener("click", interrupt);
$("composer").addEventListener("submit", generate);

drawSpectrum();
refreshHealth();
setInterval(refreshHealth, 15000);
