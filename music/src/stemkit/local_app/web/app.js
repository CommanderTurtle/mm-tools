// StemKit studio player — vanilla ES module, no build step.
// Mirrors the vendored app's intent: pick instruments, split locally,
// play every stem on its own fader against one master clock.

const $ = (id) => document.getElementById(id);

const STEM_INFO = {
  vocals: { label: "vocals", color: "#A78BFA" },
  drums: { label: "drums", color: "#F87171" },
  bass: { label: "bass", color: "#60A5FA" },
  guitar: { label: "guitar", color: "#F472B6" },
  piano: { label: "piano", color: "#FBBF24" },
  other: { label: "other", color: "#34D399" },
};
const PRESETS = {
  all: ["vocals", "drums", "bass", "other"],
  karaoke: ["drums", "bass", "other"],
  acapella: ["vocals"],
  drums_bass: ["drums", "bass"],
};
const ALL_STEMS = ["vocals", "drums", "bass", "other"];

const state = {
  track: null, // { name, buffer, seconds }
  selected: new Set(PRESETS.all),
  preset: "all",
  job: null,
  lanes: [], // { id, name, color, buffer, gain, muted, solo, vol, canvas, peaks }
  ctx: null,
  master: null,
  playing: false,
  position: 0, // seconds into the mix
  startedAt: 0, // AudioContext time when the current play segment started
  sources: [],
};

function audioContext() {
  if (!state.ctx) state.ctx = new AudioContext();
  return state.ctx;
}

function ensureMaster() {
  const ctx = audioContext();
  if (!state.master) {
    state.master = ctx.createGain();
    state.master.gain.value = Number($("masterVol").value) / 100;
    state.master.connect(ctx.destination);
  }
  return state.master;
}

const fmt = (seconds) => {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

/* ---------------- engines banner ---------------- */

async function loadEngines() {
  try {
    const response = await fetch("/api/engines");
    const info = await response.json();
    const bits = [];
    bits.push(info.gpu ? "GPU ready" : "CPU only");
    bits.push(info.demucs ? "demucs ready" : "demucs missing");
    bits.push(info.roformer_ckpt ? "roformer checkpoint ready" : "roformer checkpoint not downloaded");
    $("engines").textContent = bits.join(" · ");
  } catch {
    $("engines").textContent = "Engine check failed — is the server running?";
  }
}

/* ---------------- track source ---------------- */

function canvasSize(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, canvas.clientWidth || 320);
  const height = Math.max(1, canvas.clientHeight || 64);
  if (canvas.width !== Math.floor(width * ratio)) canvas.width = Math.floor(width * ratio);
  if (canvas.height !== Math.floor(height * ratio)) canvas.height = Math.floor(height * ratio);
  return { ratio, width, height };
}

function computePeaks(buffer, bars) {
  const data = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / bars));
  const peaks = new Float32Array(bars);
  for (let i = 0; i < bars; i += 1) {
    let peak = 0;
    const start = i * step;
    for (let j = 0; j < step; j += 32) {
      const value = Math.abs(data[start + j] || 0);
      if (value > peak) peak = value;
    }
    peaks[i] = peak;
  }
  return peaks;
}

function renderWave(canvas, peaks, color, progressRatio) {
  const { ratio, width, height } = canvasSize(canvas);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const bars = Math.min(peaks.length, Math.floor(width / 3));
  const mid = height / 2;
  const filled = Math.floor(progressRatio * bars);
  for (let i = 0; i < bars; i += 1) {
    const barHeight = Math.max(1.5, peaks[i] * (height - 6));
    context.globalAlpha = i < filled ? 0.95 : 0.4;
    context.fillStyle = color;
    context.fillRect(i * 3, mid - barHeight / 2, 2, barHeight);
  }
  context.globalAlpha = 1;
  if (progressRatio > 0) {
    context.fillStyle = "rgba(255,255,255,.9)";
    context.fillRect(progressRatio * width - 1, 0, 2, height);
  }
}

function drawPeaks(canvas, buffer, color) {
  renderWave(canvas, computePeaks(buffer, Math.max(64, Math.floor((canvas.clientWidth || 320) / 3))), color, 0);
}

function encodeWav(buffer) {
  const channels = 2;
  const rate = 44100;
  const frames = Math.ceil(buffer.duration * rate);
  const blockAlign = channels * 4;
  const payload = new ArrayBuffer(44 + frames * blockAlign);
  const view = new DataView(payload);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + payload.byteLength - 40, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true); // IEEE float
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 32, true);
  writeString(36, "data");
  view.setUint32(40, frames * blockAlign, true);
  const left = buffer.getChannelData(0);
  const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
  const scale = rate / buffer.sampleRate;
  for (let i = 0; i < frames; i += 1) {
    const li = Math.min(left.length - 1, Math.floor(i * scale));
    const ri = Math.min(right.length - 1, Math.floor(i * scale));
    view.setFloat32(44 + i * blockAlign, left[li], true);
    view.setFloat32(44 + i * blockAlign + 4, right[ri], true);
  }
  return new Blob([payload], { type: "audio/wav" });
}

async function handleFile(file) {
  if (!file) return;
  const ctx = audioContext();
  const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  state.track = { name: file.name.replace(/\.[^.]+$/, ""), buffer: decoded };
  $("trackName").textContent = file.name;
  $("trackFacts").textContent =
    `${fmt(decoded.duration)} · ${decoded.sampleRate} Hz · ${decoded.numberOfChannels} ch`;
  $("trackRow").hidden = false;
  $("overview").style.display = "block";
  drawPeaks($("overview"), decoded, "var(--accent)");
  $("splitButton").disabled = false;
  $("splitStatus").textContent = "Ready to split.";
}

$("fileInput").addEventListener("change", (event) => handleFile(event.target.files?.[0]));
const dz = $("dropzone");
["dragover", "dragenter"].forEach((type) =>
  dz.addEventListener(type, (event) => { event.preventDefault(); dz.classList.add("dragging"); }));
["dragleave", "drop"].forEach((type) =>
  dz.addEventListener(type, (event) => { event.preventDefault(); dz.classList.remove("dragging"); }));
dz.addEventListener("drop", (event) => handleFile(event.dataTransfer?.files?.[0]));
$("clearTrack").onclick = () => {
  state.track = null;
  $("trackRow").hidden = true;
  $("splitButton").disabled = true;
  $("fileInput").value = "";
  $("splitStatus").textContent = "Load a track, then split.";
};

/* ---------------- presets + stem toggles ---------------- */

const togglesRoot = $("stemToggles");
for (const stem of ALL_STEMS) {
  const chip = document.createElement("label");
  chip.className = "stem-toggle";
  chip.innerHTML =
    `<input type="checkbox" checked><i style="background:${STEM_INFO[stem].color}"></i><span>${stem}</span>`;
  chip.querySelector("input").addEventListener("change", (event) => {
    if (event.target.checked) state.selected.add(stem);
    else state.selected.delete(stem);
    syncPresetChips();
  });
  togglesRoot.append(chip);
}

function syncPresetChips() {
  const chosen = [...state.selected].sort();
  const match = Object.entries(PRESETS).find(([, stems]) =>
    stems.length === chosen.length && stems.every((s) => chosen.includes(s)));
  state.preset = match ? match[0] : "custom";
  document.querySelectorAll("#presets .chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.preset === state.preset);
  });
}

document.querySelectorAll("#presets .chip").forEach((chip) => {
  chip.onclick = () => {
    state.preset = chip.dataset.preset;
    state.selected = new Set(PRESETS[state.preset]);
    document.querySelectorAll("#stemToggles input").forEach((box, index) => {
      box.checked = state.selected.has(ALL_STEMS[index]);
    });
    syncPresetChips();
  };
});

/* ---------------- split flow ---------------- */

$("splitButton").onclick = async () => {
  if (!state.track) return;
  const button = $("splitButton");
  button.disabled = true;
  $("progressWrap").hidden = false;
  $("progressFill").style.width = "0%";
  $("progressText").textContent = "Uploading…";
  $("daudio").hidden = true;
  try {
    const blob = encodeWav(state.track.buffer);
    const form = new FormData();
    form.append("file", blob, `${state.track.name}.wav`);
    const upload = await (await fetch("/api/upload", { method: "POST", body: form })).json();
    const created = await (await fetch("/api/split", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: upload.asset_id, stems: [...state.selected] }),
    })).json();
    state.job = { id: created.job_id };
    await pollJob(created.job_id);
  } catch (error) {
    $("progressWrap").hidden = true;
    $("splitStatus").textContent = `Split failed: ${error.message}`;
    button.disabled = false;
  }
};

async function pollJob(jobId) {
  for (;;) {
    const record = await (await fetch(`/api/jobs/${jobId}`)).json();
    $("progressFill").style.width = `${record.pct}%`;
    $("progressText").textContent =
      record.stage === "log" ? record.message : `${record.stage}: ${Math.max(0, record.pct)}%${record.message ? ` — ${record.message}` : ""}`;
    if (record.status === "failed") {
      $("progressWrap").hidden = true;
      $("splitStatus").textContent = record.message || "Split failed.";
      $("splitButton").disabled = false;
      return;
    }
    if (record.status === "done") {
      $("progressWrap").hidden = true;
      $("splitStatus").textContent =
        `Done via ${record.engine} (${record.model}) — ${record.stems.length} stem(s).`;
      await buildLanes(jobId, record.stems);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

/* ---------------- lanes + transport ---------------- */

async function buildLanes(jobId, stems) {
  const ctx = audioContext();
  ensureMaster();
  state.lanes = [];
  state.position = 0;
  stopSources();
  const root = $("lanes");
  root.innerHTML = "";
  for (const name of stems) {
    const info = STEM_INFO[name] || { label: name, color: "#9ca3af" };
    const row = document.createElement("div");
    row.className = "lane";
    row.innerHTML = `
      <div class="lane-head">
        <span class="lane-dot" style="background:${info.color}"></span>
        <b>${name}</b>
        <div class="lane-controls">
          <button class="mini-button" data-action="mute" type="button" title="Mute">M</button>
          <button class="mini-button" data-action="solo" type="button" title="Solo">S</button>
          <input type="range" min="0" max="100" value="100" aria-label="${name} volume">
          <button class="mini-button" data-action="export" type="button" title="Export WAV">⤓</button>
        </div>
      </div>
      <canvas class="waveform lane-wave" aria-label="${name} waveform"></canvas>`;
    root.append(row);
    const buffer = await ctx.decodeAudioData(
      await (await fetch(`/api/stems/${jobId}/${name}`)).arrayBuffer());
    const gain = ctx.createGain();
    gain.gain.value = 1;
    gain.connect(ensureMaster());
    const lane = {
      id: jobId, name, color: info.color, buffer, gain,
      muted: false, solo: false, vol: 1,
      canvas: row.querySelector(".lane-wave"),
      peaks: null,
    };
    lane.peaks = computePeaks(buffer, Math.max(64, Math.floor((lane.canvas.clientWidth || 320) / 3)));
    renderWave(lane.canvas, lane.peaks, info.color, 0);
    state.lanes.push(lane);
    const volume = row.querySelector('input[type="range"]');
    volume.oninput = () => {
      lane.vol = Number(volume.value) / 100;
      applyMix();
    };
    row.querySelector('[data-action="mute"]').onclick = () => {
      lane.muted = !lane.muted;
      applyMix();
      paintLaneButtons(row, lane);
    };
    row.querySelector('[data-action="solo"]').onclick = () => {
      lane.solo = !lane.solo;
      applyMix();
      paintLaneButtons(row, lane);
    };
    row.querySelector('[data-action="export"]').onclick = () => exportStem(jobId, name);
    paintLaneButtons(row, lane);
  }
  $("daudio").hidden = false;
  $("seek").max = String(Math.floor(duration() * 100));
  $("clock").textContent = `00:00 / ${fmt(duration())}`;
  $("playButton").textContent = "▶ Play";
}
$("seek").addEventListener("input", () => {
  const position = Number($("seek").value) / 100;
  state.position = position;
  if (state.playing) startSources(position);
  $("clock").textContent = `${fmt(position)} / ${fmt(duration())}`;
});

function paintLaneButtons(row, lane) {
  row.querySelector('[data-action="mute"]').classList.toggle("on", lane.muted);
  row.querySelector('[data-action="solo"]').classList.toggle("on", lane.solo);
}

function applyMix() {
  const anySolo = state.lanes.some((lane) => lane.solo);
  for (const lane of state.lanes) {
    const audible = lane.muted ? 0 : (anySolo && !lane.solo ? 0 : 1);
    lane.gain.gain.value = audible * lane.vol;
  }
}

function duration() {
  return state.lanes.length ? Math.max(...state.lanes.map((l) => l.buffer.duration)) : 0;
}

function stopSources() {
  for (const source of state.sources) {
    try { source.stop(); } catch { /* already stopped */ }
    source.disconnect();
  }
  state.sources = [];
}

function startSources(atPosition) {
  const ctx = audioContext();
  const offset = Math.min(atPosition, Math.max(0, duration() - 0.05));
  const when = ctx.currentTime + 0.03;
  state.startedAt = when;
  state.position = offset;
  for (const lane of state.lanes) {
    const source = ctx.createBufferSource();
    source.buffer = lane.buffer;
    source.connect(lane.gain);
    source.start(when, Math.min(offset, lane.buffer.duration - 0.01));
    state.sources.push(source);
  }
}

$("playButton").onclick = () => {
  if (!state.lanes.length) return;
  const ctx = audioContext();
  void ctx.resume();
  if (state.playing) {
    state.position = currentPosition();
    stopSources();
    state.playing = false;
    $("playButton").textContent = "▶ Play";
  } else {
    if (state.position >= duration() - 0.05) state.position = 0;
    startSources(state.position);
    state.playing = true;
    $("playButton").textContent = "❚❚ Pause";
  }
};

function currentPosition() {
  if (!state.playing) return state.position;
  return state.position + (audioContext().currentTime - state.startedAt);
}


$("masterVol").addEventListener("input", () => {
  if (state.master) state.master.gain.value = Number($("masterVol").value) / 100;
});

$("exportAll").onclick = async () => {
  if (!state.job) return;
  for (const lane of state.lanes) {
    await exportStem(state.job.id, lane.name);
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
};

async function exportStem(jobId, name) {
  const blob = await (await fetch(`/api/stems/${jobId}/${name}`)).blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${(state.track?.name || "track").replace(/[^\w\- ]+/g, "")}-${name}.wav`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/* ---------------- animation loop ---------------- */

function paintPlayheads() {
  requestAnimationFrame(paintPlayheads);
  if (!state.lanes.length) return;
  const position = currentPosition();
  if (state.playing && position >= duration()) {
    stopSources();
    state.playing = false;
    state.position = 0;
    $("playButton").textContent = "▶ Play";
  }
  const shown = state.playing ? position : state.position;
  const ratio = duration() ? shown / duration() : 0;
  if (document.activeElement !== $("seek")) {
    $("seek").value = String(Math.round(ratio * 100));
  }
  $("clock").textContent = `${fmt(shown)} / ${fmt(duration())}`;
  for (const lane of state.lanes) {
    renderWave(lane.canvas, lane.peaks, lane.color, ratio);
  }
}
paintPlayheads();

loadEngines();
