import { STATE_FORMAT, MAX_STATE_BYTES, readControls, validateControls, validateGuideResult, parseStudioState, safeSourceUrl } from "./studio-state.js";

const $ = (id) => document.getElementById(id);

let currentAudio = null;
let audioContext = null;
let analyser = null;
let animationFrame = 0;
let lyricRows = [];
let activeLyricIndex = -1;
let guideModels = [];
let latestGuideResult = null;
let liveSessionId = "";
let selectedJobId = "";
let performanceJobId = "";
let jobPollTimer = 0;
let jobsRefreshPromise = null;
let restoringSessionState = false;
let guideBusy = false;
let guideLoaded = false;
let stateBusy = false;
const jobsById = new Map();

const LIVE_STATE_KEY = "minimax-music-studio:live-session";
const ACTIVE_JOB_STATUSES = new Set(["queued", "waiting", "generating"]);
const TERMINAL_JOB_STATUSES = new Set(["complete", "error", "cancelled"]);
const JOB_STATUS_LABELS = {
  queued: "Queued",
  waiting: "Waiting for lane",
  generating: "Rendering",
  complete: "Ready",
  error: "Failed",
  cancelled: "Cancelled",
};


function rememberedControls() {
  return Array.from(document.querySelectorAll(
    "#composer input[id], #composer textarea[id], #composer select[id], "
    + "#promptGuideForm input[id], #promptGuideForm textarea[id], #promptGuideForm select[id]",
  )).filter((control) => !["button", "submit"].includes(control.type));
}

function saveSessionState() {
  if (!liveSessionId || restoringSessionState) return;
  const controls = readControls(rememberedControls());
  const activeWorkspace = document.querySelector(".workspace-tab.active")?.dataset.tab || "songStudio";
  try {
    sessionStorage.setItem(LIVE_STATE_KEY, JSON.stringify({
      session_id: liveSessionId,
      active_workspace: activeWorkspace,
      selected_job_id: selectedJobId,
      controls,
      guide_result: latestGuideResult,
    }));
  } catch (_) {
    // Session storage can be unavailable in hardened browser contexts. The
    // server-side take ledger still restores generation history in that case.
  }
}

function restoreSessionState() {
  let saved = null;
  try {
    saved = JSON.parse(sessionStorage.getItem(LIVE_STATE_KEY) || "null");
  } catch (_) {
    saved = null;
  }
  if (!saved || saved.session_id !== liveSessionId) {
    try { sessionStorage.removeItem(LIVE_STATE_KEY); } catch (_) { /* unavailable */ }
    return;
  }

  restoringSessionState = true;
  try { applyUiState(saved); } catch (_) { /* old or corrupt draft: retain the baseline */ }
  finally { restoringSessionState = false; }
}

function initializeSessionMemory() {
  rememberedControls().forEach((control) => control.addEventListener("input", saveSessionState));
}

function uiSnapshot() {
  return {
    controls: readControls(rememberedControls()),
    active_workspace: document.querySelector(".workspace-tab.active")?.dataset.tab || "songStudio",
    selected_job_id: selectedJobId,
    guide_result: latestGuideResult,
  };
}

function applyUiState(saved) {
  const values = validateControls(saved.controls || {}, rememberedControls());
  const result = validateGuideResult(saved.guide_result);
  $("composer").reset();
  $("promptGuideForm").reset();
  for (const control of rememberedControls()) {
    if (!Object.hasOwn(values, control.id)) continue;
    if (control.type === "checkbox") control.checked = values[control.id];
    else control.value = values[control.id];
  }
  selectedJobId = saved.selected_job_id || "";
  performanceJobId = "";
  latestGuideResult = result;
  $("guideResults").hidden = !result;
  if (result) renderGuideResult(result, false);
  $("durationReadout").textContent = `${$("duration").value} s`;
  updateGuideMode();
  switchWorkspace(saved.active_workspace === "promptGuide" ? "promptGuide" : "songStudio", false);
}

function setStateBusy(busy) {
  stateBusy = busy;
  for (const id of ["exportState", "importState", "resetState"]) $(id).disabled = busy;
  updateGuideActions();
}

async function exportState() {
  if (stateBusy) return;
  setStateBusy(true);
  try {
    if (jobsRefreshPromise) await jobsRefreshPromise;
    const session = await api("/api/session");
    if (session.takes.length > 500) throw new Error("State supports up to 500 takes. Clear older takes before exporting.");
    const text = JSON.stringify({
      format: STATE_FORMAT, version: 1, exported_at: new Date().toISOString(),
      ui: uiSnapshot(), session,
    }, null, 2);
    const blob = new Blob([text], {type: "application/json"});
    if (blob.size > MAX_STATE_BYTES) throw new Error("State is too large (8 MB maximum). Clear older takes before exporting.");
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `minimax-state-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    $("stateStatus").textContent = "Exported drafts, writing result and take history. Audio bytes are not included.";
  } catch (error) {
    $("stateStatus").textContent = error.message;
  } finally { setStateBusy(false); }
}

function confirmStateReplace(label) {
  const dialog = $("stateDialog");
  $("stateDialogTitle").textContent = label + "?";
  $("stateDialogConfirm").textContent = label.startsWith("Import") ? "Import state" : "Reset state";
  dialog.returnValue = "cancel";
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), {once: true});
    dialog.showModal();
  });
}

async function replaceState(saved = null) {
  if (stateBusy) return;
  if (guideBusy) {
    $("stateStatus").textContent = "Wait for the writing assistant to finish first.";
    return;
  }
  const label = saved ? "Import this snapshot" : "Reset to the starting state";
  setStateBusy(true);
  try {
    if (!await confirmStateReplace(label)) return;
    if (jobsRefreshPromise) await jobsRefreshPromise;
    const listing = await api(saved ? "/api/session/import" : "/api/session", saved ? {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(saved.session),
    } : {method: "DELETE"});
    liveSessionId = listing.session_id;
    restoringSessionState = true;
    try {
      if (currentAudio) currentAudio.pause();
      currentAudio = null;
      document.querySelectorAll("#audioOutputs audio").forEach((audio) => audio.pause());
      $("audioOutputs").replaceChildren();
      applyUiState(saved?.ui || {controls: {}, active_workspace: "songStudio"});
      preparePerformance(requestBody());
      applyJobListing(listing.jobs);
    } finally { restoringSessionState = false; }
    saveSessionState();
    $("guideRunStatus").textContent = saved ? "Snapshot restored. Nothing was generated or applied automatically." : "Writing draft cleared.";
    $("stateStatus").textContent = saved
      ? `Imported state. ${listing.missing_audio ? `${listing.missing_audio} audio file(s) missing; history still restored.` : "Saved audio reconnected where available."}`
      : "State reset. Audio files and loaded models were left alone.";
  } catch (error) {
    $("stateStatus").textContent = error.message;
  } finally {
    setStateBusy(false);
    await refreshGuideStatus();
  }
}

async function importStateFile() {
  const file = $("stateFile").files[0];
  $("stateFile").value = "";
  if (!file) return;
  try {
    if (file.size > MAX_STATE_BYTES) throw new Error("State file is too large (8 MB maximum).");
    const saved = parseStudioState(await file.text(), rememberedControls());
    await replaceState(saved);
  } catch (error) { $("stateStatus").textContent = error.message; }
}

function setStatus(kind, text) {
  $("statusDot").className = `status-dot ${kind || ""}`;
  $("statusText").textContent = text;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = {};
  try { body = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => `${(item.loc || []).slice(1).join(".")}: ${item.msg}`).join("; ")
      : body.detail;
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
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
    try { await refreshJobs(true); } catch (_) { /* health reports engine failures */ }
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

function orderedJobs() {
  return Array.from(jobsById.values()).sort((left, right) => (
    (left.take || 0) - (right.take || 0) || left.created_at - right.created_at
  ));
}

function takeName(job) {
  return `Take ${String(job.take || 0).padStart(2, "0")}`;
}

function createTakeCard(jobId) {
  const card = document.createElement("article");
  card.className = "take-card";
  card.dataset.jobId = jobId;
  card.innerHTML = `
    <header class="take-card-header">
      <div class="take-identity"><strong class="take-number"></strong><span class="take-state"></span></div>
      <div class="take-card-actions">
        <button class="take-action take-view" type="button">Show brief</button>
        <button class="take-action take-cancel" type="button">Cancel</button>
        <button class="take-action take-clear" type="button">Clear</button>
      </div>
    </header>
    <p class="take-caption"></p>
    <p class="take-facts"></p>
    <p class="take-error" hidden></p>
    <div class="take-audio-list"></div>`;
  card.querySelector(".take-view").addEventListener("click", () => selectTake(card.dataset.jobId, true));
  card.querySelector(".take-cancel").addEventListener("click", (event) => cancelTake(card.dataset.jobId, event.currentTarget));
  card.querySelector(".take-clear").addEventListener("click", (event) => clearTake(card.dataset.jobId, event.currentTarget));
  return card;
}

function syncTakeAudio(card, job) {
  const urls = job.audio || [];
  const signature = urls.join("\n");
  if (card.dataset.audioSignature === signature) return;
  card.dataset.audioSignature = signature;
  const container = card.querySelector(".take-audio-list");
  container.replaceChildren();
  urls.forEach((url, index) => {
    const row = document.createElement("div");
    row.className = "take-audio-row";
    const label = document.createElement("span");
    label.textContent = urls.length > 1 ? `Output ${String(index + 1).padStart(2, "0")}` : "Final audio";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = url;
    audio.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach((other) => { if (other !== audio) other.pause(); });
      const currentJob = jobsById.get(card.dataset.jobId);
      if (currentJob) {
        selectTake(currentJob.id, false);
        $("nowPlaying").textContent = `${takeName(currentJob).toUpperCase()}${urls.length > 1 ? ` · OUTPUT ${String(index + 1).padStart(2, "0")}` : ""} · MINIMAX MUSIC 3`;
      }
      connectVisualizer(audio);
    });
    const download = document.createElement("a");
    download.href = url;
    download.download = "";
    download.textContent = "Download ↘";
    row.append(label, audio, download);
    container.appendChild(row);
  });
}

function updateTakeCard(card, job) {
  card.className = `take-card take-${job.status}${job.id === selectedJobId ? " selected" : ""}`;
  card.querySelector(".take-number").textContent = takeName(job);
  const state = card.querySelector(".take-state");
  state.className = `take-state take-state-${job.status}`;
  state.textContent = JOB_STATUS_LABELS[job.status] || job.status;
  card.querySelector(".take-caption").textContent = job.request?.global_metadata || "Untitled local generation";
  const created = new Date(job.created_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const request = job.request || {};
  const batch = Number(request.batch || 1);
  card.querySelector(".take-facts").textContent = [
    `${request.duration || "?"} s`,
    `seed ${request.seed ?? "?"}`,
    `${batch} output${batch === 1 ? "" : "s"}`,
    String(request.output_format || "audio").toUpperCase(),
    created,
  ].join(" · ");
  const error = card.querySelector(".take-error");
  error.hidden = !job.error;
  error.textContent = job.error || "";
  card.querySelector(".take-cancel").hidden = !ACTIVE_JOB_STATUSES.has(job.status);
  card.querySelector(".take-clear").hidden = !TERMINAL_JOB_STATUSES.has(job.status);
  syncTakeAudio(card, job);
}

function selectTake(jobId, scroll = false) {
  const job = jobsById.get(jobId);
  if (!job) return;
  selectedJobId = jobId;
  if (performanceJobId !== jobId) {
    preparePerformance(job.request || {});
    performanceJobId = jobId;
  }
  document.querySelectorAll(".take-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.jobId === jobId);
  });
  saveSessionState();
  if (scroll) $("performance").scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateRenderSummary(jobs) {
  const active = jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status));
  if (active.length) {
    const rendering = active.find((job) => job.status === "generating");
    const waiting = active.length - (rendering ? 1 : 0);
    $("renderStatus").textContent = rendering
      ? `${takeName(rendering)} is rendering${waiting ? ` · ${waiting} queued` : ""}.`
      : `${active.length} take${active.length === 1 ? "" : "s"} waiting for the local lane.`;
    return;
  }
  const latest = jobs.at(-1);
  if (!latest) {
    $("renderStatus").textContent = "Ready for a local render.";
  } else if (latest.status === "complete") {
    $("renderStatus").textContent = `${takeName(latest)} is ready.`;
  } else if (latest.status === "cancelled") {
    $("renderStatus").textContent = `${takeName(latest)} was cancelled.`;
  } else if (latest.status === "error") {
    $("renderStatus").textContent = `${takeName(latest)} failed: ${latest.error || "unknown error"}`;
  }
}

function syncTakeHistory() {
  const jobs = orderedJobs();
  const container = $("audioOutputs");
  const liveIds = new Set(jobs.map((job) => job.id));
  container.querySelectorAll(".take-card").forEach((card) => {
    if (!liveIds.has(card.dataset.jobId)) card.remove();
  });
  container.querySelector(".take-empty")?.remove();

  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.className = "take-empty";
    empty.textContent = "No takes in this live server session yet.";
    container.appendChild(empty);
    selectedJobId = "";
    performanceJobId = "";
  } else {
    if (!selectedJobId || !jobsById.has(selectedJobId)) selectedJobId = jobs.at(-1).id;
    jobs.forEach((job) => {
      let card = container.querySelector(`[data-job-id="${job.id}"]`);
      if (!card) card = createTakeCard(job.id);
      updateTakeCard(card, job);
      container.appendChild(card);
    });
    selectTake(selectedJobId, false);
  }

  const activeCount = jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status)).length;
  $("takeHistoryStatus").textContent = jobs.length
    ? `${jobs.length} take${jobs.length === 1 ? "" : "s"} in this live server session${activeCount ? ` · ${activeCount} active` : ""}.`
    : "Refresh-safe while this server runs; a server restart begins a clean session.";
  $("clearFinishedTakes").disabled = !jobs.some((job) => TERMINAL_JOB_STATUSES.has(job.status));
  updateRenderSummary(jobs);
  saveSessionState();

  if (currentAudio && !document.body.contains(currentAudio)) {
    currentAudio.pause();
    currentAudio = null;
  }
}

function applyJobListing(list) {
  jobsById.clear();
  (list || []).forEach((job) => jobsById.set(job.id, job));
  syncTakeHistory();
  if (orderedJobs().some((job) => ACTIVE_JOB_STATUSES.has(job.status))) scheduleJobPolling();
  else if (jobPollTimer) {
    clearTimeout(jobPollTimer);
    jobPollTimer = 0;
  }
}

async function refreshJobs(force = false) {
  if (stateBusy) return;
  if (jobsRefreshPromise) {
    if (!force) return jobsRefreshPromise;
    try { await jobsRefreshPromise; } catch (_) { /* force a fresh request below */ }
  }
  const pending = (async () => {
    const listing = await api("/api/jobs");
    if (liveSessionId && listing.session_id !== liveSessionId) {
      try { sessionStorage.removeItem(LIVE_STATE_KEY); } catch (_) { /* unavailable */ }
      location.reload();
      return listing;
    }
    if (!liveSessionId) {
      liveSessionId = listing.session_id;
      restoreSessionState();
    }
    applyJobListing(listing.jobs);
    return listing;
  })();
  jobsRefreshPromise = pending;
  try {
    return await pending;
  } finally {
    if (jobsRefreshPromise === pending) jobsRefreshPromise = null;
  }
}

function scheduleJobPolling() {
  if (jobPollTimer) return;
  jobPollTimer = window.setTimeout(async () => {
    jobPollTimer = 0;
    try {
      await refreshJobs();
    } catch (error) {
      $("renderStatus").textContent = `Take status unavailable: ${error.message}`;
      if (orderedJobs().some((job) => ACTIVE_JOB_STATUSES.has(job.status))) scheduleJobPolling();
    }
  }, 1200);
}

async function cancelTake(jobId, button) {
  button.disabled = true;
  try {
    const job = await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    jobsById.set(job.id, job);
    syncTakeHistory();
    if (job.interrupt_warning) $("renderStatus").textContent = `Take cancelled; engine warning: ${job.interrupt_warning}`;
    await refreshJobs(true);
  } catch (error) {
    $("renderStatus").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function clearTake(jobId, button) {
  button.disabled = true;
  try {
    await api(`/api/jobs/${jobId}`, { method: "DELETE" });
    await refreshJobs(true);
  } catch (error) {
    $("renderStatus").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function clearFinishedTakes() {
  const button = $("clearFinishedTakes");
  button.disabled = true;
  try {
    const result = await api("/api/jobs", { method: "DELETE" });
    await refreshJobs(true);
    $("renderStatus").textContent = result.removed.length
      ? `Cleared ${result.removed.length} finished take${result.removed.length === 1 ? "" : "s"} from the session. Audio files remain on disk.`
      : "No finished takes to clear.";
  } catch (error) {
    $("renderStatus").textContent = error.message;
  } finally {
    button.disabled = !orderedJobs().some((job) => TERMINAL_JOB_STATUSES.has(job.status));
  }
}

async function generate(event) {
  event.preventDefault();
  if (stateBusy) return;
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
    jobsById.set(job.id, job);
    selectedJobId = job.id;
    performanceJobId = job.id;
    syncTakeHistory();
    await refreshJobs(true);
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
  const job = jobsById.get(selectedJobId);
  const byLine = new Map();
  for (const row of Array.isArray(job?.lyric_timestamps) ? job.lyric_timestamps : []) {
    if (row && Number.isFinite(Number(row.start))) byLine.set(Number(row.line), Number(row.start));
  }
  let index = -1;
  if (byLine.size) {
    lyricRows.forEach((row, rowIndex) => {
      if (row.classList.contains("tag")) return;
      const start = byLine.get(rowIndex);
      if (start != null && start <= audio.currentTime + 0.15) index = rowIndex;
    });
  }
  if (index === -1) index = Math.min(lyricRows.length - 1, Math.floor((audio.currentTime / audio.duration) * lyricRows.length));
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

function escapeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function slugify(value) {
  const slug = String(value || "track").toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || "track";
}

function buildMinimaxPlayerHtml({ title, source, lyrics, metadata, vocals, arrangement }) {
  const caption = (label, value) => `<small>${label}</small><p>${escapeText(value || "—")}</p>`;
  const rows = lyrics
    .map((line) => `<p${/^\s*\[.*\]\s*$/.test(line) ? ' class="tag" data-tag="1"' : ""}>${escapeText(line)}</p>`)
    .join("\n    ");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeText(title)}</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;min-height:100%}
body{display:grid;place-items:center;padding:34px 22px;background:#f4f2eb;color:#1d1d1b;font:14px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}
main{width:min(980px,100%);border:1px solid #272725;background:#020202;box-shadow:0 28px 80px rgba(0,0,0,.35);padding:28px 30px 14px;display:flex;flex-direction:column}
h1{margin:0;font-size:17px;font-weight:650;letter-spacing:-.01em;color:#e8e6df}
.source{margin-top:6px}
.source a{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#686762;font:11px ui-monospace,monospace;text-decoration:none;border-bottom:1px dotted rgba(104,103,98,.5)}
.source a:hover{color:#e8e6df}
.copy{display:grid;grid-template-columns:.9fr 1.1fr;gap:36px;padding:22px 0 0;min-height:280px}
.copy article{border-right:1px solid #1d1d1c;padding-right:24px}
.copy small{display:block;color:#8f8e89;font-size:10px;letter-spacing:.16em;margin-top:14px}
.copy small:first-child{margin-top:0}
.copy p{color:#e8e6df;font-size:12px;line-height:1.7;margin:8px 0 0;white-space:pre-wrap}
.lyrics{align-self:center;max-height:300px;overflow:auto;-webkit-mask-image:linear-gradient(transparent,black 12%,black 88%,transparent);mask-image:linear-gradient(transparent,black 12%,black 88%,transparent);scrollbar-width:thin;scrollbar-color:#3a3a37 transparent}
.lyrics p{color:#454542;margin:3px 0;transition:color .2s,transform .2s,font-size .2s}
.lyrics p.tag{color:#77756f;margin-top:14px;font-size:11px;letter-spacing:.12em}
.lyrics p.active{color:#fff;transform:translateX(8px)}
canvas{display:block;width:100%;height:110px;margin-top:18px}
audio{display:block;width:100%;margin-top:14px}
footer{display:flex;justify-content:space-between;gap:12px;margin-top:10px;color:#686762;font-size:9px;letter-spacing:.13em}
@media (max-width:700px){.copy{grid-template-columns:1fr}.copy article{border-right:0;border-bottom:1px solid #1d1d1c;padding-right:0;padding-bottom:18px}}
</style>
</head>
<body>
<main>
<h1>${escapeText(title)}</h1>
<p class="source"><a href="${escapeText(source)}" target="_blank" rel="noopener">${escapeText(source)}</a></p>
<div class="copy">
<article>${caption("GLOBAL METADATA", metadata)}${caption("VOCAL DETAILS", vocals)}${caption("ARRANGEMENT", arrangement)}</article>
<section class="lyrics">${rows || '<p>Your lyrics will move with playback.</p>'}</section>
</div>
<canvas id="spectrum" aria-label="Animated audio spectrum"></canvas>
<audio id="player" src="${escapeText(source)}" crossorigin="anonymous" controls preload="metadata"></audio>
<footer><span>SELF-CONTAINED PLAYER &middot; MINIMAX MUSIC STUDIO</span><time id="playTime">00:00 / 00:00</time></footer>
</main>
<script>
"use strict";
(() => {
  var $ = function (id) { return document.getElementById(id); };
  var audio = $("player");
  var canvas = $("spectrum");
  var clock = $("playTime");
  var rows = Array.prototype.slice.call(document.querySelectorAll(".lyrics p"));
  var context = null;
  var analyser = null;
  var activeIndex = -1;
  function formatTime(seconds) {
    if (!isFinite(seconds)) return "00:00";
    var whole = Math.max(0, Math.floor(seconds));
    var m = Math.floor(whole / 60);
    var s = whole % 60;
    return (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
  }
  function drawSpectrum() {
    var ratio = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(1, canvas.clientWidth);
    var height = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) { canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio); }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    var values = new Uint8Array(analyser ? analyser.frequencyBinCount : 64);
    if (analyser && !audio.paused) analyser.getByteFrequencyData(values);
    var bars = 72;
    var gap = 3;
    var barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
    for (var i = 0; i < bars; i += 1) {
      var sample = values[Math.floor((i / bars) * values.length)] || (3 + 5 * Math.sin(i * 0.7));
      var barHeight = Math.max(2, (sample / 255) * (height - 6));
      var x = i * (barWidth + gap);
      ctx.fillStyle = "rgba(244, 242, 235, " + (0.38 + (sample / 255) * 0.62) + ")";
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    }
    clock.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
    if (!audio.paused && Number.isFinite(audio.duration) && audio.duration > 0 && rows.length) {
      var index = Math.min(rows.length - 1, Math.floor((audio.currentTime / audio.duration) * rows.length));
      if (index !== activeIndex) {
        activeIndex = index;
        rows.forEach(function (row, j) { row.classList.toggle("active", j === index); });
        var panel = rows[index].parentElement;
        panel.scrollTo({ top: Math.max(0, rows[index].offsetTop - panel.clientHeight / 2 + rows[index].clientHeight / 2), behavior: "smooth" });
      }
    }
    requestAnimationFrame(drawSpectrum);
  }
  audio.addEventListener("play", async function () {
    var AudioContextType = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextType) return;
    if (!context) {
      context = new AudioContextType();
      analyser = context.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.82;
      var node = context.createMediaElementSource(audio);
      node.connect(analyser);
      analyser.connect(context.destination);
    }
    await context.resume();
  });
  requestAnimationFrame(drawSpectrum);
})();
</script>
</body>
</html>`;
}

function exportCurrentPlayer() {
  const audio = currentAudio;
  if (!audio || !audio.src) {
    setStatus("", "Play a take first, then export its self-contained player.");
    return;
  }
  const nowPlaying = $("nowPlaying").textContent.replace(/\s*·\s*MINIMAX MUSIC 3$/i, "").trim();
  const job = selectedJobId ? jobsById.get(selectedJobId) : null;
  const title = nowPlaying || (job ? takeName(job) : "Take");
  const lyrics = lyricRows.map((row) => row.textContent).filter(Boolean);
  const html = buildMinimaxPlayerHtml({
    title,
    source: audio.src,
    lyrics,
    metadata: $("performanceMetadata").textContent.trim(),
    vocals: $("performanceVocals").textContent.trim(),
    arrangement: $("performanceArrangement").textContent.trim(),
  });
  const blob = new Blob([html], { type: "text/html" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${slugify(title)}-player.html`;
  link.click();
  URL.revokeObjectURL(link.href);
  setStatus("", "Self-contained player downloaded.");
}

function randomSeedFor(inputId) {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  $(inputId).value = String((values[0] * 0x100000 + (values[1] & 0xfffff)) % Number.MAX_SAFE_INTEGER);
  $(inputId).dispatchEvent(new Event("input", { bubbles: true }));
}

function randomSeed() {
  randomSeedFor("seed");
}

async function interrupt() {
  try {
    const result = await api("/api/interrupt", { method: "POST" });
    await refreshJobs(true);
    $("renderStatus").textContent = result.interrupt_warning
      ? `Takes cancelled; engine warning: ${result.interrupt_warning}`
      : `${result.cancelled.length} active take${result.cancelled.length === 1 ? "" : "s"} cancelled.`;
  } catch (error) {
    $("renderStatus").textContent = error.message;
  }
}

function switchWorkspace(panelId, smooth = true) {
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === panelId);
  });
  document.querySelectorAll(".workspace-panel").forEach((panel) => {
    const active = panel.id === panelId;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
  saveSessionState();
  if (smooth) window.scrollTo({ top: 0, behavior: "smooth" });
}


function flashButton(button, text) {
  const original = button.textContent;
  button.textContent = text;
  window.setTimeout(() => { button.textContent = original; }, 1200);
}


async function writeClipboard(text, button) {
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
  flashButton(button, "Copied");
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
      saveSessionState();
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

function updateGuideActions() {
  $("runPromptGuide").disabled = !guideLoaded || guideBusy || stateBusy;
  $("guideEnabled").disabled = guideBusy || stateBusy;
  $("browseGuideModels").disabled = guideLoaded || guideBusy || stateBusy;
}

async function refreshGuideStatus() {
  if (guideBusy || stateBusy) return;
  try {
    const state = await api("/api/guide/status");
    if (guideBusy || stateBusy) return;
    guideLoaded = Boolean(state.loaded);
    if (state.loaded_model) $("guideModel").value = state.loaded_model;
    $("guideEnabled").checked = guideLoaded;
    document.querySelector(".guide-runtime-card h2").textContent = guideLoaded
      ? "Ready to write" : "Writing assistant is off";
    $("guideStatus").textContent = guideLoaded ? "Your local model is ready."
      : (!state.default_exists ? "Choose an installed model in Advanced, then switch on."
        : "Switch on when you're ready to write.");
  } catch (error) {
    guideLoaded = false;
    $("guideEnabled").checked = false;
    $("guideStatus").textContent = error.message;
  }
  updateGuideActions();
}

async function toggleGuide() {
  if (guideBusy || stateBusy) return;
  const requested = $("guideEnabled").checked;
  guideBusy = true;
  updateGuideActions();
  $("guideStatus").textContent = requested ? "Loading your writing assistant…" : "Releasing the writing model…";
  try {
    await api(requested ? "/api/guide/load" : "/api/guide/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      ...(requested ? {body: JSON.stringify({ model: $("guideModel").value })} : {}),
    });
    $("guideRunStatus").textContent = requested ? "Ready. Generate text whenever you like." : "Writing assistant switched off.";
  } catch (error) {
    $("guideRunStatus").textContent = error.message;
  } finally {
    guideBusy = false;
    await refreshGuideStatus();
  }
}

function updateGuideMode() {
  const mode = $("guideMode").value;
  const hints = {
    brief: "Turn an idea into Global Metadata, Vocal Details and Arrangement.",
    keep_lyrics: "Keep every lyric as written. Generate only the surrounding musical direction.",
    song: "Draft musical direction and original lyrics from your idea.",
    ask: "A quick question, explanation or listening reference. Web lookup is optional.",
  };
  $("guideModeHint").textContent = hints[mode] || hints.brief;
  $("guideDirectionLabel").textContent = mode === "ask" ? "Your question" : "Your idea";
  $("guideDirection").placeholder = mode === "ask"
    ? 'How do the vocals and arrangement work in BLACKPINK’s "Pink Venom"?'
    : "An intimate alternative R&B verse that opens into a confident chorus…";
  $("guideLyricsField").hidden = mode === "ask";
  $("guideLyrics").required = mode === "keep_lyrics";
  $("guideLyricsLabel").textContent = mode === "keep_lyrics" ? "Your finished lyrics"
    : mode === "song" ? "Lyric ideas · optional" : "Lyrics as context · optional";
  $("guideLyricsHint").textContent = mode === "keep_lyrics" ? "Preserved exactly, including line breaks. The model only writes the other fields."
    : mode === "song" ? "The model may rework these ideas into an original draft."
    : "Used for context only. Your original lyrics stay yours.";
  $("guideResearch").hidden = mode !== "ask";
}

function promptGuideBody() {
  return {
    mode: $("guideMode").value,
    web_search: $("guideMode").value === "ask" && $("guideWebSearch").checked,
    search_query: $("guideSearchQuery").value,
    model: $("guideModel").value,
    direction: $("guideDirection").value,
    lyrics: $("guideMode").value === "ask" ? "" : $("guideLyrics").value,
    constraints: $("guideConstraints").value,
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

function renderGuideResult(result, scroll = true) {
  latestGuideResult = result;
  const sections = result.sections || {};
  for (const [id, name] of [
    ["guideGlobalOutput", "Global Metadata"], ["guideVocalOutput", "Vocal Details"],
    ["guideArrangementOutput", "Arrangement"], ["guideLyricsOutput", "Lyrics"],
  ]) {
    $(id).textContent = sections[name] || "See the full response below.";
  }
  const ask = result.mode === "ask";
  $("guideCaptionOutputs").hidden = ask;
  $("guideAnswerOutput").hidden = !ask;
  $("guideAnswerOutput").textContent = ask ? result.text : "";
  $("guideLyricResult").hidden = !Object.hasOwn(sections, "Lyrics");
  $("guideResultNote").textContent = result.mode === "keep_lyrics"
    ? "Your lyrics are unchanged. Copy the musical direction you want."
    : "Copy what you like. Song Studio is unchanged.";
  const sources = $("guideSources");
  sources.replaceChildren();
  for (const source of result.sources || []) {
    const url = safeSourceUrl(source.url);
    if (!url) continue;
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = source.title || url;
    li.append(link, document.createTextNode(source.scraped ? " · page excerpt" : " · search snippet only"));
    sources.append(li);
  }
  sources.hidden = !sources.children.length;
  $("guideRawOutput").textContent = result.text || "";
  $("guideRawDetails").hidden = ask;
  $("guideResults").hidden = false;
  saveSessionState();
  if (scroll) $("guideResults").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPromptGuide(event) {
  event.preventDefault();
  if (guideBusy || stateBusy || !guideLoaded) return;
  const request = promptGuideBody();
  guideBusy = true;
  const button = $("runPromptGuide");
  setBusy(button, true, "Writing…");
  updateGuideActions();
  $("guideRunStatus").textContent = request.web_search
    ? "Looking up sources with Firecrawl, then writing locally…" : "Writing locally…";
  try {
    const result = await api("/api/guide/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    renderGuideResult(result);
    $("guideRunStatus").textContent = "Finished. Copy only the parts you want.";
  } catch (error) {
    $("guideRunStatus").textContent = error.message;
  } finally {
    guideBusy = false;
    setBusy(button, false, "");
    await refreshGuideStatus();
  }
}

async function copyGuideText(elementId, button) {
  const text = $(elementId).textContent;
  await writeClipboard(text, button);
}

/* ---------------- StemKit pane ---------------- */

const STEM_PRESETS = [
  { id: "all", label: "All", note: "demucs htdemucs", stems: ["vocals", "drums", "bass", "other"] },
  { id: "karaoke", label: "Karaoke", note: "no-vocals mix", stems: ["drums", "bass", "other"] },
  { id: "acapella", label: "Acapella", note: "Mel-Band Roformer", stems: ["vocals"] },
  { id: "drums_bass", label: "Drums + Bass", note: "two stems", stems: ["drums", "bass"] },
];

function stemTrackChoices() {
  return orderedJobs()
    .filter((job) => job.status === "complete" && (job.audio || []).length)
    .slice(0, 24)
    .flatMap((job) => (job.audio || []).map((url, index) => ({ jobId: job.id, index, label: takeName(job), url })));
}

function renderStemPane() {
  const pick = $("stemTrackPick");
  const staleEmpty = !pick.querySelector(".stem-track") && !!pick.querySelector(".muted");
  if (!pick.childElementCount || (staleEmpty && stemTrackChoices().length)) pick.replaceChildren();
  if (!pick.childElementCount) {
    const choices = stemTrackChoices();
    if (!choices.length) {
      pick.innerHTML = "<p class='muted'>No finished takes yet — render one, or upload below.</p>";
    } else {
      choices.forEach((choice, position) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "stem-track" + (position === 0 ? " active" : "");
        button.innerHTML = `<span>♫</span><div><b>${escapeHtml(choice.label)}</b><small>${escapeHtml(new URL(choice.url, location.origin).pathname.split("/").at(-1) || choice.url)}</small></div>`;
        button.addEventListener("click", () => {
          document.querySelectorAll(".stem-track").forEach((other) => other.classList.toggle("active", other === button));
          stemKit.source = { take: choice.jobId, index: choice.index };
          $("stemStatus").textContent = "Selected — choose a preset and split.";
        });
        pick.append(button);
      });
      const first = pick.querySelector(".stem-track");
      if (first) first.click();
    }
  }
  const presets = $("stemPresets");
  if (!presets.childElementCount) {
    STEM_PRESETS.forEach((preset) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip" + (preset.id === "all" ? " active" : "");
      button.dataset.preset = preset.id;
      button.innerHTML = `<span>${preset.label}</span><small>${preset.note}</small>`;
      button.addEventListener("click", () => {
        document.querySelectorAll("#stemPresets .chip").forEach((item) => item.classList.toggle("active", item === button));
        applyStemPreset(preset.stems);
      });
      presets.append(button);
    });
    applyStemPreset(STEM_PRESETS[0].stems);
  }
  const toggles = $("stemToggles");
  if (!toggles.childElementCount) {
    [["vocals", "Vocals"], ["drums", "Drums"], ["bass", "Bass"], ["other", "Other"]].forEach(([key, label]) => {
      const wrap = document.createElement("label");
      wrap.className = "switch-row";
      wrap.innerHTML = `<input type="checkbox" value="${key}" ${key === "vocals" ? "checked" : ""}><span>${label}</span>`;
      toggles.append(wrap);
    });
  }
  if (!stemKit.bound) {
    stemKit.bound = true;
    $("stemSplit").addEventListener("click", runStemSplit);
    $("stemFile").addEventListener("change", () => { if ($("stemFile").files[0]) uploadStemSource($("stemFile").files[0]); });
    $("stemPlay").addEventListener("click", toggleStemPlayback);
    $("stemMaster").addEventListener("input", applyStemGains);
    $("stemSeek").addEventListener("input", seekStems);
  }
}

function applyStemPreset(stems) {
  document.querySelectorAll("#stemToggles input").forEach((item) => { item.checked = stems.includes(item.value); });
}

async function uploadStemSource(file) {
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const asset = await api("/api/stems/upload", { method: "POST", body: form });
    stemKit.source = asset.asset_id;
    $("stemStatus").textContent = `Uploaded ${file.name} — choose a preset and split.`;
    document.querySelectorAll(".stem-track").forEach((item) => item.classList.remove("active"));
  } catch (error) {
    $("stemStatus").textContent = error.message;
  }
}

async function runStemSplit() {
  if (!stemKit.source) { $("stemStatus").textContent = "Pick a take or upload one first."; return; }
  const stems = [...document.querySelectorAll("#stemToggles input:checked")].map((item) => item.value);
  if (!stems.length) { $("stemStatus").textContent = "Select at least one stem."; return; }
  const payload = { input: stemKit.source, stems };
  const button = $("stemSplit");
  button.disabled = true; button.firstChild.textContent = "Splitting…";
  $("stemProgressWrap").hidden = false;
  try {
    const result = await api("/api/stems/split", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    pollStemJob(result.job_id);
  } catch (error) {
    $("stemStatus").textContent = error.message;
    $("stemProgressWrap").hidden = true;
  } finally {
    button.disabled = false; button.firstChild.textContent = "Split stems";
  }
}

function pollStemJob(id) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/stems/jobs/${id}`);
      $("stemProgressFill").style.width = `${Math.round(job.pct || 0)}%`;
      $("stemProgressText").textContent = `${job.stage || job.status || "working"} · ${Math.round(job.pct || 0)}%${job.message ? ` — ${job.message}` : ""}`;
      if (job.status === "failed") {
        clearInterval(timer);
        $("stemStatus").textContent = job.error || job.message || "The split failed.";
        $("stemProgressWrap").hidden = true;
        return;
      }
      if (job.status === "done") {
        clearInterval(timer);
        renderStemLanes(job);
      }
    } catch (error) {
      clearInterval(timer);
      $("stemStatus").textContent = error.message;
      $("stemProgressWrap").hidden = true;
    }
  }, 1500);
}

function renderStemLanes(job) {
  const lanes = $("stemLanes");
  stopStemPlayback();
  stemKit.lanes = [];
  lanes.replaceChildren();
  const names = Array.isArray(job.stems) && job.stems.length ? job.stems : ["vocals", "drums", "bass", "other"];
  for (const name of names) {
    const lane = document.createElement("div");
    lane.className = "stem-lane";
    lane.innerHTML = `<span class="lane-name">${escapeHtml(name)}</span><input type="range" min="0" max="100" value="100" aria-label="volume"><span class="lane-bpm"></span>`;
    const fader = lane.querySelector("input");
    const entry = { name, url: `/api/stems/jobs/${job.id}/${name}`, gain: 1, element: fader };
    fader.addEventListener("input", () => { entry.gain = Number(fader.value) / 100; applyStemGains(); });
    const exportLink = document.createElement("a");
    exportLink.href = entry.url;
    exportLink.download = `${name}.wav`;
    exportLink.className = "lane-export";
    exportLink.textContent = "⤓";
    exportLink.title = "Download this stem";
    lane.append(exportLink);
    stemKit.lanes.push(entry);
    lanes.append(lane);
  }
  if (!stemKit.lanes.length) {
    lanes.innerHTML = "<p class='muted'>No stems found on that job.</p>";
    return;
  }
  buildStemMix();
  $("stemStatus").textContent = `${stemKit.lanes.length} lanes live (${job.engine || "local engine"}) — drag the faders, watch the spectrum.`;
}

async function buildStemMix() {
  // Fetch every lane, align lengths, and render an offline mixed master so the
  // transport plays all lanes in lockstep without N parallel <audio> elements.
  const decoded = await Promise.all(stemKit.lanes.map(async (lane) => {
    const arrayBuffer = await (await fetch(lane.url)).arrayBuffer();
    const audioCtx = new OfflineAudioContext(2, 1, 44100);
    return audioCtx.decodeAudioData(arrayBuffer.slice(0));
  }));
  const length = Math.max(...decoded.map((buffer) => buffer.length));
  const mix = new OfflineAudioContext(2, length, 44100);
  decoded.forEach((buffer, index) => {
    const source = mix.createBufferSource();
    source.buffer = buffer;
    const gain = mix.createGain();
    gain.gain.value = stemKit.lanes[index].gain;
    source.connect(gain); gain.connect(mix.destination);
    source.start(0);
  });
  const rendered = await mix.startRendering();
  const wav = audioBufferToWav(rendered);
  const blob = new Blob([wav], { type: "audio/wav" });
  if (stemKit.audio) URL.revokeObjectURL(stemKit.audio.src);
  stemKit.audio = new Audio(URL.createObjectURL(blob));
  stemKit.audio.volume = Number($("stemMaster").value) / 100;
  stemKit.audio.ontimeupdate = paintStemTransport;
  stemKit.audio.onended = () => { $("stemPlay").textContent = "▶ Play"; };
  startStemTransport();
}

function audioBufferToWav(buffer) {
  const channels = Math.min(2, buffer.numberOfChannels);
  const frames = buffer.length;
  const bytesPerSample = 2;
  const dataSize = frames * channels * bytesPerSample;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);
  const writeString = (offset, text) => { for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i)); };
  writeString(0, "RIFF"); view.setUint32(4, 36 + dataSize, true); writeString(8, "WAVE");
  writeString(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true); view.setUint16(34, 16, true);
  writeString(36, "data"); view.setUint32(40, dataSize, true);
  let offset = 44;
  const channelData = [...Array(channels)].map((_, channel) => buffer.getChannelData(channel));
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Uint8Array(arrayBuffer);
}

function startStemTransport() {
  if (!stemKit.audio) return;
  if (window.AudioContext) {
    const actx = new AudioContext();
    const source = actx.createMediaElementSource(stemKit.audio);
    const analyser = actx.createAnalyser();
    analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.8;
    source.connect(analyser); analyser.connect(actx.destination);
    stemKit.graph = { actx, analyser };
  }
  const paint = () => {
    paintStemTransport();
    const lane = document.querySelector(".stem-lane .lane-bpm");
    if (stemKit.graph?.analyser && lane) {
      const data = new Uint8Array(stemKit.graph.analyser.frequencyBinCount);
      stemKit.graph.analyser.getByteFrequencyData(data);
      const energy = data.reduce((sum, value) => sum + value, 0) / data.length / 255;
      lane.style.setProperty("--pulse", energy.toFixed(3));
    }
    stemKit.raf = requestAnimationFrame(paint);
  };
  cancelAnimationFrame(stemKit.raf);
  stemKit.raf = requestAnimationFrame(paint);
}

function paintStemTransport() {
  if (!stemKit.audio) return;
  const duration = Number.isFinite(stemKit.audio.duration) ? stemKit.audio.duration : 0;
  $("stemClock").textContent = `${formatTime(stemKit.audio.currentTime)} / ${formatTime(duration)}`;
  if (duration > 0) $("stemSeek").value = String(Math.round((stemKit.audio.currentTime / duration) * 1000));
}

function toggleStemPlayback() {
  if (!stemKit.audio) return;
  if (stemKit.audio.paused) {
    stemKit.audio.play();
    $("stemPlay").textContent = "❚❚ Pause";
  } else {
    stemKit.audio.pause();
    $("stemPlay").textContent = "▶ Play";
  }
}

function stopStemPlayback() {
  if (stemKit.audio) { stemKit.audio.pause(); stemKit.audio.src = ""; }
  stemKit.audio = null;
  if (stemKit.graph?.actx) stemKit.graph.actx.close().catch(() => {});
  stemKit.graph = null;
  cancelAnimationFrame(stemKit.raf);
  stemKit.raf = 0;
}

function seekStems() {
  if (!stemKit.audio) return;
  const duration = Number.isFinite(stemKit.audio.duration) ? stemKit.audio.duration : 0;
  if (duration <= 0) return;
  stemKit.audio.currentTime = (Number($("stemSeek").value) / 1000) * duration;
}

function applyStemGains() {
  const master = Number($("stemMaster").value) / 100;
  stemKit.lanes.forEach((lane) => {
    lane.element.style.setProperty("--level", String(lane.gain * master));
  });
  if (stemKit.audio) stemKit.audio.volume = master;
  // A full offline re-mix is expensive; rebuild when playback pauses.
  if (stemKit.audio && !stemKit.audio.paused) {
    stemKit.audio.onpause = () => buildStemMix().catch(() => {});
  }
}

renderStemPane();

async function initializeApplication() {
  drawSpectrum();
  try {
    await refreshJobs();
  } catch (error) {
    $("renderStatus").textContent = `Live session unavailable: ${error.message}`;
  }
  await Promise.all([refreshHealth(), refreshGuideModels()]);
  await refreshGuideStatus();
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
$("guideRandomSeed").addEventListener("click", () => randomSeedFor("guideSeed"));
$("refreshStatus").addEventListener("click", refreshHealth);
$("loadModel").addEventListener("click", loadModels);
$("unloadModel").addEventListener("click", unloadModels);
$("interrupt").addEventListener("click", interrupt);
$("clearFinishedTakes").addEventListener("click", clearFinishedTakes);
$("exportPlayer").addEventListener("click", exportCurrentPlayer);
$("composer").addEventListener("submit", generate);
$("guideEnabled").addEventListener("change", toggleGuide);
$("browseGuideModels").addEventListener("click", async () => {
  await refreshGuideModels();
  $("guideModelDialog").showModal();
  $("guideModelSearch").focus();
});
$("guideModelSearch").addEventListener("input", () => renderGuideModels($("guideModelSearch").value));
$("promptGuideForm").addEventListener("submit", runPromptGuide);
$("guideMode").addEventListener("change", updateGuideMode);
$("exportState").addEventListener("click", exportState);
$("importState").addEventListener("click", () => $("stateFile").click());
$("stateFile").addEventListener("change", importStateFile);
$("resetState").addEventListener("click", () => replaceState());

updateGuideMode();
initializeSessionMemory();
initializeApplication();
setInterval(() => { refreshHealth(); refreshGuideStatus(); }, 15000);
