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
