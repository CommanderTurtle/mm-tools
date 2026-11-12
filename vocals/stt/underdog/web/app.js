const $ = (selector) => document.querySelector(selector);
const statusNode = $("#status");
const preview = $("#preview");
const variant = $("#variant");
const dropzone = $("#dropzone");
let state = null;
let busy = false;
let audioFile = null;
let previewURL = "";
let recorder = null;
let stream = null;
let recordTimer = null;
let pendingMic = false;
let closing = false;
let results = {};
let activeResult = "";
let requestID = "";
let cancelPending = false;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail =
        typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail ?? payload);
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function modelDescription(row) {
  return row.operation === "intended"
    ? "Clean removes fillers, restarts and repeats."
    : "Literal speech, including fillers and repeats.";
}

function updateControls() {
  const locked =
    busy || state?.busy || pendingMic || recorder?.state === "recording";
  const selected = state?.models.find((row) => row.id === variant.value);
  $("#run").disabled = !!locked || !audioFile || !selected?.model_present;
  $("#variant").disabled = !!locked;
  $("#file").disabled = !!locked;
  $("#refresh").disabled = busy;
  $("#record").disabled = busy || state?.busy || pendingMic;
  $("#model-description").textContent = selected
    ? `${modelDescription(selected)} ${selected.compute_type} · ${selected.download} download.`
    : "Select a downloaded model.";
  for (const card of $("#models").children) {
    const row = state.models.find((item) => item.id === card.dataset.id);
    card.classList.toggle("selected", row.id === variant.value);
    card.querySelector(".choose").disabled = !!locked || !row.model_present;
    card
      .querySelector(".choose")
      .setAttribute("aria-pressed", String(row.id === variant.value));
    card.querySelector(".load").disabled = !!locked || !row.model_present;
    card.querySelector(".load").textContent = row.loaded ? "Unload" : "Load";
    card.querySelector(".state").textContent = !row.model_present
      ? "Download bundle first"
      : row.loaded
        ? "Loaded"
        : "Unloaded";
  }
}

function populateModels() {
  $("#models").replaceChildren();
  variant.replaceChildren();
  for (const row of state.models) {
    const option = new Option(row.name, row.id);
    option.disabled = !row.model_present;
    variant.add(option);
    const card = document.createElement("div");
    card.className = "model";
    card.dataset.id = row.id;
    const choice = document.createElement("button");
    choice.type = "button";
    choice.className = "secondary choose";
    choice.textContent = row.name;
    choice.onclick = () => {
      variant.value = row.id;
      updateControls();
    };
    const description = document.createElement("small");
    description.textContent = `${row.download} · ${row.operation === "intended" ? "Clean speech" : "Verbatim"}`;
    const status = document.createElement("small");
    status.className = "state";
    const load = document.createElement("button");
    load.type = "button";
    load.className = "secondary load";
    load.setAttribute("aria-label", `Load or unload ${row.name}`);
    load.onclick = () => changeLoad(row.id);
    card.append(choice, description, status, load);
    $("#models").append(card);
  }
  variant.value =
    state.models.find(
      (row) => row.id === state.default_model && row.model_present,
    )?.id ||
    state.models.find((row) => row.model_present)?.id ||
    state.default_model;
}

async function health() {
  state = await api("/api/health", { signal: AbortSignal.timeout(5000) });
  if (!$("#models").children.length) populateModels();
  for (const option of variant.options)
    option.disabled = !state.models.find((row) => row.id === option.value)
      ?.model_present;
  const loaded = state.models.filter((row) => row.loaded).length;
  $("#runtime").textContent =
    `${state.backend} · ${state.device} · ${loaded}/4 loaded · English`;
  const progress = state.progress;
  $("#cancel").disabled =
    cancelPending || !requestID || state.request_id !== requestID;
  if (busy && progress.phase === "transcribing") {
    $("#progress").max = progress.total;
    $("#progress").value = progress.done;
    statusNode.textContent = `Transcribing chunk ${progress.done + 1}/${progress.total}…`;
  } else if (busy && progress.phase === "loading") {
    $("#progress").removeAttribute("value");
    statusNode.textContent = `Loading ${state.models.find((row) => row.id === progress.variant).name}…`;
  }
  updateControls();
}

async function busyAction(action) {
  if (busy) return;
  busy = true;
  updateControls();
  $("#progress").classList.remove("hidden");
  $("#progress").removeAttribute("value");
  let monitoring = true;
  let timer = null;
  async function monitor() {
    try {
      await health();
    } catch {}
    if (monitoring) timer = setTimeout(monitor, 1000);
  }
  timer = setTimeout(monitor, 1000);
  try {
    await action();
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    monitoring = false;
    clearTimeout(timer);
    busy = false;
    requestID = "";
    cancelPending = false;
    $("#progress").classList.add("hidden");
    $("#cancel").classList.add("hidden");
    try {
      await health();
    } catch (error) {
      $("#runtime").textContent = error.message;
    }
    updateControls();
  }
}

function changeLoad(id) {
  const row = state.models.find((model) => model.id === id);
  statusNode.textContent = `${row.loaded ? "Unloading" : "Loading"} ${row.name}…`;
  return busyAction(async () => {
    const body = new FormData();
    body.set("variant", id);
    await api(row.loaded ? "/api/unload" : "/api/load", {
      method: "POST",
      body,
    });
    statusNode.textContent = `${row.name} ${row.loaded ? "unloaded" : "loaded"}.`;
  });
}

function setFile(file) {
  if (!file) return;
  if (state && file.size > state.max_upload_mb * 1048576) {
    statusNode.textContent = `Upload limit: ${state.max_upload_mb} MiB.`;
    return;
  }
  preview.pause();
  if (previewURL) URL.revokeObjectURL(previewURL);
  audioFile = file;
  previewURL = URL.createObjectURL(file);
  preview.src = previewURL;
  $("#file-label").textContent =
    `${file.name} · ${(file.size / 1048576).toFixed(1)} MiB`;
  results = {};
  activeResult = "";
  $("#results").classList.add("hidden");
  statusNode.textContent = "Ready";
  updateControls();
}

function inputLocked() {
  return busy || state?.busy || pendingMic || recorder?.state === "recording";
}
dropzone.onclick = () => {
  if (!inputLocked()) $("#file").click();
};
dropzone.onkeydown = (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dropzone.click();
  }
};
$("#file").onchange = (event) => setFile(event.target.files[0]);
for (const name of ["dragenter", "dragover"])
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    if (!inputLocked()) dropzone.classList.add("drag");
  });
for (const name of ["dragleave", "drop"])
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  });
dropzone.addEventListener("drop", (event) => {
  if (!inputLocked()) setFile(event.dataTransfer.files[0]);
});
variant.onchange = updateControls;
$("#refresh").onclick = async () => {
  try {
    await health();
  } catch (error) {
    statusNode.textContent = error.message;
  }
};

function stopTracks() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  clearInterval(recordTimer);
  recordTimer = null;
  $("#record").textContent = "● Record microphone";
  $("#record").classList.remove("recording");
}

$("#record").onclick = async () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    statusNode.textContent =
      "Microphone capture needs localhost or HTTPS. You can still upload audio.";
    return;
  }
  pendingMic = true;
  updateControls();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (closing) {
      stopTracks();
      return;
    }
    const type = [
      "audio/webm;codecs=opus",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ].find((value) => MediaRecorder.isTypeSupported(value));
    const session = new MediaRecorder(
      stream,
      type ? { mimeType: type } : undefined,
    );
    recorder = session;
    const blocks = [];
    let size = 0;
    let failed = false;
    session.ondataavailable = (event) => {
      size += event.data.size;
      if (size > (state?.max_upload_mb ?? 1024) * 1048576) {
        failed = true;
        blocks.length = 0;
        statusNode.textContent = "Recording exceeds the upload limit.";
        if (session.state === "recording") session.stop();
      } else if (!failed && event.data.size) blocks.push(event.data);
    };
    session.onerror = (event) => {
      failed = true;
      statusNode.textContent = event.error?.message || "Recording failed";
      stopTracks();
      recorder = null;
      updateControls();
    };
    session.onstop = () => {
      stopTracks();
      recorder = null;
      if (!closing && !failed && blocks.length) {
        const mime = session.mimeType || type || "audio/webm";
        const ext = mime.includes("mp4")
          ? "m4a"
          : mime.includes("ogg")
            ? "ogg"
            : "webm";
        setFile(
          new File(blocks, `recording-${Date.now()}.${ext}`, { type: mime }),
        );
      }
      updateControls();
    };
    session.start(250);
    const started = Date.now();
    $("#record-time").textContent = "00:00";
    $("#record").textContent = "■ Stop recording";
    $("#record").classList.add("recording");
    recordTimer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      $("#record-time").textContent =
        `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
      if (
        seconds >= (state?.max_audio_seconds ?? 600) &&
        session.state === "recording"
      )
        session.stop();
    }, 250);
  } catch (error) {
    stopTracks();
    recorder = null;
    statusNode.textContent = error.message;
  } finally {
    pendingMic = false;
    updateControls();
  }
};

function showResult(id) {
  activeResult = id;
  const result = results[id];
  $("#result-title").textContent = state.models.find(
    (row) => row.id === id,
  ).name;
  $("#transcript").textContent = result.text || "(No speech recognized)";
  $("#metrics").textContent =
    `${result.duration} s audio · ${result.processing_time} s inference · ${result.segments.length} chunks`;
  $("#json").textContent = JSON.stringify(result, null, 2);
  for (const tab of $("#result-tabs").children) {
    tab.classList.toggle("active", tab.dataset.id === id);
    tab.setAttribute("aria-pressed", String(tab.dataset.id === id));
  }
  $("#chunks").replaceChildren();
  for (const segment of result.segments) {
    const row = document.createElement("div");
    row.className = "chunk";
    const seek = document.createElement("button");
    seek.type = "button";
    seek.className = "secondary";
    seek.textContent = `${segment.start.toFixed(1)}–${segment.end.toFixed(1)} s`;
    seek.onclick = async () => {
      preview.currentTime = segment.start;
      try {
        await preview.play();
      } catch (error) {
        statusNode.textContent = error.message;
      }
    };
    const text = document.createElement("p");
    text.textContent = segment.text;
    row.append(seek, text);
    $("#chunks").append(row);
  }
}

$("#controls").onsubmit = (event) => {
  event.preventDefault();
  if (!audioFile || inputLocked()) return;
  const chosen = variant.value;
  const body = new FormData(event.target);
  body.set("file", audioFile);
  body.set("language", "en");
  statusNode.textContent = "Preparing audio…";
  requestID = Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  cancelPending = false;
  $("#cancel").classList.remove("hidden");
  $("#cancel").disabled = true;
  busyAction(async () => {
    const payload = await api(`/api/transcribe?request_id=${requestID}`, {
      method: "POST",
      body,
    });
    results[chosen] = payload.results[payload.operation];
    $("#result-tabs").replaceChildren();
    for (const id of Object.keys(results)) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.dataset.id = id;
      tab.textContent = state.models.find((row) => row.id === id).name;
      tab.onclick = () => showResult(id);
      $("#result-tabs").append(tab);
    }
    $("#results").classList.remove("hidden");
    showResult(chosen);
    statusNode.textContent = "Complete";
  });
};

$("#cancel").onclick = async () => {
  cancelPending = true;
  $("#cancel").disabled = true;
  try {
    const data = await api(`/api/cancel?request_id=${requestID}`, {
      method: "POST",
    });
    statusNode.textContent = data.message;
  } catch (error) {
    statusNode.textContent = error.message;
    cancelPending = false;
    $("#cancel").disabled = false;
  }
};
$("#copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(results[activeResult]?.text || "");
    statusNode.textContent = "Copied";
  } catch {
    statusNode.textContent =
      "Clipboard needs localhost or HTTPS. Select the transcript or download TXT.";
  }
};

function download(format) {
  const result = results[activeResult];
  if (!result) return;
  const text =
    format === "json" ? JSON.stringify(result, null, 2) : result.text;
  const blob = new Blob([text], {
    type: format === "json" ? "application/json" : "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${audioFile.name.replace(/\.[^.]+$/, "")}-${activeResult}.${format}`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#download-text").onclick = () => download("txt");
$("#download-json").onclick = () => download("json");
window.addEventListener("pagehide", (event) => {
  closing = true;
  preview.pause();
  if (recorder?.state === "recording") recorder.stop();
  stopTracks();
  if (!event.persisted && previewURL) URL.revokeObjectURL(previewURL);
});
window.addEventListener("pageshow", (event) => {
  closing = false;
  if (event.persisted)
    health().catch((error) => {
      statusNode.textContent = error.message;
    });
});
health().catch((error) => {
  statusNode.textContent = error.message;
  $("#runtime").textContent = "Runtime unavailable";
});
