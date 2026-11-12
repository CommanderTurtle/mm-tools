import { MotionView } from "./view.js";

const $ = (id) => document.getElementById(id);
const form = $("settings");
const video = $("video");
const terminal = new Set([
  "complete",
  "partial",
  "failed",
  "cancelled",
  "interrupted",
]);
let config,
  selected = null,
  upload = null,
  reuse = null,
  objectURL = null;
let rows = [],
  selection = null,
  selectionShown = null,
  scene = null,
  track = null;
let mesh,
  physics,
  loadedMesh = null,
  loadedPhysics = null,
  opening = false,
  generation = 0;
let playing = false,
  displayedFrame = 0,
  desiredFrame = 0,
  drawing = false,
  lastTick = 0;
let noticeTimer,
  pollTimer,
  recorder = null,
  recordingStream = null,
  recordingCanvas = null;
let recordChunks = [],
  recordRaf = 0;
const boxContext = $("boxes").getContext("2d");

function message(text) {
  $("notice").textContent = text;
  $("notice").hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("notice").hidden = true), 7000);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const content = await response.json();
  if (!response.ok)
    throw new Error(content.error || `Request failed (${response.status})`);
  return content;
}

function fileURL(id, name) {
  return `/files/${id}/${name.split("/").map(encodeURIComponent).join("/")}`;
}
function timecode(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function applySettings(values) {
  for (const [name, value] of Object.entries(values)) {
    const field = form.elements.namedItem(name);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = value;
    else field.value = value;
  }
  if (config && !config.physics) form.elements.physics.checked = false;
  $("cfgValue").textContent = Number(form.elements.cfg.value).toFixed(1);
}

function getSettings() {
  return Object.fromEntries(
    Object.keys(config.defaults).map((name) => {
      const field = form.elements.namedItem(name);
      return [
        name,
        field.type === "checkbox"
          ? field.checked
          : name === "attention"
            ? field.value
            : Number(field.value),
      ];
    }),
  );
}

function showVideo(url) {
  if (video.getAttribute("src") !== url) {
    video.src = url;
    video.load();
  }
  video.parentElement.classList.add("visible");
  $("drop").hidden = true;
  $("newSource").hidden = false;
}

function chooseFile(file) {
  if (!file) return;
  if (config && file.size > config.upload_mb * 1024 ** 2)
    return message(`Video exceeds ${config.upload_mb} MB`);
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(file);
  upload = file;
  reuse = null;
  selected = null;
  clearView();
  $("filename").textContent = file.name;
  $("sourceLabel").textContent = "New clip";
  $("generate").firstChild.textContent = "Reconstruct motion ";
  $("generate").disabled = false;
  $("jobTitle").textContent = "Ready to reconstruct";
  $("status").textContent = "Adjust the settings, then reconstruct motion.";
  $("jobActions").hidden = true;
  showVideo(objectURL);
  renderJobs();
}

$("upload").addEventListener("change", (event) =>
  chooseFile(event.target.files[0]),
);
for (const event of ["dragenter", "dragover"])
  $("drop").addEventListener(event, (e) => {
    e.preventDefault();
    $("drop").classList.add("over");
  });
for (const event of ["dragleave", "drop"])
  $("drop").addEventListener(event, (e) => {
    e.preventDefault();
    $("drop").classList.remove("over");
  });
$("drop").addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));
$("newSource").addEventListener("click", () => $("upload").click());
form.elements.cfg.addEventListener(
  "input",
  () =>
    ($("cfgValue").textContent = Number(form.elements.cfg.value).toFixed(1)),
);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!config || !form.reportValidity()) return;
  if (!upload && !reuse)
    return message("Choose a reference clip or reuse a saved job.");
  const body = new FormData();
  body.set("settings", JSON.stringify(getSettings()));
  if (reuse) body.set("reuse", reuse);
  else body.set("video", upload);
  $("generate").disabled = true;
  $("status").textContent = reuse
    ? "Preparing a new job from saved intermediates…"
    : "Uploading reference clip…";
  try {
    const job = await api("/api/jobs", { method: "POST", body });
    upload = null;
    reuse = null;
    $("upload").value = "";
    await selectJob(job);
    await refresh();
  } catch (error) {
    message(error.message);
  } finally {
    $("generate").disabled = false;
  }
});

function clearView() {
  stopRecording();
  setPlaying(false);
  generation++;
  mesh?.clear();
  physics?.clear();
  scene = null;
  track = null;
  selection = null;
  loadedMesh = null;
  loadedPhysics = null;
  opening = false;
  displayedFrame = 0;
  desiredFrame = 0;
  $("meshEmpty").hidden = false;
  $("physicsEmpty").hidden = false;
  $("meshLoading").hidden = true;
  $("metrics").hidden = true;
  $("downloads").replaceChildren();
  $("personDialog").close();
  selectionShown = null;
  for (const name of ["play", "previous", "next", "record", "timeline"])
    $(name).disabled = true;
  $("timeline").value = 0;
  $("timeline").max = 0;
  $("time").textContent = "00:00.00 / 00:00.00";
  $("frame").textContent = "FRAME —";
  $("log").textContent = "";
  drawBoxes();
}

async function selectJob(job) {
  if (selected !== job.id) clearView();
  selected = job.id;
  upload = null;
  reuse = terminal.has(job.status) ? job.id : null;
  applySettings(job.settings);
  $("filename").textContent = job.filename;
  $("sourceLabel").textContent = "Saved clip";
  showVideo(fileURL(job.id, job.input));
  await updateJob(job);
  renderJobs();
}

function renderJobs() {
  $("jobCount").textContent = rows.length;
  $("jobs").replaceChildren(
    ...rows.map((job) => {
      const button = document.createElement("button");
      button.className = `job-card${selected === job.id ? " selected" : ""}`;
      const title = document.createElement("strong");
      title.textContent = job.filename;
      title.title = job.filename;
      const detail = document.createElement("small");
      const status = document.createElement("span");
      status.className = job.status;
      status.textContent = job.status;
      const date = document.createElement("span");
      date.textContent = new Date(job.created * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      detail.append(status, date);
      button.append(title, detail);
      button.addEventListener("click", () =>
        selectJob(job).catch((error) => message(error.message)),
      );
      return button;
    }),
  );
}

async function updateJob(job) {
  if (job.id !== selected) return;
  $("jobTitle").textContent = job.filename;
  $("status").textContent =
    job.error ||
    (terminal.has(job.status)
      ? `${job.status[0].toUpperCase()}${job.status.slice(1)} · source and intermediates saved`
      : job.progress.message || "Waiting in queue");
  $("jobActions").hidden = false;
  $("cancel").hidden = terminal.has(job.status);
  $("delete").disabled = !terminal.has(job.status);
  $("reuse").disabled = !terminal.has(job.status);
  const stages = [
    "detection",
    "camera",
    "feature_extract",
    "generation",
    "viewer",
    "physics",
  ];
  const current = stages.indexOf(job.progress.stage);
  for (const element of $("stages").children) {
    const index = stages.indexOf(element.dataset.stage);
    element.classList.toggle(
      "active",
      index === current && job.status === "running",
    );
    element.classList.toggle(
      "done",
      index < current ||
        (job.viewer && index <= 4) ||
        (job.physics_viewer && index === 5),
    );
    if (index === 5) element.style.opacity = job.settings.physics ? "1" : ".35";
  }
  $("progressBar").style.width =
    `${terminal.has(job.status) ? (job.viewer ? 100 : 0) : Math.max(0, Math.min(100, job.progress.percent || 0))}%`;
  $("downloads").replaceChildren(
    ...job.downloads.map((name) => {
      const link = document.createElement("a");
      link.href = fileURL(job.id, name) + "?download=1";
      const ext = name.split(".").pop().toUpperCase();
      link.textContent = `${name.includes("physics") ? "Physics" : name.includes("origin") ? "Original" : "Corrected"} · ${ext} ↓`;
      return link;
    }),
  );
  if (job.selection) showSelection(job.selection, job);
  else if (selection) {
    selection = null;
    $("personDialog").close();
    drawBoxes();
  }
  if (job.viewer && loadedMesh !== job.id && !opening && mesh) {
    opening = true;
    const version = generation;
    try {
      const base = fileURL(job.id, "viewer/");
      const data = await api(base + "scene.json");
      const boxes = await api(base + "track.json");
      if (version !== generation) return;
      await mesh.loadMesh(base, data);
      if (version !== generation) return;
      scene = data;
      track = boxes;
      loadedMesh = job.id;
      showVideo(fileURL(job.id, data.video));
      $("meshEmpty").hidden = true;
      $("timeline").max = data.frames - 1;
      for (const name of ["play", "previous", "next", "timeline"])
        $(name).disabled = false;
      $("record").disabled =
        !window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream;
      $("range").textContent =
        `SOURCE FRAMES ${data.video_start}–${data.video_start + data.frames - 1}`;
      await seek(0);
    } catch (error) {
      if (error.name !== "AbortError") message(error.message);
    } finally {
      if (version === generation) opening = false;
    }
  }
  if (job.physics_viewer && loadedPhysics !== job.id && physics) {
    const version = generation;
    const data = await api(fileURL(job.id, "viewer/physics.json"));
    if (version !== generation) return;
    physics.loadPhysics(data);
    if (scene) physics.setPhysicsTime(displayedFrame / scene.fps);
    loadedPhysics = job.id;
    $("physicsEmpty").hidden = true;
    $("metrics").hidden = false;
    $("metrics").replaceChildren(
      ...[
        [data.metrics.tracking_score.toFixed(3), "Tracking score"],
        [`${data.metrics.mpjpe_cm.toFixed(2)} cm`, "Mean joint error"],
        [`${(data.metrics.survival * 100).toFixed(1)}%`, "Survival"],
      ].map(([value, label]) => {
        const item = document.createElement("div"),
          strong = document.createElement("strong"),
          text = document.createElement("span");
        strong.textContent = value;
        text.textContent = label;
        item.append(strong, text);
        return item;
      }),
    );
  }
  if ($("logDetails").open) await loadLog();
}

async function refresh() {
  clearTimeout(pollTimer);
  try {
    rows = await api("/api/jobs");
    renderJobs();
    const job = rows.find((row) => row.id === selected);
    if (job) await updateJob(job);
  } catch (error) {
    message(error.message);
  } finally {
    pollTimer = setTimeout(
      refresh,
      document.hidden
        ? 10000
        : rows.some((row) => !terminal.has(row.status))
          ? 1200
          : 5000,
    );
  }
}

function showSelection(data, job) {
  selection = data;
  if (selectionShown !== job.id) {
    selectionShown = job.id;
    showVideo(fileURL(job.id, "native/input/input_30fps.mp4"));
    $("candidates").replaceChildren(
      ...data.candidates.map((candidate) => {
        const button = document.createElement("button");
        button.className = `candidate${candidate.index === data.default ? " suggested" : ""}`;
        const image = document.createElement(
          candidate.thumbnail ? "img" : "div",
        );
        if (candidate.thumbnail) {
          image.src = fileURL(job.id, candidate.thumbnail);
          image.alt = `Person ${candidate.track_id}`;
        } else {
          image.className = "placeholder";
          image.textContent = candidate.track_id;
        }
        const title = document.createElement("strong"),
          detail = document.createElement("small");
        title.textContent = `Person ${candidate.track_id}${candidate.index === data.default ? " · suggested" : ""}`;
        detail.textContent = `${candidate.num_frames} frames · ${(candidate.start / 30).toFixed(1)}–${(candidate.end / 30).toFixed(1)}s`;
        button.append(image, title, detail);
        button.addEventListener("pointerenter", () => {
          video.pause();
          video.currentTime = candidate.preview_frame / 30;
        });
        button.addEventListener("focus", () => {
          video.pause();
          video.currentTime = candidate.preview_frame / 30;
        });
        button.addEventListener("click", () => choosePerson(candidate.index));
        return button;
      }),
    );
    $("personDialog").show();
  }
  $("selectionCountdown").textContent =
    `Suggested track is selected in ${Math.max(0, Math.ceil(data.deadline - Date.now() / 1000))} seconds.`;
  drawBoxes();
}

async function choosePerson(index) {
  try {
    await api(`/api/jobs/${selected}/person`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index }),
    });
    selection = null;
    $("personDialog").close();
    drawBoxes();
    await refresh();
  } catch (error) {
    message(error.message);
  }
}
$("closePerson").addEventListener("click", () => $("personDialog").close());

function visibleBoxes() {
  const frame = Math.round(video.currentTime * 30);
  if (selection)
    return selection.candidates.flatMap((candidate) => {
      const index = candidate.frames.indexOf(frame);
      return index < 0
        ? []
        : [
            {
              box: candidate.boxes[index],
              id: candidate.index,
              label: `Person ${candidate.track_id}`,
            },
          ];
    });
  if (track && frame >= track.start && frame <= track.end)
    return [{ box: track.boxes[frame - track.start], label: "Selected track" }];
  return [];
}

function boxTransform() {
  const width = video.clientWidth,
    height = video.clientHeight;
  const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
  return {
    scale,
    x: (width - video.videoWidth * scale) / 2,
    y: (height - video.videoHeight * scale) / 2,
  };
}

function drawBoxes() {
  const canvas = $("boxes");
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  canvas.classList.toggle("selecting", !!selection);
  if (!video.videoWidth) return;
  const { scale, x, y } = boxTransform();
  boxContext.font = "11px system-ui";
  boxContext.lineWidth = 2;
  for (const item of visibleBoxes()) {
    const [left, top, right, bottom] = item.box;
    boxContext.strokeStyle = "#7ee4cd";
    boxContext.fillStyle = "#7ee4cd";
    boxContext.strokeRect(
      x + left * scale,
      y + top * scale,
      (right - left) * scale,
      (bottom - top) * scale,
    );
    boxContext.fillText(
      item.label,
      x + left * scale + 4,
      Math.max(12, y + top * scale - 5),
    );
  }
}

$("boxes").addEventListener("click", (event) => {
  if (!selection) return;
  const rect = $("boxes").getBoundingClientRect(),
    { scale, x, y } = boxTransform();
  const px = (event.clientX - rect.left - x) / scale,
    py = (event.clientY - rect.top - y) / scale;
  const hit = visibleBoxes()
    .filter(({ box: [l, t, r, b] }) => px >= l && px <= r && py >= t && py <= b)
    .sort(
      (a, b) =>
        (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) -
        (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]),
    )[0];
  if (hit) choosePerson(hit.id);
  else $("personDialog").show();
});

function setPlaying(value) {
  playing = value && !!scene;
  lastTick = performance.now();
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    playing ? "Pause motion" : "Play motion",
  );
  if (!playing) video.pause();
  else {
    if (displayedFrame >= scene.frames - 1) seek(0);
    video.playbackRate = Number($("speed").value);
    video.play().catch(() => {});
  }
}

async function seek(frame, syncVideo = true) {
  if (!scene) return;
  desiredFrame = Math.min(scene.frames - 1, Math.max(0, Math.round(frame)));
  if (drawing) return;
  drawing = true;
  const version = generation;
  try {
    do {
      const next = desiredFrame;
      $("meshLoading").hidden = mesh.frame === next;
      await mesh.setFrame(next);
      if (version !== generation) return;
      displayedFrame = next;
      physics?.setPhysicsTime(next / scene.fps);
      $("timeline").value = next;
      $("frame").textContent =
        `FRAME ${String(next + 1).padStart(4, "0")} / ${scene.frames}`;
      $("time").textContent =
        `${timecode(next / scene.fps)} / ${timecode((scene.frames - 1) / scene.fps)}`;
      const videoTime = (scene.video_start + next) / scene.fps;
      if (
        syncVideo &&
        Math.abs(video.currentTime - videoTime) > (playing ? 0.15 : 0.01)
      )
        video.currentTime = videoTime;
    } while (desiredFrame !== displayedFrame);
  } catch (error) {
    if (error.name !== "AbortError") {
      setPlaying(false);
      message(error.message);
    }
  } finally {
    drawing = false;
    $("meshLoading").hidden = true;
  }
}

function tick(now) {
  if (playing && scene && !document.hidden) {
    const interval = 1000 / (scene.fps * Number($("speed").value));
    if (now - lastTick >= interval && !drawing) {
      const advance = Math.max(1, Math.floor((now - lastTick) / interval));
      lastTick = now;
      let next = displayedFrame + advance;
      if (next >= scene.frames) {
        if (recorder) {
          stopRecording();
          setPlaying(false);
          next = scene.frames - 1;
        } else if ($("loop").checked) next = 0;
        else {
          setPlaying(false);
          next = scene.frames - 1;
        }
      }
      seek(next);
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$("play").addEventListener("click", () => setPlaying(!playing));
$("previous").addEventListener("click", () => {
  setPlaying(false);
  seek(displayedFrame - 1);
});
$("next").addEventListener("click", () => {
  setPlaying(false);
  seek(displayedFrame + 1);
});
$("timeline").addEventListener("input", (event) => {
  setPlaying(false);
  seek(Number(event.target.value));
});
$("speed").addEventListener("change", () => {
  video.playbackRate = Number($("speed").value);
  lastTick = performance.now();
});
video.addEventListener("seeked", () => {
  drawBoxes();
  if (scene && !drawing)
    seek(video.currentTime * scene.fps - scene.video_start, false);
});
video.addEventListener("timeupdate", drawBoxes);
video.addEventListener("play", () => {
  if (scene && !playing) setPlaying(true);
});
video.addEventListener("pause", () => {
  if (playing) setPlaying(false);
});
video.addEventListener("loadedmetadata", () => {
  drawBoxes();
  if (scene)
    video.currentTime = (scene.video_start + displayedFrame) / scene.fps;
});
new ResizeObserver(drawBoxes).observe(video);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    setPlaying(false);
    stopRecording();
  }
});
document.addEventListener("keydown", (event) => {
  if (
    event.code === "Space" &&
    !event.target.closest("input,select,button,textarea,dialog,summary") &&
    scene
  ) {
    event.preventDefault();
    setPlaying(!playing);
  }
});
for (const key of ["trail", "wire", "skeleton", "follow"])
  $(key).addEventListener("change", () => {
    const value = { [key]: $(key).checked };
    mesh?.configure(value);
    physics?.configure(value);
  });
$("resetView").addEventListener("click", () => {
  mesh?.reset();
  physics?.reset();
});

$("cancel").addEventListener("click", async () => {
  try {
    await api(`/api/jobs/${selected}/cancel`, { method: "POST" });
    await refresh();
  } catch (error) {
    message(error.message);
  }
});
$("delete").addEventListener("click", async () => {
  if (!confirm("Delete this job, its source video and generated files?"))
    return;
  try {
    await api(`/api/jobs/${selected}`, { method: "DELETE" });
    selected = null;
    reuse = null;
    clearView();
    video.removeAttribute("src");
    video.load();
    video.parentElement.classList.remove("visible");
    $("drop").hidden = false;
    $("newSource").hidden = true;
    $("filename").textContent = "Choose another clip.";
    $("jobTitle").textContent = "Ready for a reference clip";
    $("status").textContent = "Job deleted.";
    $("jobActions").hidden = true;
    await refresh();
  } catch (error) {
    message(error.message);
  }
});
$("reuse").addEventListener("click", () => {
  const job = rows.find((row) => row.id === selected);
  if (!job || !terminal.has(job.status)) return;
  reuse = job.id;
  upload = null;
  applySettings(job.settings);
  $("generate").firstChild.textContent = "Reconstruct again ";
  message(
    "Source and compatible intermediates selected. Adjust settings, then reconstruct again.",
  );
  form.scrollIntoView({
    block: "nearest",
    behavior: matchMedia("(prefers-reduced-motion:reduce)").matches
      ? "instant"
      : "smooth",
  });
});
async function loadLog() {
  if (!selected) return;
  const id = selected,
    result = await fetch(`/api/jobs/${id}/log`);
  const text = await result.text();
  if (selected === id) $("log").textContent = text;
}
$("logDetails").addEventListener("toggle", () => {
  if ($("logDetails").open) loadLog().catch((error) => message(error.message));
});

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob),
    anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function stopRecording() {
  if (!recorder) return;
  if (recorder.state !== "inactive") recorder.stop();
  cancelAnimationFrame(recordRaf);
  recordingStream?.getTracks().forEach((track) => track.stop());
  recorder = null;
  recordingStream = null;
  recordingCanvas = null;
  $("record").textContent = "Record WebM";
}
$("record").addEventListener("click", async () => {
  if (recorder) {
    stopRecording();
    return;
  }
  if (!scene || !window.MediaRecorder) return;
  setPlaying(false);
  await seek(0);
  const mimeType = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ].find((type) => MediaRecorder.isTypeSupported(type));
  if (!mimeType)
    return message(
      "This browser cannot record WebM. NPZ and FBX exports remain available.",
    );
  try {
    recordingCanvas = document.createElement("canvas");
    recordingCanvas.width = 1280;
    recordingCanvas.height = 720;
    const ctx = recordingCanvas.getContext("2d");
    const canvas = recordingCanvas;
    recordingStream = canvas.captureStream(30);
    recordChunks = [];
    const current = new MediaRecorder(recordingStream, {
      mimeType,
      videoBitsPerSecond: 8000000,
    });
    const chunks = recordChunks;
    current.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    current.onstop = () =>
      downloadBlob(
        new Blob(chunks, { type: mimeType }),
        "flowhmr-comparison.webm",
      );
    current.onerror = () => {
      message("Browser recording stopped");
      stopRecording();
    };
    recorder = current;
    const capture = () => {
      if (!recorder) return;
      ctx.fillStyle = "#101419";
      ctx.fillRect(0, 0, 1280, 720);
      const put = (image, x, y, w, h, iw, ih) => {
        const scale = Math.min(w / iw, h / ih);
        if (Number.isFinite(scale) && scale > 0)
          ctx.drawImage(
            image,
            x + (w - iw * scale) / 2,
            y + (h - ih * scale) / 2,
            iw * scale,
            ih * scale,
          );
      };
      if (video.readyState >= 2)
        put(video, 320, 0, 640, 260, video.videoWidth, video.videoHeight);
      for (const [view, x] of [
        [mesh, 0],
        [physics, 640],
      ]) {
        const image = view.renderer.domElement;
        put(image, x, 290, 640, 430, image.width, image.height);
      }
      ctx.font = "16px system-ui";
      ctx.fillStyle = "#b9a6f5";
      ctx.fillText("Reconstruction", 24, 285);
      ctx.fillStyle = "#6cdac7";
      ctx.fillText(
        loadedPhysics ? "Physics tracking" : "Physics tracking · unavailable",
        664,
        285,
      );
      recordRaf = requestAnimationFrame(capture);
    };
    capture();
    current.start(1000);
    $("record").textContent = "Stop recording";
    setPlaying(true);
  } catch (error) {
    stopRecording();
    message(error.message);
  }
});

$("theme").addEventListener("click", () => {
  const theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("flowhmr-theme", theme);
  } catch {
    /* Browser storage may be disabled. */
  }
});
try {
  document.documentElement.dataset.theme =
    localStorage.getItem("flowhmr-theme") || "dark";
} catch {
  /* Keep the default theme. */
}

async function init() {
  try {
    mesh = new MotionView($("meshView"), 0xb9a6f5);
    physics = new MotionView($("physicsView"), 0x6cdac7);
  } catch (error) {
    message(
      `3D viewer: ${error.message}. Generated files remain downloadable.`,
    );
  }
  config = await api("/api/config");
  applySettings(config.defaults);
  form.elements.physics.disabled = !config.physics;
  $("physicsNote").textContent = config.physics
    ? "PHC ready · runs after reconstruction releases"
    : "Optional: run setup-physics.sh in the prepared Isaac Gym environment";
  document.querySelector(".status-dot").style.opacity = config.physics
    ? "1"
    : ".25";
  if (!config.camera)
    message(
      "VGGT checkpoint is missing. Select Static camera or download the flowhmr bundle.",
    );
  await refresh();
  if (rows.length) await selectJob(rows[0]);
}
init().catch((error) => message(error.message));
