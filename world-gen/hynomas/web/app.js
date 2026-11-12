import { CameraRecorder } from "./camera.js";
import { Playback } from "./playback.js";
import { structuredPrompt } from "./prompt.js";
const $ = (id) => document.getElementById(id),
  pause = (ms) => new Promise((r) => setTimeout(r, ms));
const clock = (s) => {
  s = Math.max(0, Number.isFinite(s) ? s : 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};
const formIds = [
  "world-name",
  "perspective",
  "style",
  "environment",
  "subject",
  "steps",
  "seed",
  "translation",
  "rotation",
  "structured",
];
let session = crypto.randomUUID(),
  turns = [],
  reference = null,
  busy = false,
  initializing = true,
  online = false,
  modelStatus = null,
  requestEpoch = 0,
  referenceEpoch = 0,
  pending = null;
let camera = null;
const playback = new Playback($("player"));
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
async function api(path, body, method) {
  const response = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const data = await response.json();
  if (!response.ok)
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data),
    );
  return data;
}
function download(url, name) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name || "";
  a.click();
}
function jsonFile(value, name) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  download(url, name);
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function form() {
  return Object.fromEntries(formIds.map((id) => [id, $(id).value]));
}
function persist() {
  try {
    localStorage.setItem(
      "hynomas-studio",
      JSON.stringify({ session, turns, form: form(), reference, pending }),
    );
  } catch {
    message(
      "Browser storage is full; save a world snapshot to preserve the session.",
      true,
    );
  }
}
function controls() {
  const locked = busy || initializing;
  const has = turns.length > 0;
  const resident = online && modelStatus?.worker.active_session_id === session;
  for (const id of [
    "perspective",
    "style",
    "environment",
    "subject",
    "reference",
    "clear-reference",
  ])
    $(id).disabled = locked || has;
  $("generate").disabled =
    locked ||
    pending ||
    !online ||
    (has && !resident) ||
    camera?.recording ||
    Boolean(reference && reference.caption_status !== "ready");
  $("generate").textContent = has ? "Continue world →" : "Create world →";
  for (const id of ["rollback", "snapshot", "release"])
    $(id).disabled = locked || !!pending || !resident;
  for (const id of ["download", "gallery-save", "pe-export", "share"])
    $(id).disabled = locked || !has;
  $("load").disabled = locked || online || modelStatus?.starting;
  $("adapter").disabled = locked || online || modelStatus?.starting;
  $("prepare").disabled =
    locked || online || modelStatus?.starting || $("adapter").value === "base";
  $("new-world").disabled = locked || !!pending;
  $("use-camera").disabled = locked || !camera?.path;
  $("direction").disabled = locked || $("use-camera").checked;
  $("direction").placeholder = $("use-camera").checked
    ? "Camera motion controls this turn"
    : "What happens next?";
  $("record").disabled = locked;
  $("reset-camera").disabled = locked;
  $("camera-import").disabled = locked;
  $("structured").disabled = locked;
  $("resume").hidden = !pending;
  $("resume").disabled = locked;
}
function setBusy(value, phase = "") {
  busy = value;
  $("activity").hidden = !value;
  $("phase").textContent = phase;
  controls();
}
function safe(fn, duringInit = false) {
  return async () => {
    try {
      if (initializing && !duringInit) throw Error("Restoring studio state");
      await fn();
    } catch (error) {
      message(error.message, true);
    } finally {
      controls();
    }
  };
}
function drawCamera(path) {
  $("use-camera").checked = Boolean(path);
  $("camera-summary").textContent = path
    ? `${(path.durationMs / 1000).toFixed(2)}s · ${path.frames.length} poses`
    : "No path recorded";
  $("record").textContent = "● Record";
  if (!path) {
    $("camera-line").setAttribute("d", "");
    controls();
    return;
  }
  const xs = path.frames.map((f) => f.position.x),
    zs = path.frames.map((f) => f.position.z),
    minX = Math.min(...xs),
    minZ = Math.min(...zs),
    scale = Math.min(
      210 / Math.max(1, Math.max(...xs) - minX),
      90 / Math.max(1, Math.max(...zs) - minZ),
    );
  const points = path.frames.map((f) => [
    15 + (f.position.x - minX) * scale,
    15 + (f.position.z - minZ) * scale,
  ]);
  $("camera-line").setAttribute(
    "d",
    points.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" "),
  );
  $("camera-dot").setAttribute("cx", points.at(-1)[0]);
  $("camera-dot").setAttribute("cy", points.at(-1)[1]);
  controls();
}
try {
  camera = new CameraRecorder($("camera-view"), drawCamera, (t) => {
    $("record-time").textContent = clock(t / 1000);
  });
} catch (error) {
  $("camera-view").textContent = `Camera preview unavailable: ${error.message}`;
}
function renderTurns() {
  const timeline = $("timeline"),
    history = $("history");
  timeline.replaceChildren();
  history.replaceChildren();
  turns.forEach((turn, i) => {
    const b = document.createElement("button");
    b.textContent = `T${i + 1} · ${turn.camera ? "CAMERA" : "TEXT"}`;
    const small = document.createElement("small");
    small.textContent = `${clock(turn.start_seconds)} — ${clock(turn.end_seconds)}`;
    b.append(small);
    b.onclick = () => {
      $("player").currentTime = turn.start_seconds;
      $("player")
        .play()
        .catch(() => {});
    };
    timeline.append(b);
    const article = document.createElement("article"),
      title = document.createElement("small"),
      text = document.createElement("p"),
      link = document.createElement("a");
    title.textContent = `T${i + 1} · ${clock(turn.end_seconds)}`;
    text.textContent = turn.prompt;
    link.textContent = "MP4 through this turn";
    link.href = turn.result.video_url;
    link.download = "";
    article.append(title, text, link);
    history.append(article);
  });
  $("turn-count").textContent = `T${turns.length}`;
  $("empty").hidden = turns.length > 0;
  controls();
}
function overlay() {
  return {
    version: 1,
    theme: "dark",
    turns: turns.map((turn, i) => ({
      number: i + 1,
      prompt: turn.prompt,
      start_seconds: turn.start_seconds,
      end_seconds: turn.end_seconds,
      ...(turn.camera
        ? {
            camera: {
              name: turn.camera.name,
              duration_ms: turn.camera.durationMs,
              sample_rate: turn.camera.sampleRate,
              visualization_rotation_scale: turn.rotation,
              frames: turn.camera.frames,
              key_events: turn.camera.keyEvents || [],
            },
          }
        : {}),
    })),
  };
}
async function status(expectedSession) {
  modelStatus = await api("/studio/status");
  for (
    let attempt = 0;
    expectedSession !== undefined &&
    modelStatus.worker.online &&
    (modelStatus.worker.active_session_id || null) !== expectedSession &&
    attempt < 8;
    attempt++
  ) {
    await pause(400);
    modelStatus = await api("/studio/status");
  }
  online = !!modelStatus.worker.online;
  const label = modelStatus.starting
    ? "Loading"
    : online
      ? modelStatus.worker.status
      : "Stopped";
  $("model-state").textContent = label;
  $("model-state").classList.toggle("ready", online);
  $("enhancer").textContent = `Prompt enhancer: ${modelStatus.enhancer}`;
  $("preparation").textContent =
    modelStatus.preparation.error ||
    `Adapter preparation: ${modelStatus.preparation.status}`;
  if (online || modelStatus.starting) $("adapter").value = modelStatus.adapter;
  controls();
  return modelStatus;
}
async function waitJob(job, epoch = requestEpoch) {
  let failures = 0;
  while (true) {
    if (epoch !== requestEpoch) throw Error("Request stopped");
    $("phase").textContent =
      job.status + (job.queue_position ? ` · queue ${job.queue_position}` : "");
    if (job.pe_prompt || job.pe_content_preview)
      $("pe-output").textContent = job.pe_prompt || job.pe_content_preview;
    if (job.status === "succeeded") return job.result;
    if (["failed", "cancelled"].includes(job.status)) {
      if (pending?.job === job.id) {
        pending = null;
        persist();
      }
      throw Error(job.error || "Generation stopped");
    }
    await pause(1000);
    if (epoch !== requestEpoch) throw Error("Request stopped");
    try {
      job = await api(`/studio/jobs/${encodeURIComponent(job.id)}`);
      failures = 0;
    } catch (error) {
      if (++failures >= 4) throw error;
    }
  }
}
async function snapshots() {
  const rows = await api("/studio/snapshots");
  $("snapshots").replaceChildren();
  for (const row of rows) {
    const button = document.createElement("button");
    button.textContent = row.title;
    const small = document.createElement("small");
    small.textContent = `${row.identity.adapter} · restore`;
    button.append(small);
    button.onclick = safe(async () => {
      if (busy) throw Error("Wait for the current operation");
      if (
        !confirm("Restore this saved world? Release the current world first.")
      )
        return;
      setBusy(true, "Restoring");
      try {
        const result = await waitJob(
          await api(`/studio/restore/${row.id}`, {}),
        );
        session = row.session_id;
        turns = result.studio?.turns || fromSession(result.session);
        reference = result.studio?.reference || null;
        for (const id of formIds)
          if (result.studio?.form?.[id] !== undefined)
            $(id).value = result.studio.form[id];
        pending = null;
        $("world-name").value = row.title;
        renderTurns();
        if (turns.length) await playback.show(turns.at(-1).result.video_url);
        persist();
        await status(session);
        message("World restored");
      } finally {
        setBusy(false);
      }
    });
    $("snapshots").append(button);
  }
}
function fromSession(record) {
  const results = record?.result_history || [];
  return results.map((result, i) => {
    const prompt = (record.dit_prompts || [])[i] || "";
    return {
      prompt,
      start_seconds: i ? results[i - 1].frames / results[i - 1].fps : 0,
      end_seconds: result.frames / result.fps,
      result,
    };
  });
}
async function release() {
  if (online && modelStatus.worker.active_session_id === session)
    await waitJob(
      await api(`/v1/sessions/${encodeURIComponent(session)}/release`, {}),
    );
  await status(null);
}
async function newWorld() {
  if (
    turns.length &&
    !confirm(
      "Release this world and start another? Saved snapshots and videos remain available.",
    )
  )
    return;
  setBusy(true, "Releasing");
  try {
    await release();
    session = crypto.randomUUID();
    turns = [];
    reference = null;
    pending = null;
    clearReference();
    playback.reset();
    $("pe-output").textContent = "";
    renderTurns();
    persist();
    message("New world");
  } finally {
    setBusy(false);
  }
}
function clearReference() {
  referenceEpoch++;
  reference = null;
  $("reference").value = "";
  $("reference-preview").hidden = true;
  $("reference-preview").removeAttribute("src");
  $("clear-reference").hidden = true;
  $("reference-status").textContent = "PNG / JPEG / WebP · up to 10 MB";
  controls();
}
async function caption(file) {
  if (!file) return;
  const epoch = ++referenceEpoch;
  if (file.size > 10 * 1024 * 1024)
    throw Error("Reference image must be at most 10 MB");
  setBusy(true, "Captioning image");
  try {
    const response = await fetch(
      `/v1/reference-images?perspective=${encodeURIComponent($("perspective").value)}`,
      { method: "POST", headers: { "Content-Type": file.type }, body: file },
    );
    const initial = await response.json();
    if (!response.ok) throw Error(initial.detail || "Image upload failed");
    if (epoch !== referenceEpoch) return;
    reference = initial;
    $("reference-preview").src = initial.url;
    $("reference-preview").hidden = false;
    $("clear-reference").hidden = false;
    while (reference.caption_status === "captioning") {
      if (epoch !== referenceEpoch) return;
      $("reference-status").textContent = "Captioning with local Nomas-PE…";
      await pause(1500);
      reference = {
        ...initial,
        ...(await api(`/v1/reference-images/${initial.id}`)),
      };
    }
    if (reference.caption_status !== "ready")
      throw Error(reference.caption_error || "Image caption failed");
    $("subject").value = reference.subject_description;
    $("environment").value = reference.environment_description;
    $("reference-status").textContent = "Reference ready · T0 seeds the world";
    persist();
  } finally {
    if (epoch === referenceEpoch) setBusy(false);
  }
}
async function generate() {
  if (camera?.recording) throw Error("Stop camera recording first");
  if (pending) throw Error("Reconnect to the pending turn or stop it first");
  const structured = structuredPrompt($("structured").value);
  const first = !turns.length,
    path = $("use-camera").checked ? camera?.path : null,
    prompt = path ? "" : $("direction").value.trim();
  if (structured && !first)
    throw Error("Create a new world for standalone structured JSON");
  if (structured && (reference || path))
    throw Error(
      "Use structured JSON on a text turn without a reference image or camera path",
    );
  if (first && reference && path)
    throw Error("Create the image reference turn before adding camera motion");
  if (
    first &&
    !structured &&
    !reference &&
    (!$("environment").value.trim() || !$("subject").value.trim())
  )
    throw Error("Describe the environment and subject first");
  if (!first && !prompt && !path && !structured)
    throw Error("Enter a direction or record a camera path");
  const seed = $("seed").value;
  if (
    seed !== "" &&
    (!Number.isInteger(Number(seed)) ||
      Number(seed) < 0 ||
      Number(seed) > 2147483647)
  )
    throw Error("Prompt seed must be an integer from 0 to 2147483647");
  const translation = Number($("translation").value),
    rotation = Number($("rotation").value);
  if (
    !Number.isFinite(translation) ||
    !Number.isFinite(rotation) ||
    translation <= 0 ||
    rotation <= 0
  )
    throw Error("Camera scales must be positive numbers");
  const pe = {
    session_id: session,
    prompt_enhancer_model: "Nomas-PE",
    prompt_enhancer_reasoning_effort: "none",
    ...(seed !== "" ? { prompt_enhancer_seed: Number(seed) } : {}),
    ...(first
      ? {
          perspective: $("perspective").value,
          ...(!reference
            ? {
                subject_description: $("subject").value.trim(),
                environment_description: `${$("environment").value.trim()}\nVisual style: ${$("style").value}.${prompt ? `\nInitial direction: ${prompt}` : ""}`,
                style: $("style").value,
              }
            : {}),
        }
      : path
        ? {}
        : { action: prompt }),
    ...(path
      ? { camera_controlled: true, camera_duration_ms: path.durationMs }
      : {}),
  };
  const body = {
    ...(structured || { pe_input: pe }),
    session_id: session,
    reset_session: first,
    num_inference_steps: Number($("steps").value),
    segment_has_camera: Array(
      structured ? structured.segment_boundaries.length - 1 : 1,
    ).fill(!!path),
    user_prompts: Array(
      structured ? structured.segment_boundaries.length - 1 : 1,
    ).fill(path ? "" : prompt),
    ...(path
      ? {
          camera_trajectory: JSON.stringify(path),
          trajectory_translation_scale: translation,
          trajectory_rotation_scale: rotation,
        }
      : {}),
    ...(first && reference ? { reference_image_id: reference.id } : {}),
  };
  const epoch = requestEpoch;
  setBusy(true, "Preparing");
  message("");
  try {
    const job = await api("/v1/generate", body);
    pending = {
      job: job.id,
      prompt: path
        ? "Camera motion"
        : structured
          ? $("structured").value
          : prompt || (first ? $("environment").value : "Camera motion"),
      camera: path ? structuredClone(path) : null,
      rotation,
    };
    persist();
    const result = await waitJob(job, epoch);
    await completed(result);
    $("direction").value = "";
    $("structured").value = "";
    persist();
    message("Turn complete");
  } finally {
    if (epoch === requestEpoch) setBusy(false);
  }
}
async function completed(result) {
  const turn = {
    ...pending,
    start_seconds:
      result.playback_start_seconds ?? (turns.at(-1)?.end_seconds || 0),
    end_seconds: result.frames / result.fps,
    result,
  };
  delete turn.job;
  turns.push(turn);
  pending = null;
  renderTurns();
  persist();
  await playback.append(result);
  await status(result.prompt_enhanced ? session : undefined);
}
$("resume").onclick = safe(async () => {
  if (!pending) return;
  const epoch = requestEpoch;
  setBusy(true, "Reconnecting");
  try {
    await completed(
      await waitJob(await api(`/studio/jobs/${pending.job}`), epoch),
    );
  } finally {
    if (epoch === requestEpoch) setBusy(false);
  }
});
async function gallery() {
  const data = await api("/v1/gallery");
  $("gallery").replaceChildren();
  for (const item of data.items || []) {
    const card = document.createElement("article"),
      video = document.createElement("video"),
      title = document.createElement("p"),
      actions = document.createElement("div"),
      remove = document.createElement("button"),
      link = document.createElement("a");
    video.src = "/" + item.video;
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    title.textContent = item.title;
    actions.className = "actions";
    remove.textContent = "Remove";
    remove.onclick = safe(async () => {
      if (
        confirm(
          "Remove this gallery entry? The generated original remains in outputs.",
        )
      ) {
        await api(
          `/v1/gallery/${encodeURIComponent(item.id)}`,
          undefined,
          "DELETE",
        );
        await gallery();
      }
    });
    link.textContent = "MP4 ↓";
    link.href = video.src;
    link.download = "";
    actions.append(link, remove);
    card.append(video, title, actions);
    $("gallery").append(card);
  }
}
$("record").onclick = safe(() => {
  if (!camera) throw Error("WebGL camera preview unavailable");
  if (camera.recording) camera.stop();
  else {
    camera.start();
    $("record").textContent = "■ Finish";
    controls();
  }
});
$("reset-camera").onclick = () => camera?.reset();
$("camera-import").onchange = safe(async () => {
  const file = $("camera-import").files[0];
  if (file) {
    if (file.size > 10 * 1024 * 1024) throw Error("Path is larger than 10 MB");
    camera?.import(JSON.parse(await file.text()));
  }
});
$("camera-export").onclick = safe(() => {
  if (!camera?.path) throw Error("Record or import a camera path first");
  jsonFile(camera.path, "camera-trajectory.json");
});
$("load").onclick = safe(async () => {
  await api("/studio/load", { adapter: $("adapter").value });
  let state = await status();
  while (state.starting) {
    await pause(1500);
    state = await status();
  }
  if (!state.worker.online)
    throw Error("Model stopped during loading; see Runtime log");
  message("Model ready");
});
$("stop").onclick = safe(async () => {
  requestEpoch++;
  referenceEpoch++;
  setBusy(true, "Stopping");
  try {
    await api("/studio/stop", {});
    pending = null;
    await status();
    message(
      "Model and prompt enhancer released. Restore a snapshot or start a new world.",
    );
  } finally {
    setBusy(false);
    persist();
  }
}, true);
$("prepare").onclick = safe(async () => {
  setBusy(true, "Preparing INT8 adapter");
  try {
    await api("/studio/prepare", { adapter: $("adapter").value });
    let s = await status();
    while (s.preparation.status === "running") {
      await pause(2000);
      s = await status();
    }
    if (s.preparation.error) throw Error(s.preparation.error);
    message("INT8 adapter cached");
  } finally {
    setBusy(false);
  }
});
$("adapter").onchange = controls;
$("use-camera").onchange = controls;
$("generate").onclick = safe(generate);
$("new-world").onclick = safe(newWorld);
$("reference").onchange = safe(() => caption($("reference").files[0]));
$("clear-reference").onclick = () => {
  clearReference();
  persist();
};
$("rollback").onclick = safe(async () => {
  setBusy(true, "Rolling back");
  try {
    const result = await waitJob(
      await api(`/v1/sessions/${session}/rollback`, {}),
    );
    turns.pop();
    pending = null;
    renderTurns();
    persist();
    if (result.video_url) await playback.show(result.video_url);
    else {
      playback.reset();
      persist();
    }
    await status();
    message("Last turn removed from the live world");
  } finally {
    setBusy(false);
  }
});
$("snapshot").onclick = safe(async () => {
  setBusy(true, "Saving world state");
  try {
    await waitJob(
      await api("/studio/snapshot", {
        session_id: session,
        title: $("world-name").value,
        studio: { turns, form: form(), reference },
      }),
    );
    await snapshots();
    message("World snapshot saved");
  } finally {
    setBusy(false);
  }
});
$("release").onclick = safe(async () => {
  setBusy(true, "Releasing world");
  try {
    await release();
    online = false;
    message("World released; create a new world or restore a snapshot.");
    await status();
  } finally {
    setBusy(false);
  }
});
$("download").onclick = safe(async () => {
  setBusy(true, "Preparing MP4");
  try {
    const result = await api("/v1/downloads", {
      video_url: turns.at(-1).result.video_url,
      include_overlay: $("overlay").checked,
      overlay: overlay(),
    });
    download(result.download_path, result.filename);
  } finally {
    setBusy(false);
  }
});
$("gallery-save").onclick = safe(async () => {
  await api("/v1/gallery", {
    video_url: turns.at(-1).result.video_url,
    title: $("world-name").value,
    perspective: $("perspective").value,
    subject: $("subject").value,
    attributes: $("environment").value,
    direction: turns.at(-1).prompt,
    duration_seconds: turns.at(-1).end_seconds,
    prompt_timeline: overlay().turns,
  });
  await gallery();
  message("Saved to local gallery");
});
$("gallery-refresh").onclick = safe(gallery);
$("pe-export").onclick = safe(async () =>
  jsonFile(
    await api(`/v1/sessions/${session}/pe-export`),
    "world-prompts.json",
  ),
);
$("share").onclick = safe(async () => {
  const result = await api("/v1/shares", {
    video_url: turns.at(-1).result.video_url,
    overlay: overlay(),
  });
  const url = new URL(result.share_path, location.href).href;
  try {
    await navigator.clipboard.writeText(url);
    message(
      "Local share link copied; this address stays on your studio server.",
    );
  } catch {
    message(url);
  }
});
for (const id of formIds) $(id).addEventListener("change", persist);
$("player").addEventListener("timeupdate", () => {
  const time = $("player").currentTime;
  $("clock").textContent = `${clock(time)} / ${clock($("player").duration)}`;
  [...$("timeline").children].forEach((el, i) =>
    el.classList.toggle(
      "active",
      time >= turns[i].start_seconds && time < turns[i].end_seconds,
    ),
  );
});
window.addEventListener("pagehide", persist);
async function restoreUI() {
  const epoch = requestEpoch;
  try {
    const saved = JSON.parse(localStorage.getItem("hynomas-studio") || "null");
    if (saved) {
      session = saved.session || session;
      turns = saved.turns || [];
      reference = saved.reference || null;
      pending = saved.pending || null;
      for (const id of formIds)
        if (saved.form?.[id] !== undefined) $(id).value = saved.form[id];
    }
  } catch {
    message("Saved browser state could not be read", true);
  }
  renderTurns();
  await status();
  await snapshots();
  await gallery();
  if (epoch !== requestEpoch) return;
  if (reference) {
    $("reference-preview").src = reference.url;
    $("reference-preview").hidden = false;
    $("clear-reference").hidden = false;
    $("reference-status").textContent = reference.caption_status;
  }
  if (pending) {
    setBusy(true, "Reconnecting");
    try {
      await completed(
        await waitJob(await api(`/studio/jobs/${pending.job}`), epoch),
      );
    } finally {
      if (epoch === requestEpoch) setBusy(false);
    }
  } else if (turns.length) await playback.show(turns.at(-1).result.video_url);
  if (turns.length && modelStatus.worker.active_session_id !== session)
    message(
      "Saved playback is available. Restore a world snapshot or start a new world to continue.",
      true,
    );
}
async function init() {
  controls();
  try {
    await restoreUI();
  } finally {
    initializing = false;
    controls();
  }
}
safe(init, true)();
