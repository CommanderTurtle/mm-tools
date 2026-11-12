const $ = (id) => document.getElementById(id);
const form = $("form"),
  player = $("player"),
  vocal = new Audio();
let config,
  current = null,
  active = null,
  file = null,
  reuse = null,
  epoch = 0,
  timer;
let objectURL,
  recorder,
  recordingStream,
  recording = false,
  busy = false,
  mediaKey = "",
  rowKey = "";
let cues = [],
  cueNodes = [],
  activeCue = -1,
  activeWord = null,
  peaks = [];
const notes = {
  karaoke:
    "CPU vocal separation runs first. Lyrics are aligned to the separated vocal track; all playback tracks share the original timeline.",
  ask: "Ask about this recording. Continue conversation keeps the selected session’s earlier turns; reprocess starts a fresh conversation.",
  nearfield:
    "Experimental native recipe. The nearest speaker instruction depends on the recording and model response.",
  translate:
    "Edit the target language in the instruction. Word alignment describes source speech, so translated text is exported without forced source timings.",
  language:
    "Language identification returns the native model response. This is a single request, not continuous detection.",
};
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function time(t) {
  if (!Number.isFinite(t)) t = 0;
  return `${String(Math.floor((t || 0) / 60)).padStart(2, "0")}:${String(Math.floor((t || 0) % 60)).padStart(2, "0")}`;
}
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function error(e) {
  $("status").textContent = e.message || String(e);
}
function settings() {
  const out = {};
  for (const [key, val] of Object.entries(config.defaults)) {
    const field = form.elements.namedItem(key);
    out[key] =
      key === "source_url"
        ? $("source-url").value.trim()
        : typeof val === "boolean"
          ? field.checked
          : typeof val === "number"
            ? Number(field.value)
            : field.value;
  }
  return out;
}
function applySettings(cfg) {
  for (const [key, value] of Object.entries(cfg)) {
    const field = form.elements.namedItem(key);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = value;
    else field.value = value;
  }
  $("source-url").value = "";
  taskState();
}
function taskState() {
  const task = $("task").value;
  const noTiming = ["ask", "language", "translate"].includes(task);
  $("timestamps").disabled = noTiming;
  if (noTiming) $("timestamps").checked = false;
  $("thinking").disabled = task !== "ask";
  if (task !== "ask") $("thinking").checked = false;
  $("continue-label").hidden = task !== "ask";
  $("continue-chat").disabled =
    !reuse ||
    current?.settings.task !== "ask" ||
    !current?.conversation?.length;
  if ($("continue-chat").disabled) $("continue-chat").checked = false;
  $("run").replaceChildren(
    document.createTextNode(
      task === "ask" ? "Ask audio " : "Run transcription ",
    ),
    el("span", "↗"),
  );
  $("recipe-note").textContent =
    notes[task] ||
    "Native punctuation, capitalization and speaker tags are preserved. Timing and subtitle exports use the model’s returned CTC alignment.";
}
function locked(value) {
  busy = value;
  $("run").disabled = value || recording;
  $("upload").disabled = value;
  $("record").disabled = value;
  $("new").disabled = value;
  $("source-url").disabled = value;
  $("refresh").disabled = value;
  $("history")
    .querySelectorAll("button")
    .forEach((button) => {
      button.disabled = value || recording;
    });
}
function source(path, video = false, position = 0) {
  player.pause();
  vocal.pause();
  player.src = path;
  player.style.visibility = video ? "visible" : "hidden";
  $("wave").hidden = video;
  $("empty-media").hidden = true;
  player.onloadedmetadata = () => {
    player.currentTime = Math.min(position, player.duration || 0);
    tick();
  };
}
function clearMedia() {
  player.pause();
  vocal.pause();
  player.removeAttribute("src");
  player.load();
  vocal.removeAttribute("src");
  vocal.load();
  player.onloadedmetadata = null;
  mediaKey = "";
  peaks = [];
  $("wave").hidden = false;
  $("empty-media").hidden = false;
  $("subtitle").replaceChildren();
  $("mix-panel").hidden = true;
  draw();
}
function empty() {
  rowKey = "";
  cues = [];
  cueNodes = [];
  activeCue = -1;
  activeWord = null;
  $("warnings").textContent = "";
  $("exports").replaceChildren();
  $("subtitle").replaceChildren();
  $("reasoning-panel").hidden = true;
  $("speaker").replaceChildren(new Option("All speakers", ""));
  $("search").value = "";
  $("language-panel").hidden = true;
  $("language-filter").replaceChildren(new Option("All languages", ""));
  const box = el("div", null, "empty-result");
  box.append(
    el("span", "01 / INPUT"),
    el("h2", "Start with a recording."),
    el(
      "p",
      "Choose a recipe, adjust the instruction, then listen with the transcript.",
    ),
  );
  $("transcript").replaceChildren(box);
}
function reset() {
  if (!config) return;
  epoch++;
  current = null;
  reuse = null;
  file = null;
  rowKey = "";
  cues = [];
  cueNodes = [];
  activeCue = -1;
  activeWord = null;
  clearMedia();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  $("upload").value = "";
  $("source-url").value = "";
  $("input-name").textContent = "Drop audio or video here";
  $("title").textContent = "Untitled recording";
  $("duration").textContent = "—";
  $("warnings").textContent = "";
  $("exports").replaceChildren();
  $("reasoning-panel").hidden = true;
  $("remove").disabled = true;
  $("log-text").hidden = true;
  empty();
  applySettings(config.defaults);
  $("status").textContent = active ? "Another session is processing" : "Ready";
  history().catch(error);
}
function setFile(next) {
  if (!next || busy || active || recording) return;
  reset();
  file = next;
  objectURL = URL.createObjectURL(next);
  $("input-name").textContent = next.name;
  $("title").textContent = next.name;
  source(objectURL, next.type.startsWith("video/"));
}
$("upload").addEventListener("change", (e) => setFile(e.target.files[0]));
$("stage").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("stage").classList.add("drag");
});
$("stage").addEventListener("dragleave", () =>
  $("stage").classList.remove("drag"),
);
$("stage").addEventListener("drop", (e) => {
  e.preventDefault();
  $("stage").classList.remove("drag");
  setFile(e.dataTransfer.files[0]);
});
$("new").onclick = reset;
$("task").onchange = () => {
  $("prompt").value = config.prompts[$("task").value];
  $("timestamps").checked = !["ask", "translate", "language"].includes(
    $("task").value,
  );
  taskState();
};
$("thinking").onchange = () => {
  if ($("thinking").checked)
    form.elements.max_tokens.value = Math.max(
      20480,
      Number(form.elements.max_tokens.value),
    );
};
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy || active || recording) return;
  const token = ++epoch;
  locked(true);
  try {
    const data = new FormData();
    const cfg = settings();
    if (cfg.source_url) cfg.continue_chat = false;
    else if (file) data.set("media", file);
    else if (reuse) data.set("reuse", reuse);
    data.set("settings", JSON.stringify(cfg));
    const run = await api("/api/runs", { method: "POST", body: data });
    active = run.id;
    if (token === epoch) {
      current = run;
      reuse = run.id;
      file = null;
      $("source-url").value = "";
      if (objectURL) URL.revokeObjectURL(objectURL);
      objectURL = null;
      clearMedia();
      empty();
      render(run);
    }
    schedule();
    await history();
  } catch (e) {
    error(e);
  } finally {
    locked(false);
    $("run").disabled = !!active;
  }
});
$("cancel").onclick = async () => {
  if (!active) return;
  try {
    await api(`/api/runs/${active}/cancel`, { method: "POST" });
    $("status").textContent = "Stopping…";
  } catch (e) {
    error(e);
  }
};
async function history() {
  const rows = await api("/api/runs");
  $("history").replaceChildren(
    ...rows.map((row) => {
      const b = el(
        "button",
        row.title,
        `history-entry${row.id === current?.id ? " active" : ""}`,
      );
      b.title = row.title;
      b.append(
        el("small", `${config.tasks[row.settings.task]} · ${row.status}`),
      );
      b.disabled = busy || recording;
      b.onclick = () => openRun(row.id);
      return b;
    }),
  );
}
async function openRun(id, restore = false) {
  if ((busy && !restore) || recording) return;
  const token = ++epoch;
  try {
    const run = await api(`/api/runs/${id}`);
    if (token !== epoch) return;
    clearMedia();
    empty();
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = null;
    current = run;
    if (run.status === "running") {
      active = id;
      schedule();
    }
    reuse = id;
    file = null;
    rowKey = "";
    $("upload").value = "";
    $("source-url").value = "";
    $("log-text").hidden = true;
    applySettings(run.settings);
    render(run);
    await history();
  } catch (e) {
    if (token === epoch) error(e);
  }
}
$("refresh").onclick = () => history().catch(error);
function schedule() {
  clearTimeout(timer);
  if (active) timer = setTimeout(poll, 1200);
}
async function poll() {
  if (!active) return;
  const id = active,
    token = epoch;
  try {
    const run = await api(`/api/runs/${id}`);
    if (active !== id) return;
    if (current?.id === id && token === epoch) {
      current = run;
      render(run);
    }
    if (run.status !== "running") {
      active = null;
      document.body.classList.remove("busy");
      $("cancel").hidden = true;
      $("run").disabled = false;
      $("remove").disabled = !current || current.status === "running";
      taskState();
      await history();
    }
  } catch (e) {
    error(e);
  }
  schedule();
}
function render(run) {
  $("title").textContent = run.title;
  $("input-name").textContent = "Saved input · reuse or open another recording";
  $("remove").disabled = run.id === active;
  $("status").textContent =
    run.error ||
    (run.status === "running"
      ? run.progress.stage || "Queued"
      : run.status === "complete"
        ? "Saved · model released"
        : run.status);
  document.body.classList.toggle("busy", !!active);
  $("cancel").hidden = !active;
  $("run").disabled = !!active;
  if (run.media) {
    $("duration").textContent =
      `${time(run.media.duration)} / ${config.tasks[run.settings.task]}`;
    peaks = run.media.peaks || [];
    const path = run.files.includes("original.wav")
      ? "original.wav"
      : run.files.includes("preview.mp4")
        ? "preview.mp4"
        : "audio.wav";
    const key = run.id + path;
    if (mediaKey !== key) {
      mediaKey = key;
      source(`/files/${run.id}/${path}`, path === "preview.mp4");
      $("track").value = "original.wav";
      $("mix-panel").hidden = !run.files.includes("music.wav");
      $("stem-downloads").replaceChildren(
        ...["vocals.wav", "music.wav"]
          .filter((n) => run.files.includes(n))
          .map((n) => link(run.id, n, n.split(".")[0])),
      );
    }
    draw();
  }
  if (run.result && rowKey !== run.id + (run.result.text || "").length) {
    rowKey = run.id + (run.result.text || "").length;
    result(run);
  }
  taskState();
}
function link(id, name, label) {
  const a = el("a", label);
  a.href = `/files/${id}/${name}?download=1`;
  a.download = name;
  return a;
}
function result(run) {
  cues = run.result.cues || [];
  activeCue = -1;
  activeWord = null;
  $("warnings").textContent = (run.result.warnings || []).join("\n");
  $("result-heading").textContent =
    run.settings.task === "ask"
      ? "Conversation"
      : run.settings.task === "karaoke"
        ? "Lyrics"
        : "Transcript";
  const speakers = [
    ...new Set(cues.map((c) => c.speaker).filter((v) => v != null)),
  ];
  $("speaker").replaceChildren(
    new Option("All speakers", ""),
    ...speakers.map((s) => new Option(String(s), String(s))),
  );
  $("exports").replaceChildren(
    ...run.result.exports.map((n) =>
      link(
        run.id,
        n,
        n === "result.zip"
          ? "Transcript bundle ↗"
          : n.split(".").pop().toUpperCase() +
              (n === "words.json"
                ? " · words"
                : n === "conversation.json"
                  ? " · chat"
                  : ""),
      ),
    ),
  );
  $("reasoning-panel").hidden = !run.result.reasoning;
  $("reasoning").textContent = run.result.reasoning || "";
  languageMap();
  renderTranscript();
}
function languageMap() {
  const labels = current?.annotations || {};
  const field = $("language-filter"),
    selected = field.value;
  field.replaceChildren(
    new Option("All languages", ""),
    ...[...new Set(Object.values(labels))]
      .sort()
      .map((label) => new Option(label, label)),
  );
  field.value = [...field.options].some((o) => o.value === selected)
    ? selected
    : "";
  $("language-panel").hidden = !cues.length;
  $("language-export").hidden = !Object.keys(labels).length;
  $("language-export").href =
    `/files/${current.id}/annotations.json?download=1`;
  const duration = current.media?.duration || cues.at(-1)?.end || 1;
  $("language-map").replaceChildren(
    ...Object.entries(labels).flatMap(([key, label]) => {
      const cue = cues[Number(key)];
      if (!cue) return [];
      const bar = el("button", label);
      bar.title = `${label} · ${time(cue.start)} · ${cue.text}`;
      bar.style.left = `${(100 * cue.start) / duration}%`;
      bar.style.width = `${(100 * (cue.end - cue.start)) / duration}%`;
      bar.onclick = () => {
        player.currentTime = cue.start;
        tick();
      };
      return [bar];
    }),
  );
}
async function annotate(index) {
  const id = current?.id;
  if (!id || id === active) return;
  const label = prompt(
    "Language note for this cue (en, es, zh-Hans…). Empty clears it.",
    current.annotations?.[index] || "",
  );
  if (label === null) return;
  try {
    const data = await api(`/api/runs/${id}/language`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cue: index, language: label.trim() }),
    });
    if (current?.id !== id) return;
    current.annotations = data.annotations;
    languageMap();
    renderTranscript();
  } catch (e) {
    error(e);
  }
}
function renderTranscript() {
  cueNodes = [];
  const target = $("transcript");
  target.replaceChildren();
  if (current?.settings.task === "ask") {
    for (const message of current.conversation || []) {
      const article = el("article", null, "message");
      const content = Array.isArray(message.content)
        ? message.content
            .filter((p) => p.type === "text")
            .map((p) => p.text)
            .join("\n")
        : message.content;
      article.append(el("h3", message.role), el("div", content));
      target.append(article);
    }
    return;
  }
  if (!cues.length) {
    target.append(
      el(
        "div",
        current?.result?.text || "Transcript will appear here.",
        "plain",
      ),
    );
    return;
  }
  const query = $("search").value.toLowerCase(),
    speaker = $("speaker").value,
    language = $("language-filter").value;
  for (const [index, cue] of cues.entries()) {
    const row = el("article", null, "cue");
    const stamp = el("button", time(cue.start), "time");
    stamp.onclick = () => {
      player.currentTime = cue.start;
      tick();
    };
    const words = el("div", null, "words");
    words.replaceChildren(
      ...cue.words.map((word) => {
        const b = el("button", word.word, "word");
        b.onclick = () => {
          player.currentTime = word.start;
          tick();
        };
        return b;
      }),
    );
    const label = current.annotations?.[index] || "";
    const languageButton = el("button", label || "+ language", "language-note");
    languageButton.title = "Edit user annotation";
    languageButton.onclick = () => annotate(index);
    const heading = el("div", null, "cue-heading");
    heading.append(
      el("span", cue.speaker ?? "Speech", "speaker-label"),
      languageButton,
    );
    row.append(stamp, heading, words);
    row.hidden =
      (speaker !== "" && String(cue.speaker) !== speaker) ||
      (language !== "" && label !== language) ||
      (query && !cue.text.toLowerCase().includes(query));
    cueNodes[index] = row;
    target.append(row);
  }
  activeCue = -1;
  activeWord = null;
  tick();
}
$("search").oninput = renderTranscript;
$("speaker").onchange = renderTranscript;
$("language-filter").onchange = renderTranscript;
function draw() {
  const canvas = $("wave");
  if (canvas.hidden) return;
  const width = canvas.clientWidth,
    height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const position = player.duration ? player.currentTime / player.duration : 0;
  const step = width / Math.max(1, peaks.length);
  for (const [i, peak] of peaks.entries()) {
    ctx.fillStyle = i / peaks.length <= position ? "#c3f277" : "#536747";
    const h = Math.max(2, peak * height * 0.7);
    ctx.fillRect(i * step, (height - h) / 2, Math.max(1, step - 1), h);
  }
}
function tick() {
  const now = player.currentTime || 0;
  $("position").textContent = time(now);
  $("length").textContent = time(player.duration);
  $("seek").max = Number.isFinite(player.duration) ? player.duration : 1;
  $("seek").value = now;
  draw();
  if (
    $("track").value === "mix" &&
    !player.paused &&
    Math.abs(vocal.currentTime - now) > 0.15
  )
    vocal.currentTime = now;
  let low = 0,
    high = cues.length - 1,
    found = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (cues[mid].start <= now) {
      found = mid;
      low = mid + 1;
    } else high = mid - 1;
  }
  if (found >= 0 && now > cues[found].end) found = -1;
  if (found !== activeCue) {
    cueNodes[activeCue]?.classList.remove("current");
    activeCue = found;
    const row = cueNodes[found];
    row?.classList.add("current");
    if (row && !row.hidden && $("follow").checked && !player.paused) {
      const pane = $("transcript");
      pane.scrollTo({
        top: row.offsetTop - pane.offsetTop - pane.clientHeight / 3,
        behavior: matchMedia("(prefers-reduced-motion:reduce)").matches
          ? "instant"
          : "smooth",
      });
    }
  }
  activeWord?.classList.remove("active");
  activeWord = null;
  $("subtitle").replaceChildren();
  if (found >= 0) {
    const cue = cues[found];
    const index = cue.words.findIndex((w) => w.start <= now && w.end >= now);
    activeWord = cueNodes[found]?.querySelectorAll(".word")[index];
    activeWord?.classList.add("active");
    if ($("captions").checked) {
      const nodes = cue.words.map((w, i) =>
        el("span", w.word + " ", i === index ? "spoken" : ""),
      );
      $("subtitle").replaceChildren(...nodes);
    }
  }
}
player.addEventListener("timeupdate", tick);
player.addEventListener("seeked", () => {
  if ($("track").value === "mix") vocal.currentTime = player.currentTime;
  tick();
});
player.addEventListener("play", () => {
  $("play").textContent = "❚❚";
  $("play").ariaLabel = "Pause";
  if ($("track").value === "mix") {
    vocal.currentTime = player.currentTime;
    vocal.playbackRate = player.playbackRate;
    vocal.volume = Number($("vocal-volume").value) * Number($("volume").value);
    vocal.play().catch(error);
  }
});
player.addEventListener("pause", () => {
  $("play").textContent = "▶";
  $("play").ariaLabel = "Play";
  vocal.pause();
});
player.addEventListener("ended", () => vocal.pause());
$("play").onclick = () => {
  if (!player.getAttribute("src")) return;
  if (player.paused) player.play().catch(error);
  else player.pause();
};
$("back").onclick = () => {
  player.currentTime = Math.max(0, player.currentTime - 15);
};
$("forward").onclick = () => {
  if (Number.isFinite(player.duration))
    player.currentTime = Math.min(player.duration, player.currentTime + 15);
};
$("seek").oninput = () => {
  if (player.getAttribute("src")) player.currentTime = Number($("seek").value);
};
$("speed").onchange = () => {
  player.playbackRate = vocal.playbackRate = Number($("speed").value);
};
$("captions").onchange = tick;
$("vocal-volume").oninput = () => {
  vocal.volume = Number($("vocal-volume").value) * Number($("volume").value);
};
$("volume").oninput = () => {
  player.volume = Number($("volume").value);
  $("vocal-volume").oninput();
};
$("track").onchange = () => {
  if (!current) return;
  const pos = player.currentTime,
    mix = $("track").value === "mix";
  vocal.pause();
  vocal.src = mix ? `/files/${current.id}/vocals.wav` : "";
  source(
    `/files/${current.id}/${mix ? "music.wav" : $("track").value}`,
    false,
    pos,
  );
};
new ResizeObserver(draw).observe($("stage"));
$("copy").onclick = () =>
  navigator.clipboard.writeText(current?.result?.text || "").catch(error);
$("log").onclick = async () => {
  if (!current) return;
  const id = current.id;
  try {
    const response = await fetch(`/api/runs/${id}/log`);
    if (!response.ok) throw new Error("Unable to read log");
    const text = await response.text();
    if (current?.id !== id) return;
    $("log-text").textContent = text || "No log yet.";
    $("log-text").hidden = false;
  } catch (e) {
    error(e);
  }
};
$("remove").onclick = async () => {
  if (
    !current ||
    current.id === active ||
    !confirm("Delete this session and its local media?")
  )
    return;
  const id = current.id;
  try {
    await api(`/api/runs/${id}`, { method: "DELETE" });
    if (current?.id === id) reset();
    else await history();
  } catch (e) {
    error(e);
  }
};
$("record").onclick = async () => {
  if (recording) {
    if (recorder?.state === "recording") recorder.stop();
    return;
  }
  if (active || busy) return;
  const token = ++epoch;
  locked(true);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    if (token !== epoch) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    recordingStream = stream;
    const chunks = [];
    const capture = new MediaRecorder(stream);
    recorder = capture;
    capture.onerror = (event) =>
      error(event.error || "Microphone recording stopped");
    capture.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    capture.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      recording = false;
      $("record").textContent = "Record microphone";
      locked(false);
      if (token !== epoch) return;
      const type = capture.mimeType;
      setFile(
        new File(chunks, `Recording.${type.includes("ogg") ? "ogg" : "webm"}`, {
          type,
        }),
      );
    };
    capture.start(1000);
    recording = true;
    locked(false);
    $("new").disabled = $("upload").disabled = true;
    $("record").textContent = "Stop recording";
    $("run").disabled = true;
  } catch (e) {
    recordingStream?.getTracks().forEach((t) => t.stop());
    recording = false;
    locked(false);
    error(e);
  } finally {
    if (!recording) locked(false);
  }
};
window.addEventListener("pagehide", () => {
  epoch++;
  clearTimeout(timer);
  if (recorder?.state === "recording") recorder.stop();
  recordingStream?.getTracks().forEach((t) => t.stop());
  player.pause();
  vocal.pause();
});
window.addEventListener("pageshow", (e) => {
  if (e.persisted) schedule();
});
async function init() {
  config = await api("/api/config");
  for (const [key, label] of Object.entries(config.tasks))
    $("task").add(new Option(label, key));
  applySettings(config.defaults);
  $("source-pin").textContent = config.source.revision.slice(0, 7);
  active = config.active;
  await history();
  if (active) {
    await openRun(active, true);
    schedule();
  }
  locked(false);
  $("run").disabled = !!active;
}
locked(true);
init().catch(error);
