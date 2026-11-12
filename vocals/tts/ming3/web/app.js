const $ = (id) => document.getElementById(id);
const numeric = [
  "seed",
  "cfg",
  "sigma",
  "temperature",
  "flow_steps",
  "max_length",
  "max_decode_steps",
  "min_decode_steps",
  "stop_threshold",
  "min_frames",
  "max_frames",
];
const nullable = new Set([
  "seed",
  "max_decode_steps",
  "min_decode_steps",
  "stop_threshold",
  "min_frames",
  "max_frames",
]);
const toggles = ["normalization", "trim", "raw"];
const fields = [
  "model",
  "text",
  "language",
  "ip",
  "instruction",
  "style",
  "emotion",
  "dialect",
  "speed",
  "pitch",
  "volume",
  ...numeric,
  ...toggles,
];
const colors = ["#b8a0e3", "#efbc80", "#8fc9bb", "#d495a5", "#8fb4dc"];
const tags = [
  "Breathing",
  "Cough",
  "Sigh",
  "Laughter",
  "Surprise-ah",
  "Surprise-oh",
  "Dissatisfaction-hnn",
  "Uhm",
  "Confirmation-en",
  "Question-ei",
  "Question-ah",
  "Question-en",
  "Question-oh",
  "Question-huh",
];
const help = {
  seed: "Clone a reference voice in Chinese or English, including across the two languages.",
  multilingual:
    "Speak Japanese, Korean, German, French or Spanish using a reference voice.",
  voice_design:
    "Describe the voice in words. This task uses the description alone.",
  freeform:
    "Direct emotion, rhythm and emphasis in natural language. A reference voice is optional.",
  ip: "Choose a built-in voice. Voice identities and emotional variants come from the published catalogue.",
  emotion: "Apply one emotion to a reference voice.",
  basic: "Set speed, pitch and volume together on a reference voice.",
  dialect: "Use a reference recording in the requested dialect.",
  paralinguistic:
    "Place vocal tags inside Chinese or English speech on a cloned voice.",
  podcast: "Cast the speakers, then write their turns. Available on tiny.",
};
let config,
  task = "ip",
  speakers = [],
  current = null,
  active = null,
  result = null;
let busy = true,
  recording = null,
  epoch = 0,
  importTicket = 0,
  timer = null,
  hidden = false,
  voiceEntries = [];
const player = $("player"),
  wave = $("wave");
function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
async function api(path, options) {
  const r = await fetch(path, options);
  const v = await r.json();
  if (!r.ok) throw Error(v.error || r.statusText);
  return v;
}
function guard(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      status(e.message, true);
    }
  };
}
function newSpeaker(voice = "lingguang") {
  return {
    voice,
    transcript: config.presets[voice]?.transcript || "",
    file: null,
    url: null,
    reuse: null,
  };
}
function release(s) {
  if (s.url?.startsWith("blob:")) URL.revokeObjectURL(s.url);
}
function settings() {
  const v = { task };
  for (const k of fields)
    v[k] = toggles.includes(k)
      ? $(k).checked
      : numeric.includes(k)
        ? nullable.has(k) && $(k).value === ""
          ? null
          : Number($(k).value)
        : $(k).value;
  v.speakers = speakers.map((s) => ({
    voice: s.voice,
    transcript: s.transcript,
  }));
  return v;
}
function save() {
  if (!config) return;
  try {
    localStorage.setItem(
      "ming3-draft",
      JSON.stringify({
        settings: settings(),
        references: speakers.map((s) => ({ reuse: s.reuse })),
        current,
      }),
    );
  } catch {}
}
function lockUI() {
  const locked = busy || !!active;
  const casting = !!recording;
  $("editor").disabled = locked;
  $("generate").disabled =
    locked || casting || !config?.installed[$("model").value];
  $("new").disabled = locked || casting;
  $("cancel").hidden = !active;
  $("refresh").disabled = busy || casting;
  $("model").disabled = locked || casting;
  $("import").disabled = locked || casting;
  document
    .querySelectorAll("#tasks button")
    .forEach(
      (b) =>
        (b.disabled =
          locked ||
          casting ||
          (b.dataset.task === "podcast" && $("model").value !== "tiny")),
    );
  document
    .querySelectorAll("#history button")
    .forEach((b) => (b.disabled = busy || casting || !!active));
  document
    .querySelectorAll(".speaker select,.speaker input,.speaker .remove")
    .forEach(
      (n) =>
        (n.disabled =
          locked ||
          casting ||
          (n.classList.contains("remove") && speakers.length <= 2)),
    );
  $("add-speaker").disabled = locked || casting || speakers.length >= 16;
  document
    .querySelectorAll(".record")
    .forEach(
      (n) =>
        (n.disabled =
          locked || (casting && n.dataset.index !== String(recording.index))),
    );
  document.body.classList.toggle("working", !!active);
}
function populateVoices(selected) {
  const q = $("voice-search").value.toLocaleLowerCase();
  $("ip").replaceChildren();
  let count = 0;
  for (const [group, voices] of voiceEntries) {
    const g = document.createElement("optgroup");
    g.label = group;
    for (const v of voices)
      if (v === selected || `${group} ${v}`.toLocaleLowerCase().includes(q)) {
        const o = new Option(v, v);
        o.selected = v === selected;
        g.append(o);
        count++;
      }
    if (g.childNodes.length) $("ip").append(g);
  }
  if (
    selected &&
    !Array.from($("ip").options).some((o) => o.value === selected)
  )
    $("ip").add(new Option(selected, selected, true, true));
  if (!count && !selected) $("ip").add(new Option("No matches", ""));
}
function updateMode() {
  const tiny = $("model").value === "tiny";
  if (task === "podcast" && !tiny) task = "seed";
  $("task-title").textContent = config.tasks[task];
  $("task-help").textContent = help[task];
  document
    .querySelectorAll("[data-modes]")
    .forEach((n) => (n.hidden = !n.dataset.modes.split(" ").includes(task)));
  document.querySelectorAll(".tiny-only").forEach((n) => (n.hidden = !tiny));
  document
    .querySelectorAll(".decode-only")
    .forEach(
      (n) =>
        (n.hidden =
          task === "podcast" || (n.classList.contains("tiny-only") && !tiny)),
    );
  $("chunk-label").hidden = task === "podcast";
  $("defaults-note").textContent =
    `Seed: ${tiny ? (task === "podcast" ? 1235 : 1234) : 1895} when blank; -1 for random. ${tiny ? "Native release: BF16 weights, FP32 recurrent state, SDPA." : "Native BF16 synthesis with SDPA."} Raw text bypasses normalization and segmentation.`;
  $("engine").textContent =
    `${$("model").value} / ${config.installed[$("model").value] ? "ready" : "download + setup required"}`;
  $("reference-section").hidden = ["ip", "voice_design"].includes(task);
  const oldLanguage = $("language").value;
  const langs =
    task === "multilingual"
      ? config.languages.slice(2)
      : config.languages.slice(0, 2);
  $("language").replaceChildren(...langs.map((v) => new Option(v, v)));
  if (langs.includes(oldLanguage)) $("language").value = oldLanguage;
  if (task === "podcast") {
    while (speakers.length < 2)
      speakers.push(newSpeaker(speakers.length ? "db30" : "lingguang"));
  } else {
    speakers.splice(1).forEach(release);
    if (!speakers.length)
      speakers.push(newSpeaker(task === "freeform" ? "none" : "lingguang"));
    if (task !== "freeform" && speakers[0].voice === "none")
      speakers[0] = newSpeaker();
  }
  document
    .querySelectorAll("#tasks button")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.task === task)),
    );
  renderSpeakers();
  updateText();
  lockUI();
}
function updateText() {
  const text = $("text").value;
  $("count").textContent = `${text.length.toLocaleString()} characters`;
  $("turns").replaceChildren();
  if (task === "podcast")
    for (const line of text.split("\n").filter((l) => l.trim())) {
      const m = line.match(/^\s*speaker_(\d+):\s*(.*)$/);
      const row = el("div", undefined, "turn");
      if (m) {
        row.style.setProperty(
          "--speaker",
          colors[(Number(m[1]) - 1) % colors.length],
        );
        row.append(el("strong", `S${m[1]}`), el("span", m[2]));
      } else row.textContent = line;
      $("turns").append(row);
    }
}
function renderSpeakers() {
  $("speakers")
    .querySelectorAll("audio")
    .forEach((a) => a.pause());
  $("speakers").replaceChildren();
  speakers.forEach((s, i) => {
    const card = el("article", undefined, "speaker");
    card.style.setProperty("--speaker", colors[i % colors.length]);
    const head = el("div", undefined, "speaker-head"),
      select = document.createElement("select");
    select.setAttribute("aria-label", `Speaker ${i + 1} reference`);
    if (task === "freeform") select.add(new Option("No reference", "none"));
    for (const [key, p] of Object.entries(config.presets))
      select.add(new Option(p.name, key));
    select.add(new Option("Upload / record", "custom"));
    select.value = s.voice;
    select.onchange = () => {
      release(s);
      Object.assign(s, newSpeaker(select.value));
      renderSpeakers();
      save();
    };
    head.append(el("strong", `S${String(i + 1).padStart(2, "0")}`), select);
    if (task === "podcast") {
      const remove = el("button", "Remove", "remove");
      remove.type = "button";
      remove.disabled = speakers.length <= 2;
      remove.onclick = () => {
        release(s);
        speakers.splice(i, 1);
        renderSpeakers();
        save();
      };
      head.append(remove);
    }
    card.append(head);
    if (s.voice === "custom") {
      const row = el("div", undefined, "reference-controls"),
        upload = document.createElement("input");
      upload.type = "file";
      upload.accept = "audio/*,video/*";
      upload.setAttribute("aria-label", `Upload speaker ${i + 1}`);
      upload.onchange = () => {
        if (!upload.files[0]) return;
        release(s);
        s.file = upload.files[0];
        s.url = URL.createObjectURL(s.file);
        s.reuse = null;
        s.transcript = "";
        renderSpeakers();
        save();
      };
      const record = el("button", "Record", "record");
      record.type = "button";
      record.dataset.index = String(i);
      record.onclick = guard(() => recordAudio(i, record));
      row.append(upload, record);
      card.append(row);
      card.append(
        el(
          "div",
          s.file?.name ||
            (s.reuse ? "Saved reference" : "Upload or record a reference"),
          "filename",
        ),
      );
    }
    const src = s.voice in config.presets ? `/voices/${s.voice}` : s.url;
    if (src) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = src;
      card.append(audio);
    }
    if (["seed", "multilingual", "paralinguistic", "podcast"].includes(task)) {
      const label = el(
          "label",
          $("model").value === "tiny"
            ? "Exact transcript"
            : "Exact transcript · optional",
        ),
        input = document.createElement("textarea");
      input.rows = 2;
      input.value = s.transcript;
      input.maxLength = 4000;
      input.placeholder = "The words spoken in this reference recording";
      input.readOnly = s.voice !== "custom";
      input.oninput = () => {
        s.transcript = input.value;
        save();
      };
      label.append(input);
      card.append(label);
    }
    $("speakers").append(card);
  });
  lockUI();
  document
    .querySelectorAll(".remove")
    .forEach(
      (b) =>
        (b.disabled = busy || !!active || !!recording || speakers.length <= 2),
    );
}
async function recordAudio(index, button) {
  if (recording) {
    recording.recorder?.stop();
    if (!recording.recorder) {
      recording = null;
      epoch++;
      lockUI();
    }
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
    throw Error(
      "Microphone recording is unavailable; upload a recording instead",
    );
  const ticket = epoch,
    s = speakers[index],
    session = { index, recorder: null, stream: null, timer: null };
  recording = session;
  button.textContent = "Cancel permission";
  lockUI();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (epoch !== ticket || recording !== session) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    session.stream = stream;
    const recorder = new MediaRecorder(stream),
      chunks = [];
    session.recorder = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      clearTimeout(session.timer);
      stream.getTracks().forEach((t) => t.stop());
      if (recording === session) recording = null;
      if (
        !session.failed &&
        epoch === ticket &&
        speakers[index] === s &&
        chunks.length
      ) {
        release(s);
        const blob = new Blob(chunks, { type: recorder.mimeType });
        s.file = new File([blob], "recording.webm", { type: blob.type });
        s.url = URL.createObjectURL(blob);
        s.reuse = null;
        s.transcript = "";
        renderSpeakers();
        save();
        status("Recording ready · enter its exact transcript when requested");
      }
      lockUI();
    };
    recorder.onerror = () => {
      session.failed = true;
      stream.getTracks().forEach((t) => t.stop());
      clearTimeout(session.timer);
      if (recording === session) recording = null;
      button.textContent = "Record";
      button.classList.remove("recording");
      lockUI();
      status("Microphone recording failed", true);
    };
    recorder.start();
    button.textContent = "Stop recording";
    button.classList.add("recording");
    session.timer = setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, 28000);
    lockUI();
  } catch (e) {
    session.stream?.getTracks().forEach((t) => t.stop());
    if (recording === session) recording = null;
    button.textContent = "Record";
    lockUI();
    throw e;
  }
}
function stopRecording() {
  const s = recording;
  recording = null;
  if (s) {
    clearTimeout(s.timer);
    if (s.recorder && s.recorder.state !== "inactive") s.recorder.stop();
    s.stream?.getTracks().forEach((t) => t.stop());
  }
}
function fill(cfg, refs = []) {
  epoch++;
  stopRecording();
  speakers.forEach(release);
  task = cfg.task || "ip";
  $("voice-search").value = "";
  populateVoices(cfg.ip);
  for (const k of fields) {
    if (toggles.includes(k)) $(k).checked = cfg[k];
    else $(k).value = cfg[k] ?? "";
  }
  speakers = (cfg.speakers || []).map((s, i) => ({
    ...newSpeaker(s.voice),
    ...s,
    reuse: refs[i]?.reuse || null,
    url: referenceURL(refs[i]?.reuse),
  }));
  updateMode();
}
function referenceURL(reuse) {
  const match = /^([a-f0-9]{32}):([1-9]|1[0-6])$/.exec(reuse || "");
  return match ? `/files/${match[1]}/reference-${match[2]}.wav` : null;
}
function clearOutput() {
  player.pause();
  player.removeAttribute("src");
  player.load();
  result = null;
  $("output").hidden = true;
  $("log").textContent = "";
}
function showResult(key, data) {
  result = data;
  $("output").hidden = !data;
  if (!data) return;
  player.src = `/files/${key}/speech.wav`;
  player.playbackRate = Number($("playback-rate").value);
  $("duration").textContent =
    `${data.duration.toFixed(2)} s · ${(data.sample_rate / 1000).toFixed(1)} kHz`;
  $("take-seed").textContent = `/ ${data.seed}`;
  $("warning").textContent = data.warning || "";
  $("downloads").replaceChildren();
  for (const [name, title] of [
    ["speech.wav", "WAV · float32"],
    ["speech.flac", "FLAC · PCM24"],
    ["manifest.json", "Manifest"],
    ["result.zip", "Take bundle"],
  ]) {
    const a = el("a", title);
    a.href = `/files/${key}/${name}?download=1`;
    a.download = name;
    $("downloads").append(a);
  }
  draw();
}
function draw() {
  const dpr = window.devicePixelRatio || 1,
    r = wave.getBoundingClientRect();
  if (!r.width) return;
  wave.width = r.width * dpr;
  wave.height = 130 * dpr;
  const c = wave.getContext("2d");
  c.scale(dpr, dpr);
  c.clearRect(0, 0, r.width, 130);
  if (!result) return;
  const peaks = result.peaks || [],
    position = player.currentTime / result.duration,
    scale = Math.max(1, ...peaks);
  c.lineWidth = 1;
  peaks.forEach((p, i) => {
    const x = (i / peaks.length) * r.width,
      h = Math.max(1, (p / scale) * 52);
    c.strokeStyle = i / peaks.length <= position ? "#efbc80" : "#726380";
    c.beginPath();
    c.moveTo(x, 65 - h);
    c.lineTo(x, 65 + h);
    c.stroke();
  });
  c.fillStyle = "#efbc80";
  c.fillRect(position * r.width, 8, 1, 114);
  wave.setAttribute("aria-valuenow", String(Math.round(position * 100) || 0));
  wave.setAttribute(
    "aria-valuetext",
    `${player.currentTime.toFixed(1)} of ${result.duration.toFixed(1)} seconds`,
  );
}
function seek(e) {
  if (!result || !Number.isFinite(player.duration)) return;
  const r = wave.getBoundingClientRect();
  player.currentTime =
    Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * player.duration;
  draw();
}
wave.onpointerdown = (e) => {
  wave.setPointerCapture(e.pointerId);
  seek(e);
};
wave.onpointermove = (e) => {
  if (wave.hasPointerCapture(e.pointerId)) seek(e);
};
wave.onpointerup = (e) => {
  if (wave.hasPointerCapture(e.pointerId))
    wave.releasePointerCapture(e.pointerId);
};
wave.onkeydown = (e) => {
  if (!result || !Number.isFinite(player.duration)) return;
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
    e.preventDefault();
    player.currentTime =
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? player.duration
          : Math.max(
              0,
              Math.min(
                player.duration,
                player.currentTime + (e.key === "ArrowRight" ? 5 : -5),
              ),
            );
    draw();
  }
};
player.ontimeupdate = draw;
player.onloadedmetadata = draw;
$("playback-rate").onchange = () => {
  player.playbackRate = Number($("playback-rate").value);
};
new ResizeObserver(draw).observe(wave);
async function history() {
  const rows = await api("/api/runs");
  $("history").replaceChildren();
  for (const row of rows) {
    const line = el("div", undefined, "history-row"),
      open = el("button", row.title),
      remove = el("button", "×");
    open.title = row.title;
    open.append(
      el(
        "small",
        `${row.settings.model} · ${row.settings.task} · ${row.status}`,
      ),
    );
    open.onclick = guard(() => loadTake(row.id));
    remove.title = "Delete take";
    remove.onclick = guard(async () => {
      if (
        busy ||
        active ||
        recording ||
        !confirm("Delete this saved take and its recordings?")
      )
        return;
      busy = true;
      lockUI();
      try {
        await api(`/api/runs/${row.id}`, { method: "DELETE" });
        if (current === row.id) {
          current = null;
          speakers.forEach((s) => {
            if (s.reuse?.startsWith(row.id + ":")) {
              s.reuse = null;
              s.url = null;
            }
          });
          clearOutput();
          renderSpeakers();
          save();
        }
        await history();
      } finally {
        busy = false;
        lockUI();
      }
    });
    line.append(open, remove);
    $("history").append(line);
  }
  lockUI();
}
async function loadTake(key) {
  if (busy || recording || (active && active !== key)) return;
  busy = true;
  lockUI();
  try {
    const row = await api(`/api/runs/${key}`);
    clearOutput();
    current = key;
    fill(
      row.settings,
      row.settings.speakers.map((_, i) => ({
        reuse: `${key}:${i + 1}`,
        url: `/files/${key}/reference-${i + 1}.wav`,
      })),
    );
    showResult(key, row.result);
    status(row.error || row.status, row.status === "failed");
    save();
    await fetchLog(key);
  } finally {
    busy = false;
    lockUI();
  }
}
async function fetchLog(key) {
  if (!key) return;
  const ticket = epoch;
  const r = await fetch(`/api/runs/${key}/log`);
  if (!r.ok) return;
  const text = await r.text();
  if (key === current && ticket === epoch) $("log").textContent = text;
}
function schedule() {
  clearTimeout(timer);
  if (active && !hidden) timer = setTimeout(() => guard(poll)(), 1000);
}
async function poll() {
  const key = active;
  if (!key) return;
  try {
    const row = await api(`/api/runs/${key}`);
    if (active !== key) return;
    status(
      row.error || row.progress.message || row.status,
      row.status === "failed",
    );
    if ($("log-panel").open) await fetchLog(key);
    if (row.status !== "running") {
      active = null;
      showResult(key, row.result);
      await history();
      await fetchLog(key);
      lockUI();
      save();
      $("footer-status").textContent = "Model process released";
    }
  } finally {
    schedule();
  }
}
$("form").onsubmit = guard(async (e) => {
  e.preventDefault();
  if (busy || active || recording) return;
  busy = true;
  lockUI();
  try {
    save();
    const body = new FormData(),
      cfg = settings();
    body.set("settings", JSON.stringify(cfg));
    const references = !["ip", "voice_design"].includes(task);
    if (references)
      speakers.forEach((s, i) => {
        if (s.file) body.set(`reference-${i + 1}`, s.file);
        if (s.reuse) body.set(`reuse-${i + 1}`, s.reuse);
      });
    const row = await api("/api/runs", { method: "POST", body });
    clearOutput();
    current = active = row.id;
    row.settings.speakers.forEach((_, i) => {
      speakers[i].reuse = `${row.id}:${i + 1}`;
    });
    status("Preparing take");
    $("footer-status").textContent = "Native worker active";
    save();
    await history();
    schedule();
  } finally {
    busy = false;
    lockUI();
  }
});
$("cancel").onclick = guard(async () => {
  if (active) {
    await api(`/api/runs/${active}/cancel`, { method: "POST" });
    status("Cancelling and releasing the model");
  }
});
$("refresh").onclick = guard(history);
$("new").onclick = () => {
  if (busy || active || recording) return;
  current = null;
  clearOutput();
  fill(structuredClone(config.defaults));
  save();
  status("Ready");
};
$("model").onchange = () => {
  if (busy || active || recording) return;
  updateMode();
  save();
};
$("voice-search").oninput = () => {
  const selected = $("ip").value;
  populateVoices(selected);
};
$("add-speaker").onclick = () => {
  if (recording || busy || active || speakers.length >= 16) return;
  speakers.push(newSpeaker("custom"));
  renderSpeakers();
  save();
};
$("add-turn").onclick = () => {
  const next =
    ($("text")
      .value.split("\n")
      .filter((l) => l.trim()).length %
      speakers.length) +
    1;
  $("text").value += ($("text").value ? "\n" : "") + `speaker_${next}: `;
  updateText();
  save();
  $("text").focus();
};
$("text").oninput = () => {
  updateText();
  save();
};
$("editor").addEventListener("input", save);
$("editor").addEventListener("change", save);
$("import").onchange = guard(async () => {
  const f = $("import").files[0],
    ticket = epoch,
    request = ++importTicket,
    previousText = $("text").value;
  if (!f || busy || active || recording) return;
  if (f.size > 100000) throw Error("Use a script smaller than 100 KB");
  const value = await f.text();
  if (
    ticket !== epoch ||
    request !== importTicket ||
    previousText !== $("text").value ||
    busy ||
    active ||
    recording
  )
    return;
  if (value.length > 24000) throw Error("Use at most 24000 characters");
  $("text").value = value;
  updateText();
  save();
  $("import").value = "";
});
$("log-panel").ontoggle = () => {
  if ($("log-panel").open) guard(() => fetchLog(current))();
};
document.addEventListener("visibilitychange", () => {
  hidden = document.hidden;
  if (hidden) clearTimeout(timer);
  else schedule();
});
window.addEventListener("pagehide", (e) => {
  save();
  epoch++;
  stopRecording();
  player.pause();
  document.querySelectorAll(".speaker audio").forEach((a) => a.pause());
  clearTimeout(timer);
  if (!e.persisted) speakers.forEach(release);
});
window.addEventListener("pageshow", () => {
  hidden = document.hidden;
  if (config && !recording) renderSpeakers();
  lockUI();
  schedule();
});
guard(async () => {
  config = await api("/api/config");
  voiceEntries = Object.entries(config.voices);
  $("source").textContent =
    `inclusionAI/Ming-3.0-tts\n${config.source.revision}`;
  for (const [key, title] of Object.entries(config.tasks)) {
    const b = el("button");
    b.type = "button";
    b.dataset.task = key;
    b.append(
      el("span", String($("tasks").children.length + 1).padStart(2, "0")),
      el("strong", title),
    );
    b.onclick = () => {
      if (busy || active || recording) return;
      task = key;
      epoch++;
      updateMode();
      save();
    };
    $("tasks").append(b);
  }
  for (const tag of tags) {
    const b = el("button", `[${tag}]`);
    b.type = "button";
    b.onclick = () => {
      const text = $("text"),
        start = text.selectionStart,
        end = text.selectionEnd;
      text.setRangeText(` [${tag}] `, start, end, "end");
      updateText();
      save();
      text.focus();
    };
    $("tags").append(b);
  }
  let draft;
  try {
    draft = JSON.parse(localStorage.getItem("ming3-draft"));
  } catch {}
  fill(config.defaults);
  if (draft?.settings) {
    fill({ ...config.defaults, ...draft.settings }, draft.references);
    current = draft.current || null;
  }
  active = config.active;
  if (active) {
    const row = await api(`/api/runs/${active}`);
    current = active;
    fill(
      row.settings,
      row.settings.speakers.map((_, i) => ({
        reuse: `${active}:${i + 1}`,
        url: `/files/${active}/reference-${i + 1}.wav`,
      })),
    );
    status(row.progress.message || "Working");
    schedule();
  }
  await history();
  const key = current,
    ticket = epoch;
  if (key && !active) {
    try {
      const row = await api(`/api/runs/${key}`);
      if (current !== key || epoch !== ticket) return;
      showResult(key, row.result);
      status(row.error || row.status, row.status === "failed");
      await fetchLog(key);
    } catch {
      if (current === key && epoch === ticket) {
        current = null;
        save();
      }
    }
  }
})().finally(() => {
  busy = false;
  lockUI();
});
