const $ = (id) => document.getElementById(id);
const examples = {
  en: "Somewhere beyond the city lights, the road opens up, and for a moment, everything feels possible.",
  zh: "雨停了，窗外的树叶还挂着水珠。我们泡一壶茶，慢慢聊聊今天发生的事吧。",
  ja: "雨がやみました。窓を開けると、涼しい風と一緒に、遠くの電車の音が聞こえてきます。",
  es: "El sol empieza a esconderse detrás de las montañas. Es el momento perfecto para hacer una pausa y escuchar.",
};
let config,
  selected = null,
  active = null,
  timer = null,
  pollBusy = false,
  submitting = false;
let initializing = true,
  selecting = false,
  deleting = false;
let reference = null,
  referenceURL = null,
  reuse = null,
  recorder = null,
  stream = null,
  micTimer = null,
  recording = false;
let micEpoch = 0;
let generation = 0,
  rendered = "",
  players = [],
  waveObservers = [];
const error = (e) => {
  $("error").textContent = e?.message || String(e || "");
};
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function button(text, action, cls = "") {
  const el = document.createElement("button");
  el.type = "button";
  el.textContent = text;
  el.className = cls;
  el.onclick = () => Promise.resolve().then(action).catch(error);
  return el;
}
function link(text, name, key = selected) {
  const el = document.createElement("a");
  el.textContent = text;
  el.className = "download";
  el.href = `/files/${key}/${name}?download=1`;
  el.download = name;
  return el;
}
function settings() {
  return Object.fromEntries(
    Object.entries(config.defaults).map(([key, value]) => [
      key,
      typeof value === "boolean"
        ? $(key).checked
        : typeof value === "number"
          ? Number($(key).value)
          : $(key).value,
    ]),
  );
}
function setSettings(values) {
  for (const [key, value] of Object.entries(config.defaults)) {
    if (typeof value === "boolean") $(key).checked = values[key] ?? value;
    else $(key).value = values[key] ?? value;
  }
  counts();
  outputs();
}
function counts() {
  $("count").textContent = `${$("text").value.length} characters`;
}
function outputs() {
  document
    .querySelectorAll("output[for]")
    .forEach((el) => (el.value = $(el.htmlFor).value));
}
function controls() {
  const busy = initializing || selecting || submitting || deleting;
  $("form")
    .querySelectorAll("input,select,textarea,button")
    .forEach((el) => {
      el.disabled = busy;
    });
  $("generate").disabled = Boolean(active || busy || recording);
  $("cancel").hidden = !active;
  $("record").disabled = busy;
  $("new").disabled = recording || busy;
  $("refresh").disabled = busy;
  $("voice").disabled = recording || busy;
  $("reference").disabled = recording || busy;
  document.querySelectorAll("[data-open-session]").forEach((el) => {
    el.disabled = recording || busy;
  });
  document.querySelectorAll("[data-delete-session]").forEach((el) => {
    el.disabled =
      recording ||
      busy ||
      el.dataset.running === "true" ||
      el.dataset.deleteSession === active;
  });
}
function resetReference() {
  if (referenceURL) URL.revokeObjectURL(referenceURL);
  referenceURL = null;
  reference = null;
  reuse = null;
  $("reference").value = "";
  $("referencePlayer").pause();
  $("referencePlayer").removeAttribute("src");
  $("referencePlayer").load();
}
function voiceChanged(preserve = false) {
  const key = $("voice").value;
  if (!preserve) resetReference();
  $("custom").hidden = key !== "custom";
  $("referenceArea").hidden = key === "none";
  $("noVoice").hidden = key !== "none";
  $("transcript").readOnly = key !== "custom";
  if (config.voices[key]) {
    $("transcript").value = config.voices[key].transcript;
    $("referencePlayer").src = `/voices/${key}`;
    $("referenceName").textContent = config.voices[key].name;
  } else if (!preserve) {
    $("transcript").value = "";
    $("referenceName").textContent =
      key === "custom" ? "Choose or record a voice reference." : "";
  }
}
function useReference(file) {
  if (file.size > 32 * 1024 ** 2)
    throw new Error("Reference must be at most 32 MB");
  resetReference();
  $("transcript").value = "";
  reference = file;
  referenceURL = URL.createObjectURL(file);
  $("referencePlayer").src = referenceURL;
  $("referenceName").textContent = file.name;
}
async function toggleMic() {
  if (recording) {
    recorder?.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
    throw new Error(
      "Microphone recording is unavailable in this browser; upload a recording instead.",
    );
  recording = true;
  controls();
  $("record").disabled = true;
  const epoch = ++micEpoch;
  try {
    const media = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (epoch !== micEpoch) {
      media.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = media;
    const chunks = [];
    const capture = new MediaRecorder(media);
    recorder = capture;
    capture.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    capture.onstop = () => {
      const mime = capture.mimeType || "audio/webm";
      media.getTracks().forEach((track) => track.stop());
      if (epoch !== micEpoch) return;
      stream = null;
      clearInterval(micTimer);
      recording = false;
      $("record").textContent = "Record microphone";
      try {
        useReference(
          new File(
            chunks,
            "microphone." + (mime.includes("mp4") ? "m4a" : "webm"),
            { type: mime },
          ),
        );
      } catch (e) {
        error(e);
      }
      controls();
    };
    capture.onerror = (e) => {
      if (epoch !== micEpoch) return;
      error(e.error || "Recording failed");
      stopMic();
    };
    capture.start();
    const start = performance.now();
    $("record").textContent = "Stop recording";
    $("record").disabled = false;
    micTimer = setInterval(() => {
      const seconds = (performance.now() - start) / 1000;
      $("recordTime").textContent = `${seconds.toFixed(1)} / 15 s`;
      if (seconds >= 14.8 && capture.state === "recording") capture.stop();
    }, 100);
  } catch (e) {
    if (epoch === micEpoch) {
      stopMic();
      throw e;
    }
  }
}
function stopMic() {
  micEpoch++;
  if (recorder?.state === "recording") recorder.stop();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  clearInterval(micTimer);
  recording = false;
  $("record").textContent = "Record microphone";
  controls();
}
function clearPlayers() {
  players.forEach((p) => {
    p.pause();
    p.removeAttribute("src");
    p.load();
  });
  players = [];
  waveObservers.forEach((o) => o.disconnect());
  waveObservers = [];
}
function showTakes(takes, key) {
  const signature = key + ":" + takes.map((t) => t.stem).join(",");
  if (signature === rendered) return;
  rendered = signature;
  clearPlayers();
  $("takes").replaceChildren();
  if (!takes.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Takes appear here as each passage finishes.";
    $("takes").append(empty);
    return;
  }
  for (const [index, take] of takes.entries()) {
    const card = document.createElement("article");
    card.className = "take";
    const head = document.createElement("div");
    head.className = "take-head";
    head.textContent = `TAKE ${String(index + 1).padStart(2, "0")} · ${take.seconds.toFixed(1)} s · ${(take.sample_rate / 1000).toFixed(1)} kHz`;
    const title = document.createElement("h3");
    title.textContent = take.text;
    const audio = new Audio(`/files/${key}/${take.stem}.wav`);
    audio.preload = "metadata";
    players.push(audio);
    const wave = document.createElement("div");
    wave.className = "wave";
    const canvas = document.createElement("canvas"),
      seek = document.createElement("input");
    seek.type = "range";
    seek.min = "0";
    seek.max = String(take.seconds);
    seek.step = "0.01";
    seek.value = "0";
    seek.ariaLabel = `Seek take ${index + 1}`;
    wave.append(canvas, seek);
    const transport = document.createElement("div");
    transport.className = "transport";
    const play = button("Play", async () => {
      if (audio.paused) {
        players.forEach((p) => {
          if (p !== audio) p.pause();
        });
        await audio.play();
      } else audio.pause();
    });
    const clock = document.createElement("span");
    clock.className = "time";
    const speed = document.createElement("select");
    speed.ariaLabel = `Playback speed for take ${index + 1}`;
    [0.75, 1, 1.25, 1.5, 2].forEach((rate) =>
      speed.add(new Option(`${rate}×`, rate, rate === 1, rate === 1)),
    );
    speed.onchange = () => {
      audio.playbackRate = Number(speed.value);
    };
    transport.append(play, clock, speed);
    const volume = document.createElement("input");
    volume.type = "range";
    volume.min = "0";
    volume.max = "1";
    volume.step = "0.01";
    volume.value = "1";
    volume.className = "volume";
    volume.ariaLabel = `Volume for take ${index + 1}`;
    const mute = button("Mute", () => {
      audio.muted = !audio.muted;
      mute.textContent = audio.muted ? "Unmute" : "Mute";
    });
    volume.oninput = () => {
      audio.volume = Number(volume.value);
    };
    transport.append(mute, volume);
    const draw = () => {
      const rect = canvas.getBoundingClientRect(),
        dpr = devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.round(76 * dpr);
      const ctx = canvas.getContext("2d"),
        width = canvas.width,
        height = canvas.height;
      const bars = Math.max(1, Math.floor(width / (4 * dpr))),
        peaks = take.peaks,
        max = Math.max(0.01, ...peaks);
      for (let i = 0; i < bars; i++) {
        const p = peaks[Math.floor((i * peaks.length) / bars)] || 0;
        const h = Math.max(2 * dpr, (p / max) * height * 0.8);
        ctx.fillStyle =
          i / bars < audio.currentTime / take.seconds ? "#a6edc6" : "#3b6351";
        ctx.fillRect(
          (i * width) / bars,
          (height - h) / 2,
          Math.max(1, width / bars - dpr),
          h,
        );
      }
    };
    const tick = () => {
      seek.value = audio.currentTime;
      clock.textContent = `${audio.currentTime.toFixed(1)} / ${take.seconds.toFixed(1)} s`;
      draw();
    };
    seek.oninput = () => {
      if (Number.isFinite(audio.duration))
        audio.currentTime = Math.min(audio.duration, Number(seek.value));
      tick();
    };
    audio.ontimeupdate = tick;
    audio.onloadedmetadata = tick;
    audio.onplay = () => {
      play.textContent = "Pause";
    };
    audio.onpause = audio.onended = () => {
      play.textContent = "Play";
    };
    audio.onerror = () =>
      error(
        "Audio could not be opened; download the WAV or inspect the worker log.",
      );
    const downloads = document.createElement("div");
    downloads.className = "links";
    downloads.append(
      link("WAV", take.stem + ".wav", key),
      link("FLAC", take.stem + ".flac", key),
    );
    if (take.codes) downloads.append(link("Tokens", take.stem + ".npy", key));
    card.append(head, title, wave, transport, downloads);
    if (!take.finished) {
      const warning = document.createElement("p");
      warning.className = "warning";
      warning.textContent =
        "Frame/context limit reached before the end token. Shorten this passage or reference; this take may be incomplete.";
      card.append(warning);
    }
    $("takes").append(card);
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    waveObservers.push(observer);
    tick();
  }
}
function display(job) {
  $("status").textContent =
    job.status === "running"
      ? job.progress.stage || "Starting"
      : job.status[0].toUpperCase() + job.status.slice(1);
  const p = job.progress;
  $("detail").textContent =
    job.status === "running"
      ? [
          p.first
            ? `Take ${p.first}${p.count > 1 ? `–${p.first + p.count - 1}` : ""} of ${p.total}`
            : "",
          p.frames
            ? `${p.frames} / ${p.budget} frames${p.retry ? " · retry" : ""}`
            : "",
        ]
          .filter(Boolean)
          .join(" · ") || "Owned worker is active."
      : job.error ||
        "Model worker released. Inputs and completed takes remain saved.";
  $("progress").hidden = job.status !== "running";
  if (p.frames && p.budget) {
    $("progress").max = p.budget;
    $("progress").value = p.frames;
  } else $("progress").removeAttribute("value");
  showTakes(job.result?.takes || job.partial || [], job.id);
  $("downloads").replaceChildren();
  if (job.result)
    $("downloads").append(
      link("Download session", "result.zip", job.id),
      link("Manifest", "manifest.json", job.id),
    );
}
async function select(key, restore = true) {
  if (recording || submitting || initializing || selecting || deleting) return;
  selecting = true;
  controls();
  const token = ++generation;
  try {
    const job = await api(`/api/runs/${key}`);
    if (token !== generation) return;
    selected = key;
    display(job);
    if (restore && !recording) {
      resetReference();
      setSettings(job.settings);
      voiceChanged(true);
      if (job.settings.voice === "custom") {
        reuse = key;
        $("referenceName").textContent = "Saved reference";
        $("referencePlayer").src = `/files/${key}/reference.wav`;
      }
    }
    if (job.status === "running") active = key;
    controls();
    await history();
    if ($("logs").open) await logs();
    schedule();
  } finally {
    selecting = false;
    controls();
  }
}
async function history() {
  const rows = await api("/api/runs");
  $("history").replaceChildren();
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "history-row" + (selected === row.id ? " selected" : "");
    const open = button("", () => select(row.id));
    open.dataset.openSession = "";
    open.disabled = recording || submitting || initializing || selecting;
    const title = document.createElement("b");
    title.textContent = row.title;
    const small = document.createElement("small");
    small.textContent = `${new Date(row.created * 1000).toLocaleString()} · ${row.status}`;
    open.append(title, small);
    const del = button(
      "×",
      async () => {
        if (recording || submitting || initializing || selecting || deleting)
          return;
        if (!confirm("Delete this saved session and its audio?")) return;
        deleting = true;
        controls();
        try {
          await api(`/api/runs/${row.id}`, { method: "DELETE" });
          if (selected === row.id) newDraft();
          await history();
        } finally {
          deleting = false;
          controls();
        }
      },
      "quiet",
    );
    del.ariaLabel = "Delete session";
    del.dataset.deleteSession = row.id;
    del.dataset.running = String(row.status === "running");
    del.disabled =
      row.id === active ||
      row.status === "running" ||
      recording ||
      submitting ||
      initializing ||
      selecting ||
      deleting;
    line.append(open, del);
    $("history").append(line);
  }
  if (!rows.length) $("history").textContent = "No saved sessions yet.";
}
function schedule() {
  clearTimeout(timer);
  if (active) timer = setTimeout(poll, 900);
}
async function poll() {
  if (pollBusy || !active) return;
  pollBusy = true;
  const key = active;
  try {
    const job = await api(`/api/runs/${key}`);
    if (selected === key) display(job);
    if (job.status !== "running" && active === key) {
      active = null;
      controls();
      await history();
      if ($("logs").open) await logs();
    }
  } catch (e) {
    error(e);
  } finally {
    pollBusy = false;
    schedule();
  }
}
async function logs() {
  const key = selected;
  if (!key) return;
  const response = await fetch(`/api/runs/${key}/log`);
  const text = await response.text();
  if (key === selected) $("log").textContent = text || "No log output yet.";
}
function newDraft() {
  generation++;
  selected = null;
  rendered = "";
  resetReference();
  setSettings(config.defaults);
  voiceChanged();
  clearPlayers();
  $("takes").replaceChildren();
  $("downloads").replaceChildren();
  $("status").textContent = active ? "Worker active" : "Ready";
  $("detail").textContent = active
    ? "Open the running session below to see progress."
    : "Models load for a take, then release.";
  $("progress").hidden = true;
  $("log").textContent = "Select a session.";
  error("");
  controls();
}
$("form").onsubmit = async (e) => {
  e.preventDefault();
  if (
    active ||
    submitting ||
    recording ||
    initializing ||
    selecting ||
    deleting
  )
    return;
  generation++;
  submitting = true;
  controls();
  error("");
  try {
    const data = new FormData();
    data.set("settings", JSON.stringify(settings()));
    if (reference) data.set("reference", reference);
    if (reuse) data.set("reuse", reuse);
    const job = await api("/api/runs", { method: "POST", body: data });
    generation++;
    active = selected = job.id;
    display(job);
    await history();
    schedule();
  } catch (e) {
    error(e);
    if (e.message.includes("active take")) {
      const state = await api("/api/config").catch(() => null);
      if (state?.active) {
        active = state.active;
        schedule();
      }
    }
  } finally {
    submitting = false;
    controls();
  }
};
$("cancel").onclick = () => {
  if (active)
    api(`/api/runs/${active}/cancel`, { method: "POST" }).catch(error);
};
$("voice").onchange = () => voiceChanged();
$("text").oninput = counts;
document
  .querySelectorAll("input[type=range]")
  .forEach((el) => (el.oninput = outputs));
document.querySelectorAll("[data-example]").forEach(
  (el) =>
    (el.onclick = () => {
      $("text").value = examples[el.dataset.example];
      counts();
    }),
);
$("reference").onchange = () => {
  try {
    if ($("reference").files[0]) useReference($("reference").files[0]);
  } catch (e) {
    error(e);
  }
};
$("record").onclick = () => toggleMic().catch(error);
$("refresh").onclick = () => history().catch(error);
$("new").onclick = () => {
  newDraft();
  history().catch(error);
};
$("import").onclick = () => $("textFile").click();
$("textFile").onchange = async () => {
  try {
    const file = $("textFile").files[0];
    if (!file) return;
    if (file.size > 128000) throw new Error("Text file is too large");
    const token = generation,
      current = $("text").value;
    const text = await file.text();
    if (
      token !== generation ||
      current !== $("text").value ||
      submitting ||
      selecting ||
      deleting
    )
      return;
    $("text").value = text;
    counts();
  } catch (e) {
    error(e);
  }
};
$("logs").ontoggle = () => {
  if ($("logs").open) logs().catch(error);
};
addEventListener("pagehide", (event) => {
  stopMic();
  clearTimeout(timer);
  players.forEach((player) => player.pause());
  $("referencePlayer").pause();
  if (!event.persisted) {
    clearPlayers();
    if (referenceURL) URL.revokeObjectURL(referenceURL);
  }
});
addEventListener("pageshow", (event) => {
  if (event.persisted) {
    schedule();
    history().catch(error);
  }
});
async function init() {
  controls();
  config = await api("/api/config");
  active = config.active;
  for (const [key, voice] of Object.entries(config.voices))
    $("voice").add(new Option(voice.name, key));
  $("voice").add(new Option("Your recording", "custom"));
  $("voice").add(new Option("No reference", "none"));
  $("languages").textContent = config.languages.join(" · ");
  $("pin").textContent = config.source.revision.slice(0, 7);
  setSettings(config.defaults);
  voiceChanged();
  controls();
  await history();
  initializing = false;
  controls();
  if (active) await select(active);
  schedule();
}
init().catch(error);
