// Copyright 2026 Inworld AI — Apache-2.0
const state = {
  take: null,
  sequence: 0,
  queue: [],
  flight: false,
  capturing: false,
  stopping: false,
  finishing: false,
  flushing: false,
  starting: false,
  listen: false,
  context: null,
  stream: null,
  node: null,
  source: null,
  gain: null,
  timer: null,
  samples: 0,
  error: "",
  audioSeconds: 0,
  generation: 0,
  disposed: false,
  device: "",
  rate: 0,
};
const $ = (selector) => element.querySelector(selector);
// Gradio may mount the app, and WaveSurfer its native players, in open shadows.
const appRoot = element.getRootNode();
const mediaPlayers = new Set(),
  mediaRoots = new Set(),
  playedEvents = new WeakSet();
let observer = null,
  discoveryTimer = null;
const panel = () => $(".live-studio");
const current = (generation) =>
  !state.disposed && element.isConnected && generation === state.generation;
function discoverPlayers() {
  const pending = [document],
    roots = new Set(),
    players = new Set();
  if (appRoot !== document) pending.push(appRoot);
  // Bound traversal work even when this demo is embedded in a larger page.
  let remaining = 10000;
  while (pending.length && remaining > 0 && roots.size < 128) {
    const root = pending.pop();
    if (roots.has(root)) continue;
    roots.add(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    let node;
    while (remaining > 0 && (node = walker.nextNode())) {
      remaining -= 1;
      if (node instanceof HTMLMediaElement) players.add(node);
      if (node.shadowRoot) pending.push(node.shadowRoot);
    }
  }
  observer?.disconnect();
  for (const player of mediaPlayers)
    if (!players.has(player)) player.removeEventListener("play", onPlay, true);
  for (const root of mediaRoots)
    if (!roots.has(root)) root.removeEventListener("play", onPlay, true);
  for (const player of players)
    if (!mediaPlayers.has(player))
      player.addEventListener("play", onPlay, true);
  for (const root of roots) {
    if (!mediaRoots.has(root)) root.addEventListener("play", onPlay, true);
    observer?.observe(root, { childList: true, subtree: true });
  }
  mediaPlayers.clear();
  players.forEach((player) => mediaPlayers.add(player));
  mediaRoots.clear();
  roots.forEach((root) => mediaRoots.add(root));
  return players;
}
function insidePlayer(player, id) {
  for (
    let node = player, depth = 0;
    node && depth < 64;
    node = node.parentNode || node.host, depth++
  ) {
    if (node.id === id) return true;
  }
  return false;
}
function forgetPlayers() {
  clearTimeout(discoveryTimer);
  discoveryTimer = null;
  observer?.disconnect();
  mediaPlayers.forEach((player) =>
    player.removeEventListener("play", onPlay, true),
  );
  mediaRoots.forEach((root) => root.removeEventListener("play", onPlay, true));
  mediaPlayers.clear();
  mediaRoots.clear();
}
const pausePlayers = () => discoverPlayers().forEach((audio) => audio.pause());
const status = (title, detail) => {
  if ($("[data-status]")) $("[data-status]").textContent = title;
  if (detail && $("[data-detail]")) $("[data-detail]").textContent = detail;
};
function controls() {
  if (!panel()) return;
  panel().dataset.capturing = String(state.capturing);
  $("[data-action=record]").disabled =
    state.capturing || state.starting || state.stopping || state.flight;
  $("[data-action=record]").textContent = state.take
    ? "● New recording"
    : "● Start recording";
  $("[data-action=listen]").disabled = !state.capturing;
  $("[data-action=stop]").disabled = !state.capturing;
  $("[data-error]").textContent = state.error;
}
function clearFlight() {
  state.flight = false;
  clearTimeout(state.timer);
}
function send(name, payload) {
  state.flight = true;
  const generation = state.generation;
  // A broken connection must release the microphone instead of recording forever.
  state.timer = setTimeout(() => {
    if (current(generation))
      fail("The server stopped responding. Please start a new recording.");
  }, 45000);
  trigger(name, payload);
}
// Little-endian 16-bit PCM in base64: about a tenth of the same samples as
// JSON numbers, so a take keeps up over an ordinary uplink.
function encodePcm(pieces, total) {
  const bytes = new Uint8Array(total * 2),
    view = new DataView(bytes.buffer);
  let offset = 0;
  for (const piece of pieces)
    for (let i = 0; i < piece.length; i++, offset += 2)
      view.setInt16(
        offset,
        Math.max(-32768, Math.min(32767, Math.round(piece[i] * 32768))),
        true,
      );
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}
function pump() {
  if (state.flight || !state.take || state.finishing) return;
  if (state.queue.length) {
    // Everything waiting, up to one second (the server's limit per event):
    // a slow answer is caught up in one round trip.
    const rate = state.queue[0].rate,
      pieces = [];
    let total = 0;
    while (
      state.queue.length &&
      state.queue[0].rate === rate &&
      (!pieces.length || total + state.queue[0].samples.length <= rate)
    ) {
      const piece = state.queue.shift().samples;
      pieces.push(piece);
      total += piece.length;
    }
    state.sequence += 1;
    send("audio_chunk", {
      take: state.take,
      sequence: state.sequence,
      rate,
      pcm: encodePcm(pieces, total),
    });
  } else if (state.stopping && !state.flushing) {
    state.finishing = true;
    send("record_finish", {
      take: state.take,
      sequence: state.sequence,
      listen: state.listen,
    });
  }
  controls();
}
async function release() {
  const { stream, node, source, gain, context } = state;
  // Detach before awaiting close: an older teardown must not clear a new take.
  state.context = state.stream = state.node = state.source = state.gain = null;
  stream?.getTracks().forEach((track) => track.stop());
  node?.disconnect();
  source?.disconnect();
  gain?.disconnect();
  if (context && context.state !== "closed")
    await context.close().catch(() => {});
}
async function fail(message) {
  const generation = ++state.generation;
  state.error = message;
  state.capturing = false;
  state.stopping = false;
  state.finishing = false;
  state.starting = false;
  state.flushing = false;
  state.queue = [];
  state.take = null;
  clearFlight();
  await release();
  if (current(generation)) {
    controls();
    status("Recording paused", "Start again when you are ready.");
  }
}
async function stop(listen = false) {
  if (!state.capturing) return;
  const generation = state.generation;
  state.capturing = false;
  state.stopping = true;
  state.flushing = true;
  state.listen = listen;
  // Stop acquisition now, before awaiting the worklet's already captured tail.
  state.stream?.getTracks().forEach((track) => track.stop());
  status(
    "Finishing your audio…",
    "Microphone paused. Processing the last samples.",
  );
  controls();
  // The worklet sends its partial final block before acknowledging this flush.
  if (state.node) {
    const flushed = await new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(false), 1000);
      state.flushed = () => {
        clearTimeout(timeout);
        resolve(true);
      };
      state.node.port.postMessage("flush");
    });
    if (!flushed && current(generation))
      state.error =
        "The microphone did not return its final audio block. Your recording may be slightly shortened.";
  }
  if (!current(generation)) return;
  await release();
  if (!current(generation)) return;
  state.flushing = false;
  pump();
}
async function startCapture() {
  const generation = state.generation,
    context = state.context,
    stream = state.stream,
    take = state.take;
  if (
    !current(generation) ||
    !state.starting ||
    !context ||
    !stream ||
    state.node
  )
    return;
  const worklet = `class Capture extends AudioWorkletProcessor {
    constructor() { super(); this.size=Math.round(sampleRate*.2); this.buffer=new Float32Array(this.size); this.offset=0;
      this.port.onmessage=()=>{this.emit();this.port.postMessage({flushed:true});}; }
    emit(){if(this.offset){const block=this.buffer.slice(0,this.offset);this.port.postMessage({samples:block},[block.buffer]);this.offset=0;}}
    process(inputs){const x=inputs[0]?.[0];if(x){let i=0;while(i<x.length){const n=Math.min(this.size-this.offset,x.length-i);
      this.buffer.set(x.subarray(i,i+n),this.offset);this.offset+=n;i+=n;if(this.offset===this.size)this.emit();}}return true;}}
    registerProcessor('voice-capture',Capture);`;
  const url = URL.createObjectURL(
    new Blob([worklet], { type: "application/javascript" }),
  );
  try {
    await context.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
  if (
    !current(generation) ||
    state.context !== context ||
    state.stream !== stream
  )
    return;
  const node = new AudioWorkletNode(context, "voice-capture");
  state.node = node;
  node.port.onmessage = ({ data }) => {
    if (!current(generation) || state.node !== node || state.take !== take)
      return;
    if (data.flushed) {
      state.flushed?.();
      state.flushed = null;
      return;
    }
    if (!data.samples || !state.take) return;
    const rate = context.sampleRate;
    const remaining = Math.max(0, Math.round(rate * 60) - state.samples);
    const samples = data.samples.subarray(0, remaining);
    if (samples.length) {
      state.samples += samples.length;
      state.queue.push({ rate, samples });
    }
    if (state.queue.length > 20) {
      state.error =
        "This device or connection is falling behind. Capture paused so the remaining audio can finish.";
      stop(false);
    }
    if (state.samples >= rate * 60 && state.capturing) stop(false);
    pump();
  };
  // A player may have started while permission or the server was pending.
  pausePlayers();
  // An unplugged microphone ends its track (stop() does not fire "ended").
  stream.getAudioTracks().forEach((track) =>
    track.addEventListener("ended", () => {
      if (!current(generation) || !state.capturing) return;
      state.error =
        "The microphone was disconnected. The recording up to that point is kept.";
      stop(false);
    }),
  );
  state.source = context.createMediaStreamSource(stream);
  state.gain = context.createGain();
  state.gain.gain.value = 0;
  state.source.connect(node).connect(state.gain).connect(context.destination);
  state.starting = false;
  state.capturing = true;
  status("Recording live", "Your voice is being isolated as you speak.");
  controls();
}
function microphoneMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access is blocked. Allow it for this site (the icon in the address bar), then start again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found. Connect one, then start again.";
    case "NotReadableError":
    case "AbortError":
      return "The microphone could not be opened. Another application may be using it.";
    default:
      return error?.message || "Please allow microphone access and try again.";
  }
}
async function start() {
  if (
    state.disposed ||
    !element.isConnected ||
    state.flight ||
    state.starting ||
    state.capturing ||
    state.stopping
  )
    return;
  const generation = ++state.generation;
  state.starting = true;
  controls();
  state.error = "";
  state.sequence = 0;
  state.samples = 0;
  state.rate = 0;
  state.audioSeconds = 0;
  state.queue = [];
  state.finishing = false;
  state.listen = false;
  state.take = null;
  pausePlayers();
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("Microphone access needs HTTPS or localhost.");
    const Context = window.AudioContext || window.webkitAudioContext;
    // The device's own rate: Firefox refuses a microphone whose rate differs
    // from a forced one, and the server resamples whatever arrives.
    const context = new Context({ latencyHint: "interactive" });
    state.context = context;
    // Kept past the take's end, when the context is closed, for the readout.
    state.rate = context.sampleRate;
    await context.resume();
    if (!current(generation)) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    if (!current(generation)) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    state.take = crypto.randomUUID();
    status("Preparing your recording…", "Both models are already loaded.");
    send("record_start", { take: state.take });
    controls();
  } catch (error) {
    if (current(generation)) await fail(microphoneMessage(error));
  }
}
const palette = Array.from({ length: 256 }, (_, i) => {
  const t = i / 255;
  const stops = [
    [7, 13, 27],
    [44, 34, 99],
    [121, 53, 137],
    [225, 98, 107],
    [255, 210, 125],
  ];
  const s = Math.min(3, Math.floor(t * 4)),
    u = t * 4 - s;
  return stops[s].map((v, k) => Math.round(v * (1 - u) + stops[s + 1][k] * u));
});
function drawWave(canvas, points, color) {
  if (!canvas) return;
  const width = Math.max(
      100,
      Math.round(canvas.clientWidth * devicePixelRatio),
    ),
    height = 94 * devicePixelRatio;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0c1425";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#26344d";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  if (!points?.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, (width / points.length) * 0.65);
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i * width) / points.length;
    ctx.moveTo(x, height * (0.5 - Math.max(-1, p[0]) * 0.46));
    ctx.lineTo(x, height * (0.5 - Math.min(1, p[1]) * 0.46));
  });
  ctx.stroke();
}
function drawSpec(canvas, bins) {
  if (!canvas) return;
  const rows = bins?.length || 96,
    cols = bins?.[0]?.length || 80;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d"),
    data = ctx.createImageData(cols, rows);
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++) {
      const rgb = palette[bins?.[rows - 1 - y]?.[x] || 0],
        p = (y * cols + x) * 4;
      data.data.set([...rgb, 255], p);
    }
  ctx.putImageData(data, 0, 0);
}
function render(value = {}) {
  const viz = value.visualization || {};
  for (const name of ["input", "output"]) {
    drawWave(
      $(`[data-wave=${name}]`),
      viz[name]?.waveform,
      name === "output" ? "#75f1c5" : "#91a5d5",
    );
    drawSpec($(`[data-spec=${name}]`), viz[name]?.spectrogram);
  }
  const seconds = value.input_seconds || 0,
    whole = Math.floor(seconds);
  $("[data-clock]").innerHTML =
    `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}<span> / 01:00</span>`;
  // Error responses carry no device; keep the last one the server named.
  if (typeof value.device === "string" && value.device)
    state.device = value.device;
  if (value.processing_seconds > 0)
    $("[data-speed]").textContent =
      `${(seconds / value.processing_seconds).toFixed(1)}× real time${state.device ? ` on ${state.device}` : ""} · ${Math.max(0, state.samples / (state.rate || 48000) - seconds).toFixed(1)} s pending`;
  controls();
}
// Resolves true if the player stops rewound to the start within two seconds:
// Gradio's player stops itself (pause, then seek to 0) once its waveform is
// ready, which can land just after playback started. A listener's own pause
// leaves the position where it was, and is respected.
function rewoundSoon(audio) {
  return new Promise((resolve) => {
    const done = (value) => {
      clearTimeout(timer);
      audio.removeEventListener("pause", onPause);
      resolve(value);
    };
    // The seek follows the pause in the same call; read the position after it.
    const onPause = () =>
      setTimeout(() => done(audio.currentTime === 0 && !audio.ended), 0);
    const timer = setTimeout(() => done(false), 2000);
    audio.addEventListener("pause", onPause);
  });
}
async function playCleanRecording() {
  const generation = state.generation;
  const wanted = () =>
    current(generation) && !state.starting && !state.capturing;
  // The HTML response arrives before the Audio component finishes loading its WAV.
  for (let attempt = 0; attempt < 100; attempt++) {
    if (!wanted()) return;
    const audio = Array.from(discoverPlayers()).find(
      (player) =>
        insidePlayer(player, "online-result") &&
        player.currentSrc &&
        player.readyState >= 1,
    );
    if (audio) {
      for (let tries = 0; tries < 3 && wanted(); tries++) {
        try {
          await audio.play();
        } catch {
          if (current(generation))
            status(
              "Ready to listen",
              "Press play below to hear your clean recording.",
            );
          return;
        }
        if (!(await rewoundSoon(audio))) return;
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}
watch("value", async () => {
  const value = props.value || {};
  if (state.disposed || value.take !== state.take || !state.take) return;
  if (
    ["active", "finished"].includes(value.phase) &&
    value.sequence !== state.sequence
  )
    return;
  render(value);
  clearFlight();
  if (value.phase === "error") {
    await fail(value.message);
    return;
  }
  if (value.phase === "ready") {
    const generation = state.generation;
    try {
      await startCapture();
    } catch (error) {
      if (current(generation)) await fail(error.message);
    }
  } else if (value.phase === "finished") {
    state.stopping = false;
    state.finishing = false;
    status(
      "Ready to listen",
      "Microphone paused. Play your clean voice below, or start a new recording.",
    );
    controls();
    if (state.listen) playCleanRecording();
  } else {
    if (value.limit_reached) await stop(false);
    pump();
  }
});
element.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "record") start();
  if (action === "listen") stop(true);
  if (action === "stop") stop(false);
});
// Pause capture before any player on either tab can feed speakers into the mic.
const onPlay = async (event) => {
  const audio =
    event.composedPath?.().find((node) => node instanceof HTMLMediaElement) ||
    event.target;
  if (!(audio instanceof HTMLMediaElement) || playedEvents.has(event)) return;
  playedEvents.add(event);
  if (audio instanceof HTMLMediaElement && state.capturing) {
    const generation = state.generation;
    audio.pause();
    await stop(false);
    // Track release has completed before the speaker is allowed to resume.
    if (current(generation)) audio.play().catch(() => {});
  }
};
function dispose() {
  state.disposed = true;
  state.generation += 1;
  state.take = null;
  state.capturing =
    state.starting =
    state.stopping =
    state.finishing =
    state.flushing =
      false;
  state.queue = [];
  clearFlight();
  release();
}
const onLeave = () => dispose();
const onReturn = () => {
  if (element.isConnected && state.disposed) {
    state.disposed = false;
    controls();
    status("Ready to record", "Start a new recording when you are ready.");
  }
};
window.addEventListener("pagehide", onLeave);
window.addEventListener("pageshow", onReturn);
observer = new MutationObserver((records) => {
  if (!element.isConnected) {
    dispose();
    forgetPlayers();
    window.removeEventListener("pagehide", onLeave);
    window.removeEventListener("pageshow", onReturn);
    return;
  }
  // Canvas/clock updates do not change the player inventory.
  if (
    discoveryTimer === null &&
    records.some((record) => !element.contains(record.target))
  ) {
    discoveryTimer = setTimeout(() => {
      discoveryTimer = null;
      discoverPlayers();
    }, 0);
  }
});
discoverPlayers();
render(props.value || {});
