/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const FRAME_MS = 1000 / 50;
  const HOLD_MS = 400;
  const labels = {
    forward: "Move forward", backward: "Move backward", left: "Move left",
    right: "Move right", yaw_left: "Pan left", yaw_right: "Pan right", idle: "Hold pose",
  };
  const keys = {
    KeyW: "forward", KeyS: "backward", KeyA: "left", KeyD: "right",
    ArrowLeft: "yaw_left", ArrowRight: "yaw_right",
  };
  const buttons = [...document.querySelectorAll("[data-action]")];
  const state = {
    socket: null, connected: false, ready: false, access: "available",
    active: false, preparing: false, stopping: false, waiting: false, awaitingStart: false,
    playing: false, pending: null, held: null, holdTimer: null,
    sequence: 0, epoch: 0, frames: 0, chunks: 0, image: null,
    examples: [], requestTimes: new Map(), metadata: [],
    frameURL: null, cancelLoad: null, lastActivity: -Infinity,
  };
  let toastTimer;

  function send(message) {
    if (state.socket?.readyState !== WebSocket.OPEN) return false;
    state.socket.send(JSON.stringify(message));
    return true;
  }

  function status(message) { $("stateLabel").textContent = message; }

  function error(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 9000);
  }

  function canControl() {
    return state.connected && state.active && !state.preparing && !state.stopping && state.access === "owner";
  }

  function updateControls() {
    const allowed = state.access === "owner" || state.access === "available";
    $("startButton").disabled = !state.connected || !allowed || state.preparing || state.stopping;
    $("startButtonLabel").textContent = state.preparing ? "Preparing…" : state.active ? "Restart world" : "Start world";
    $("stopButton").disabled = !state.connected || (!state.active && !state.preparing) || state.stopping;
    buttons.forEach((button) => { button.disabled = !canControl(); });
    $("keyboardStatus").textContent = canControl() ? "KEYBOARD LIVE" : "STANDBY";
    $("keyboardStatus").classList.toggle("active", canControl());
    $("transportLight").classList.toggle("active", state.active);
    $("takeDot").classList.toggle("active", state.active);
    $("takeLabel").textContent = state.preparing ? "PREPARING" : state.active ? "LIVE WORLD" : "STANDBY";
  }

  function clearHeld() {
    state.held = null;
    clearTimeout(state.holdTimer);
    buttons.forEach((button) => button.classList.remove("pressed"));
  }

  function cancelPlayback() {
    state.epoch += 1;
    state.playing = false;
    state.waiting = false;
    state.pending = null;
    state.requestTimes.clear();
    state.cancelLoad?.();
    state.cancelLoad = null;
    clearHeld();
  }

  function resetTiming() {
    for (const id of ["modelLatency", "nonModelLatency", "totalLatency"]) $(id).textContent = "—";
    $("latencySample").textContent = "Make a camera move to measure";
  }

  function activity() {
    const now = performance.now();
    if (state.access === "owner" && now - state.lastActivity > 5000) {
      send({ type: "activity" });
      state.lastActivity = now;
    }
  }

  function start() {
    if ($("startButton").disabled) return;
    const prompt = $("promptInput").value.trim();
    const seed = Number($("seedInput").value);
    if (!prompt) { error("Describe your scene before starting."); $("promptInput").focus(); return; }
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || $("seedInput").value === "") {
      error("Choose a whole-number seed between 0 and 4294967295."); $("seedInput").focus(); return;
    }
    const example = $("exampleSelect").value;
    if (!state.image && !example) { error("Choose an example or upload a starting image."); return; }
    cancelPlayback();
    resetTiming();
    state.active = true;
    state.preparing = true;
    state.awaitingStart = true;
    state.waiting = true;
    state.frames = 0;
    state.chunks = 0;
    $("frameCount").textContent = "0000";
    $("chunkCount").textContent = "0";
    $("actionReadout").textContent = "Preparing your world";
    $("toast").hidden = true;
    const message = { type: "start", prompt, seed };
    if (state.image) message.image = state.image;
    else message.example = example;
    send(message);
    status("Preparing your world…");
    updateControls();
  }

  function stop() {
    if ($("stopButton").disabled) return;
    cancelPlayback();
    state.active = false;
    state.preparing = false;
    state.stopping = true;
    send({ type: "stop" });
    status("Stopping session…");
    updateControls();
  }

  function dispatch(action, inputAt) {
    if (!canControl()) return;
    const requestId = `camera-${++state.sequence}`;
    if (inputAt !== null) state.requestTimes.set(requestId, inputAt);
    state.waiting = true;
    $("actionReadout").textContent = labels[action];
    status(`Generating · ${labels[action].toLowerCase()}`);
    if (!send({ type: "action", action, request_id: requestId })) {
      state.requestTimes.delete(requestId);
      state.waiting = false;
    }
  }

  function requestAction(action, inputAt) {
    if (!canControl()) return;
    if (state.waiting || state.playing) {
      // Only the newest unsent input is kept; a fast key sequence cannot build a backlog.
      state.pending = { action, inputAt };
      return;
    }
    dispatch(action, inputAt);
  }

  function continueHeld() {
    clearTimeout(state.holdTimer);
    if (!canControl() || state.waiting || state.playing) return;
    if (state.pending) {
      const pending = state.pending;
      state.pending = null;
      dispatch(pending.action, pending.inputAt);
      return;
    }
    if (!state.held) return;
    const remaining = HOLD_MS - (performance.now() - state.held.since);
    if (remaining > 0) state.holdTimer = setTimeout(continueHeld, remaining);
    else dispatch(state.held.action, null);
  }

  function press(action, source) {
    if (!canControl() || state.held?.source === source) return;
    clearHeld();
    const inputAt = performance.now();
    state.held = { action, source, since: inputAt };
    buttons.forEach((button) => button.classList.toggle("pressed", button.dataset.action === action));
    requestAction(action, inputAt);
    state.holdTimer = setTimeout(continueHeld, HOLD_MS);
  }

  function release(source) {
    if (state.held?.source === source) clearHeld();
  }

  function showTiming(metadata, loadedAt) {
    const model = Number.isFinite(metadata.model_ms) ? metadata.model_ms : null;
    const inputAt = state.requestTimes.get(metadata.request_id);
    state.requestTimes.delete(metadata.request_id);
    $("modelLatency").textContent = model === null ? "—" : `${Math.round(model)} ms`;
    const total = inputAt === undefined ? null : loadedAt - inputAt;
    $("totalLatency").textContent = total === null ? "—" : `${Math.round(total)} ms`;
    // No input timestamp means no end-to-end sample. Never use model time as a fallback.
    const overhead = total === null || model === null ? null : total - model;
    $("nonModelLatency").textContent = overhead === null ? "—" : `${Math.round(Math.max(0, overhead))} ms`;
    $("latencySample").textContent = total === null ? "Model time · no new camera input" : "Latest camera input · first frame loaded";
  }

  function loadFrame(blob, epoch) {
    return new Promise((resolve, reject) => {
      if (epoch !== state.epoch) { resolve(null); return; }
      const frame = $("worldFrame");
      const url = URL.createObjectURL(blob);
      const previousURL = state.frameURL;
      state.frameURL = url;
      let finished = false;
      function clean() {
        finished = true;
        frame.onload = null;
        frame.onerror = null;
        state.cancelLoad = null;
        if (previousURL) URL.revokeObjectURL(previousURL);
      }
      state.cancelLoad = () => {
        if (finished) return;
        clean();
        resolve(null);
      };
      frame.onload = () => {
        const loadedAt = performance.now();
        clean();
        if (epoch !== state.epoch) { resolve(null); return; }
        frame.hidden = false;
        $("stagePlaceholder").hidden = true;
        state.frames += 1;
        $("frameCount").textContent = String(state.frames).padStart(4, "0");
        resolve(loadedAt);
      };
      frame.onerror = () => { clean(); reject(new Error("The browser could not decode a generated frame.")); };
      frame.src = url;
    });
  }

  async function playChunk(metadata, buffer) {
    if (!state.active || state.stopping || state.awaitingStart) return;
    const sizes = metadata.frame_sizes;
    if (!Array.isArray(sizes) || !sizes.length || sizes.some((n) => !Number.isInteger(n) || n <= 0) || sizes.reduce((a, b) => a + b, 0) !== buffer.byteLength) {
      error("The server sent an invalid frame block. Restart the session.");
      stop();
      return;
    }
    const epoch = state.epoch;
    state.preparing = false;
    state.playing = true;
    if (metadata.request_id === null) $("actionReadout").textContent = "Hold pose";
    updateControls();
    if (metadata.width && metadata.height) $("resolutionLabel").textContent = `${metadata.width} × ${metadata.height}`;
    let offset = 0;
    let firstLoadedAt;
    try {
      for (let index = 0; index < sizes.length; index += 1) {
        if (epoch !== state.epoch) return;
        if (index) {
          const delay = firstLoadedAt + index * FRAME_MS - performance.now();
          if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
        }
        if (epoch !== state.epoch) return;
        const loadedAt = await loadFrame(new Blob([buffer.slice(offset, offset + sizes[index])], { type: "image/jpeg" }), epoch);
        offset += sizes[index];
        if (loadedAt === null || epoch !== state.epoch) return;
        if (index === 0) { firstLoadedAt = loadedAt; showTiming(metadata, loadedAt); }
      }
      state.chunks += 1;
      $("chunkCount").textContent = String(state.chunks);
      state.playing = false;
      state.waiting = false;
      // The next held action is sent only after this block's final frame has loaded.
      send({ type: "displayed", chunk_index: metadata.chunk_index });
      if (metadata.terminal) {
        state.active = false;
        state.pending = null;
        clearHeld();
        status("World complete · start another scene");
        updateControls();
      } else {
        status("Frame held · move to continue");
        continueHeld();
      }
    } catch (failure) {
      if (epoch !== state.epoch) return;
      error(failure.message);
      stop();
    }
  }

  function queueState(message) {
    const previous = state.access;
    state.access = message.status;
    if (previous === "owner" && state.access !== "owner") {
      cancelPlayback();
      state.active = false;
      state.preparing = false;
      state.stopping = false;
      status("Session released · start when the server is available");
    }
    const queued = message.status === "queued";
    const busy = message.status === "busy";
    const waiting = Number(message.waiting_count) || 0;
    $("waitingRoom").hidden = !(queued || busy || (message.status === "owner" && waiting > 0));
    $("joinQueueButton").hidden = !busy;
    $("leaveQueueButton").hidden = !queued;
    $("waitingRoomTitle").textContent = queued ? `Waiting room · position ${message.position}` : busy ? "Server in use" : `${waiting} ${waiting === 1 ? "person" : "people"} waiting`;
    $("waitingRoomMessage").textContent = queued ? "Your controls will unlock when it is your turn." : busy ? "One person can explore at a time. Join the queue to take a turn." : "Keep exploring, or stop your session to let the next person play.";
    $("waitingRoomTimer").textContent = message.status === "owner" && Number.isFinite(message.idle_remaining_seconds) ? `Idle timeout: ${Math.ceil(message.idle_remaining_seconds)}s` : "";
    if (previous === "queued" && message.status === "owner") status("Your turn · start a world to explore");
    updateControls();
  }

  function handleStatus(message) {
    // Frames already in transit from the old world must not undo a restart.
    if (state.awaitingStart && message.state !== "preparing" && message.state !== "stopped") return;
    if (message.message) status(message.message);
    if (message.state === "preparing") { state.preparing = true; state.awaitingStart = false; }
    if (message.state === "ready") { state.ready = true; state.preparing = false; }
    if (message.state === "stopping") { state.stopping = true; clearHeld(); }
    if (message.state === "stopped") {
      cancelPlayback();
      state.active = false;
      state.preparing = false;
      state.stopping = false;
      state.awaitingStart = false;
    }
    // A terminal block is acknowledged after playback; a completion status may
    // arrive before its binary frames, so it must not cancel that playback.
    if (message.state === "complete" && !state.waiting && !state.playing) {
      state.active = false;
      clearHeld();
    }
    updateControls();
  }

  function connect() {
    const url = new URL("ws", window.location.href);
    url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    state.socket = socket;
    socket.onopen = () => {
      state.connected = true;
      $("connectionLabel").textContent = "Connected";
      $("connectionPill").classList.add("online");
      updateControls();
    };
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") {
        const metadata = state.metadata.shift();
        if (metadata) void playChunk(metadata, event.data);
        else error("Received frames without timing metadata. Restart the session.");
        return;
      }
      let message;
      try { message = JSON.parse(event.data); }
      catch { error("The server sent an unreadable message."); return; }
      switch (message.type) {
        case "hello":
          state.ready = Boolean(message.ready);
          $("gpuLabel").textContent = `${message.gpus} GPU${message.gpus === 1 ? "" : "s"}`;
          status(state.ready ? "Ready · set the scene and start your world" : "Ready · first start warms up the model");
          updateControls();
          break;
        case "queue_state": queueState(message); break;
        case "session":
          if (message.status === "stopped") {
            handleStatus({ state: "stopped", message: message.reason === "idle_timeout" ? "Session released after inactivity" : "Session stopped · start a new world" });
          }
          break;
        case "status": handleStatus(message); break;
        case "chunk": state.metadata.push(message); break;
        case "error":
          error(message.message || "The server could not complete the request.");
          cancelPlayback();
          state.preparing = false;
          state.stopping = false;
          state.active = false;
          state.awaitingStart = false;
          status("Session interrupted · restart to try again");
          updateControls();
          break;
        default: break;
      }
    };
    socket.onclose = () => {
      cancelPlayback();
      state.connected = false;
      state.ready = false;
      state.active = false;
      state.preparing = false;
      state.stopping = false;
      state.awaitingStart = false;
      state.metadata = [];
      state.access = "available";
      $("waitingRoom").hidden = true;
      $("connectionLabel").textContent = "Reconnecting";
      $("connectionPill").classList.remove("online");
      status("Connection lost · reconnecting to your server…");
      updateControls();
      setTimeout(connect, 2000);
    };
    socket.onerror = () => { /* onclose handles reconnecting without duplicate alerts. */ };
  }

  function updatePrompt() {
    $("promptCount").textContent = `${$("promptInput").value.length} / 1600`;
    activity();
  }

  function selectExample() {
    const example = state.examples.find((item) => String(item.id) === $("exampleSelect").value);
    if (!example) return;
    state.image = null;
    $("imageInput").value = "";
    $("imagePreview").src = example.image_url;
    $("imagePreview").hidden = false;
    $("imagePlaceholder").hidden = true;
    $("imageName").textContent = example.label;
    $("imageName").hidden = false;
    $("promptInput").value = example.prompt || "";
    updatePrompt();
  }

  async function loadExamples() {
    try {
      const response = await fetch(new URL("api/examples", window.location.href));
      if (!response.ok) throw new Error("Examples unavailable");
      const examples = await response.json();
      if (!Array.isArray(examples)) throw new Error("Invalid examples response");
      state.examples = examples;
      $("exampleSelect").replaceChildren();
      const custom = new Option(examples.length ? "Choose an example…" : "Upload an image to begin", "");
      $("exampleSelect").add(custom);
      examples.forEach((example) => $("exampleSelect").add(new Option(example.label, String(example.id))));
      if (examples.length) { $("exampleSelect").value = String(examples[0].id); selectExample(); }
    } catch {
      $("exampleSelect").replaceChildren(new Option("Examples unavailable · upload an image", ""));
    }
  }

  $("startButton").addEventListener("click", start);
  $("stopButton").addEventListener("click", stop);
  $("joinQueueButton").addEventListener("click", () => send({ type: "join_queue" }));
  $("leaveQueueButton").addEventListener("click", () => send({ type: "leave_queue" }));
  $("promptInput").addEventListener("input", updatePrompt);
  $("seedInput").addEventListener("input", activity);
  $("exampleSelect").addEventListener("change", selectExample);
  $("randomSeedButton").addEventListener("click", () => {
    $("seedInput").value = crypto.getRandomValues(new Uint32Array(1))[0];
    activity();
  });
  $("uploadButton").addEventListener("click", () => $("imageInput").click());
  $("imageInput").addEventListener("change", () => {
    const file = $("imageInput").files[0];
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { error("Choose a JPEG, PNG, or WebP image."); return; }
    if (file.size > 8 * 1024 * 1024) { error("Choose an image smaller than 8 MiB."); return; }
    const reader = new FileReader();
    reader.onload = () => {
      state.image = reader.result;
      $("imagePreview").src = reader.result;
      $("imagePreview").hidden = false;
      $("imagePlaceholder").hidden = true;
      $("imageName").textContent = file.name;
      $("imageName").hidden = false;
      $("exampleSelect").value = "";
      activity();
    };
    reader.onerror = () => error("Your browser could not read that image. Try another file.");
    reader.readAsDataURL(file);
  });
  $("imagePreview").addEventListener("error", () => {
    $("imagePreview").hidden = true;
    $("imagePlaceholder").hidden = false;
    $("imagePlaceholder").textContent = "Image preview unavailable";
  });

  buttons.forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || button.disabled) return;
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      press(button.dataset.action, `pointer-${event.pointerId}`);
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => {
      button.addEventListener(name, (event) => release(`pointer-${event.pointerId}`));
    });
    // Enter/Space and assistive-technology activation have no pointer event.
    button.addEventListener("click", (event) => {
      if (event.detail === 0) requestAction(button.dataset.action, performance.now());
    });
  });

  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); start(); return; }
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    if (event.target.closest("input, textarea, select, [contenteditable='true']")) return;
    const action = keys[event.code];
    if (!action || !canControl()) return;
    event.preventDefault();
    press(action, event.code);
  });
  window.addEventListener("keyup", (event) => release(event.code));
  window.addEventListener("blur", () => { clearHeld(); state.pending = null; });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { clearHeld(); state.pending = null; }
  });

  void loadExamples();
  connect();
})();
