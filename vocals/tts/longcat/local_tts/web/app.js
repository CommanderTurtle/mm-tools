const SENTENCE_ENDINGS = new Set([".", "—", "!", "?"]);
const SENTENCE_CLOSERS = new Set(['"', "'", "”", "’", ")", "]", "}", "»", "」", "』"]);

function voiceSpans(source) {
  const spans = [];
  const pattern = /\{\{([\s\S]*?)\}\}|\{([\s\S]*?)\}/g;
  let cursor = 0;
  let match;
  const add = (voice, text) => {
    if (text.length) spans.push({ voice, text });
  };
  while ((match = pattern.exec(source)) !== null) {
    add(1, source.slice(cursor, match.index));
    const trailing = source.slice(pattern.lastIndex).match(/^[.—!?]["'”’)\]»」』]*/)?.[0] ?? "";
    add(match[1] === undefined ? 2 : 3, `${match[1] ?? match[2]}${trailing}`);
    pattern.lastIndex += trailing.length;
    cursor = pattern.lastIndex;
  }
  add(1, source.slice(cursor));
  return spans;
}

function characterCount(value) {
  return Array.from(value).length;
}

function spokenCharacterCount(source) {
  return voiceSpans(source).reduce((total, span) => total + characterCount(span.text), 0);
}

function sentenceBoundaries(text) {
  const characters = Array.from(text);
  const boundaries = [];
  let pending = null;
  characters.forEach((character, index) => {
    if (SENTENCE_ENDINGS.has(character)) {
      pending = index + 1;
    } else if (pending !== null && SENTENCE_CLOSERS.has(character)) {
      pending = index + 1;
    } else if (pending !== null) {
      boundaries.push(pending);
      pending = null;
    }
  });
  if (pending !== null) boundaries.push(pending);
  return boundaries;
}

function splitVoiceSpan(span, target) {
  const fragments = [];
  let remaining = span.text.trim();
  while (remaining) {
    const characters = Array.from(remaining);
    if (characters.length <= target) {
      fragments.push({ voice: span.voice, text: remaining });
      break;
    }
    const boundaries = sentenceBoundaries(remaining);
    const before = boundaries.filter((boundary) => boundary <= target).at(-1);
    const split = before ?? boundaries.find((boundary) => boundary > target);
    if (split === undefined) {
      fragments.push({ voice: span.voice, text: remaining });
      break;
    }
    const head = characters.slice(0, split).join("").trim();
    const tail = characters.slice(split).join("").trim();
    if (!head || !tail) {
      fragments.push({ voice: span.voice, text: remaining });
      break;
    }
    fragments.push({ voice: span.voice, text: head });
    remaining = tail;
  }
  return fragments;
}

function planFragments(source, target) {
  const safeTarget = Math.max(1, Math.trunc(Number(target) || 1));
  return voiceSpans(source).flatMap((span) => splitVoiceSpan(span, safeTarget));
}

const $ = (selector) => document.querySelector(selector);
const form = $("#tts-form");
const textInput = $("#text");
const statusDot = $("#status-dot");
const modelState = $("#model-state");
const modelDetail = $("#model-detail");
const modelToggle = $("#model-toggle");
const secondaryToggle = $("#secondary-toggle");
const synthesizeButton = $("#synthesize");
const message = $("#message");
const player = $("#player");
const result = $("#result");
const finalResult = $("#final-result");
const download = $("#download");
const hoist = $("#hoist");
const autoConcatenate = $("#auto-concatenate");
const concatenateOptions = $("#concatenate-options");
const characterTarget = $("#character-target");
const characterTargetSlider = $("#character-target-slider");
const fragmentEstimate = $("#fragment-estimate");
const fragmentSession = $("#fragment-session");
const fragmentList = $("#fragment-list");
const voiceMarkers = $("#voice-markers");
const additionalVoices = $("#additional-voices");
const voiceReferenceTitle = $("#voice-reference-title");

const voiceRoutes = {
  1: { audio: $("#prompt-audio"), transcript: $("#prompt-text"), fileName: $("#file-name") },
  2: { audio: $("#voice-2-audio"), transcript: $("#voice-2-text"), fileName: $("#voice-2-file-name") },
  3: { audio: $("#voice-3-audio"), transcript: $("#voice-3-text"), fileName: $("#voice-3-file-name") },
};

let loaded = false;
let localLoaded = false;
let activeBase = "";
let workflowBusy = false;
let uiConfig = { secondary_port: 8230, secondary_scheme: "http", router_url: "http://127.0.0.1:8182" };
let resultUrl = null;
let resultBlob = null;
let fragments = [];

async function apiAt(base, path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return response;
}

const api = (path, options = {}) => apiAt(activeBase, path, options);
const localApi = (path, options = {}) => apiAt("", path, options);

function secondaryBase() {
  const bareHost = location.hostname.replace(/^\[|\]$/g, "");
  const host = bareHost.includes(":") ? `[${bareHost}]` : bareHost;
  return `${uiConfig.secondary_scheme}://${host}:${uiConfig.secondary_port}`;
}

function renderStatus(state, source) {
  loaded = state.loaded;
  statusDot.className = state.busy ? "busy" : loaded ? "ready" : "";
  modelState.textContent = state.busy ? "Synthesizing" : loaded ? `${source} model resident` : `${source} model unloaded`;
  modelDetail.textContent = `${state.dtype} · ${state.cuda ?? state.device}${state.vram_allocated_gb ? ` · ${state.vram_allocated_gb} GB` : ""}`;
}

async function refreshStatus() {
  try {
    const localState = await (await localApi("/api/status")).json();
    localLoaded = localState.loaded;
    modelToggle.textContent = localLoaded ? "Unload local" : "Load";
    modelToggle.disabled = localState.busy || workflowBusy;
    secondaryToggle.disabled = workflowBusy;
    secondaryToggle.textContent = activeBase ? "Use local model" : "Check Secondary Load";

    if (activeBase) {
      try {
        const secondaryState = await (await apiAt(activeBase, "/api/status")).json();
        renderStatus(secondaryState, "Secondary HTTP");
        return;
      } catch (error) {
        activeBase = "";
        secondaryToggle.textContent = "Check Secondary Load";
        message.textContent = `Secondary service disconnected; using local UI. ${error.message}`;
      }
    }
    renderStatus(localState, "UI-local");
  } catch (error) {
    modelState.textContent = "UI service unavailable";
    modelDetail.textContent = error.message;
  }
}

modelToggle.addEventListener("click", async () => {
  modelToggle.disabled = true;
  activeBase = "";
  secondaryToggle.textContent = "Check Secondary Load";
  message.textContent = localLoaded ? "Releasing the UI-local model…" : "Loading the UI-local 3.5B checkpoint…";
  try {
    await localApi(localLoaded ? "/api/unload" : "/api/load", { method: "POST" });
    message.textContent = localLoaded ? "UI-local model unloaded." : "UI-local model ready.";
  } catch (error) { message.textContent = error.message; }
  await refreshStatus();
});

secondaryToggle.addEventListener("click", async () => {
  if (activeBase) {
    activeBase = "";
    message.textContent = "Using the UI-local model path.";
    await refreshStatus();
    return;
  }
  secondaryToggle.disabled = true;
  message.textContent = "Checking the secondary HTTP model…";
  try {
    const candidate = secondaryBase();
    const state = await (await apiAt(candidate, "/api/status")).json();
    if (!state.loaded) throw new Error("Secondary service is running, but its model is unloaded.");
    activeBase = candidate;
    message.textContent = "Attached to the already-loaded secondary HTTP model.";
  } catch (error) {
    message.textContent = `Secondary load unavailable: ${error.message}`;
  } finally {
    secondaryToggle.disabled = false;
    await refreshStatus();
  }
});

function targetCharacters() {
  const value = Number(characterTarget.value);
  return Number.isFinite(value) && value >= 1 ? Math.trunc(value) : 1;
}

function updateTextPlan() {
  if (!autoConcatenate.checked) {
    $("#characters").textContent = `${textInput.value.length} characters`;
    return;
  }
  const planned = planFragments(textInput.value, targetCharacters());
  $("#characters").textContent = `${spokenCharacterCount(textInput.value)} spoken characters`;
  fragmentEstimate.textContent = `${planned.length} planned fragment${planned.length === 1 ? "" : "s"}`;
}

function syncCharacterTarget(value, source) {
  const parsed = Math.min(100000, Math.max(1, Math.trunc(Number(value) || 1)));
  if (source !== characterTarget) characterTarget.value = String(parsed);
  if (parsed > Number(characterTargetSlider.max)) characterTargetSlider.max = String(parsed);
  if (source !== characterTargetSlider) characterTargetSlider.value = String(parsed);
  updateTextPlan();
}

textInput.addEventListener("input", updateTextPlan);
characterTarget.addEventListener("input", () => syncCharacterTarget(characterTarget.value, characterTarget));
characterTarget.addEventListener("change", () => syncCharacterTarget(characterTarget.value));
characterTargetSlider.addEventListener("input", () => syncCharacterTarget(characterTargetSlider.value, characterTargetSlider));

Object.values(voiceRoutes).forEach((route) => {
  route.audio.addEventListener("change", () => {
    route.fileName.textContent = route.audio.files[0]?.name ?? "Choose WAV, M4A, or MP3";
  });
});

const duration = form.elements.duration_scale;
duration.addEventListener("input", () => { $("#duration-value").textContent = `${Number(duration.value).toFixed(2)}×`; });

document.querySelectorAll("[data-wrap-voice]").forEach((button) => {
  button.addEventListener("click", () => {
    const voice = Number(button.dataset.wrapVoice);
    const prefix = voice === 3 ? "{{" : "{";
    const suffix = voice === 3 ? "}}" : "}";
    const start = textInput.selectionStart;
    const end = textInput.selectionEnd;
    const selected = textInput.value.slice(start, end);
    textInput.setRangeText(`${prefix}${selected}${suffix}`, start, end, "end");
    if (!selected) textInput.setSelectionRange(start + prefix.length, start + prefix.length);
    textInput.focus();
    updateTextPlan();
  });
});

autoConcatenate.addEventListener("change", () => {
  const enabled = autoConcatenate.checked;
  concatenateOptions.hidden = !enabled;
  voiceMarkers.hidden = !enabled;
  additionalVoices.hidden = !enabled;
  voiceReferenceTitle.textContent = enabled ? "Voice 1 reference" : "Voice reference";
  synthesizeButton.querySelector("span").textContent = enabled ? "Generate & concatenate" : "Generate speech";
  renderFragments();
  updateTextPlan();
});

function parseSeed(value) {
  const seed = Number(value);
  if (!Number.isSafeInteger(seed) || seed < 0) throw new Error("Seed must be a non-negative safe integer.");
  return seed;
}

function validateVoiceRoute(voice, required) {
  const route = voiceRoutes[voice];
  const audio = route.audio.files[0];
  const transcript = route.transcript.value.trim();
  if (required && (!audio || !transcript)) {
    throw new Error(`Voice ${voice} needs both reference audio and its exact transcript.`);
  }
  if (audio && !transcript) throw new Error(`Add the exact transcript for Voice ${voice}.`);
  if (!audio && transcript) throw new Error(`Add reference audio for Voice ${voice}.`);
}

function synthesisBody(text, seed, voice) {
  const body = new FormData(form);
  body.set("text", text);
  body.set("seed", String(seed));
  if (voice !== 1) {
    const route = voiceRoutes[voice];
    const audio = route.audio.files[0];
    body.delete("prompt_audio");
    body.delete("prompt_text");
    body.set("prompt_audio", audio, audio.name);
    body.set("prompt_text", route.transcript.value.trim());
  }
  return body;
}

function responseMetadata(response) {
  try { return JSON.parse(response.headers.get("X-LongCat-Metadata") ?? "{}"); }
  catch { return {}; }
}

async function synthesizeFragment(fragment, seed = fragment.pendingSeed) {
  const response = await api("/api/synthesize", {
    method: "POST",
    body: synthesisBody(fragment.text, seed, fragment.voice),
  });
  return { blob: await response.blob(), metadata: responseMetadata(response), seed };
}

async function concatenateFragments(source) {
  const body = new FormData();
  source.forEach((fragment, index) => body.append("chunks", fragment.blob, `longcat-part-${String(index + 1).padStart(2, "0")}.wav`));
  const response = await api("/api/concatenate", { method: "POST", body });
  return { blob: await response.blob(), metadata: responseMetadata(response) };
}

function setFinalResult(blob, metadataText, filename = "longcat.wav") {
  const nextUrl = URL.createObjectURL(blob);
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = nextUrl;
  resultBlob = blob;
  player.src = resultUrl;
  download.href = resultUrl;
  download.download = filename;
  $("#result-meta").textContent = metadataText;
  finalResult.hidden = false;
  result.hidden = false;
}

function clearFragments() {
  fragments.forEach((fragment) => {
    if (fragment.url) URL.revokeObjectURL(fragment.url);
  });
  fragments = [];
  renderFragments();
}

function fragmentMetadataText(fragment) {
  if (fragment.state === "queued") return "Queued";
  if (fragment.state === "generating") return "Synthesizing…";
  if (fragment.state === "rerolling") return "Rerolling; prior WAV retained…";
  if (fragment.state === "error") return fragment.error;
  if (fragment.error) return `Prior WAV retained · ${fragment.error}`;
  return `${fragment.metadata?.audio_seconds ?? "?"}s · seed ${fragment.seed}`;
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderFragments() {
  fragmentSession.hidden = !autoConcatenate.checked || fragments.length === 0;
  fragmentList.replaceChildren();
  fragments.forEach((fragment, index) => {
    const card = makeElement("article", "fragment-card");
    card.dataset.state = fragment.error ? "error" : fragment.state;

    const head = makeElement("div", "fragment-head");
    const heading = makeElement("div", "fragment-heading");
    heading.append(makeElement("b", "", `Part ${String(index + 1).padStart(2, "0")}`));
    heading.append(makeElement("span", "voice-badge", `VOICE ${fragment.voice}`));
    head.append(heading, makeElement("small", "fragment-status", fragmentMetadataText(fragment)));
    card.append(head, makeElement("p", "fragment-text", fragment.text));

    if (fragment.url) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = fragment.url;
      card.append(audio);
    }

    const controls = makeElement("div", "fragment-controls");
    const rangeLabel = makeElement("label");
    const pendingSeed = fragment.pendingSeed ?? fragment.seed;
    const seedOutput = makeElement("span", "fragment-active-seed", `Rendered: ${fragment.seed ?? "pending"}`);
    rangeLabel.append(document.createTextNode("Seed"), seedOutput);
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(Math.max(0, pendingSeed - 2048));
    range.max = String(Math.max(4096, pendingSeed + 2048));
    range.step = "1";
    range.value = String(pendingSeed);
    rangeLabel.append(range);

    const number = document.createElement("input");
    number.type = "number";
    number.min = "0";
    number.step = "1";
    number.value = String(pendingSeed);
    number.setAttribute("aria-label", `Seed for part ${index + 1}`);
    const setPendingSeed = (value) => {
      const parsed = Number(value);
      if (!Number.isSafeInteger(parsed) || parsed < 0) return;
      fragment.pendingSeed = parsed;
      number.value = String(parsed);
      if (parsed < Number(range.min)) range.min = String(parsed);
      if (parsed > Number(range.max)) range.max = String(parsed);
      range.value = String(parsed);
      seedOutput.textContent = `Rendered: ${fragment.seed ?? "pending"}${parsed !== fragment.seed ? ` · next: ${parsed}` : ""}`;
    };
    range.addEventListener("input", () => setPendingSeed(range.value));
    number.addEventListener("input", () => setPendingSeed(number.value));
    number.addEventListener("change", () => {
      if (!Number.isSafeInteger(Number(number.value)) || Number(number.value) < 0) number.value = String(fragment.pendingSeed ?? fragment.seed);
    });
    controls.append(rangeLabel, number);

    if (fragment.url) {
      const partDownload = makeElement("a", "download", "Download WAV");
      partDownload.href = fragment.url;
      partDownload.download = `longcat-part-${String(index + 1).padStart(2, "0")}.wav`;
      controls.append(partDownload);
    }
    const reroll = makeElement("button", "", "Reroll");
    reroll.type = "button";
    reroll.disabled = workflowBusy || !fragment.blob;
    reroll.addEventListener("click", () => rerollFragment(index));
    controls.append(reroll);
    card.append(controls);
    fragmentList.append(card);
  });
}

function setWorkflowBusy(busy) {
  workflowBusy = busy;
  synthesizeButton.disabled = busy;
  autoConcatenate.disabled = busy;
  modelToggle.disabled = busy;
  secondaryToggle.disabled = busy;
  synthesizeButton.classList.toggle("working", busy);
  renderFragments();
}

function finalMetadataText(merged, source) {
  const renderSeconds = source.reduce((total, fragment) => total + Number(fragment.metadata?.generation_seconds ?? 0), 0);
  return `${merged.audio_seconds ?? "?"}s audio · ${renderSeconds.toFixed(2)}s render · ${source.length} fragment${source.length === 1 ? "" : "s"}`;
}

async function generateSingle() {
  const prompt = voiceRoutes[1].audio.files[0];
  if (prompt && !voiceRoutes[1].transcript.value.trim()) throw new Error("Add the exact transcript for the reference audio.");
  clearFragments();
  result.hidden = true;
  message.textContent = loaded ? "Diffusion synthesis is running…" : "Loading the UI-local model, then synthesizing…";
  const response = await api("/api/synthesize", { method: "POST", body: new FormData(form) });
  const blob = await response.blob();
  const metadata = responseMetadata(response);
  setFinalResult(blob, `${metadata.audio_seconds ?? "?"}s audio · ${metadata.generation_seconds ?? "?"}s render`);
}

async function generateConcatenated() {
  const planned = planFragments(textInput.value, targetCharacters());
  if (!planned.length) throw new Error("Text is empty after voice selectors are removed.");
  const usedVoices = new Set(planned.map((fragment) => fragment.voice));
  validateVoiceRoute(1, false);
  if (usedVoices.has(2)) validateVoiceRoute(2, true);
  if (usedVoices.has(3)) validateVoiceRoute(3, true);
  const seed = parseSeed(form.elements.seed.value);

  clearFragments();
  fragments = planned.map((fragment) => ({ ...fragment, seed, pendingSeed: seed, state: "queued", blob: null, url: null, metadata: null, error: "" }));
  finalResult.hidden = true;
  result.hidden = false;
  renderFragments();

  for (let index = 0; index < fragments.length; index += 1) {
    const fragment = fragments[index];
    fragment.state = "generating";
    message.textContent = `${loaded ? "Synthesizing" : "Loading the model, then synthesizing"} part ${index + 1} of ${fragments.length}…`;
    renderFragments();
    try {
      const generated = await synthesizeFragment(fragment, seed);
      fragment.blob = generated.blob;
      fragment.url = URL.createObjectURL(generated.blob);
      fragment.metadata = generated.metadata;
      fragment.seed = generated.seed;
      fragment.pendingSeed = generated.seed;
      fragment.state = "ready";
      loaded = true;
      renderFragments();
    } catch (error) {
      fragment.state = "error";
      fragment.error = error.message;
      renderFragments();
      throw new Error(`Part ${index + 1} failed: ${error.message}`);
    }
  }

  message.textContent = "Joining fragment WAVs…";
  const merged = await concatenateFragments(fragments);
  setFinalResult(merged.blob, finalMetadataText(merged.metadata, fragments), "longcat-concatenated.wav");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setWorkflowBusy(true);
  try {
    if (autoConcatenate.checked) await generateConcatenated();
    else await generateSingle();
    message.textContent = autoConcatenate.checked ? "Fragments and final WAV are ready." : "Generation complete.";
    loaded = true;
  } catch (error) {
    message.textContent = error.message;
  } finally {
    setWorkflowBusy(false);
    await refreshStatus();
  }
});

async function rerollFragment(index) {
  const fragment = fragments[index];
  if (!fragment?.blob || workflowBusy) return;
  try {
    validateVoiceRoute(fragment.voice, fragment.voice !== 1);
    fragment.pendingSeed = parseSeed(fragment.pendingSeed);
  } catch (error) {
    message.textContent = error.message;
    return;
  }

  setWorkflowBusy(true);
  fragment.state = "rerolling";
  fragment.error = "";
  renderFragments();
  message.textContent = `Rerolling part ${index + 1}; the current final WAV stays available…`;
  try {
    const generated = await synthesizeFragment(fragment, fragment.pendingSeed);
    const candidate = fragments.map((part, partIndex) => partIndex === index ? { ...part, ...generated } : part);
    const merged = await concatenateFragments(candidate);

    const nextUrl = URL.createObjectURL(generated.blob);
    if (fragment.url) URL.revokeObjectURL(fragment.url);
    Object.assign(fragment, generated, { url: nextUrl, pendingSeed: generated.seed, state: "ready", error: "" });
    setFinalResult(merged.blob, finalMetadataText(merged.metadata, fragments), "longcat-concatenated.wav");
    message.textContent = `Part ${index + 1} rerolled and the final WAV rebuilt.`;
  } catch (error) {
    fragment.state = "ready";
    fragment.error = error.message;
    message.textContent = `Reroll failed; the prior fragment and final WAV were kept. ${error.message}`;
  } finally {
    setWorkflowBusy(false);
    renderFragments();
    await refreshStatus();
  }
}

hoist.addEventListener("click", async () => {
  if (!resultBlob) {
    message.textContent = "Generate speech before hoisting it.";
    return;
  }
  hoist.disabled = true;
  message.textContent = "Hoisting generated audio to the Vox microphone router…";
  try {
    await apiAt(uiConfig.router_url, "/v1/forward", {
      method: "POST",
      headers: { "Content-Type": resultBlob.type || "audio/wav" },
      body: resultBlob,
    });
    message.textContent = "Audio queued on the virtual microphone.";
  } catch (error) {
    message.textContent = `Vox router unavailable: ${error.message}`;
  } finally {
    hoist.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  fragments.forEach((fragment) => { if (fragment.url) URL.revokeObjectURL(fragment.url); });
});

async function initialize() {
  try { uiConfig = await (await localApi("/api/ui-config")).json(); } catch {}
  updateTextPlan();
  await refreshStatus();
}

initialize();
