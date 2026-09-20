"use strict";

const $ = (id) => document.getElementById(id);
const qsa = (selector, root = document) => [...root.querySelectorAll(selector)];
const ACTIVE = new Set(["queued", "running", "cancelling"]);
const TERMINAL = new Set(["complete", "failed", "cancelled", "interrupted"]);
const pages = ["create", "prompt", "queue", "library", "compare", "presets", "help", "stems"];

const state = {
  meta: null,
  manifest: null,
  promptLoadedFor: null,
  mode: null,
  group: "all",
  values: {},
  jobs: [],
  assets: [],
  localFiles: new Map(),
  presets: [],
  page: "create",
  libraryFilter: "all",
  eventSource: null,
  pollTimer: 0,
  apiCode: "curl",
  token: localStorage.getItem("mm-tools-studio:token") || "",
};

function projectKey(suffix) {
  const project = state.manifest?.id || "pending";
  return `mm-tools:${project}:${suffix}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** power)).toFixed(power ? 1 : 0)} ${units[power]}`;
}

function formatTime(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value * 1000));
}

function relativeTime(value) {
  if (!value) return "—";
  const delta = Date.now() / 1000 - value;
  if (delta < 60) return `${Math.max(0, Math.round(delta))}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return formatTime(value);
}

function toast(message, kind = "info", duration = 4200) {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = String(message ?? "");
  $("toastRegion").append(item);
  setTimeout(() => item.remove(), duration);
}

async function api(path, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  if (options.body && !(options.body instanceof Blob) && !(options.body instanceof ArrayBuffer) && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401 && retry) {
    const supplied = window.prompt("This private-LAN studio requires its MM_STUDIO_TOKEN:", state.token);
    if (supplied) {
      state.token = supplied.trim();
      localStorage.setItem("mm-tools-studio:token", state.token);
      return api(path, options, false);
    }
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* response is not JSON */ }
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    throw new Error(detail);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("mm-tools-studio:theme", theme);
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
}

function hexToRgb(value) {
  const hex = String(value || "#90efb8").replace("#", "");
  const full = hex.length === 3 ? [...hex].map((x) => x + x).join("") : hex;
  const parsed = Number.parseInt(full, 16);
  if (!Number.isFinite(parsed)) return "144, 239, 184";
  return `${(parsed >> 16) & 255}, ${(parsed >> 8) & 255}, ${parsed & 255}`;
}

function setPage(name, push = true) {
  if (!pages.includes(name)) name = "create";
  state.page = name;
  qsa(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
  qsa(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === name));
  $("pageTitle").textContent = name === "help" ? "Guide & API" : name[0].toUpperCase() + name.slice(1);
  if (push) history.replaceState(null, "", `#${name}`);
  if (name === "queue") renderQueue();
  if (name === "library") renderLibrary();
  if (name === "compare") renderCompare();
  if (name === "presets") renderPresets();
  if (name === "prompt") renderPromptStudio();
  if (name === "stems") renderStemsPage();
}

function modeById(id) {
  return (state.manifest?.modes || []).find((mode) => mode.id === id);
}

function modeGroup(id) {
  return (state.manifest?.groups || []).find((group) => group.id === id);
}

function modeFields(mode, zone = "fields") {
  if (!mode) return [];
  const references = mode[`${zone}_refs`] || [];
  const shared = references.flatMap((name) => state.manifest?.field_sets?.[name] || []);
  return [...shared, ...(mode[zone] || [])];
}

function restoreLocalState() {
  try {
    state.presets = JSON.parse(localStorage.getItem(projectKey("presets")) || "[]");
    const draft = JSON.parse(localStorage.getItem(projectKey("draft")) || "null");
    if (draft && typeof draft === "object") {
      state.values = draft.values || {};
      state.group = draft.group || "all";
      state.mode = modeById(draft.mode) || null;
    }
  } catch (_) {
    state.presets = [];
    state.values = {};
  }
}

function saveDraft() {
  if (!state.manifest) return;
  localStorage.setItem(projectKey("draft"), JSON.stringify({
    mode: state.mode?.id || null,
    group: state.group,
    values: state.values,
  }));
}

function defaultsForMode(mode) {
  const fields = [...modeFields(mode), ...modeFields(mode, "advanced")];
  return Object.fromEntries(fields.filter((field) => field.id).map((field) => {
    let value = field.default;
    if (value === undefined) {
      if (field.type === "toggle") value = false;
      else if (field.type === "emotion_mixer") value = Object.fromEntries((field.options || []).map((item) => [item.value, 0]));
      else if (["nonverbal_events", "collection", "tags", "multi_choice", "multi_asset"].includes(field.type)) value = [];
      else value = "";
    }
    return [field.id, structuredClone(value)];
  }));
}

function controlsForMode(mode = state.mode) {
  if (!mode) return {};
  if (!state.values[mode.id]) state.values[mode.id] = defaultsForMode(mode);
  return state.values[mode.id];
}

function applyManifest() {
  const manifest = state.manifest;
  document.title = `${manifest.title} · mm-tools`;
  document.documentElement.style.setProperty("--accent", manifest.theme?.accent || "#90efb8");
  document.documentElement.style.setProperty("--accent-rgb", hexToRgb(manifest.theme?.accent));
  document.documentElement.style.setProperty("--accent-2", manifest.theme?.secondary || "#d6a86f");
  $("brandName").textContent = manifest.short_name || manifest.title;
  $("brandKind").textContent = manifest.kind || "LOCAL STUDIO";
  $("projectEyebrow").textContent = (manifest.id || "MM-TOOLS").toUpperCase();
  $("heroKicker").textContent = manifest.kicker || "LOCAL CREATIVE RUNTIME";
  $("heroTitle").textContent = manifest.hero || manifest.title;
  $("heroCopy").textContent = manifest.description || "Private local inference studio.";
  $("modelFootprint").textContent = manifest.vram || "GPU";
  $("runtimeContract").textContent = manifest.runtime_contract || "GPU-only · one job at a time";
  $("guideTitle").textContent = `${manifest.title} guide & API.`;
  $("guideCopy").textContent = manifest.guide || manifest.description || "Exact local setup and API contract.";
  $("heroBadges").replaceChildren(...(manifest.badges || []).map((text) => {
    const badge = document.createElement("span"); badge.textContent = text; return badge;
  }));
  renderGroupTabs();
  renderTaskCards();
  renderHelp();
  if (state.mode) selectMode(state.mode.id, false);
}

function renderGroupTabs() {
  const groups = [{ id: "all", title: "All workflows" }, ...(state.manifest.groups || [])];
  $("taskGroupTabs").replaceChildren(...groups.map((group) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = group.title;
    button.classList.toggle("active", group.id === state.group);
    button.onclick = () => {
      state.group = group.id;
      renderGroupTabs();
      renderTaskCards();
      saveDraft();
    };
    return button;
  }));
}

function renderTaskCards() {
  const search = $("taskSearch").value.trim().toLowerCase();
  const modes = (state.manifest.modes || []).filter((mode) => {
    const groupMatches = state.group === "all" || mode.group === state.group;
    const haystack = `${mode.title} ${mode.description || ""} ${(mode.keywords || []).join(" ")}`.toLowerCase();
    return groupMatches && (!search || haystack.includes(search));
  });
  if (!modes.length) {
    $("taskCards").innerHTML = '<div class="empty-state"><div><b>No matching native workflow</b>Try another task name or capability.</div></div>';
    return;
  }
  $("taskCards").replaceChildren(...modes.map((mode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "task-card";
    button.classList.toggle("active", mode.id === state.mode?.id);
    button.innerHTML = `<span class="task-icon">${escapeHtml(mode.icon || "✦")}</span><small>${escapeHtml(mode.badge || mode.engine || "NATIVE")}</small><b>${escapeHtml(mode.title)}</b><p>${escapeHtml(mode.short || mode.description || "")}</p>`;
    button.onclick = () => selectMode(mode.id);
    return button;
  }));
}

function selectMode(id, scroll = true) {
  const mode = modeById(id);
  if (!mode) return;
  state.mode = mode;
  if (!state.values[id]) state.values[id] = defaultsForMode(mode);
  if (state.group !== "all" && state.group !== mode.group) state.group = mode.group;
  renderGroupTabs();
  renderTaskCards();
  $("modeTitle").textContent = mode.title;
  $("modeDescription").textContent = mode.description || "";
  renderForm();
  saveDraft();
  if (scroll) document.querySelector(".studio-deck").scrollIntoView({ behavior: "smooth", block: "start" });
}

function fieldClass(field) {
  const span = field.span || "half";
  return `field ${span === "full" ? "full" : span === "third" ? "third" : span === "quarter" ? "quarter" : ""}`.trim();
}

function fieldLabel(field) {
  const label = document.createElement("span");
  label.innerHTML = `${escapeHtml(field.label || field.id)}${field.required ? " <i>required</i>" : field.unit ? ` <i>${escapeHtml(field.unit)}</i>` : ""}`;
  return label;
}

function fieldHint(field) {
  if (!field.hint) return null;
  const hint = document.createElement("small");
  hint.textContent = field.hint;
  return hint;
}

function setValue(field, value) {
  controlsForMode()[field.id] = value;
  updateVisibility();
  updateRecipe();
  validateForm();
  saveDraft();
}

function renderInputField(field, value) {
  const wrapper = document.createElement("label");
  wrapper.className = fieldClass(field);
  wrapper.dataset.field = field.id;
  wrapper.append(fieldLabel(field));
  let input;
  if (field.type === "textarea" || field.type === "code") {
    input = document.createElement("textarea");
    input.rows = field.rows || (field.type === "code" ? 10 : 5);
    input.spellcheck = field.type !== "code";
    if (field.type === "code") input.style.fontFamily = "ui-monospace, monospace";
  } else if (field.type === "select") {
    input = document.createElement("select");
    for (const option of field.options || []) {
      const node = document.createElement("option");
      node.value = option.value;
      node.textContent = option.label;
      input.append(node);
    }
  } else {
    input = document.createElement("input");
    input.type = field.type === "number" ? "number" : field.type === "date" ? "date" : "text";
  }
  input.id = `field-${field.id}`;
  input.value = value ?? "";
  if (field.placeholder) input.placeholder = field.placeholder;
  if (field.min !== undefined) input.min = field.min;
  if (field.max !== undefined) input.max = field.max;
  if (field.step !== undefined) input.step = field.step;
  if (field.maxlength) input.maxLength = field.maxlength;
  input.addEventListener("input", () => setValue(field, field.type === "number" ? (input.value === "" ? "" : Number(input.value)) : input.value));
  wrapper.append(input);
  if (field.auto_video_asset) {
    const action = document.createElement("button");
    action.type = "button";
    action.className = "field-action";
    action.textContent = field.auto_label || "Match input video";
    action.onclick = async (event) => {
      event.preventDefault();
      const assetId = String(controlsForMode()[field.auto_video_asset] || "").trim();
      if (!assetId) return toast("Add the driving video first.", "error");
      action.disabled = true;
      action.textContent = "Reading video…";
      try {
        const probe = await api(`/api/assets/${encodeURIComponent(assetId)}/probe`);
        const controls = controlsForMode();
        controls[field.id] = probe.wan_frames;
        if (field.auto_fps_field && Number.isFinite(probe.fps)) {
          controls[field.auto_fps_field] = Math.max(1, Math.min(60, Math.round(probe.fps)));
        }
        if (field.auto_width_field) controls[field.auto_width_field] = probe.output_width;
        if (field.auto_height_field) controls[field.auto_height_field] = probe.output_height;
        renderForm();
        saveDraft();
        const capped = probe.source_frames > probe.wan_frames ? " · capped to this workflow's limit" : "";
        toast(`${probe.source_width}×${probe.source_height} at ${probe.fps.toFixed(3)} FPS → ${probe.output_width}×${probe.output_height}, ${probe.wan_frames} frames (4n+1)${capped}.`);
      } catch (error) {
        toast(error.message, "error");
        action.disabled = false;
        action.textContent = field.auto_label || "Match input video";
      }
    };
    wrapper.append(action);
  }
  const hint = fieldHint(field); if (hint) wrapper.append(hint);
  return wrapper;
}

function renderRange(field, value) {
  const wrapper = document.createElement("label");
  wrapper.className = fieldClass(field);
  wrapper.dataset.field = field.id;
  wrapper.append(fieldLabel(field));
  const row = document.createElement("div"); row.className = "range-wrap";
  const input = document.createElement("input"); input.type = "range";
  input.min = field.min ?? 0; input.max = field.max ?? 100; input.step = field.step ?? 1; input.value = value ?? field.default ?? input.min;
  const output = document.createElement("output"); output.className = "range-readout";
  const render = () => { output.textContent = `${input.value}${field.suffix || ""}`; };
  input.oninput = () => { render(); setValue(field, Number(input.value)); };
  render(); row.append(input, output); wrapper.append(row);
  const hint = fieldHint(field); if (hint) wrapper.append(hint);
  return wrapper;
}

function renderToggle(field, value) {
  const wrapper = document.createElement("div");
  wrapper.className = `${fieldClass(field)} toggle-field`;
  wrapper.dataset.field = field.id;
  wrapper.append(fieldLabel(field));
  const label = document.createElement("label"); label.className = "switch";
  const input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(value);
  const knob = document.createElement("i");
  const text = document.createElement("span"); text.textContent = field.toggle_label || field.hint || "Enabled";
  input.onchange = () => setValue(field, input.checked);
  label.append(input, knob, text); wrapper.append(label);
  return wrapper;
}

function renderChoiceGrid(field, value) {
  const wrapper = document.createElement("div");
  wrapper.className = fieldClass({ ...field, span: field.span || "full" });
  wrapper.dataset.field = field.id;
  wrapper.append(fieldLabel(field));
  const grid = document.createElement("div"); grid.className = "choice-grid";
  const selected = field.multiple ? new Set(Array.isArray(value) ? value : []) : null;
  for (const option of field.options || []) {
    const button = document.createElement("button"); button.type = "button"; button.className = "choice-card";
    const active = field.multiple ? selected.has(option.value) : value === option.value;
    button.classList.toggle("active", active);
    button.innerHTML = `<span>${escapeHtml(option.icon || "•")}</span><b>${escapeHtml(option.label)}</b>`;
    button.title = option.description || option.label;
    button.onclick = () => {
      if (field.multiple) {
        active ? selected.delete(option.value) : selected.add(option.value);
        setValue(field, [...selected]);
      } else {
        setValue(field, option.value);
      }
      renderForm();
    };
    grid.append(button);
  }
  wrapper.append(grid);
  const hint = fieldHint(field); if (hint) wrapper.append(hint);
  return wrapper;
}

function assetById(id) {
  return state.assets.find((asset) => asset.id === id);
}

function uploadAsset(file, field, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/assets");
    xhr.setRequestHeader("X-File-Name", file.name);
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    if (state.token) xhr.setRequestHeader("Authorization", `Bearer ${state.token}`);
    xhr.upload.onprogress = (event) => progress(event.lengthComputable ? event.loaded / event.total : 0.2);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const asset = JSON.parse(xhr.responseText);
        state.assets.unshift(asset);
        resolve(asset);
      } else {
        try { reject(new Error(JSON.parse(xhr.responseText).detail)); } catch (_) { reject(new Error(`Upload failed (${xhr.status})`)); }
      }
    };
    xhr.onerror = () => reject(new Error("Upload connection failed."));
    xhr.send(file);
  });
}

function previewAsset(container, field, asset, localUrl = null, input = null) {
  container.replaceChildren();
  const preview = document.createElement("div"); preview.className = "asset-preview";
  const url = localUrl || asset?.url;
  const media = asset?.media_type || "";
  if (url && media.startsWith("audio/")) {
    const node = document.createElement("audio"); node.controls = true; node.src = url; preview.append(node);
  } else if (url && media.startsWith("video/")) {
    const node = document.createElement("video"); node.controls = true; node.src = url; preview.append(node);
  } else if (url && media.startsWith("image/")) {
    const node = document.createElement("img"); node.src = url; node.alt = asset?.name || "Selected image"; preview.append(node);
  } else {
    const glyph = document.createElement("span"); glyph.className = "drop-icon"; glyph.textContent = "⇧"; preview.append(glyph);
  }
  const name = document.createElement("b"); name.textContent = asset?.name || "Uploading…"; preview.append(name);
  const detail = document.createElement("small"); detail.textContent = asset ? `${formatBytes(asset.size)} · ${asset.media_type}` : "Preparing local upload"; preview.append(detail);
  const actions = document.createElement("div"); actions.className = "asset-actions";
  const replace = document.createElement("button"); replace.type = "button"; replace.textContent = "Replace";
  const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "Remove";
  replace.onclick = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!input) return;
    input.value = "";
    input.click();
  };
  remove.onclick = (event) => { event.preventDefault(); setValue(field, field.type === "multi_asset" ? [] : ""); renderForm(); };
  actions.append(replace, remove); preview.append(actions);
  container.append(preview);
  if (input) container.append(input);
}

function renderAsset(field, value) {
  const wrapper = document.createElement("div"); wrapper.className = fieldClass(field); wrapper.dataset.field = field.id;
  wrapper.append(fieldLabel(field));
  const drop = document.createElement("div"); drop.className = "asset-drop";
  const input = document.createElement("input"); input.type = "file"; input.accept = field.accept || "*/*";
  if (field.type === "multi_asset") input.multiple = true;
  const icon = document.createElement("span"); icon.className = "drop-icon"; icon.textContent = field.icon || "⇧";
  const title = document.createElement("b"); title.textContent = field.drop_label || "Drop a local file or browse";
  const detail = document.createElement("small"); detail.textContent = field.hint || "Stored only in this studio runtime";
  drop.append(icon, title, detail, input);
  const existingId = field.type === "multi_asset" ? value?.[0] : value;
  const existing = assetById(existingId);
  if (existing) previewAsset(drop, field, existing, null, input);
  const handle = async (files) => {
    const selected = [...files];
    if (!selected.length) return;
    if (field.max_files && selected.length > field.max_files) return toast(`Choose at most ${field.max_files} files.`, "error");
    const progress = document.createElement("div"); progress.className = "asset-progress"; progress.innerHTML = "<i></i>";
    drop.append(progress);
    try {
      const uploaded = [];
      for (const file of selected) {
        const localUrl = URL.createObjectURL(file);
        previewAsset(drop, field, { name: file.name, size: file.size, media_type: file.type }, localUrl, input);
        drop.append(progress);
        const asset = await uploadAsset(file, field, (fraction) => { progress.firstElementChild.style.width = `${fraction * 100}%`; });
        uploaded.push(asset.id);
        URL.revokeObjectURL(localUrl);
      }
      setValue(field, field.type === "multi_asset" ? uploaded : uploaded[0]);
      if (field.video_probe && uploaded.length === 1) {
        try {
          const probe = await api(`/api/assets/${encodeURIComponent(uploaded[0])}/probe`);
          const controls = controlsForMode();
          const targets = field.video_probe;
          if (targets.frames_field) controls[targets.frames_field] = probe.wan_frames;
          if (targets.fps_field) controls[targets.fps_field] = Math.max(1, Math.min(60, Math.round(probe.fps)));
          if (targets.width_field) controls[targets.width_field] = probe.output_width;
          if (targets.height_field) controls[targets.height_field] = probe.output_height;
          saveDraft();
          toast(`${probe.source_width}×${probe.source_height} · ${probe.source_frames} frames matched automatically.`);
        } catch (error) {
          toast(`Upload ready; automatic video matching failed: ${error.message}`, "error");
        }
      }
      renderForm();
      toast(`${selected.length === 1 ? selected[0].name : `${selected.length} files`} ready.`);
    } catch (error) {
      toast(error.message, "error"); renderForm();
    }
  };
  input.onchange = () => handle(input.files);
  for (const eventName of ["dragenter", "dragover"]) drop.addEventListener(eventName, (event) => { event.preventDefault(); drop.classList.add("drag"); });
  for (const eventName of ["dragleave", "drop"]) drop.addEventListener(eventName, (event) => { event.preventDefault(); drop.classList.remove("drag"); });
  drop.addEventListener("drop", (event) => handle(event.dataTransfer.files));
  wrapper.append(drop);
  return wrapper;
}

function renderEmotionMixer(field, value) {
  const wrapper = document.createElement("div"); wrapper.className = "emotion-mixer"; wrapper.dataset.field = field.id;
  const title = document.createElement("div"); title.className = "field-title"; title.innerHTML = `<span>${escapeHtml(field.label)}</span><small>${escapeHtml(field.hint || "Blend one or more emotions into a natural-language performance direction.")}</small>`;
  wrapper.append(title);
  const values = value && typeof value === "object" ? value : {};
  for (const option of field.options || []) {
    const row = document.createElement("div"); row.className = "emotion-row";
    const icon = document.createElement("span"); icon.textContent = option.icon;
    const label = document.createElement("b"); label.textContent = option.label;
    const input = document.createElement("input"); input.type = "range"; input.min = 0; input.max = 100; input.step = 5; input.value = values[option.value] || 0;
    const output = document.createElement("output"); output.textContent = `${input.value}%`;
    input.oninput = () => {
      const next = { ...controlsForMode()[field.id], [option.value]: Number(input.value) };
      output.textContent = `${input.value}%`;
      setValue(field, next);
    };
    row.append(icon, label, input, output); wrapper.append(row);
  }
  return wrapper;
}

function blankNonverbal(field) {
  return {
    operation: field.defaults?.operation || "add",
    sound: field.defaults?.sound || "breath",
    placement: field.defaults?.placement || "before",
    anchor: "",
  };
}

function renderNonverbal(field, value) {
  const wrapper = document.createElement("div"); wrapper.className = "nonverbal-builder"; wrapper.dataset.field = field.id;
  const title = document.createElement("div"); title.className = "field-title"; title.innerHTML = `<span>${escapeHtml(field.label)}</span><small>${escapeHtml(field.hint || "Layer as many additions or removals as the edit needs.")}</small>`;
  wrapper.append(title);
  const rows = Array.isArray(value) && value.length ? value : [blankNonverbal(field)];
  if (!Array.isArray(value) || !value.length) controlsForMode()[field.id] = rows;
  rows.forEach((item, index) => {
    const row = document.createElement("div"); row.className = "collection-row";
    const operation = selectFor(["add", "remove"], item.operation, (next) => update(index, "operation", next));
    const sound = selectFor(field.sounds || [], item.sound, (next) => update(index, "sound", next), true);
    const placement = selectFor(["beginning", "end", "before", "after", "all"], item.placement, (next) => update(index, "placement", next));
    const anchor = document.createElement("input"); anchor.type = "text"; anchor.placeholder = "Anchor words (optional)"; anchor.value = item.anchor || ""; anchor.oninput = () => update(index, "anchor", anchor.value, false);
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.title = "Remove event";
    remove.onclick = () => { const next = [...rows]; next.splice(index, 1); setValue(field, next.length ? next : [blankNonverbal(field)]); renderForm(); };
    row.append(operation, sound, placement, anchor, remove); wrapper.append(row);
  });
  const add = document.createElement("button"); add.type = "button"; add.className = "add-row"; add.textContent = "＋ Add another nonverbal edit";
  add.onclick = () => { setValue(field, [...rows, blankNonverbal(field)]); renderForm(); };
  wrapper.append(add);
  function update(index, key, next, rerender = true) {
    const updated = rows.map((item, itemIndex) => itemIndex === index ? { ...item, [key]: next } : item);
    setValue(field, updated); if (rerender) renderForm();
  }
  return wrapper;
}

function selectFor(options, value, callback, objects = false) {
  const select = document.createElement("select");
  for (const raw of options) {
    const item = objects ? raw : { value: raw, label: raw[0].toUpperCase() + raw.slice(1) };
    const option = document.createElement("option"); option.value = item.value; option.textContent = `${item.icon || ""} ${item.label}`.trim(); select.append(option);
  }
  select.value = value; select.onchange = () => callback(select.value); return select;
}

function renderCollection(field, value) {
  const wrapper = document.createElement("div"); wrapper.className = "collection-builder"; wrapper.dataset.field = field.id;
  const title = document.createElement("div"); title.className = "field-title"; title.innerHTML = `<span>${escapeHtml(field.label)}</span><small>${escapeHtml(field.hint || "")}</small>`; wrapper.append(title);
  const rows = Array.isArray(value) ? value : [];
  rows.forEach((item, index) => {
    const row = document.createElement("div"); row.className = "collection-row"; row.style.gridTemplateColumns = `repeat(${Math.min((field.fields || []).length, 4)}, minmax(0, 1fr)) auto`;
    for (const sub of field.fields || []) {
      let input;
      if (sub.type === "select") input = selectFor(sub.options || [], item[sub.id] ?? sub.default ?? "", (next) => update(index, sub.id, next), true);
      else {
        input = document.createElement("input"); input.type = sub.type === "number" ? "number" : "text"; input.placeholder = sub.label; input.value = item[sub.id] ?? sub.default ?? "";
        if (sub.min !== undefined) input.min = sub.min; if (sub.max !== undefined) input.max = sub.max; if (sub.step !== undefined) input.step = sub.step;
        input.oninput = () => update(index, sub.id, sub.type === "number" ? Number(input.value) : input.value, false);
      }
      input.title = sub.label; row.append(input);
    }
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.onclick = () => { const next = [...rows]; next.splice(index, 1); setValue(field, next); renderForm(); }; row.append(remove); wrapper.append(row);
  });
  const add = document.createElement("button"); add.type = "button"; add.className = "add-row"; add.textContent = field.add_label || "＋ Add row";
  add.onclick = () => { const item = Object.fromEntries((field.fields || []).map((sub) => [sub.id, sub.default ?? ""])); setValue(field, [...rows, item]); renderForm(); }; wrapper.append(add);
  function update(index, key, next, rerender = true) { setValue(field, rows.map((item, itemIndex) => itemIndex === index ? { ...item, [key]: next } : item)); if (rerender) renderForm(); }
  return wrapper;
}

function renderTags(field, value) {
  const wrapper = document.createElement("div"); wrapper.className = fieldClass(field); wrapper.dataset.field = field.id; wrapper.append(fieldLabel(field));
  const box = document.createElement("div"); box.className = "tags-input";
  const tags = Array.isArray(value) ? value : [];
  tags.forEach((tag, index) => {
    const chip = document.createElement("span"); chip.textContent = tag;
    const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.onclick = () => { setValue(field, tags.filter((_, item) => item !== index)); renderForm(); }; chip.append(remove); box.append(chip);
  });
  const input = document.createElement("input"); input.placeholder = field.placeholder || "Type and press Enter";
  input.onkeydown = (event) => {
    if ((event.key === "Enter" || event.key === ",") && input.value.trim()) {
      event.preventDefault(); const tag = input.value.trim().replace(/,$/, ""); if (!tags.includes(tag)) setValue(field, [...tags, tag]); renderForm();
    }
  };
  box.append(input); wrapper.append(box); const hint = fieldHint(field); if (hint) wrapper.append(hint); return wrapper;
}

function renderField(field) {
  const value = controlsForMode()[field.id];
  if (field.type === "section") {
    const section = document.createElement("div"); section.className = "form-section"; section.textContent = field.label; return section;
  }
  if (["text", "textarea", "code", "number", "date", "select"].includes(field.type)) return renderInputField(field, value);
  if (field.type === "range") return renderRange(field, value);
  if (field.type === "toggle") return renderToggle(field, value);
  if (["choice", "multi_choice"].includes(field.type)) return renderChoiceGrid({ ...field, multiple: field.type === "multi_choice" }, value);
  if (["asset", "multi_asset"].includes(field.type)) return renderAsset(field, value);
  if (field.type === "emotion_mixer") return renderEmotionMixer(field, value);
  if (field.type === "nonverbal_events") return renderNonverbal(field, value);
  if (field.type === "collection") return renderCollection(field, value);
  if (field.type === "tags") return renderTags(field, value);
  return renderInputField({ ...field, type: "text" }, value);
}

function conditionMatches(condition, controls) {
  if (!condition) return true;
  const value = controls[condition.field];
  if (Object.hasOwn(condition, "equals")) return value === condition.equals;
  if (condition.in) return condition.in.includes(value);
  if (condition.not_in) return !condition.not_in.includes(value);
  if (condition.truthy) return Boolean(value);
  if (condition.nonzero) return Number(value) !== 0;
  return true;
}

function updateVisibility() {
  if (!state.mode) return;
  const controls = controlsForMode();
  for (const field of [...modeFields(state.mode), ...modeFields(state.mode, "advanced")]) {
    if (!field.id || !field.show_if) continue;
    const node = document.querySelector(`[data-field="${CSS.escape(field.id)}"]`);
    if (node) node.hidden = !conditionMatches(field.show_if, controls);
  }
}

function renderForm() {
  if (!state.mode) return;
  const primary = modeFields(state.mode);
  const advanced = modeFields(state.mode, "advanced");
  $("dynamicForm").replaceChildren(...primary.map(renderField));
  $("advancedForm").replaceChildren(...advanced.map(renderField));
  $("advancedPanel").hidden = !advanced.length;
  updateVisibility(); updateRecipe(); validateForm();
}

function displayValue(field, value) {
  if (value === undefined || value === null || value === "") return "—";
  if (field.type === "choice" || field.type === "select") return (field.options || []).find((option) => option.value === value)?.label || String(value);
  if (field.type === "multi_choice") return value.map((item) => (field.options || []).find((option) => option.value === item)?.label || item).join(", ") || "—";
  if (field.type === "emotion_mixer") return Object.entries(value).filter(([, strength]) => strength > 0).map(([name, strength]) => `${name} ${strength}%`).join(", ") || "neutral";
  if (field.type === "nonverbal_events") return value.map((item) => `${item.operation} ${item.sound}${item.anchor ? ` ${item.placement} “${item.anchor}”` : ` at ${item.placement}`}`).join("; ");
  if (["asset", "multi_asset"].includes(field.type)) {
    const ids = Array.isArray(value) ? value : [value]; return ids.filter(Boolean).map((id) => assetById(id)?.name || id).join(", ") || "—";
  }
  if (Array.isArray(value)) return value.join(", ") || "—";
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "On" : "Off";
  return `${value}${field.suffix || ""}`;
}

function compilePreview() {
  if (!state.mode) return "";
  const controls = controlsForMode();
  const fields = [...modeFields(state.mode), ...modeFields(state.mode, "advanced")];
  const byId = Object.fromEntries(fields.filter((field) => field.id).map((field) => [field.id, field]));
  let template = state.mode.preview_template || "";
  template = template.replace(/\{\{\s*([\w-]+)\s*\}\}/g, (_, id) => displayValue(byId[id] || {}, controls[id]));
  const emotions = fields.find((field) => field.type === "emotion_mixer");
  if (emotions && !template.includes("emotion")) {
    const summary = displayValue(emotions, controls[emotions.id]); if (summary !== "neutral") template += `\nEmotion blend: ${summary}.`;
  }
  const events = fields.find((field) => field.type === "nonverbal_events");
  if (events && !template.includes("nonverbal")) template += `\nNonverbal edits: ${displayValue(events, controls[events.id])}.`;
  return template.trim();
}

function updateRecipe() {
  const mode = state.mode;
  if (!mode) return;
  $("recipeIcon").textContent = mode.icon || "✦";
  $("recipeTitle").textContent = mode.title;
  $("recipeSummary").textContent = mode.recipe || mode.description || "";
  const facts = [
    ["Engine", mode.engine || state.manifest.engine || "Native"],
    ["Input", mode.input_label || "Structured controls"],
    ["Output", mode.output_label || "Local artifact"],
    ["Residency", mode.residency || state.manifest.residency || "Explicit"],
  ];
  $("recipeFacts").replaceChildren(...facts.map(([name, value]) => {
    const row = document.createElement("div"); row.innerHTML = `<span>${escapeHtml(name)}</span><b>${escapeHtml(value)}</b>`; return row;
  }));
  const preview = compilePreview();
  $("compiledPrompt").hidden = !preview;
  $("compiledPrompt").querySelector("pre").textContent = preview;
  updateApiExample();
}

function validateForm() {
  if (!state.mode) return;
  const controls = controlsForMode();
  const failures = [];
  for (const field of [...modeFields(state.mode), ...modeFields(state.mode, "advanced")]) {
    if (!field.id || !conditionMatches(field.show_if, controls)) continue;
    const value = controls[field.id];
    const empty = value === "" || value === null || value === undefined || (Array.isArray(value) && !value.length);
    if (field.required && empty) failures.push(`${field.label} is required`);
    if (!empty && field.type === "number" && field.min !== undefined && value < field.min) failures.push(`${field.label} is below ${field.min}`);
    if (!empty && field.type === "number" && field.max !== undefined && value > field.max) failures.push(`${field.label} is above ${field.max}`);
  }
  if (state.mode.requires_any && !state.mode.requires_any.some((id) => {
    const value = controls[id]; return value !== "" && value !== null && value !== undefined && (!Array.isArray(value) || value.length);
  })) failures.push(state.mode.requires_any_message || "One source input is required");
  $("validationDot").className = `validation-dot ${failures.length ? "error" : "ready"}`;
  $("validationText").textContent = failures[0] || "Recipe is complete and stays local.";
  $("queueJob").disabled = Boolean(failures.length);
  return failures;
}

function currentRequest() {
  if (!state.mode) return null;
  return {
    mode: state.mode.id,
    controls: structuredClone(controlsForMode()),
    client: { source: "mm-tools-studio", schema: 1, queued_at: new Date().toISOString() },
  };
}

async function queueCurrent(event) {
  event?.preventDefault();
  const failures = validateForm();
  if (failures.length) return toast(failures[0], "error");
  const button = $("queueJob"); button.disabled = true; button.querySelector("span").textContent = "Queuing…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(currentRequest()) });
    state.jobs.unshift(job); renderCounts(); renderQueue();
    toast(`${state.mode.title} entered the private GPU queue.`);
    setPage("queue");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.querySelector("span").textContent = "Queue local render"; validateForm();
  }
}

function renderCounts() {
  const active = state.jobs.filter((job) => ACTIVE.has(job.status)).length;
  const outputs = state.jobs.reduce((sum, job) => sum + (job.outputs?.length || 0), 0);
  $("queueNavCount").textContent = `${active} active`;
  $("libraryNavCount").textContent = `${outputs} output${outputs === 1 ? "" : "s"}`;
}

function jobMode(job) { return modeById(job.mode) || { title: job.mode, icon: "✦" }; }

function renderQueue() {
  const counts = {
    active: state.jobs.filter((job) => ACTIVE.has(job.status)).length,
    queued: state.jobs.filter((job) => job.status === "queued").length,
    complete: state.jobs.filter((job) => job.status === "complete").length,
    failed: state.jobs.filter((job) => ["failed", "interrupted"].includes(job.status)).length,
  };
  $("queueSummary").innerHTML = Object.entries(counts).map(([label, count]) => `<div class="stat"><small>${escapeHtml(label.toUpperCase())}</small><b>${count}</b></div>`).join("");
  if (!state.jobs.length) {
    $("queueList").innerHTML = '<div class="empty-state"><div><b>The GPU queue is clear.</b>Choose a native workflow in Create.</div></div>';
    return;
  }
  $("queueList").replaceChildren(...state.jobs.map((job) => {
    const mode = jobMode(job); const card = document.createElement("article"); card.className = "job-card";
    card.innerHTML = `<div><span class="status-chip ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span><h3>${escapeHtml(mode.icon)} ${escapeHtml(mode.title)}</h3><small>${escapeHtml(relativeTime(job.created_at))} · ${escapeHtml(job.id.slice(0, 8))}</small></div><div class="job-stage"><p><span>${escapeHtml(job.stage || job.status)}</span><b>${Math.round((job.progress || 0) * 100)}%</b></p><div class="progress"><i style="width:${Math.max(0, Math.min(100, (job.progress || 0) * 100))}%"></i></div></div>`;
    const actions = document.createElement("div"); actions.className = "job-actions";
    const inspect = document.createElement("button"); inspect.textContent = "Details"; inspect.onclick = () => showJob(job.id); actions.append(inspect);
    if (ACTIVE.has(job.status)) { const cancel = document.createElement("button"); cancel.className = "cancel"; cancel.textContent = "Stop"; cancel.onclick = () => cancelJob(job.id); actions.append(cancel); }
    if (job.status === "complete") { const library = document.createElement("button"); library.textContent = "Outputs"; library.onclick = () => { state.libraryFilter = job.mode; setPage("library"); }; actions.append(library); }
    if (TERMINAL.has(job.status)) { const remove = document.createElement("button"); remove.textContent = "Hide"; remove.onclick = () => deleteJob(job.id); actions.append(remove); }
    card.append(actions); return card;
  }));
}

function mediaPlaying() {
  return [...document.querySelectorAll("audio, video")].some((element) => !element.paused && !element.ended);
}

function renderVisiblePages() {
  if (state.page === "queue") renderQueue();
  if (state.page === "library") renderLibrary();
  if (state.page === "compare") renderCompare();
}

function refreshVisiblePages() {
  // A re-rendered page replaces live <audio> nodes and drops playback position;
  // hold the refresh until the user pauses or the track ends.
  if (mediaPlaying()) { state.pendingRender = true; return; }
  state.pendingRender = false;
  renderVisiblePages();
}

function settleRender() {
  if (!state.pendingRender || mediaPlaying()) return;
  state.pendingRender = false;
  renderVisiblePages();
}

async function refreshJobs() {
  try { state.jobs = await api("/api/jobs?limit=200"); renderCounts(); refreshVisiblePages(); updatePromptResult(); }
  catch (error) { toast(error.message, "error"); }
}

async function cancelJob(id) {
  try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); await refreshJobs(); toast("Safe cancellation requested."); }
  catch (error) { toast(error.message, "error"); }
}

async function deleteJob(id) {
  try { await api(`/api/jobs/${id}`, { method: "DELETE" }); state.jobs = state.jobs.filter((job) => job.id !== id); renderCounts(); renderQueue(); renderLibrary(); }
  catch (error) { toast(error.message, "error"); }
}

function showJob(id) {
  const job = state.jobs.find((item) => item.id === id); if (!job) return;
  const mode = jobMode(job); $("jobDialogMode").textContent = `${job.status.toUpperCase()} · ${job.id.slice(0, 8)}`; $("jobDialogTitle").textContent = mode.title;
  const body = $("jobDialogBody"); body.replaceChildren();
  const grid = document.createElement("div"); grid.className = "job-detail-grid";
  const request = document.createElement("div"); request.innerHTML = "<small>VISIBLE RECIPE</small>"; const requestCode = document.createElement("pre"); requestCode.className = "job-log"; requestCode.textContent = JSON.stringify(job.request?.controls || {}, null, 2); request.append(requestCode);
  const log = document.createElement("div"); log.innerHTML = "<small>NATIVE LOG</small>"; const logCode = document.createElement("pre"); logCode.className = "job-log"; logCode.textContent = (job.log || []).map((entry) => `[${new Date(entry.at * 1000).toLocaleTimeString()}] ${entry.level}: ${entry.message}`).join("\n") || "No log entries yet."; log.append(logCode);
  grid.append(request, log); body.append(grid);
  if (job.error) { const error = document.createElement("p"); error.style.color = "var(--danger)"; error.textContent = job.error; body.append(error); }
  $("jobDialog").showModal();
}

function outputs() {
  return state.jobs.flatMap((job) => (job.outputs || []).map((output) => ({ ...output, job, mode: jobMode(job) })));
}

function renderLibrary() {
  const all = outputs();
  const modes = [...new Set(all.map((item) => item.job.mode))];
  $("libraryFilters").replaceChildren(...[{ id: "all", title: "All outputs" }, ...modes.map((id) => ({ id, title: modeById(id)?.title || id }))].map((item) => {
    const button = document.createElement("button"); button.textContent = item.title; button.classList.toggle("active", state.libraryFilter === item.id); button.onclick = () => { state.libraryFilter = item.id; renderLibrary(); }; return button;
  }));
  const search = $("librarySearch").value.trim().toLowerCase();
  const filtered = all.filter((item) => (state.libraryFilter === "all" || item.job.mode === state.libraryFilter) && (!search || JSON.stringify(item).toLowerCase().includes(search)));
  if (!filtered.length) { $("libraryGrid").innerHTML = '<div class="empty-state"><div><b>No matching artifacts.</b>Finished native renders appear here.</div></div>'; return; }
  $("libraryGrid").replaceChildren(...filtered.map(outputCard));
}

function outputCard(item) {
  const card = document.createElement("article"); card.className = "output-card";
  const preview = document.createElement("div"); preview.className = "output-preview";
  preview.append(item.kind === "audio" ? audioStudioNode(item) : mediaNode(item, false));
  const copy = document.createElement("div"); copy.className = "output-copy";
  copy.innerHTML = `<header><h3>${escapeHtml(item.label || item.name)}</h3><small>${escapeHtml(formatBytes(item.size))}</small></header><p>${escapeHtml(item.mode.title)} · ${escapeHtml(formatTime(item.job.finished_at))}</p>`;
  const actions = document.createElement("div"); actions.className = "output-actions";
  const download = document.createElement("a"); download.href = `${item.url}?download=true`; download.textContent = "Download";
  const details = document.createElement("button"); details.textContent = "Recipe"; details.onclick = () => showJob(item.job.id);
  const reopen = document.createElement("button"); reopen.textContent = "Reopen"; reopen.onclick = () => reopenJob(item.job);
  actions.append(download, details, reopen);
  if (item.metadata?.krea) {
    const useInCreate = document.createElement("button"); useInCreate.textContent = "Use in Create";
    useInCreate.onclick = () => {
      const mode = modeById("compose_full"); if (!mode) return toast("The Create workflow is not available.", "error");
      const values = state.values[mode.id] || defaultsForMode(mode); state.values[mode.id] = values;
      if (item.metadata.style) values.style = item.metadata.style;
      if (item.metadata.lyrics) values.lyrics = item.metadata.lyrics;
      selectMode(mode.id, false); setPage("create"); toast("Krea direction carried into Create.");
    };
    actions.append(useInCreate);
  }
  copy.append(actions); card.append(preview, copy); return card;
}

function mediaNode(item, autoplay = false) {
  const kind = item.kind || "file";
  if (kind === "audio") { const node = document.createElement("audio"); node.controls = true; node.preload = "metadata"; node.src = item.url; if (autoplay) node.autoplay = true; return node; }
  if (kind === "video") { const node = document.createElement("video"); node.controls = true; node.preload = "metadata"; node.src = item.url; return node; }
  if (kind === "image") { const node = document.createElement("img"); node.loading = "lazy"; node.src = item.url; node.alt = item.label || item.name; return node; }
  if (kind === "motion" || String(item.name || "").toLowerCase().endsWith(".motion.json")) return motionStudioNode(item);
  if (kind === "model" || String(item.name || "").toLowerCase().endsWith(".glb")) return modelStudioNode(item);
  const glyph = document.createElement("span"); glyph.className = "file-glyph"; glyph.textContent = kind === "model" ? "◇" : kind === "midi" ? "♬" : kind === "text" ? "¶" : "▧"; glyph.title = item.name; return glyph;
}

function motionStudioNode(item) {
  const shell = document.createElement("div"); shell.className = "motion-stage";
  const canvas = document.createElement("canvas"); canvas.className = "motion-canvas"; canvas.setAttribute("aria-label", `Interactive motion preview of ${item.label || item.name}`);
  const status = document.createElement("div"); status.className = "model-status"; status.textContent = "Loading native skeleton…";
  const toolbar = document.createElement("div"); toolbar.className = "motion-toolbar";
  const play = document.createElement("button"); play.type = "button"; play.textContent = "Ⅱ Pause"; play.classList.add("active");
  const trail = document.createElement("button"); trail.type = "button"; trail.textContent = "⌁ Trail";
  const home = document.createElement("button"); home.type = "button"; home.textContent = "⌂ Reset";
  const fullscreen = document.createElement("button"); fullscreen.type = "button"; fullscreen.textContent = "⛶ Full";
  const timeline = document.createElement("input"); timeline.type = "range"; timeline.min = "0"; timeline.max = "1"; timeline.step = "1"; timeline.value = "0"; timeline.setAttribute("aria-label", "Motion frame");
  toolbar.append(play, trail, home, fullscreen, timeline); shell.append(canvas, status, toolbar);
  const view = { yaw: .65, pitch: -.18, zoom: 1, playing: true, trail: false, drag: null, frame: 0, data: null, previous: performance.now() };
  play.onclick = () => { view.playing = !view.playing; play.textContent = view.playing ? "Ⅱ Pause" : "▶ Play"; play.classList.toggle("active", view.playing); };
  trail.onclick = () => { view.trail = !view.trail; trail.classList.toggle("active", view.trail); };
  home.onclick = () => Object.assign(view, { yaw: .65, pitch: -.18, zoom: 1, frame: 0 });
  fullscreen.onclick = () => shell.requestFullscreen?.();
  timeline.oninput = () => { view.frame = Number(timeline.value); view.playing = false; play.textContent = "▶ Play"; play.classList.remove("active"); };
  canvas.addEventListener("pointerdown", (event) => { view.drag = { x: event.clientX, y: event.clientY, yaw: view.yaw, pitch: view.pitch }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!view.drag) return; view.yaw = view.drag.yaw + (event.clientX - view.drag.x) * .009; view.pitch = Math.max(-1.25, Math.min(1.25, view.drag.pitch + (event.clientY - view.drag.y) * .009)); });
  canvas.addEventListener("pointerup", () => { view.drag = null; }); canvas.addEventListener("pointercancel", () => { view.drag = null; });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); view.zoom = Math.max(.45, Math.min(3.5, view.zoom * Math.exp(-event.deltaY * .001))); }, { passive: false });

  const paint = (now) => {
    if (!canvas.isConnected) return;
    requestAnimationFrame(paint);
    const data = view.data; if (!data?.frames?.length) return;
    const elapsed = Math.min(.1, (now - view.previous) / 1000); view.previous = now;
    if (view.playing) view.frame = (view.frame + elapsed * Number(data.fps || 20)) % data.frames.length;
    timeline.value = String(Math.floor(view.frame));
    const ratio = Math.min(devicePixelRatio || 1, 2); const width = Math.max(1, canvas.clientWidth); const height = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) { canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio); }
    const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    const frame = data.frames[Math.floor(view.frame)] || data.frames[0]; const root = frame.find((_, index) => Number(data.parents[index]) < 0) || frame[0];
    const cosine = Math.cos(view.yaw); const sine = Math.sin(view.yaw); const pitchCos = Math.cos(view.pitch); const pitchSin = Math.sin(view.pitch);
    const transform = (point) => { const x = point[0] - root[0]; const y = point[1] - root[1]; const z = point[2] - root[2]; const rx = x * cosine - z * sine; const rz = x * sine + z * cosine; const ry = y * pitchCos - rz * pitchSin; const depth = y * pitchSin + rz * pitchCos; const scale = Math.min(width, height) * .38 * view.zoom / Math.max(.55, 1 + depth * .08); return [width * .5 + rx * scale, height * .72 - ry * scale, depth]; };
    const points = frame.map(transform); const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#79efaa";
    context.lineCap = "round"; context.lineJoin = "round";
    if (view.trail) { context.strokeStyle = "rgba(122,239,170,.2)"; context.lineWidth = 1; context.beginPath(); const rootIndex = Math.max(0, data.parents.findIndex((value) => Number(value) < 0)); for (let offset = 0; offset < Math.min(data.frames.length, 80); offset += 4) { const trailFrame = data.frames[(Math.floor(view.frame) - offset + data.frames.length) % data.frames.length]; const p = transform(trailFrame[rootIndex]); if (!offset) context.moveTo(p[0], p[1]); else context.lineTo(p[0], p[1]); } context.stroke(); }
    const bones = points.map((point, index) => ({ point, parent: Number(data.parents[index]), depth: point[2], index })).filter((bone) => bone.parent >= 0).sort((a, b) => a.depth - b.depth);
    for (const bone of bones) { const parent = points[bone.parent]; context.globalAlpha = Math.max(.32, Math.min(1, .72 + bone.depth * .04)); context.strokeStyle = accent; context.lineWidth = Math.max(2, Math.min(7, 4.2 - bone.depth * .08)); context.beginPath(); context.moveTo(parent[0], parent[1]); context.lineTo(bone.point[0], bone.point[1]); context.stroke(); }
    context.globalAlpha = 1; for (const point of points) { context.fillStyle = accent; context.beginPath(); context.arc(point[0], point[1], 2.4, 0, Math.PI * 2); context.fill(); }
    context.strokeStyle = "rgba(130,155,145,.22)"; context.lineWidth = 1; context.beginPath(); context.moveTo(width * .12, height * .78); context.lineTo(width * .88, height * .78); context.stroke();
    status.textContent = `${data.names?.length || frame.length} joints · frame ${Math.floor(view.frame) + 1}/${data.frames.length} · ${Number(data.source_fps || data.fps || 0).toFixed(1)} FPS · drag to orbit · wheel to zoom`;
  };
  fetch(item.url).then((response) => { if (!response.ok) throw new Error(`motion request failed (${response.status})`); return response.json(); }).then((data) => {
    if (data.format !== "mm-tools-motion-v1" || !Array.isArray(data.frames) || !data.frames.length) throw new Error("unsupported motion document");
    view.data = data; timeline.max = String(data.frames.length - 1); view.previous = performance.now(); requestAnimationFrame(paint);
  }).catch((error) => { status.textContent = `Preview unavailable · ${error.message}`; status.classList.add("error"); });
  return shell;
}

function modelStudioNode(item) {
  const shell = document.createElement("div"); shell.className = "model-stage";
  const canvas = document.createElement("canvas"); canvas.className = "model-canvas"; canvas.setAttribute("aria-label", `Interactive 3D preview of ${item.label || item.name}`);
  const status = document.createElement("div"); status.className = "model-status"; status.textContent = "Loading local GLB…";
  const toolbar = document.createElement("div"); toolbar.className = "model-toolbar";
  const orbit = document.createElement("button"); orbit.type = "button"; orbit.textContent = "↻ Auto orbit"; orbit.classList.add("active");
  const wire = document.createElement("button"); wire.type = "button"; wire.textContent = "⌗ Wire";
  const home = document.createElement("button"); home.type = "button"; home.textContent = "⌂ Reset";
  const fullscreen = document.createElement("button"); fullscreen.type = "button"; fullscreen.textContent = "⛶ Full";
  toolbar.append(orbit, wire, home, fullscreen); shell.append(canvas, status, toolbar);
  const state3d = { yaw: .65, pitch: -.24, distance: 2.7, auto: true, wire: false, drag: null, renderer: null };
  orbit.onclick = () => { state3d.auto = !state3d.auto; orbit.classList.toggle("active", state3d.auto); };
  wire.onclick = () => { state3d.wire = !state3d.wire; wire.classList.toggle("active", state3d.wire); };
  home.onclick = () => Object.assign(state3d, { yaw: .65, pitch: -.24, distance: 2.7 });
  fullscreen.onclick = () => shell.requestFullscreen?.();
  canvas.addEventListener("pointerdown", (event) => { state3d.drag = { x: event.clientX, y: event.clientY, yaw: state3d.yaw, pitch: state3d.pitch }; state3d.auto = false; orbit.classList.remove("active"); canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!state3d.drag) return; state3d.yaw = state3d.drag.yaw + (event.clientX - state3d.drag.x) * .009; state3d.pitch = Math.max(-1.45, Math.min(1.45, state3d.drag.pitch + (event.clientY - state3d.drag.y) * .009)); });
  canvas.addEventListener("pointerup", () => { state3d.drag = null; });
  canvas.addEventListener("pointercancel", () => { state3d.drag = null; });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); state3d.distance = Math.max(1.15, Math.min(8, state3d.distance * Math.exp(event.deltaY * .001))); }, { passive: false });
  fetch(item.url).then((response) => {
    if (!response.ok) throw new Error(`GLB request failed (${response.status})`);
    return response.arrayBuffer();
  }).then((buffer) => createGlbRenderer(canvas, buffer, state3d, status)).catch((error) => {
    status.textContent = `Preview unavailable · ${error.message}`; status.classList.add("error");
  });
  return shell;
}

function glbChunks(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2) throw new Error("Not a GLB 2.0 file");
  let offset = 12; let json = null; let binary = null;
  while (offset + 8 <= view.byteLength) {
    const length = view.getUint32(offset, true); const type = view.getUint32(offset + 4, true); offset += 8;
    if (offset + length > view.byteLength) throw new Error("Truncated GLB chunk");
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, length)).replace(/\u0000+$/g, ""));
    else if (type === 0x004e4942) binary = buffer.slice(offset, offset + length);
    offset += length;
  }
  if (!json || !binary) throw new Error("GLB needs JSON and binary chunks");
  return { json, binary };
}

function accessorData(doc, binary, index) {
  const accessor = doc.accessors?.[index]; if (!accessor) throw new Error(`Missing accessor ${index}`);
  const view = doc.bufferViews?.[accessor.bufferView]; if (!view) throw new Error(`Accessor ${index} has no buffer view`);
  const sizes = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }; const components = sizes[accessor.type];
  const types = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }; const Type = types[accessor.componentType];
  if (!Type || !components || accessor.sparse) throw new Error("Unsupported sparse or typed accessor");
  const bytes = Type.BYTES_PER_ELEMENT; const stride = view.byteStride || components * bytes; const start = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const result = new Type(accessor.count * components);
  if (stride === components * bytes && start % bytes === 0) result.set(new Type(binary, start, accessor.count * components));
  else {
    const source = new DataView(binary);
    const readers = { 5120: "getInt8", 5121: "getUint8", 5122: "getInt16", 5123: "getUint16", 5125: "getUint32", 5126: "getFloat32" }; const reader = readers[accessor.componentType];
    for (let row = 0; row < accessor.count; row += 1) for (let column = 0; column < components; column += 1) result[row * components + column] = source[reader](start + row * stride + column * bytes, true);
  }
  return { array: result, components, componentType: accessor.componentType, normalized: !!accessor.normalized, count: accessor.count, min: accessor.min, max: accessor.max };
}

function computedNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  const triangles = indices || Uint32Array.from({ length: positions.length / 3 }, (_, index) => index);
  for (let index = 0; index + 2 < triangles.length; index += 3) {
    const a = triangles[index] * 3; const b = triangles[index + 1] * 3; const c = triangles[index + 2] * 3;
    const abx = positions[b] - positions[a]; const aby = positions[b + 1] - positions[a + 1]; const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a]; const acy = positions[c + 1] - positions[a + 1]; const acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy; const ny = abz * acx - abx * acz; const nz = abx * acy - aby * acx;
    for (const vertex of [a, b, c]) { normals[vertex] += nx; normals[vertex + 1] += ny; normals[vertex + 2] += nz; }
  }
  for (let index = 0; index < normals.length; index += 3) { const length = Math.hypot(normals[index], normals[index + 1], normals[index + 2]) || 1; normals[index] /= length; normals[index + 1] /= length; normals[index + 2] /= length; }
  return normals;
}

function matrixPerspective(field, aspect, near, far) {
  const f = 1 / Math.tan(field / 2); const range = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * range, -1, 0, 0, far * near * 2 * range, 0]);
}

function matrixLookAt(eye, center, up) {
  let zx = eye[0] - center[0]; let zy = eye[1] - center[1]; let zz = eye[2] - center[2]; let length = Math.hypot(zx, zy, zz) || 1; zx /= length; zy /= length; zz /= length;
  let xx = up[1] * zz - up[2] * zy; let xy = up[2] * zx - up[0] * zz; let xz = up[0] * zy - up[1] * zx; length = Math.hypot(xx, xy, xz) || 1; xx /= length; xy /= length; xz /= length;
  const yx = zy * xz - zz * xy; const yy = zz * xx - zx * xz; const yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]), -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}

function matrixMultiply(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) for (let row = 0; row < 4; row += 1) out[column * 4 + row] = a[row] * b[column * 4] + a[4 + row] * b[column * 4 + 1] + a[8 + row] * b[column * 4 + 2] + a[12 + row] * b[column * 4 + 3];
  return out;
}

function shaderProgram(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => { const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader); if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader)); return shader; };
  const program = gl.createProgram(); gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource)); gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource)); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program)); return program;
}

async function createGlbRenderer(canvas, source, controls, status) {
  const { json: doc, binary } = glbChunks(source); const gl = canvas.getContext("webgl2", { antialias: true, alpha: true }); if (!gl) throw new Error("WebGL 2 is unavailable");
  const vertexSource = `#version 300 es
  in vec3 a_position; in vec3 a_normal; in vec2 a_uv; uniform mat4 u_view_projection; uniform vec3 u_center; uniform float u_scale; out vec3 v_normal; out vec2 v_uv;
  void main(){ vec3 p=(a_position-u_center)*u_scale; gl_Position=u_view_projection*vec4(p,1.0); v_normal=normalize(a_normal); v_uv=a_uv; }`;
  const fragmentSource = `#version 300 es
  precision highp float; in vec3 v_normal; in vec2 v_uv; uniform vec4 u_color; uniform sampler2D u_texture; uniform bool u_has_texture; uniform bool u_wire; out vec4 color;
  void main(){ vec3 n=normalize(v_normal); float key=max(dot(n,normalize(vec3(.35,.78,.52))),0.0); float rim=pow(1.0-max(n.z,0.0),2.0); vec4 base=u_has_texture?texture(u_texture,v_uv)*u_color:u_color; vec3 lit=base.rgb*(.24+.74*key)+vec3(.22,.38,.31)*rim*.28; color=vec4(u_wire?mix(lit,vec3(.55,1.0,.79),.58):lit,base.a); }`;
  const program = shaderProgram(gl, vertexSource, fragmentSource); gl.useProgram(program);
  const locations = Object.fromEntries(["a_position", "a_normal", "a_uv", "u_view_projection", "u_center", "u_scale", "u_color", "u_texture", "u_has_texture", "u_wire"].map((name) => [name, name.startsWith("a_") ? gl.getAttribLocation(program, name) : gl.getUniformLocation(program, name)]));
  const textures = await Promise.all((doc.images || []).map(async (image) => {
    let bytes; let type = image.mimeType || "image/png";
    if (Number.isInteger(image.bufferView)) { const view = doc.bufferViews[image.bufferView]; bytes = binary.slice(view.byteOffset || 0, (view.byteOffset || 0) + view.byteLength); }
    else if (String(image.uri || "").startsWith("data:")) { const [head, payload] = image.uri.split(",", 2); type = head.match(/^data:([^;]+)/)?.[1] || type; bytes = Uint8Array.from(atob(payload), (character) => character.charCodeAt(0)).buffer; }
    else return null;
    const bitmap = await createImageBitmap(new Blob([bytes], { type })); const texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap); gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT); return texture;
  }));
  const primitives = []; const minimum = [Infinity, Infinity, Infinity]; const maximum = [-Infinity, -Infinity, -Infinity];
  for (const mesh of doc.meshes || []) for (const primitive of mesh.primitives || []) {
    if (primitive.mode != null && primitive.mode !== 4) continue;
    const position = accessorData(doc, binary, primitive.attributes.POSITION); const indices = Number.isInteger(primitive.indices) ? accessorData(doc, binary, primitive.indices) : null;
    const normal = Number.isInteger(primitive.attributes.NORMAL) ? accessorData(doc, binary, primitive.attributes.NORMAL).array : computedNormals(position.array, indices?.array);
    const uv = Number.isInteger(primitive.attributes.TEXCOORD_0) ? accessorData(doc, binary, primitive.attributes.TEXCOORD_0).array : new Float32Array(position.count * 2);
    const makeBuffer = (target, data) => { const buffer = gl.createBuffer(); gl.bindBuffer(target, buffer); gl.bufferData(target, data, gl.STATIC_DRAW); return buffer; };
    const item = { position: makeBuffer(gl.ARRAY_BUFFER, position.array), normal: makeBuffer(gl.ARRAY_BUFFER, normal), uv: makeBuffer(gl.ARRAY_BUFFER, uv), count: indices?.count || position.count, type: indices?.componentType || 0, indices: indices ? makeBuffer(gl.ELEMENT_ARRAY_BUFFER, indices.array) : null, material: doc.materials?.[primitive.material] || {} };
    const boundsMin = position.min || [0, 0, 0]; const boundsMax = position.max || [0, 0, 0]; for (let axis = 0; axis < 3; axis += 1) { minimum[axis] = Math.min(minimum[axis], boundsMin[axis]); maximum[axis] = Math.max(maximum[axis], boundsMax[axis]); }
    primitives.push(item);
  }
  if (!primitives.length) throw new Error("GLB contains no triangle meshes");
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2); const extent = Math.max(...maximum.map((value, axis) => value - minimum[axis])) || 1; const scale = 1.65 / extent;
  status.textContent = `${primitives.length} mesh${primitives.length === 1 ? "" : "es"} · ${Math.round(source.byteLength / 1048576)} MB · drag to orbit · wheel to zoom`;
  const resize = () => { const ratio = Math.min(devicePixelRatio || 1, 2); const width = Math.max(canvas.clientWidth, 1); const height = Math.max(canvas.clientHeight, 1); const realWidth = Math.floor(width * ratio); const realHeight = Math.floor(height * ratio); if (canvas.width !== realWidth || canvas.height !== realHeight) { canvas.width = realWidth; canvas.height = realHeight; gl.viewport(0, 0, realWidth, realHeight); } return width / height; };
  const bindAttribute = (location, buffer, components) => { gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, components, gl.FLOAT, false, 0, 0); };
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.uniform1i(locations.u_texture, 0);
  let previous = performance.now();
  const paint = (now) => {
    if (!canvas.isConnected) return;
    const elapsed = Math.min((now - previous) / 1000, .05); previous = now; if (controls.auto) controls.yaw += elapsed * .22;
    const aspect = resize(); const radius = controls.distance; const eye = [Math.sin(controls.yaw) * Math.cos(controls.pitch) * radius, Math.sin(controls.pitch) * radius, Math.cos(controls.yaw) * Math.cos(controls.pitch) * radius];
    const matrix = matrixMultiply(matrixPerspective(.67, aspect, .01, 100), matrixLookAt(eye, [0, 0, 0], [0, 1, 0]));
    const dark = document.documentElement.dataset.theme !== "light"; gl.clearColor(dark ? .025 : .91, dark ? .035 : .93, dark ? .03 : .91, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.useProgram(program); gl.uniformMatrix4fv(locations.u_view_projection, false, matrix); gl.uniform3fv(locations.u_center, center); gl.uniform1f(locations.u_scale, scale); gl.uniform1i(locations.u_wire, controls.wire ? 1 : 0);
    for (const primitive of primitives) {
      bindAttribute(locations.a_position, primitive.position, 3); bindAttribute(locations.a_normal, primitive.normal, 3); bindAttribute(locations.a_uv, primitive.uv, 2);
      const pbr = primitive.material.pbrMetallicRoughness || {}; const factor = pbr.baseColorFactor || [1, 1, 1, 1]; gl.uniform4fv(locations.u_color, factor);
      const textureIndex = pbr.baseColorTexture?.index; const imageIndex = Number.isInteger(textureIndex) ? doc.textures?.[textureIndex]?.source : null; const texture = Number.isInteger(imageIndex) ? textures[imageIndex] : null; gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture); gl.uniform1i(locations.u_has_texture, texture ? 1 : 0);
      if (primitive.indices) { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, primitive.indices); gl.drawElements(controls.wire ? gl.LINE_STRIP : gl.TRIANGLES, primitive.count, primitive.type, 0); }
      else gl.drawArrays(controls.wire ? gl.LINE_STRIP : gl.TRIANGLES, 0, primitive.count);
    }
    requestAnimationFrame(paint);
  };
  requestAnimationFrame(paint);
}

function slugify(value) {
  const slug = String(value || "track").toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || "track";
}

function downloadBlob(filename, blob) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function trackLyrics(item) {
  return String(item.metadata?.lyrics || item.job?.request?.controls?.lyrics || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function buildAudioPlayerHtml({ title, source, accent, lyrics, timestamps }) {
  const accentRgb = hexToRgb(accent);
  const rows = lyrics.map((line) => '<p' + (/^\[.+\]$/.test(line) ? ' class="section" data-section="1"' : "") + ">" + escapeHtml(line) + "</p>").join("\n    ");
  const css = [
    ":root{--accent:" + accent + ";--pulse:0}",
    "*{box-sizing:border-box}",
    "html,body{margin:0;min-height:100%}",
    "body{display:grid;place-items:center;padding:28px 20px;background:radial-gradient(120% 90% at 50% 0%,#101513 0%,#070908 60%);color:#e9eeeb;font:14px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}",
    "main{position:relative;width:min(880px,100%);border:1px solid rgba(" + accentRgb + ",.16);border-radius:16px;background:linear-gradient(150deg,rgba(" + accentRgb + ",.07),transparent 52%),#0e1110;padding:26px 26px 16px;overflow:hidden}",
    "h1{margin:0;font-size:19px;font-weight:650;letter-spacing:-.01em}",
    ".source{margin:6px 0 0}",
    ".source a{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#98a29c;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;text-decoration:none;border-bottom:1px dotted rgba(152,162,156,.4)}",
    ".source a:hover{color:var(--accent)}",
    ".stage{position:relative;margin-top:20px}",
    ".orbit{position:absolute;top:-42%;left:12%;width:76%;aspect-ratio:1;border:1px solid rgba(" + accentRgb + ",.18);border-radius:50%;box-shadow:0 0 70px rgba(" + accentRgb + ",calc(.1 + var(--pulse) * .22)),inset 0 0 50px rgba(" + accentRgb + ",calc(.05 + var(--pulse) * .08));opacity:calc(.5 + var(--pulse) * .5);transform:scale(calc(1 + var(--pulse) * .05));pointer-events:none}",
    "canvas{display:block;width:100%;height:170px;opacity:.92}",
    ".lyrics{position:relative;max-height:190px;overflow:auto;margin-top:14px;padding:14px 4px 6px;-webkit-mask-image:linear-gradient(transparent,black 14%,black 86%,transparent);mask-image:linear-gradient(transparent,black 14%,black 86%,transparent);scrollbar-width:thin;scrollbar-color:rgba(" + accentRgb + ",.35) transparent;text-align:center}",
    ".lyrics p{margin:4px 0;color:#98a29c;font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;transition:color .2s ease,transform .2s ease,font-size .2s ease}",
    ".lyrics p.section{margin-top:12px;color:rgba(" + accentRgb + ",.6);font-size:10px;letter-spacing:.14em;text-transform:uppercase}",
    ".lyrics p.active{color:#e9eeeb;font-size:14px;transform:translateX(10px)}",
    "audio{display:block;width:100%;margin-top:16px}",
    "footer{display:flex;justify-content:space-between;gap:12px;margin-top:10px;color:#98a29c;font:10px ui-monospace,monospace;letter-spacing:.1em}",
  ].join("\n");
  const script = [
    '"use strict";',
    "(() => {",
    "  var $ = function (id) { return document.getElementById(id); };",
    "  var audio = $(\"player\");",
    "  var canvas = $(\"spectrum\");",
    "  var clock = $(\"clock\");",
    "  var rows = Array.prototype.slice.call(document.querySelectorAll(\".lyrics p\"));",
    "  var candidates = rows.filter(function (row) { return row.dataset.section !== \"1\"; });",
    "  var times = (function () {",
    "    var map = {};",
    "    var data = " + JSON.stringify(Array.isArray(timestamps) ? timestamps : []) + ";",
    "    for (var i = 0; i < data.length; i += 1) {",
    "      if (data[i] && isFinite(Number(data[i].start))) map[Number(data[i].line)] = Number(data[i].start);",
    "    }",
    "    return map;",
    "  })();",
    "  var context = null;",
    "  var analyser = null;",
    "  var animation = 0;",
    "  var activeIndex = -1;",
    "  var ACCENT = \"" + accent + "\";",
    "  function formatTime(seconds) {",
    "    if (!isFinite(seconds)) return \"00:00\";",
    "    var whole = Math.max(0, Math.floor(seconds));",
    "    var m = Math.floor(whole / 60);",
    "    var s = whole % 60;",
    "    return (m < 10 ? \"0\" + m : m) + \":\" + (s < 10 ? \"0\" + s : s);",
    "  }",
    "  function paint() {",
    "    var ratio = Math.min(window.devicePixelRatio || 1, 2);",
    "    var width = Math.max(canvas.clientWidth, 1);",
    "    var height = Math.max(canvas.clientHeight, 1);",
    "    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) { canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio); }",
    "    var draw = canvas.getContext(\"2d\");",
    "    draw.setTransform(ratio, 0, 0, ratio, 0, 0);",
    "    draw.clearRect(0, 0, width, height);",
    "    var bins = new Uint8Array(analyser ? analyser.frequencyBinCount : 64);",
    "    if (analyser) analyser.getByteFrequencyData(bins);",
    "    var lowCount = Math.min(8, bins.length);",
    "    var low = 0;",
    "    for (var i = 0; i < lowCount; i += 1) low += bins[i];",
    "    low = lowCount ? low / lowCount : 0;",
    "    document.documentElement.style.setProperty(\"--pulse\", String(Math.min(1, low / 80)));",
    "    var count = 46;",
    "    var gap = 3;",
    "    var bar = Math.max(2, (width - gap * (count - 1)) / count);",
    "    for (var index = 0; index < count; index += 1) {",
    "      var sample = bins[Math.floor((index / count) * bins.length)] || (audio.paused ? 18 + Math.sin(index * 0.65) * 9 : 0);",
    "      var level = Math.max(3, (sample / 255) * (height - 8));",
    "      var x = index * (bar + gap);",
    "      var y = height - level;",
    "      var gradient = draw.createLinearGradient(0, y, 0, height);",
    "      gradient.addColorStop(0, ACCENT);",
    "      gradient.addColorStop(1, ACCENT + \"33\");",
    "      draw.fillStyle = gradient;",
    "      draw.beginPath();",
    "      if (draw.roundRect) draw.roundRect(x, y, bar, level, Math.min(bar / 2, 3)); else draw.rect(x, y, bar, level);",
    "      draw.fill();",
    "    }",
    "    if (!audio.paused && !audio.ended) animation = requestAnimationFrame(paint);",
    "  }",
    "  function syncLyrics() {",
    "    if (!candidates.length || !isFinite(audio.duration) || audio.duration <= 0) return;",
    "    var known = Object.keys(times).length > 0;",
    "    var selected = null;",
    "    if (known) {",
    "      for (var k = 0; k < candidates.length; k += 1) {",
    "        var rowIndex = rows.indexOf(candidates[k]);",
    "        if (times[rowIndex] != null && times[rowIndex] <= audio.currentTime + 0.15) selected = candidates[k];",
    "      }",
    "    } else {",
    "      selected = candidates[Math.min(candidates.length - 1, Math.floor((audio.currentTime / audio.duration) * candidates.length))];",
    "    }",
    "    if (!selected) return;",
    "    var index = rows.indexOf(selected);",
    "    if (index === activeIndex) return;",
    "    activeIndex = index;",
    "    rows.forEach(function (row, i) { row.classList.toggle(\"active\", i === index); });",
    "    var panel = selected.parentElement;",
    "    panel.scrollTo({ top: Math.max(0, selected.offsetTop - panel.clientHeight / 2), behavior: \"smooth\" });",
    "  }",
    "  audio.addEventListener(\"play\", async function () {",
    "    var AudioContextType = window.AudioContext || window.webkitAudioContext;",
    "    if (!AudioContextType) return;",
    "    if (!context) {",
    "      context = new AudioContextType();",
    "      analyser = context.createAnalyser();",
    "      analyser.fftSize = 256;",
    "      analyser.smoothingTimeConstant = 0.84;",
    "      var node = context.createMediaElementSource(audio);",
    "      node.connect(analyser);",
    "      analyser.connect(context.destination);",
    "    }",
    "    await context.resume();",
    "    cancelAnimationFrame(animation);",
    "    paint();",
    "  });",
    "  audio.addEventListener(\"pause\", function () { cancelAnimationFrame(animation); paint(); });",
    "  audio.addEventListener(\"ended\", function () { cancelAnimationFrame(animation); paint(); });",
    "  audio.addEventListener(\"timeupdate\", function () {",
    "    clock.textContent = formatTime(audio.currentTime) + \" / \" + formatTime(audio.duration);",
    "    syncLyrics();",
    "  });",
    "  requestAnimationFrame(paint);",
    "})();",
  ].join("\n");
  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    "<title>" + escapeHtml(title) + "</title>",
    "<style>",
    css,
    "</style>",
    "</head>",
    "<body>",
    "<main>",
    "<h1>" + escapeHtml(title) + "</h1>",
    '<p class="source"><a href="' + escapeHtml(source) + '" target="_blank" rel="noopener">' + escapeHtml(source) + "</a></p>",
    '<div class="stage"><div class="orbit"></div><canvas id="spectrum" aria-label="Live audio spectrum"></canvas></div>',
    (lyrics.length ? '<section class="lyrics">\n    ' + rows + "\n</section>" : ""),
    '<audio id="player" src="' + escapeHtml(source) + '" crossorigin="anonymous" controls preload="metadata"></audio>',
    '<footer><span>SELF-CONTAINED PLAYER &middot; MM-TOOLS</span><time id="clock">00:00 / 00:00</time></footer>',
    "</main>",
    "<script>",
    script,
    "</scr" + "ipt>",
    "</body>",
    "</html>",
  ].join("\n");
}

function exportAudioPlayer(item) {
  const source = new URL(item.url, location.origin).href;
  const accentRaw = state.manifest?.theme?.accent;
  const accent = /^#[0-9a-fA-F]{6}$/.test(String(accentRaw || "")) ? accentRaw : "#90efb8";
  const title = item.label || item.mode?.title || "Track";
  const html = buildAudioPlayerHtml({ title, source, accent, lyrics: trackLyrics(item), timestamps: item.metadata?.lyric_timestamps });
  downloadBlob(`${slugify(title)}-player.html`, new Blob([html], { type: "text/html" }));
  toast("Self-contained player downloaded.", "info", 3200);
}

function audioStudioNode(item) {
  const shell = document.createElement("div"); shell.className = "audio-stage";
  const canvas = document.createElement("canvas"); canvas.className = "audio-spectrum"; canvas.setAttribute("aria-label", "Live audio spectrum");
  const glow = document.createElement("div"); glow.className = "audio-orbit";
  const lyrics = trackLyrics(item);
  let lyricRows = []; let lyricsPanel = null;
  if (lyrics.length) {
    const panel = document.createElement("div"); panel.className = "audio-lyrics"; lyricsPanel = panel;
    lyricRows = lyrics.map((line) => {
      const row = document.createElement("p"); row.textContent = line; row.classList.toggle("section", /^\[.+\]$/.test(line)); panel.append(row); return row;
    });
    shell.append(panel);
  }
  const audio = mediaNode(item, false); audio.classList.add("studio-audio");
  const toolbar = document.createElement("div"); toolbar.className = "audio-toolbar";
  const big = document.createElement("button"); big.type = "button"; big.textContent = "⛶ Performance";
  big.onclick = () => openPerformance(item);
  const exportButton = document.createElement("button"); exportButton.type = "button"; exportButton.textContent = "⤓ Export"; exportButton.title = "Download a self-contained player (.html) for this track";
  exportButton.onclick = () => exportAudioPlayer(item);
  toolbar.append(big, exportButton);
  document.addEventListener("fullscreenchange", () => { if (document.fullscreenElement === shell && shell.isConnected) paint(); });
  let context; let analyser; let animation = 0; let source;
  const paint = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(canvas.clientWidth, 1); const height = Math.max(canvas.clientHeight, 1);
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) { canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio); }
    const draw = canvas.getContext("2d"); draw.setTransform(ratio, 0, 0, ratio, 0, 0); draw.clearRect(0, 0, width, height);
    const bins = new Uint8Array(analyser?.frequencyBinCount || 64); if (analyser) analyser.getByteFrequencyData(bins);
    const count = 46; const gap = 3; const bar = Math.max(2, (width - gap * (count - 1)) / count); const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#90efb8";
    for (let index = 0; index < count; index += 1) {
      const sample = bins[Math.floor((index / count) * bins.length)] || (audio.paused ? 18 + Math.sin(index * .65) * 9 : 0);
      const level = Math.max(3, (sample / 255) * (height - 8)); const x = index * (bar + gap); const y = height - level;
      const gradient = draw.createLinearGradient(0, y, 0, height); gradient.addColorStop(0, accent); gradient.addColorStop(1, `${accent}33`); draw.fillStyle = gradient; draw.beginPath(); draw.roundRect(x, y, bar, level, Math.min(bar / 2, 3)); draw.fill();
    }
    if (!audio.paused && !audio.ended) animation = requestAnimationFrame(paint);
  };
  const times = lyricTimes(lyrics, item.metadata?.lyric_timestamps);
  const syncLyrics = () => {
    if (!lyricRows.length) return;
    const candidates = lyricRows.map((row, index) => ({ row, index })).filter(({ row }) => !row.classList.contains("section"));
    if (!candidates.length) return;
    const known = times && candidates.some(({ index }) => times[index] != null);
    let selected;
    if (known && Number.isFinite(audio.currentTime)) {
      selected = null;
      for (const candidate of candidates) {
        if (times[candidate.index] != null && times[candidate.index] <= audio.currentTime + 0.15) selected = candidate;
      }
    } else if (Number.isFinite(audio.duration) && audio.duration > 0) {
      selected = candidates[Math.min(candidates.length - 1, Math.floor((audio.currentTime / audio.duration) * candidates.length))];
    }
    if (!selected) return;
    lyricRows.forEach((row) => row.classList.toggle("active", row === selected.row));
    lyricsPanel?.scrollTo({ top: Math.max(0, selected.row.offsetTop - lyricsPanel.clientHeight / 2), behavior: "smooth" });
  };
  audio.addEventListener("play", async () => {
    if (!context) {
      const AudioContextType = window.AudioContext || window.webkitAudioContext; if (!AudioContextType) return;
      context = new AudioContextType(); analyser = context.createAnalyser(); analyser.fftSize = 256; analyser.smoothingTimeConstant = .84;
      source = context.createMediaElementSource(audio); source.connect(analyser); analyser.connect(context.destination);
    }
    await context.resume(); cancelAnimationFrame(animation); paint();
  });
  audio.addEventListener("pause", () => { cancelAnimationFrame(animation); paint(); });
  audio.addEventListener("ended", () => { cancelAnimationFrame(animation); paint(); });
  audio.addEventListener("timeupdate", syncLyrics);
  shell.append(glow, canvas, audio, toolbar); requestAnimationFrame(paint); return shell;
}

function reopenJob(job) {
  const mode = modeById(job.mode); if (!mode) return toast("This workflow is no longer in the active manifest.", "error");
  state.values[mode.id] = structuredClone(job.request.controls || defaultsForMode(mode)); selectMode(mode.id, false); setPage("create"); document.querySelector(".studio-deck").scrollIntoView({ behavior: "smooth" }); toast("The exact visible recipe has been restored.");
}

function renderCompare() {
  const media = outputs().filter((item) => ["audio", "video", "image", "model", "motion"].includes(item.kind));
  const fill = (select) => {
    const current = select.value; select.innerHTML = '<option value="">Choose an output…</option>';
    media.forEach((item, index) => { const option = document.createElement("option"); option.value = `${item.job.id}:${item.relative}`; option.textContent = `${item.mode.title} · ${item.label || item.name} · ${index + 1}`; select.append(option); });
    if ([...select.options].some((option) => option.value === current)) select.value = current;
  };
  fill($("compareA")); fill($("compareB")); updateCompareSide("A"); updateCompareSide("B");
}

function selectedCompare(side) {
  const value = $(`compare${side}`).value; if (!value) return null;
  const split = value.indexOf(":"); const jobId = value.slice(0, split); const relative = value.slice(split + 1);
  return outputs().find((item) => item.job.id === jobId && item.relative === relative);
}

function updateCompareSide(side) {
  const item = selectedCompare(side); const media = $(`compareMedia${side}`); media.replaceChildren();
  if (!item) { media.innerHTML = "<p>Choose an output.</p>"; $(`compareMeta${side}`).textContent = ""; return; }
  media.append(mediaNode(item)); $(`compareMeta${side}`).textContent = JSON.stringify({ mode: item.job.mode, controls: item.job.request.controls, output: item.metadata }, null, 2); updateCrossfade();
}

function updateCrossfade() {
  const fraction = Number($("crossfade").value) / 100;
  const audioA = $("compareMediaA").querySelector("audio,video"); const audioB = $("compareMediaB").querySelector("audio,video");
  if (audioA) audioA.volume = Math.cos(fraction * Math.PI / 2);
  if (audioB) audioB.volume = Math.sin(fraction * Math.PI / 2);
}

function syncedPlay() {
  const media = [$("compareMediaA").querySelector("audio,video"), $("compareMediaB").querySelector("audio,video")].filter(Boolean);
  if (!media.length) return toast("Choose audio or video outputs first.", "error");
  const playing = media.some((item) => !item.paused);
  if (playing) media.forEach((item) => item.pause());
  else { const position = Math.min(...media.map((item) => item.currentTime || 0)); media.forEach((item) => { item.currentTime = position; item.play(); }); }
}

function saveCurrentPreset() {
  if (!state.mode) return toast("Choose a workflow first.", "error");
  $("presetName").value = state.mode.title;
  $("presetNote").value = "";
  $("presetDialog").showModal();
}

function confirmPreset(event) {
  event.preventDefault();
  const name = $("presetName").value.trim(); if (!name) return toast("Give this recipe a name.", "error");
  state.presets.unshift({ id: crypto.randomUUID(), name, note: $("presetNote").value.trim(), mode: state.mode.id, controls: structuredClone(controlsForMode()), created: Date.now() });
  localStorage.setItem(projectKey("presets"), JSON.stringify(state.presets)); $("presetDialog").close(); renderPresets(); toast("Preset saved locally in this browser.");
}

function renderPresets() {
  if (!state.presets.length) { $("presetGrid").innerHTML = '<div class="empty-state"><div><b>No saved recipes yet.</b>Save any visible workflow state without hiding a default.</div></div>'; return; }
  $("presetGrid").replaceChildren(...state.presets.map((preset) => {
    const mode = modeById(preset.mode); const card = document.createElement("article"); card.className = "preset-card";
    card.innerHTML = `<header><h3>${escapeHtml(preset.name)}</h3><small>${escapeHtml(mode?.icon || "✦")}</small></header><p>${escapeHtml(preset.note || mode?.description || "")}</p><small>${escapeHtml(mode?.title || preset.mode)} · ${new Date(preset.created).toLocaleDateString()}</small>`;
    const footer = document.createElement("footer");
    const use = document.createElement("button"); use.textContent = "Use recipe"; use.onclick = () => { state.values[preset.mode] = structuredClone(preset.controls); selectMode(preset.mode, false); setPage("create"); toast("Preset restored."); };
    const remove = document.createElement("button"); remove.textContent = "Delete"; remove.onclick = () => { state.presets = state.presets.filter((item) => item.id !== preset.id); localStorage.setItem(projectKey("presets"), JSON.stringify(state.presets)); renderPresets(); };
    footer.append(use, remove); card.append(footer); return card;
  }));
}

function renderHelp() {
  const manifest = state.manifest;
  $("capabilityMap").replaceChildren(...(manifest.modes || []).map((mode) => {
    const item = document.createElement("div"); item.innerHTML = `<b>${escapeHtml(mode.icon || "✦")} ${escapeHtml(mode.title)}</b><small>${escapeHtml(mode.short || mode.description || "")}</small>`; return item;
  }));
  $("modelChecklist").replaceChildren(...(manifest.models || []).map((model) => {
    const item = document.createElement("div"); item.className = "check-item"; item.innerHTML = `<span>${model.optional ? "○" : "✓"}</span><div><b>${escapeHtml(model.label)}</b><small>${escapeHtml(model.path || model.note || "")}</small></div>`; return item;
  }));
  $("setupSteps").replaceChildren(...(manifest.setup_steps || []).map((step) => { const item = document.createElement("li"); item.innerHTML = escapeHtml(step).replaceAll(/`([^`]+)`/g, "<code>$1</code>"); return item; }));
  updateApiExample();
}

function apiExamples() {
  const request = currentRequest() || { mode: state.manifest?.modes?.[0]?.id || "workflow", controls: {} };
  const json = JSON.stringify(request, null, 2);
  return {
    curl: `curl -sS http://127.0.0.1:${state.manifest.port}/api/jobs \\\n+  -H 'Content-Type: application/json' \\\n+  --data-binary @- <<'JSON'\n${json}\nJSON`,
    python: `import requests\n\npayload = ${JSON.stringify(request, null, 4)}\njob = requests.post(\n    "http://127.0.0.1:${state.manifest.port}/api/jobs",\n    json=payload,\n    timeout=30,\n).json()\nprint(job["id"], job["status"])`,
    js: `const payload = ${json};\nconst response = await fetch("http://127.0.0.1:${state.manifest.port}/api/jobs", {\n  method: "POST",\n  headers: {"Content-Type": "application/json"},\n  body: JSON.stringify(payload),\n});\nconsole.log(await response.json());`,
  };
}

function updateApiExample() {
  if (!state.manifest) return;
  $("apiExample").textContent = apiExamples()[state.apiCode];
}

function exportState() {
  const payload = { schema: 1, project: state.manifest.id, exported_at: new Date().toISOString(), mode: state.mode?.id || null, values: state.values, presets: state.presets };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${state.manifest.id}-studio-state.json`; link.click(); URL.revokeObjectURL(link.href);
}

async function importStateFile(file) {
  try {
    const payload = JSON.parse(await file.text());
    if (payload.project !== state.manifest.id) throw new Error(`This state belongs to ${payload.project || "another project"}.`);
    state.values = payload.values || {}; state.presets = payload.presets || []; state.mode = modeById(payload.mode) || state.mode;
    localStorage.setItem(projectKey("presets"), JSON.stringify(state.presets)); saveDraft();
    if (state.mode) selectMode(state.mode.id, false); renderPresets(); toast("Studio state imported.");
  } catch (error) { toast(error.message, "error"); }
}

async function refreshHealth() {
  try {
    const health = await api("/api/health"); const adapter = health.adapter || {};
    $("runtimePill").className = `runtime-pill ${health.ok ? (health.busy ? "warn" : "") : "error"}`;
    const gpu = health.gpu?.devices?.[0];
    $("runtimeText").textContent = health.busy ? `${health.queue} waiting · ${adapter.loaded ? "loaded" : "loading"}` : adapter.loaded ? `${gpu ? `${Math.round(gpu.memory_used_mib / 1024)} GB` : "GPU"} · loaded` : health.ok ? "Ready · cold" : "Needs attention";
    $("loadModels").disabled = health.busy || adapter.loaded;
    $("unloadModels").disabled = health.busy || !adapter.loaded;
  } catch (error) { $("runtimePill").className = "runtime-pill error"; $("runtimeText").textContent = error.message; }
}

async function modelAction(action) {
  const button = action === "load" ? $("loadModels") : $("unloadModels"); button.disabled = true;
  try { const health = await api(`/api/models/${action}`, { method: "POST" }); toast(action === "load" ? "Models loaded into the local runtime." : "Models released from GPU memory."); await refreshHealth(); return health; }
  catch (error) { toast(error.message, "error"); } finally { button.disabled = false; }
}

function connectEvents() {
  if (state.meta.privacy.token_enabled) {
    clearInterval(state.pollTimer); state.pollTimer = setInterval(refreshJobs, 1800); return;
  }
  state.eventSource?.close();
  const source = new EventSource("/api/events"); state.eventSource = source;
  source.addEventListener("jobs", (event) => {
    const payload = JSON.parse(event.data); state.jobs = payload.jobs || []; renderCounts();
    refreshVisiblePages();
    refreshHealth();
  });
  source.onerror = () => { source.close(); clearInterval(state.pollTimer); state.pollTimer = setInterval(refreshJobs, 2000); };
}

function bindEvents() {
  qsa(".nav-item").forEach((item) => item.onclick = () => setPage(item.dataset.page));
  $("themeToggle").onclick = toggleTheme;
  $("railCollapse").onclick = () => document.body.classList.toggle("rail-folded");
  $("taskSearch").oninput = renderTaskCards;
  $("generationForm").onsubmit = queueCurrent;
  $("promptForm").onsubmit = runPromptPlan;
  $("perfClose").onclick = () => { if ($("perfAudio").paused) $("perfAudio").pause(); $("performanceDialog").close(); };
  $("perfSyncV1").onclick = () => runPerformanceSync(false);
  $("perfSyncV2").onclick = () => runPerformanceSync(true);
  $("perfExportPlayer").onclick = () => { if (currentPerfItem) exportAudioPlayer(currentPerfItem); };
  $("resetMode").onclick = () => { if (!state.mode) return; state.values[state.mode.id] = defaultsForMode(state.mode); renderForm(); saveDraft(); toast("Workflow controls reset."); };
  $("runtimeRefresh").onclick = refreshHealth;
  $("loadModels").onclick = () => modelAction("load");
  $("unloadModels").onclick = () => modelAction("unload");
  $("librarySearch").oninput = renderLibrary;
  $("compareA").onchange = () => updateCompareSide("A");
  $("compareB").onchange = () => updateCompareSide("B");
  $("crossfade").oninput = updateCrossfade;
  $("comparePlay").onclick = syncedPlay;
  $("savePresetQuick").onclick = saveCurrentPreset;
  $("newPreset").onclick = saveCurrentPreset;
  $("confirmPreset").onclick = confirmPreset;
  $("exportStudio").onclick = exportState;
  $("importStudio").onclick = () => $("studioStateFile").click();
  $("studioStateFile").onchange = () => { if ($("studioStateFile").files[0]) importStateFile($("studioStateFile").files[0]); };
  qsa(".api-tabs button").forEach((button) => button.onclick = () => { state.apiCode = button.dataset.code; qsa(".api-tabs button").forEach((item) => item.classList.toggle("active", item === button)); updateApiExample(); });
  $("copyApi").onclick = async () => { await navigator.clipboard.writeText($("apiExample").textContent); toast("API example copied."); };
  window.addEventListener("hashchange", () => setPage(location.hash.slice(1), false));
  document.addEventListener("pause", settleRender);
  document.addEventListener("ended", settleRender);
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state.page === "create") queueCurrent(event);
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state.page === "prompt") runPromptPlan(event);
    if (event.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { event.preventDefault(); setPage("create"); $("taskSearch").focus(); }
    if (!event.ctrlKey && !event.metaKey && !event.altKey && /^[1-6]$/.test(event.key) && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) setPage(pages[Number(event.key) - 1]);
  });
}

/* ---------------- Prompt Studio (Krea2 lane) ---------------- */

const PROMPT_MODES = [
  { value: "brief", icon: "✒", title: "Refine brief", note: "Turn the brief into the three caption sections; any lyrics you supply stay context only." },
  { value: "keep_lyrics", icon: "𝄞", title: "Keep my lyrics", note: "Finish the captions around your finished lyrics, quoted verbatim." },
  { value: "song", icon: "♪", title: "Draft song", note: "Caption sections plus an original lyric draft with section tags." },
  { value: "ask", icon: "?", title: "Ask the model", note: "Direct music questions — concise answers, no caption headings." },
];

const PROMPT_SAMPLING_FIELDS = [
  { id: "krea_temperature", label: "Temperature", type: "range", min: 0, max: 2, step: 0.01, value: 0.7 },
  { id: "krea_top_p", label: "Top-p", type: "range", min: 0.01, max: 1, step: 0.01, value: 0.95 },
  { id: "krea_top_k", label: "Top-k", type: "number", min: 1, max: 4096, step: 1, value: 64 },
  { id: "krea_min_p", label: "Min-p", type: "range", min: 0, max: 0.5, step: 0.01, value: 0.05 },
  { id: "krea_repetition_penalty", label: "Repetition penalty", type: "range", min: 1, max: 2, step: 0.01, value: 1.05 },
  { id: "krea_presence_penalty", label: "Presence penalty", type: "range", min: -2, max: 2, step: 0.01, value: 0 },
  { id: "krea_seed", label: "Seed", type: "number", min: 0, max: 1000000000, step: 1, value: 0 },
  { id: "krea_max_length", label: "Max tokens", type: "number", min: 64, max: 8192, step: 1, value: 1024 },
];

async function rawFile(url) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(url, { headers });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

function renderPromptStudio() {
  const modes = $("promptModes");
  if (!modes.childElementCount) {
    modes.replaceChildren(...PROMPT_MODES.map((mode) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip" + (mode.value === "brief" ? " active" : "");
      button.dataset.value = mode.value;
      button.innerHTML = `<span>${mode.icon}</span><b>${mode.title}</b><small>${escapeHtml(mode.note)}</small>`;
      button.onclick = () => qsa("#promptModes .chip").forEach((item) => item.classList.toggle("active", item === button));
      return button;
    }));
  }
  const sampling = $("promptSampling");
  if (!sampling.childElementCount) {
    sampling.replaceChildren(...PROMPT_SAMPLING_FIELDS.map((field) => {
      const wrap = document.createElement("label");
      wrap.className = "field";
      wrap.innerHTML = `<span>${field.label}</span><input id="${field.id}" type="${field.type}" min="${field.min}" max="${field.max}" step="${field.step}" value="${field.value}">`;
      return wrap;
    }));
  }
  updatePromptResult();
}

async function runPromptPlan(event) {
  event?.preventDefault();
  const brief = $("promptBrief").value.trim();
  if (!brief) { $("promptStatus").textContent = "Describe the musical direction first."; return; }
  const mode = document.querySelector("#promptModes .chip.active")?.dataset.value || "brief";
  const lyrics = $("promptLyrics").value;
  if (mode === "keep_lyrics" && !lyrics.trim()) { toast("“Keep my lyrics” planning needs the finished lyrics.", "error"); return; }
  const controls = {
    krea_mode: mode,
    krea_brief: brief,
    krea_lyrics: lyrics,
    krea_constraints: $("promptConstraints").value,
    krea_research: $("promptResearch").checked,
    krea_research_query: $("promptResearchQuery").value.trim(),
  };
  for (const field of PROMPT_SAMPLING_FIELDS) {
    const raw = $(field.id).value;
    if (raw === "") continue;
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) controls[field.id] = parsed;
  }
  state.promptJobId = null;
  const button = $("runPromptPlan");
  button.disabled = true; button.querySelector("span").textContent = "Planning…";
  $("promptStatus").textContent = "Queued on the private GPU queue — shared Krea2 lane.";
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ mode: "krea_plan", controls, client: { source: "mm-tools-studio", schema: 1, queued_at: new Date().toISOString() } }) });
    state.promptJobId = job.id;
    state.jobs.unshift(job);
    renderCounts(); refreshVisiblePages(); updatePromptResult();
  } catch (error) {
    toast(error.message, "error");
    button.disabled = false; button.querySelector("span").textContent = "Run Krea plan";
  }
}

function promptJob() {
  if (!state.promptJobId) return null;
  return state.jobs.find((job) => job.id === state.promptJobId) || null;
}

function updatePromptResult() {
  const job = promptJob();
  const status = $("promptStatus");
  const dot = $("promptDot");
  if (!job) return;
  if (ACTIVE.has(job.status)) {
    dot.className = "validation-dot busy";
    status.textContent = `Planning · ${Math.round((job.progress || 0) * 100)}% — ${job.stage || "queued"}`;
    return;
  }
  if (!TERMINAL.has(job.status)) return;
  dot.className = job.status === "complete" ? "validation-dot ok" : "validation-dot bad";
  if (job.status !== "complete") {
    status.textContent = job.error || "The plan failed — adjust the brief and retry.";
    return;
  }
  status.textContent = "Plan complete — copy a section or carry it into Create.";
  loadPromptPlan(job);
}

async function loadPromptPlan(job) {
  if (state.promptLoadedFor === job.id) return;
  state.promptLoadedFor = job.id;
  const outputs = job.outputs || [];
  const planOutput = outputs.find((item) => item.relative === "krea_plan.json");
  const direction = outputs.find((item) => item.metadata?.krea);
  const meta = { style: direction?.metadata?.style || "", lyrics: direction?.metadata?.lyrics || "" };
  if (!planOutput) {
    $("promptResults").innerHTML = "<p class='muted'>The plan text is missing from this job.</p>";
    return;
  }
  try {
    const payload = { ...JSON.parse(await rawFile(planOutput.url)), ...meta };
    if (direction) {
      try { payload.raw = await rawFile(direction.url); } catch (_) { /* metadata only */ }
    }
    renderPromptResults(payload);
  } catch (error) {
    $("promptResults").innerHTML = `<p class="muted">Could not read the plan: ${escapeHtml(error.message)}</p>`;
  }
}

function renderPromptResults(payload) {
  const grid = $("promptResults");
  const rawSections = payload.sections;
  const sections = Array.isArray(rawSections)
    ? rawSections
    : rawSections && typeof rawSections === "object"
      ? Object.entries(rawSections).map(([heading, text]) => ({ heading, text: String(text) }))
      : [];
  $("promptResultMode").textContent = payload.mode || "";
  const hasSections = sections.length > 0;
  grid.replaceChildren();
  if (!hasSections) {
    const card = document.createElement("article");
    card.className = "prompt-section-card";
    const text = typeof payload.raw === "string" && payload.raw.trim() ? payload.raw : JSON.stringify(payload, null, 2);
    card.innerHTML = `<small>ANSWER</small><pre class="prompt-raw">${escapeHtml(text)}</pre>`;
    grid.append(card);
  } else {
    for (const section of sections) {
      const card = document.createElement("article");
      card.className = "prompt-section-card";
      const copy = document.createElement("button");
      copy.type = "button"; copy.textContent = "⧉ Copy";
      copy.onclick = async () => { await navigator.clipboard.writeText(String(section.text || "")); toast(`${section.heading} copied.`); };
      card.innerHTML = `<div class="recipe-head"><small>${escapeHtml(section.heading || "SECTION")}</small></div><p>${escapeHtml(section.text || "")}</p>`;
      card.append(copy);
      grid.append(card);
    }
    if (Array.isArray(payload.sources) && payload.sources.length) {
      const sources = document.createElement("article");
      sources.className = "prompt-section-card prompt-sources";
      sources.innerHTML = `<div class="recipe-head"><small>SOURCES</small></div><ul>${payload.sources.map((source) => `<li>${escapeHtml(source)}</li>`).join("")}</ul>`;
      grid.append(sources);
    }
    if (typeof payload.raw === "string" && payload.raw.trim()) {
      const rawCard = document.createElement("details");
      rawCard.className = "prompt-raw-wrap";
      rawCard.innerHTML = `<summary>Raw response</summary><pre class="prompt-raw">${escapeHtml(payload.raw)}</pre>`;
      grid.append(rawCard);
    }
  }
  $("useInCreate").hidden = !(typeof payload.style === "string" && payload.style.trim());
  $("copyRawPrompt").hidden = !(typeof payload.raw === "string" && payload.raw.trim());
  $("useInCreate").onclick = () => {
    const mode = modeById("compose_full");
    if (!mode) return toast("The Create workflow is not available.", "error");
    const values = state.values[mode.id] || defaultsForMode(mode);
    state.values[mode.id] = values;
    values.style = payload.style;
    if (payload.lyrics) values.lyrics = payload.lyrics;
    selectMode(mode.id, false); setPage("create"); toast("Krea direction carried into Create.");
  };
  $("copyRawPrompt").onclick = async () => { await navigator.clipboard.writeText(payload.raw); toast("Raw response copied."); };
}

/* ---------------- Performance view ---------------- */

const performanceGraph = { ctx: null, source: null, analyser: null };

function ensurePerformanceGraph() {
  if (!performanceGraph.ctx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    performanceGraph.ctx = new Ctx();
    performanceGraph.source = performanceGraph.ctx.createMediaElementSource($("perfAudio"));
    performanceGraph.analyser = performanceGraph.ctx.createAnalyser();
    performanceGraph.analyser.fftSize = 256;
    performanceGraph.analyser.smoothingTimeConstant = 0.82;
    performanceGraph.source.connect(performanceGraph.analyser);
    performanceGraph.analyser.connect(performanceGraph.ctx.destination);
  }
  return performanceGraph;
}

const perfLyricState = { candidates: [], times: [] };
let currentPerfItem = null;
let chrisperTimer = 0;

function openPerformance(item) {
  const dialog = $("performanceDialog");
  const audio = $("perfAudio");
  currentPerfItem = item;
  $("perfStyle").textContent = item.metadata?.style || item.job?.request?.controls?.style || "—";
  $("perfEngine").textContent = jobMode(item.job)?.title || "YuE2";
  $("perfNowPlaying").textContent = item.label || item.mode?.title || "Take";
  renderPerformanceLyrics(item);
  audio.src = item.url;
  const graph = ensurePerformanceGraph();
  const canvas = $("perfSpectrum");
  const ctx = canvas.getContext("2d");
  let perfRaf = 0;
  const paint = () => {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, canvas.clientWidth || 900);
    if (canvas.width !== width * dpr) { canvas.width = width * dpr; canvas.height = 220 * dpr; }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (graph?.analyser) {
      const data = new Uint8Array(graph.analyser.frequencyBinCount);
      graph.analyser.getByteFrequencyData(data);
      const bars = 96;
      const step = Math.max(1, Math.floor(data.length / bars));
      const barWidth = canvas.width / bars;
      for (let bar = 0; bar < bars; bar += 1) {
        const value = data[bar * step] / 255;
        const height = Math.max(3, value * canvas.height * 0.92);
        const hue = 150 + value * 90;
        ctx.fillStyle = `hsla(${hue}, 62%, ${34 + value * 30}%, ${0.55 + value * 0.4})`;
        const x = bar * barWidth + 1;
        ctx.fillRect(x, canvas.height - height, Math.max(2, barWidth - 2), height);
      }
    }
    if (!audio.paused) {
      const current = audio.currentTime;
      const candidates = perfLyricState.candidates;
      const candidateTime = perfLyricState.times;
      const known = candidateTime.some((value) => value != null);
      $("perfClock").textContent = `${formatClock(current)} / ${formatClock(Number.isFinite(audio.duration) ? audio.duration : 0)}`;
      let selected = -1;
      if (known) {
        candidates.forEach((node, index) => { if (candidateTime[index] != null && candidateTime[index] <= current + 0.15) selected = index; });
      } else if (Number.isFinite(audio.duration) && audio.duration > 0) {
        selected = Math.min(candidates.length - 1, Math.floor((current / audio.duration) * candidates.length));
      }
      candidates.forEach((node, index) => node.classList.toggle("active", index === selected));
      const activeNode = candidates[selected];
      if (activeNode) {
        const offset = activeNode.offsetTop - lyricsBox.clientHeight / 2 + activeNode.clientHeight / 2;
        if (Math.abs(lyricsBox.scrollTop - offset) > 4) lyricsBox.scrollTo({ top: offset, behavior: "smooth" });
      }
    }
    perfRaf = requestAnimationFrame(paint);
  };
  const stop = () => {
    cancelAnimationFrame(perfRaf);
    clearInterval(chrisperTimer);
    chrisperTimer = 0;
    if (performanceGraph.ctx && performanceGraph.ctx.state === "running") performanceGraph.ctx.suspend().catch(() => {});
  };
  audio.onended = stop;
  dialog.addEventListener("close", stop, { once: true });
  if (performanceGraph.ctx && performanceGraph.ctx.state === "suspended") performanceGraph.ctx.resume().catch(() => {});
  audio.play().catch(() => {});
  paint();
  refreshChrisperStatus();
  chrisperTimer = setInterval(refreshChrisperStatus, 4000);
  dialog.showModal();
}

function renderPerformanceLyrics(item) {
  const rows = trackLyrics(item);
  const times = lyricTimes(rows, item.metadata?.lyric_timestamps);
  const lyricsBox = $("perfLyrics");
  lyricsBox.replaceChildren();
  if (!rows.length) {
    lyricsBox.innerHTML = "<p class='muted'>No lyrics attached to this take.</p>";
    perfLyricState.candidates = [];
    perfLyricState.times = [];
    return;
  }
  rows.forEach((line, index) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    paragraph.dataset.index = index;
    if (/^\[.+\]$/.test(line)) paragraph.className = "section";
    lyricsBox.append(paragraph);
  });
  perfLyricState.candidates = [...lyricsBox.querySelectorAll("p:not(.section)")];
  perfLyricState.times = perfLyricState.candidates.map((node) => {
    const rowIndex = Number(node.dataset.index);
    return times && times[rowIndex] != null ? times[rowIndex] : null;
  });
}

async function refreshChrisperStatus() {
  const node = $("chrisperStatus");
  if (!node) return;
  const label = node.querySelector("span");
  try {
    const data = await api("/api/chrisper/status");
    node.classList.toggle("ready", Boolean(data.up && data.loaded));
    node.classList.toggle("warm", Boolean(data.up && !data.loaded));
    node.classList.toggle("offline", !data.up);
    label.textContent = !data.up ? `CRISPER OFFLINE · :${data.port}` : data.loaded ? "CRISPER READY" : "CRISPER LOADING";
  } catch (_) {
    node.classList.add("offline");
    node.classList.remove("ready", "warm");
    label.textContent = "CRISPER OFFLINE";
  }
}

async function runPerformanceSync(chrisper) {
  const item = currentPerfItem;
  if (!item || !item.job?.id || !item.relative) { toast("No take selected.", "error"); return; }
  const button = $(chrisper ? "perfSyncV2" : "perfSyncV1");
  const idle = button.textContent;
  button.disabled = true;
  button.textContent = chrisper ? "Refining…" : "Syncing…";
  try {
    const payload = { job_id: item.job.id, relative: item.relative };
    if (chrisper) payload.language = $("chrisperLanguage").value || "auto";
    const data = await api(chrisper ? "/api/performance/chrisper" : "/api/performance/vocal-sync", { method: "POST", body: JSON.stringify(payload) });
    item.metadata = { ...(item.metadata || {}), lyric_timestamps: data.rows };
    renderPerformanceLyrics(item);
    if (chrisper) toast(`Chrisper v2: ${data.matched}/${data.total} lines re-timed (${data.language}, ${data.snippets} passes).`, "info", 4200);
    else toast("Vocal sync applied.", "info", 3200);
  } catch (error) {
    toast(error.message, "error", 4200);
  } finally {
    button.disabled = false;
    button.textContent = idle;
  }
}

function formatClock(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const rest = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function lyricTimes(lyricLines, timestamps) {
  if (!Array.isArray(timestamps) || !timestamps.length) return null;
  const byLine = new Map();
  for (const row of timestamps) {
    if (row && Number.isFinite(Number(row.start))) byLine.set(Number(row.line), Number(row.start));
  }
  if (!byLine.size) return null;
  return lyricLines.map((_, index) => (byLine.has(index) ? byLine.get(index) : null));
}

/* ---------------- Stems page ---------------- */

const stemKit = {
  sourceInput: null,
  lanes: [],
  audio: null,
  graph: null,
  raf: 0,
};

const STEM_PRESETS = [
  { id: "all", label: "Full 4-stem", note: "demucs htdemucs", stems: ["vocals", "drums", "bass", "other"] },
  { id: "acapella", label: "Acapella", note: "Mel-Band Roformer", stems: ["vocals"] },
  { id: "karaoke", label: "Instrumental", note: "no-vocals mix", stems: ["drums", "bass", "other"] },
  { id: "drums_bass", label: "Drums + Bass", note: "two stems", stems: ["drums", "bass"] },
];

function stemTracks() {
  return outputs().filter((item) => item.kind === "audio").slice(0, 24);
}

function renderStemsPage() {
  const pick = $("stemTrackPick");
  if (!pick.childElementCount) {
    const tracks = stemTracks();
    if (!tracks.length) {
      pick.innerHTML = "<p class='muted'>No rendered tracks yet — create one, or upload below.</p>";
    } else {
      tracks.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "stem-track" + (index === 0 ? " active" : "");
        button.innerHTML = `<span>♫</span><div><b>${escapeHtml(item.label || item.name)}</b><small>${escapeHtml(formatBytes(item.size))}</small></div>`;
        button.onclick = () => {
          qsa(".stem-track").forEach((other) => other.classList.toggle("active", other === button));
          stemKit.sourceInput = { job: item.job?.id, output: item.relative };
          $("stemStatus").textContent = "Selected — choose a preset and split.";
        };
        pick.append(button);
      });
      const first = pick.querySelector(".stem-track");
      if (first) first.click();
    }
  }
  const presets = $("stemPresets");
  if (!presets.childElementCount) {
    presets.replaceChildren(...STEM_PRESETS.map((preset) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip" + (preset.id === "all" ? " active" : "");
      button.dataset.preset = preset.id;
      button.innerHTML = `<span>${preset.label}</span><small>${preset.note}</small>`;
      button.onclick = () => {
        qsa("#stemPresets .chip").forEach((item) => item.classList.toggle("active", item === button));
        applyPresetToChecks(preset.stems);
      };
      return button;
    }));
    applyPresetToChecks(STEM_PRESETS[0].stems);
  }
  const toggles = $("stemToggles");
  if (!toggles.childElementCount) {
    [["vocals", "Vocals"], ["drums", "Drums"], ["bass", "Bass"], ["other", "Other"]].forEach(([key, label]) => {
      const wrap = document.createElement("label");
      wrap.className = "switch-row";
      wrap.innerHTML = `<input type="checkbox" value="${key}" ${key === "vocals" ? "checked" : ""}><span>${label}</span>`;
      toggles.append(wrap);
    });
  }
  if (!stemKit.bound) {
    stemKit.bound = true;
    $("stemSplit").onclick = runStemSplit;
    $("stemFile").onchange = () => { if ($("stemFile").files[0]) uploadStemSource($("stemFile").files[0]); };
    $("stemPlay").onclick = toggleStemPlayback;
    $("stemMaster").oninput = applyStemGains;
    $("stemSeek").oninput = seekStems;
  }
}

function applyPresetToChecks(stems) {
  qsa("#stemToggles input").forEach((item) => { item.checked = stems.includes(item.value); });
}

async function uploadStemSource(file) {
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const asset = await api("/api/stems/upload", { method: "POST", body: form });
    stemKit.sourceInput = asset.asset_id;
    $("stemStatus").textContent = `Uploaded ${file.name} — choose a preset and split.`;
    qsa(".stem-track").forEach((item) => item.classList.remove("active"));
  } catch (error) {
    toast(error.message, "error");
  }
}

async function runStemSplit() {
  if (!stemKit.sourceInput) { $("stemStatus").textContent = "Pick a track or upload one first."; return; }
  const stems = [...$("stemToggles").querySelectorAll("input:checked")].map((item) => item.value);
  if (!stems.length) { $("stemStatus").textContent = "Select at least one stem."; return; }
  const payload = { input: stemKit.sourceInput, stems };
  const button = $("stemSplit");
  button.disabled = true; button.querySelector("span").textContent = "Splitting…";
  $("stemProgressWrap").hidden = false;
  try {
    const result = await api("/api/stems/split", { method: "POST", body: JSON.stringify(payload) });
    pollStemJob(result.job_id);
  } catch (error) {
    toast(error.message, "error");
    $("stemStatus").textContent = error.message;
    $("stemProgressWrap").hidden = true;
  } finally {
    button.disabled = false; button.querySelector("span").textContent = "Split stems";
  }
}

function pollStemJob(id) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/stems/jobs/${id}`);
      $("stemProgressFill").style.width = `${Math.round(job.pct || 0)}%`;
      $("stemProgressText").textContent = `${job.stage || job.status || "working"} · ${Math.round(job.pct || 0)}%${job.message ? ` — ${job.message}` : ""}`;
      if (job.status === "failed") {
        clearInterval(timer);
        $("stemStatus").textContent = job.error || job.message || "The split failed.";
        $("stemProgressWrap").hidden = true;
        return;
      }
      if (job.status === "done") {
        clearInterval(timer);
        renderStemLanes(job);
      }
    } catch (error) {
      clearInterval(timer);
      $("stemStatus").textContent = error.message;
      $("stemProgressWrap").hidden = true;
    }
  }, 1500);
}

function renderStemLanes(job) {
  const lanes = $("stemLanes");
  stopStemPlayback();
  stemKit.lanes = [];
  lanes.replaceChildren();
  const names = Array.isArray(job.stems) && job.stems.length ? job.stems : ["vocals", "drums", "bass", "other"];
  for (const name of names) {
    const lane = document.createElement("div");
    lane.className = "stem-lane";
    lane.innerHTML = `<span class="lane-name">${escapeHtml(name)}</span><input type="range" min="0" max="100" value="100" aria-label="volume"><span class="lane-bpm"></span>`;
    const fader = lane.querySelector("input");
    const entry = { name, url: `/api/stems/jobs/${job.id}/${name}`, gain: 1, element: fader };
    fader.oninput = () => { entry.gain = Number(fader.value) / 100; applyStemGains(); };
    const exportLink = document.createElement("a");
    exportLink.href = entry.url;
    exportLink.download = `${name}.wav`;
    exportLink.className = "lane-export";
    exportLink.textContent = "⤓";
    exportLink.title = "Download this stem";
    lane.append(exportLink);
    stemKit.lanes.push(entry);
    lanes.append(lane);
  }
  if (!stemKit.lanes.length) {
    lanes.innerHTML = "<p class='muted'>No stems found on that job.</p>";
    return;
  }
  buildStemMix();
  $("stemStatus").textContent = `${stemKit.lanes.length} lanes live (${job.engine || "local engine"}) — drag the faders, watch the spectrum.`;
}

async function buildStemMix() {
  // Fetch every lane, align lengths, and render an offline mixed master so the
  // transport plays all lanes in lockstep without N parallel <audio> elements.
  const decoded = await Promise.all(stemKit.lanes.map(async (lane) => {
    const arrayBuffer = await (await fetch(lane.url, { headers: authHeaders() })).arrayBuffer();
    const audioCtx = new OfflineAudioContext(2, 1, 44100);
    return audioCtx.decodeAudioData(arrayBuffer.slice(0));
  }));
  const length = Math.max(...decoded.map((buffer) => buffer.length));
  const mix = new OfflineAudioContext(2, length, 44100);
  decoded.forEach((buffer, index) => {
    const source = mix.createBufferSource();
    source.buffer = buffer;
    const gain = mix.createGain();
    gain.gain.value = stemKit.lanes[index].gain;
    source.connect(gain); gain.connect(mix.destination);
    source.start(0);
  });
  const rendered = await mix.startRendering();
  const wav = audioBufferToWav(rendered);
  const blob = new Blob([wav], { type: "audio/wav" });
  if (stemKit.audio) URL.revokeObjectURL(stemKit.audio.src);
  stemKit.audio = new Audio(URL.createObjectURL(blob));
  stemKit.audio.volume = Number($("stemMaster").value) / 100;
  stemKit.audio.ontimeupdate = paintStemTransport;
  stemKit.audio.onended = () => { $("stemPlay").textContent = "▶ Play"; };
  startStemTransport();
}

function audioBufferToWav(buffer) {
  const channels = Math.min(2, buffer.numberOfChannels);
  const frames = buffer.length;
  const bytesPerSample = 2;
  const dataSize = frames * channels * bytesPerSample;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);
  const writeString = (offset, text) => { for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i)); };
  writeString(0, "RIFF"); view.setUint32(4, 36 + dataSize, true); writeString(8, "WAVE");
  writeString(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true); view.setUint16(34, 16, true);
  writeString(36, "data"); view.setUint32(40, dataSize, true);
  let offset = 44;
  const channelData = [...Array(channels)].map((_, channel) => buffer.getChannelData(channel));
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Uint8Array(arrayBuffer);
}

function authHeaders() {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  return headers;
}

function startStemTransport() {
  if (!stemKit.audio) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (Ctx) {
    const actx = new Ctx();
    const source = actx.createMediaElementSource(stemKit.audio);
    const analyser = actx.createAnalyser();
    analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.8;
    source.connect(analyser); analyser.connect(actx.destination);
    stemKit.graph = { actx, analyser };
  }
  const paint = () => {
    paintStemTransport();
    const lane = document.querySelector(".stem-lane .lane-bpm");
    if (stemKit.graph?.analyser && lane) {
      const data = new Uint8Array(stemKit.graph.analyser.frequencyBinCount);
      stemKit.graph.analyser.getByteFrequencyData(data);
      const energy = data.reduce((sum, value) => sum + value, 0) / data.length / 255;
      lane.style.setProperty("--pulse", energy.toFixed(3));
    }
    stemKit.raf = requestAnimationFrame(paint);
  };
  cancelAnimationFrame(stemKit.raf);
  stemKit.raf = requestAnimationFrame(paint);
}

function paintStemTransport() {
  if (!stemKit.audio) return;
  const duration = Number.isFinite(stemKit.audio.duration) ? stemKit.audio.duration : 0;
  $("stemClock").textContent = `${formatClock(stemKit.audio.currentTime)} / ${formatClock(duration)}`;
  if (duration > 0) $("stemSeek").value = String(Math.round((stemKit.audio.currentTime / duration) * 1000));
}

function toggleStemPlayback() {
  if (!stemKit.audio) return;
  if (stemKit.audio.paused) {
    stemKit.audio.play();
    $("stemPlay").textContent = "❚❚ Pause";
  } else {
    stemKit.audio.pause();
    $("stemPlay").textContent = "▶ Play";
  }
}

function stopStemPlayback() {
  if (stemKit.audio) { stemKit.audio.pause(); stemKit.audio.src = ""; }
  stemKit.audio = null;
  if (stemKit.graph?.actx) stemKit.graph.actx.close().catch(() => {});
  stemKit.graph = null;
  cancelAnimationFrame(stemKit.raf);
  stemKit.raf = 0;
}

function seekStems() {
  if (!stemKit.audio) return;
  const duration = Number.isFinite(stemKit.audio.duration) ? stemKit.audio.duration : 0;
  if (duration <= 0) return;
  stemKit.audio.currentTime = (Number($("stemSeek").value) / 1000) * duration;
}

function applyStemGains() {
  const master = Number($("stemMaster").value) / 100;
  stemKit.lanes.forEach((lane) => {
    lane.element.style.setProperty("--level", String(lane.gain * master));
  });
  if (stemKit.audio) stemKit.audio.volume = master;
  // A full offline re-mix is expensive; rebuild when playback pauses.
  if (stemKit.audio && !stemKit.audio.paused) {
    stemKit.audio.onpause = () => buildStemMix().catch(() => {});
  }
}

async function initialize() {
  applyTheme(localStorage.getItem("mm-tools-studio:theme") || "dark");
  bindEvents();
  try {
    state.meta = await api("/api/meta"); state.manifest = state.meta.manifest; state.jobs = state.meta.jobs || []; state.assets = state.meta.assets || [];
    restoreLocalState(); applyManifest(); renderCounts(); renderQueue(); renderLibrary(); renderCompare(); renderPresets();
    if (!state.mode && state.manifest.modes?.length) selectMode(state.manifest.default_mode || state.manifest.modes[0].id, false);
    setPage(location.hash.slice(1) || "create", false); await refreshHealth(); connectEvents();
  } catch (error) {
    $("heroTitle").textContent = "Studio could not initialize."; $("heroCopy").textContent = error.message; toast(error.message, "error", 9000);
  }
}

initialize();
