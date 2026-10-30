"use strict";
const $ = id => document.getElementById(id);
const ids = ["file", "drop", "convert", "error", "output", "serializationPanel", "resolution", "resolutionOut", "alphabet", "tracingPanel", "cleanupPanel", "mode", "preset", "colors", "colorsOut", "smoothness", "smoothOut", "speckle", "speckleOut", "threshold", "thresholdOut", "monoPanel", "invert", "removeBackground", "tolerance", "toleranceOut", "maxDimension", "empty", "stats", "copy", "dataUri", "download", "resultTab", "codeTab"];
const el = Object.fromEntries(ids.map(id => [id, $(id)]));
Object.assign(el, { vector: $("vectorPreview"), raster: $("rasterPreview"), code: $("codePreview"), text: $("textPreview") });
let file = null, result = null, resultUrl = "", rasterUrl = "", tab = "raster", request = null;
const format = n => new Intl.NumberFormat().format(n);
const textMode = () => el.output.value === "text";

function clearResult() {
  request?.abort();
  request = null;
  result = null;
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = "";
  el.vector.removeAttribute("src");
  el.code.querySelector("code").textContent = "";
  el.text.textContent = "";
  el.error.textContent = "";
  el.stats.textContent = file ? `${file.name} · ${format(file.size)} bytes` : "Ready";
  tab = "raster";
  sync();
}

function setFile(next) {
  file = next || null;
  if (rasterUrl) URL.revokeObjectURL(rasterUrl);
  rasterUrl = file ? URL.createObjectURL(file) : "";
  if (rasterUrl) el.raster.src = rasterUrl;
  else el.raster.removeAttribute("src");
  clearResult();
}

function render() {
  const text = textMode();
  el.empty.hidden = Boolean(file);
  el.vector.hidden = !(result && !text && tab === "vector");
  el.text.hidden = !(result && text && tab === "vector");
  el.raster.hidden = !(file && tab === "raster");
  el.code.hidden = !(result && !text && tab === "code");
  el.resultTab.textContent = text ? "Text" : "Vector";
  el.codeTab.hidden = text;
  el.copy.textContent = text ? "Copy text" : "Copy SVG";
  el.download.textContent = text ? "Download TXT" : "Download SVG";
  for (const button of document.querySelectorAll(".tab")) button.classList.toggle("active", button.dataset.tab === tab);
  for (const button of [el.copy, el.dataUri, el.download]) button.disabled = !result || Boolean(request);
  el.convert.disabled = !file || Boolean(request);
  el.convert.classList.toggle("busy", Boolean(request));
  el.convert.textContent = request ? (text ? "Serializing…" : "Tracing…") : (text ? "Serialize image" : "Vectorize image");
}

function sync() {
  el.colorsOut.textContent = `${el.colors.value} colors`;
  el.smoothOut.textContent = el.smoothness.value;
  el.speckleOut.textContent = `${el.speckle.value} px`;
  el.thresholdOut.textContent = el.threshold.value;
  el.toleranceOut.textContent = el.tolerance.value;
  el.resolutionOut.textContent = `${el.resolution.value} px`;
  el.serializationPanel.hidden = !textMode();
  el.tracingPanel.hidden = textMode();
  el.cleanupPanel.hidden = textMode();
  el.monoPanel.hidden = textMode() || el.mode.value !== "mono";
  el.colors.disabled = el.mode.value === "mono";
  render();
}

for (const event of ["dragenter", "dragover"]) el.drop.addEventListener(event, e => { e.preventDefault(); el.drop.classList.add("drag"); });
for (const event of ["dragleave", "drop"]) el.drop.addEventListener(event, e => { e.preventDefault(); el.drop.classList.remove("drag"); });
el.drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
el.file.addEventListener("change", () => setFile(el.file.files[0]));
el.output.addEventListener("change", clearResult);
for (const control of document.querySelectorAll("select,input[type=range],input[type=checkbox]")) control.addEventListener("input", sync);
document.querySelector(".tabs").addEventListener("click", e => {
  const button = e.target.closest(".tab");
  if (button) { tab = button.dataset.tab; render(); }
});

el.convert.addEventListener("click", async () => {
  if (!file || request) return;
  const controller = new AbortController(), source = file, text = textMode();
  request = controller;
  render();
  el.error.textContent = "";
  try {
    const options = text ? { resolution: Number(el.resolution.value), style: el.alphabet.value } : {
      mode: el.mode.value, preset: el.preset.value, colors: Number(el.colors.value),
      smoothness: Number(el.smoothness.value), speckle: Number(el.speckle.value),
      threshold: Number(el.threshold.value), invert: el.invert.checked,
      remove_background: el.removeBackground.checked, background_tolerance: Number(el.tolerance.value),
      max_dimension: Number(el.maxDimension.value), path_precision: 2
    };
    const form = new FormData();
    form.append("image", source, source.name);
    form.append("options", JSON.stringify(options));
    const response = await fetch(text ? "/api/serialize" : "/api/vectorize", { method: "POST", body: form, signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (request !== controller) return;
    result = { content: text ? data.text : data.svg, mime: text ? "text/plain;charset=utf-8" : "image/svg+xml", extension: text ? ".txt" : ".svg", name: source.name };
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = URL.createObjectURL(new Blob([result.content], { type: result.mime }));
    el.code.querySelector("code").textContent = result.content;
    if (text) el.text.textContent = result.content;
    else el.vector.src = resultUrl;
    tab = "vector";
    el.stats.textContent = text
      ? `${data.source_width}×${data.source_height} → ${data.columns} columns × ${data.rows} rows · ${data.style} · ${format(data.output_bytes)} bytes · ${data.elapsed_ms} ms`
      : `${data.source_width}×${data.source_height} → ${data.traced_width}×${data.traced_height} · ${format(data.path_count)} paths · ${format(data.output_bytes)} bytes · ${data.elapsed_ms} ms`;
  } catch (error) {
    if (request === controller && !controller.signal.aborted) el.error.textContent = error.message;
  } finally {
    if (request === controller) { request = null; render(); }
  }
});

async function copy(dataUri) {
  if (!result) return;
  try {
    const value = dataUri ? `data:${result.mime},${encodeURIComponent(result.content)}` : result.content;
    await navigator.clipboard.writeText(value);
  } catch { el.error.textContent = "Clipboard unavailable. Use Download, or select and copy the output."; }
}
el.copy.addEventListener("click", () => void copy(false));
el.dataUri.addEventListener("click", () => void copy(true));
el.download.addEventListener("click", () => {
  if (!result) return;
  const link = document.createElement("a");
  link.href = resultUrl;
  link.download = result.name.replace(/\.[^.]+$/, "") + result.extension;
  link.click();
});
window.addEventListener("pagehide", event => {
  if (event.persisted) return;
  request?.abort();
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  if (rasterUrl) URL.revokeObjectURL(rasterUrl);
});
sync();
