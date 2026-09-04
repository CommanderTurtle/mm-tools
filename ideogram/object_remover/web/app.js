const $ = selector => document.querySelector(selector);
const source = $('#source');
const mask = $('#mask');
const selection = $('#selection');
const lx = selection.getContext('2d');
const sx = source.getContext('2d');
const mx = mask.getContext('2d', { willReadFrequently: true });

let file = null;
let tool = 'brush';
let drawing = false;
let strokePointer = null;
let lastPoint = null;
let lassoPoints = [];
let resultBlob = null;
let resultUrl = null;
let downloadSuffix = 'clean';
let undo = [];
let modelState = null;
let operationBusy = false;

function lockEditing(value) {
  operationBusy = value;
  endStroke();
  document.querySelectorAll('aside input, aside select, aside textarea, aside button, header button').forEach(el => {
    if (value) { el.dataset.wasDisabled = String(el.disabled); el.disabled = true; }
    else { el.disabled = el.dataset.wasDisabled === 'true'; delete el.dataset.wasDisabled; }
  });
}

async function errorMessage(response) {
  const text = await response.text();
  if (!text) return `${response.status} ${response.statusText}`;
  try { return JSON.parse(text).detail || text; } catch { return text; }
}

function setBusy(message) {
  $('#status').textContent = message;
}

function showResult(blob, suffix) {
  resultBlob = blob;
  downloadSuffix = suffix;
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(blob);
  $('#result').src = resultUrl;
  $('#download').disabled = false;
  setView('result');
}

function setImage(selected) {
  if (operationBusy) return;
  file = selected;
  resultBlob = null;
  const image = new Image();
  const sourceUrl = URL.createObjectURL(selected);
  image.onload = () => {
    endStroke();
    source.width = mask.width = selection.width = image.naturalWidth;
    source.height = mask.height = selection.height = image.naturalHeight;
    sx.drawImage(image, 0, 0);
    mx.clearRect(0, 0, mask.width, mask.height);
    undo = [];
    $('#viewport').className = 'viewport';
    $('#remove').disabled = false;
    $('#background').disabled = false;
    $('#cutout').disabled = false;
    $('#caption').value = '';
    $('#download-mask').disabled = false;
    $('#download-comfy').disabled = false;
    $('#copy-comfy').disabled = false;
    $('#download').disabled = true;
    setView('edit');
    setBusy(`${image.naturalWidth} × ${image.naturalHeight}`);
    URL.revokeObjectURL(sourceUrl);
  };
  image.onerror = () => { URL.revokeObjectURL(sourceUrl); setBusy('Could not decode that image in the browser.'); };
  image.src = sourceUrl;
}

$('#file').addEventListener('change', event => event.target.files[0] && setImage(event.target.files[0]));
const drop = $('#dropzone');
for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, event => event.preventDefault());
drop.addEventListener('drop', event => {
  event.preventDefault();
  if (event.dataTransfer.files[0]) setImage(event.dataTransfer.files[0]);
});

document.querySelectorAll('.tool').forEach(button => button.addEventListener('click', () => {
  tool = button.dataset.tool;
  document.querySelectorAll('.tool').forEach(item => item.classList.toggle('active', item === button));
}));

$('#brush-size').addEventListener('input', event => $('#brush-value').textContent = `${event.target.value} px`);
$('#tolerance').addEventListener('input', event => $('#tolerance-value').textContent = event.target.value);

function point(event) {
  const rect = mask.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * mask.width / rect.width,
    y: (event.clientY - rect.top) * mask.height / rect.height,
  };
}

function snapshot() {
  // Full 8K masks are large: cap undo by bytes, not only by stroke count.
  const limit = Math.max(1, Math.min(20, Math.floor(256 * 1024 * 1024 / (mask.width * mask.height * 4))));
  while (undo.length >= limit) undo.shift();
  undo.push(mx.getImageData(0, 0, mask.width, mask.height));
}

function paint(p, previous = null) {
  const size = Number($('#brush-size').value) * mask.width / mask.getBoundingClientRect().width;
  mx.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over';
  mx.strokeStyle = '#ff3b30';
  mx.fillStyle = '#ff3b30';
  mx.lineWidth = size;
  mx.lineCap = 'round';
  mx.lineJoin = 'round';
  if (previous) {
    mx.beginPath();
    mx.moveTo(previous.x, previous.y);
    mx.lineTo(p.x, p.y);
    mx.stroke();
    return;
  }
  mx.beginPath();
  mx.arc(p.x, p.y, size / 2, 0, Math.PI * 2);
  mx.fill();
}

function lassoPath(context, points) {
  if (!points.length) return;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (const p of points.slice(1)) context.lineTo(p.x, p.y);
  context.closePath();
}

function previewLasso() {
  lx.clearRect(0, 0, selection.width, selection.height);
  const scale = mask.width / mask.getBoundingClientRect().width;
  lx.lineWidth = 2 * scale;
  lx.setLineDash([5 * scale, 4 * scale]);
  lx.strokeStyle = '#ffe9b7';
  lx.fillStyle = 'rgba(255, 59, 48, .14)';
  lassoPath(lx, lassoPoints);
  lx.fill(); lx.stroke();
}

mask.addEventListener('pointerdown', async event => {
  if (!file || operationBusy || drawing || event.button !== 0 || event.isPrimary === false) return;
  if (tool === 'fuzzy') {
    snapshot();
    await fuzzy(point(event));
    return;
  }
  snapshot();
  drawing = true;
  strokePointer = event.pointerId;
  mask.setPointerCapture(event.pointerId);
  lastPoint = point(event);
  if (tool === 'lasso') { lassoPoints = [lastPoint]; previewLasso(); }
  else paint(lastPoint);
});
mask.addEventListener('pointermove', event => {
  if (!drawing || event.pointerId !== strokePointer) return;
  const next = point(event);
  if (tool === 'lasso') { lassoPoints.push(next); previewLasso(); }
  else paint(next, lastPoint);
  lastPoint = next;
});
function endStroke(commit = false) {
  if (lassoPoints.length) {
    if (commit && lassoPoints.length >= 3) {
      mx.globalCompositeOperation = 'source-over';
      mx.fillStyle = '#ff3b30';
      lassoPath(mx, lassoPoints);
      mx.fill();
    } else {
      // Cancelled lasso did not modify the mask; discard its unused undo snapshot.
      undo.pop();
    }
    lassoPoints = [];
    lx.clearRect(0, 0, selection.width, selection.height);
  }
  drawing = false;
  lastPoint = null;
  const pointer = strokePointer;
  strokePointer = null;
  if (pointer !== null && mask.hasPointerCapture(pointer)) mask.releasePointerCapture(pointer);
}
mask.addEventListener('pointerup', event => {
  if (event.pointerId === strokePointer) endStroke(true);
});
for (const name of ['pointercancel', 'lostpointercapture']) mask.addEventListener(name, event => {
  if (event.pointerId === strokePointer) endStroke(false);
});

async function fuzzy(p) {
  setBusy('Selecting connected color…');
  try {
    const body = new FormData();
    body.set('image', file);
    body.set('x', Math.round(p.x));
    body.set('y', Math.round(p.y));
    body.set('tolerance', $('#tolerance').value);
    const response = await fetch('/api/fuzzy', { method: 'POST', body });
    if (!response.ok) throw new Error(await errorMessage(response));
    const image = new Image();
    const url = URL.createObjectURL(await response.blob());
    image.onload = () => {
      const temp = document.createElement('canvas');
      temp.width = mask.width;
      temp.height = mask.height;
      const context = temp.getContext('2d', { willReadFrequently: true });
      context.drawImage(image, 0, 0);
      const data = context.getImageData(0, 0, temp.width, temp.height).data;
      const current = mx.getImageData(0, 0, mask.width, mask.height);
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] > 0) {
          current.data[i] = 255;
          current.data[i + 1] = 59;
          current.data[i + 2] = 48;
          current.data[i + 3] = 255;
        }
      }
      mx.putImageData(current, 0, 0);
      URL.revokeObjectURL(url);
      setBusy('Fuzzy selection added');
    };
    image.src = url;
  } catch (error) {
    setBusy(error.message);
  }
}

$('#undo').addEventListener('click', () => {
  const prior = undo.pop();
  if (prior) mx.putImageData(prior, 0, 0);
});
$('#clear').addEventListener('click', () => {
  snapshot();
  mx.clearRect(0, 0, mask.width, mask.height);
});

function maskBlob() {
  const out = document.createElement('canvas');
  out.width = mask.width;
  out.height = mask.height;
  const context = out.getContext('2d');
  const data = mx.getImageData(0, 0, mask.width, mask.height);
  for (let i = 0; i < data.data.length; i += 4) {
    const alpha = data.data[i + 3];
    data.data[i] = data.data[i + 1] = data.data[i + 2] = alpha;
    data.data[i + 3] = 255;
  }
  context.putImageData(data, 0, 0);
  return new Promise((resolve, reject) => out.toBlob(
    blob => blob ? resolve(blob) : reject(new Error('Could not encode the mask.')),
    'image/png',
  ));
}

function comfyBlob() {
  const out = document.createElement('canvas');
  out.width = source.width;
  out.height = source.height;
  const context = out.getContext('2d');
  const pixels = sx.getImageData(0, 0, source.width, source.height);
  const selection = mx.getImageData(0, 0, mask.width, mask.height).data;
  // ComfyUI's Load Image node returns MASK = 1 - alpha. Encoding the painted
  // selection as transparent therefore produces the expected white MASK while
  // preserving the source RGB in the same portable PNG.
  for (let i = 0; i < pixels.data.length; i += 4) pixels.data[i + 3] = 255 - selection[i + 3];
  context.putImageData(pixels, 0, 0);
  return new Promise((resolve, reject) => out.toBlob(
    blob => blob ? resolve(blob) : reject(new Error('Could not encode the Comfy PNG.')),
    'image/png',
  ));
}

function downloadBlob(blob, suffix) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${file.name.replace(/\.[^.]+$/, '')}-${suffix}.png`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('#download-mask').addEventListener('click', async () => {
  try {
    downloadBlob(await maskBlob(), 'mask');
    setBusy('Full-resolution black-and-white mask downloaded');
  } catch (error) {
    setBusy(error.message);
  }
});

$('#download-comfy').addEventListener('click', async () => {
  try {
    downloadBlob(await comfyBlob(), 'comfy-mask');
    setBusy('ComfyUI image + embedded mask downloaded');
  } catch (error) {
    setBusy(error.message);
  }
});

$('#copy-comfy').addEventListener('click', async () => {
  const blob = await comfyBlob().catch(error => {
    setBusy(error.message);
    return null;
  });
  if (!blob) return;
  try {
    if (!navigator.clipboard?.write || !window.ClipboardItem) throw new Error('binary clipboard access is unavailable');
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    setBusy('Image + mask copied — paste onto the ComfyUI canvas');
  } catch (error) {
    downloadBlob(blob, 'comfy-mask');
    setBusy(`Clipboard denied (${error.message}). Comfy PNG downloaded instead.`);
  }
});

function renderModelState(state) {
  modelState = state;
  const engines = state.engines;
  for (const engine of ['objectclear', 'birefnet', 'ideogram']) {
    const info = engines[engine];
    const label = $(`#${engine}-state`);
    label.textContent = info.loaded ? (engine === 'ideogram' ? 'Private engine ready' : 'Loaded') : info.available ? 'Ready on disk' : 'Files missing';
    label.title = info.missing?.join('\n') || info.model_path;
    label.className = info.loaded ? 'loaded' : info.available ? 'ready' : 'missing';
    document.querySelector(`.model-load[data-engine="${engine}"]`).disabled = operationBusy || info.loaded || !info.available;
    document.querySelector(`.model-unload[data-engine="${engine}"]`).disabled = operationBusy || !info.loaded;
  }
  const vram = state.vram_free_gib == null ? '' : ` · ${state.vram_free_gib}/${state.vram_total_gib} GiB free`;
  $('#device-state').textContent = `Device: ${state.device}${vram}`;
}

async function refreshModels() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error(await errorMessage(response));
    renderModelState(await response.json());
  } catch (error) {
    $('#device-state').textContent = `Models unavailable: ${error.message}`;
  }
}

async function changeModel(engine, action) {
  setBusy(`${action === 'load' ? 'Loading' : 'Unloading'} ${engine}…`);
  const response = await fetch(`/api/models/${engine}/${action}`, { method: 'POST' });
  if (!response.ok) throw new Error(await errorMessage(response));
  renderModelState(await response.json());
  setBusy(`${engine} ${action === 'load' ? (engine === 'ideogram' ? 'engine ready; weights load when editing' : 'loaded') : 'unloaded'}`);
}

async function manualModelChange(engine, action) {
  if (operationBusy) return;
  lockEditing(true);
  try { await changeModel(engine, action); } catch (error) { setBusy(error.message); }
  finally { lockEditing(false); await refreshModels(); }
}

document.querySelectorAll('.model-load').forEach(button => button.addEventListener('click', async () => {
  await manualModelChange(button.dataset.engine, 'load');
}));
document.querySelectorAll('.model-unload').forEach(button => button.addEventListener('click', async () => {
  await manualModelChange(button.dataset.engine, 'unload');
}));
$('#unload-all').addEventListener('click', async () => {
  await manualModelChange('all', 'unload');
});

async function ensureLoaded(engine) {
  if (!modelState) await refreshModels();
  const info = modelState?.engines?.[engine];
  if (!info?.available) throw new Error(`${engine} model files are not installed. See the model path shown in the project documentation.`);
  if (!info.loaded) await changeModel(engine, 'load');
}

$('#remove').addEventListener('click', async () => {
  if (operationBusy || !file) return;
  lockEditing(true);
  try {
    const engine = $('#engine').value;
    if (engine === 'objectclear') await ensureLoaded('objectclear');
    setBusy(engine === 'objectclear' ? 'Reconstructing the selected region locally…' : 'Repairing selected pixels locally…');
    const body = new FormData();
    body.set('image', file);
    body.set('mask', await maskBlob(), 'mask.png');
    body.set('engine', engine);
    for (const id of ['method', 'radius', 'grow', 'feather', 'steps', 'guidance', 'seed']) body.set(id, $(`#${id}`).value);
    if (engine === 'ideogram') {
      ideogramFields(body);
      body.set('caption', $('#caption').value);
      body.set('strength', $('#strength').value);
      body.set('guidance', $('#ideogram-guidance').value);
      setBusy($('#caption').value.trim() ? 'Editing selected crop with local Ideogram…' : 'Drafting caption, then editing with local Ideogram…');
    }
    const response = await fetch('/api/remove', { method: 'POST', body });
    if (!response.ok) throw new Error(await errorMessage(response));
    showResult(await response.blob(), engine === 'ideogram' ? 'ideogram-edit' : engine === 'objectclear' ? 'object-cleared' : 'clean');
    setBusy('Removal complete');
    await refreshModels();
  } catch (error) {
    setBusy(error.message);
  } finally {
    lockEditing(false);
    $('#download').disabled = !resultBlob;
    await refreshModels();
  }
});

$('#background').addEventListener('click', async () => {
  if (operationBusy || !file) return;
  lockEditing(true);
  try {
    await ensureLoaded('birefnet');
    setBusy('Separating foreground and reconstructing alpha…');
    const body = new FormData();
    body.set('image', file);
    const response = await fetch('/api/background', { method: 'POST', body });
    if (!response.ok) throw new Error(await errorMessage(response));
    showResult(await response.blob(), 'background-removed');
    setBusy('Transparent background ready');
    await refreshModels();
  } catch (error) {
    setBusy(error.message);
  } finally {
    lockEditing(false);
    $('#download').disabled = !resultBlob;
    await refreshModels();
  }
});

$('#download').addEventListener('click', () => {
  if (!resultBlob) return;
  downloadBlob(resultBlob, downloadSuffix);
});

function setView(view) {
  $('#viewport').classList.toggle('result-view', view === 'result');
  document.querySelectorAll('.tab').forEach(button => button.classList.toggle('active', button.dataset.view === view));
}

document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
$('#engine').addEventListener('change', event => {
  const ai = event.target.value !== 'opencv';
  $('#ai-controls').classList.toggle('hidden', !ai);
  $('#opencv-controls').classList.toggle('hidden', ai);
  $('#ideogram-controls').classList.toggle('hidden', event.target.value !== 'ideogram');
  $('#guidance').closest('label').classList.toggle('hidden', event.target.value === 'ideogram');
  $('#remove').textContent = event.target.value === 'ideogram' ? 'Apply masked edit' : 'Remove selection';
});

function ideogramFields(body) {
  for (const id of ['instruction', 'resolution', 'padding', 'mask_feather']) body.set(id, $(`#${id}`).value);
  body.set('invert', $('#invert').checked);
}

$('#caption-generate').addEventListener('click', async () => {
  if (operationBusy || !file) return;
  lockEditing(true);
  try {
    const body = new FormData();
    body.set('image', file); body.set('mask', await maskBlob(), 'mask.png'); ideogramFields(body);
    setBusy('Drafting an edit caption with the local vision model…');
    const response = await fetch('/api/ideogram/caption', {method: 'POST', body});
    if (!response.ok) throw new Error(await errorMessage(response));
    $('#caption').value = JSON.stringify(JSON.parse((await response.json()).caption), null, 2);
    $('#caption').closest('details').open = true;
    setBusy('Caption ready to review. No image was generated.');
  } catch (error) { setBusy(error.message); }
  finally { lockEditing(false); await refreshModels(); }
});

$('#cutout').addEventListener('click', async () => {
  if (operationBusy || !file) return;
  lockEditing(true);
  try {
    const body = new FormData();
    body.set('image', file); body.set('mask', await maskBlob(), 'mask.png');
    setBusy('Keeping the selected foreground at original resolution…');
    const response = await fetch('/api/cutout', {method: 'POST', body});
    if (!response.ok) throw new Error(await errorMessage(response));
    showResult(await response.blob(), 'selected-cutout');
    setBusy('Transparent cutout ready');
  } catch (error) { setBusy(error.message); }
  finally { lockEditing(false); $('#download').disabled = !resultBlob; }
});

refreshModels();
