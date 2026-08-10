const $ = selector => document.querySelector(selector);
const source = $('#source');
const mask = $('#mask');
const sx = source.getContext('2d');
const mx = mask.getContext('2d', { willReadFrequently: true });

let file = null;
let tool = 'brush';
let drawing = false;
let lastPoint = null;
let resultBlob = null;
let resultUrl = null;
let downloadSuffix = 'clean';
let undo = [];
let modelState = null;

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
  file = selected;
  resultBlob = null;
  const image = new Image();
  const sourceUrl = URL.createObjectURL(selected);
  image.onload = () => {
    source.width = mask.width = image.naturalWidth;
    source.height = mask.height = image.naturalHeight;
    sx.drawImage(image, 0, 0);
    mx.clearRect(0, 0, mask.width, mask.height);
    undo = [];
    $('#viewport').className = 'viewport';
    $('#remove').disabled = false;
    $('#background').disabled = false;
    $('#download-mask').disabled = false;
    $('#download-comfy').disabled = false;
    $('#copy-comfy').disabled = false;
    $('#download').disabled = true;
    setView('edit');
    setBusy(`${image.naturalWidth} × ${image.naturalHeight}`);
    URL.revokeObjectURL(sourceUrl);
  };
  image.onerror = () => setBusy('Could not decode that image in the browser.');
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
  undo.push(mx.getImageData(0, 0, mask.width, mask.height));
  if (undo.length > 20) undo.shift();
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

mask.addEventListener('pointerdown', async event => {
  if (!file) return;
  if (tool === 'fuzzy') {
    snapshot();
    await fuzzy(point(event));
    return;
  }
  snapshot();
  drawing = true;
  mask.setPointerCapture(event.pointerId);
  lastPoint = point(event);
  paint(lastPoint);
});
mask.addEventListener('pointermove', event => {
  if (!drawing) return;
  const next = point(event);
  paint(next, lastPoint);
  lastPoint = next;
});
function endStroke() {
  drawing = false;
  lastPoint = null;
}
mask.addEventListener('pointerup', endStroke);
mask.addEventListener('pointercancel', endStroke);

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
  for (const engine of ['objectclear', 'birefnet']) {
    const info = engines[engine];
    const label = $(`#${engine}-state`);
    label.textContent = info.loaded ? 'Loaded' : info.available ? 'Ready on disk' : 'Files missing';
    label.className = info.loaded ? 'loaded' : info.available ? 'ready' : 'missing';
    document.querySelector(`.model-load[data-engine="${engine}"]`).disabled = info.loaded || !info.available;
    document.querySelector(`.model-unload[data-engine="${engine}"]`).disabled = !info.loaded;
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
  setBusy(`${engine} ${action === 'load' ? 'loaded' : 'unloaded'}`);
}

document.querySelectorAll('.model-load').forEach(button => button.addEventListener('click', async () => {
  try { await changeModel(button.dataset.engine, 'load'); } catch (error) { setBusy(error.message); await refreshModels(); }
}));
document.querySelectorAll('.model-unload').forEach(button => button.addEventListener('click', async () => {
  try { await changeModel(button.dataset.engine, 'unload'); } catch (error) { setBusy(error.message); await refreshModels(); }
}));
$('#unload-all').addEventListener('click', async () => {
  try { await changeModel('all', 'unload'); } catch (error) { setBusy(error.message); await refreshModels(); }
});

async function ensureLoaded(engine) {
  if (!modelState) await refreshModels();
  const info = modelState?.engines?.[engine];
  if (!info?.available) throw new Error(`${engine} model files are not installed. See the model path shown in the project documentation.`);
  if (!info.loaded) await changeModel(engine, 'load');
}

$('#remove').addEventListener('click', async () => {
  const button = $('#remove');
  button.disabled = true;
  try {
    const engine = $('#engine').value;
    if (engine === 'objectclear') await ensureLoaded('objectclear');
    setBusy(engine === 'objectclear' ? 'Reconstructing the selected region locally…' : 'Repairing selected pixels locally…');
    const body = new FormData();
    body.set('image', file);
    body.set('mask', await maskBlob(), 'mask.png');
    body.set('engine', engine);
    for (const id of ['method', 'radius', 'grow', 'feather', 'steps', 'guidance', 'seed']) body.set(id, $(`#${id}`).value);
    const response = await fetch('/api/remove', { method: 'POST', body });
    if (!response.ok) throw new Error(await errorMessage(response));
    showResult(await response.blob(), engine === 'objectclear' ? 'object-cleared' : 'clean');
    setBusy('Removal complete');
    await refreshModels();
  } catch (error) {
    setBusy(error.message);
  } finally {
    button.disabled = false;
  }
});

$('#background').addEventListener('click', async () => {
  const button = $('#background');
  button.disabled = true;
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
    button.disabled = false;
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
  const ai = event.target.value === 'objectclear';
  $('#ai-controls').classList.toggle('hidden', !ai);
  $('#opencv-controls').classList.toggle('hidden', ai);
});

refreshModels();
