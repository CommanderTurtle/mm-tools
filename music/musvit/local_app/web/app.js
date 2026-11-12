const $ = (selector) => document.querySelector(selector);
const statusNode = $('#status');
const fileInput = $('#score-file');
let loaded = false;

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(message);
  }
  return response;
}

async function refreshHealth() {
  try {
    const state = await (await request('/api/health')).json();
    loaded = state.loaded;
    $('#signal').classList.toggle('active', loaded);
    $('#runtime-title').textContent = loaded ? 'MuSViT model loaded' : 'MuSViT model unloaded';
    $('#runtime-detail').textContent = `${state.device} · ${state.dtype}${state.model_present ? '' : ' · checkpoint missing'}`;
    $('#model-toggle').textContent = loaded ? 'Unload model' : 'Load model';
    $('#model-toggle').disabled = !state.model_present;
  } catch (error) {
    $('#runtime-title').textContent = 'Runtime unavailable';
    $('#runtime-detail').textContent = error.message;
  }
}

function setFile(file) {
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  $('#file-title').textContent = file.name;
  $('#file-detail').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB · ready to process`;
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
const dropZone = $('#drop-zone');
for (const name of ['dragenter', 'dragover']) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  });
}
for (const name of ['dragleave', 'drop']) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
  });
}
dropZone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));

$('#model-toggle').addEventListener('click', async () => {
  const button = $('#model-toggle');
  button.disabled = true;
  statusNode.textContent = loaded ? 'Unloading model…' : 'Loading model…';
  try {
    await request(loaded ? '/api/unload' : '/api/load', { method: 'POST' });
    statusNode.textContent = loaded ? 'Model unloaded.' : 'Model ready.';
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    await refreshHealth();
  }
});

$('#convert-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) return;
  const button = $('#convert');
  const body = new FormData(event.currentTarget);
  if (!body.get('page')) body.delete('page');
  if (!body.get('max_tokens')) body.delete('max_tokens');
  body.set('write_svg', event.currentTarget.write_svg.checked ? 'true' : 'false');
  button.disabled = true;
  statusNode.textContent = 'Reading notation. Complex pages can take a while…';
  try {
    const response = await request('/api/convert', { method: 'POST', body });
    const blob = await response.blob();
    const disposition = response.headers.get('content-disposition') || '';
    const name = disposition.match(/filename="?([^";]+)"?/i)?.[1] || 'musvit-score.zip';
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    URL.revokeObjectURL(url);
    statusNode.textContent = 'Complete. Your local score bundle is downloading.';
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    button.disabled = false;
    await refreshHealth();
  }
});

refreshHealth();
