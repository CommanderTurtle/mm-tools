const $ = (selector) => document.querySelector(selector);
const fileInput = $('#file');
const dropzone = $('#dropzone');
const preview = $('#preview');
const operation = $('#operation');
const statusNode = $('#status');
let audioFile = null;
let recorder = null;
let timer = null;
let elapsed = 0;
let results = {};
let activeResult = '';
let localLoaded = false;
let activeBase = '';
let uiConfig = { secondary_port: 8172, secondary_scheme: 'http' };

async function apiAt(base, path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return response;
}

const localApi = (path, options = {}) => apiAt('', path, options);
const api = (path, options = {}) => apiAt(activeBase, path, options);

function secondaryBase() {
  const bareHost = location.hostname.replace(/^\[|\]$/g, '');
  const host = bareHost.includes(':') ? `[${bareHost}]` : bareHost;
  return `${uiConfig.secondary_scheme}://${host}:${uiConfig.secondary_port}`;
}

function renderRuntime(data, source) {
  $('#runtime').textContent = `${source} · ${data.backend} · ${data.device}/${data.compute_type} · ${data.loaded ? 'model loaded' : 'model unloaded'}${data.model_present ? '' : ' · MODEL MISSING'}`;
}

async function health() {
  try {
    const local = await (await localApi('/api/health')).json();
    localLoaded = local.loaded;
    $('#model-toggle').textContent = localLoaded ? 'Unload local model' : 'Load GPU model';
    $('#model-toggle').disabled = !local.model_present;
    $('#secondary-toggle').textContent = activeBase ? 'Use local model' : 'Check Secondary Load';

    if (activeBase) {
      try {
        const secondary = await (await apiAt(activeBase, '/api/health')).json();
        renderRuntime(secondary, 'Secondary HTTP');
        return;
      } catch (error) {
        activeBase = '';
        $('#secondary-toggle').textContent = 'Check Secondary Load';
        statusNode.textContent = `Secondary service disconnected; using local UI. ${error.message}`;
      }
    }
    renderRuntime(local, 'UI-local');
  } catch { $('#runtime').textContent = 'UI runtime unavailable'; }
}

function setFile(file) {
  audioFile = file;
  $('#file-label').textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MiB`;
  preview.src = URL.createObjectURL(file);
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => fileInput.files[0] && setFile(fileInput.files[0]));
for (const event of ['dragenter', 'dragover']) dropzone.addEventListener(event, e => { e.preventDefault(); dropzone.classList.add('drag'); });
for (const event of ['dragleave', 'drop']) dropzone.addEventListener(event, e => { e.preventDefault(); dropzone.classList.remove('drag'); });
dropzone.addEventListener('drop', e => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

operation.addEventListener('change', () => {
  $('#transcript-wrap').classList.toggle('hidden', !['verbatimize', 'align'].includes(operation.value));
});

$('#record').addEventListener('click', async () => {
  if (recorder?.state === 'recording') { recorder.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia) {
    statusNode.textContent = 'Microphone capture needs localhost or HTTPS. File upload still works.';
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    statusNode.textContent = `Microphone unavailable: ${error.message}`;
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    setFile(new File([blob], `recording-${Date.now()}.webm`, { type: blob.type }));
    stream.getTracks().forEach(track => track.stop());
    clearInterval(timer); $('#record').textContent = '● Record microphone'; $('#record').classList.remove('recording');
  };
  recorder.start(250); elapsed = 0;
  $('#record').textContent = '■ Stop recording'; $('#record').classList.add('recording');
  timer = setInterval(() => { elapsed += 1; $('#record-time').textContent = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`; }, 1000);
});

function showResult(name) {
  activeResult = name;
  const result = results[name];
  $('#result-title').textContent = name.replace('_', ' ');
  $('#result-tabs').querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.name === name));
  const words = result.words || [];
  $('#transcript').innerHTML = words.length
    ? words.map(w => `<span class="word" title="${w.start?.toFixed(2)}–${w.end?.toFixed(2)} s">${escapeHtml(w.word)}</span>`).join(' ')
    : escapeHtml(result.text || '');
  $('#metrics').innerHTML = `<span>${result.duration ?? '—'} s audio</span><span>${result.processing_time ?? '—'} s inference</span><span>${words.length} timed words</span>`;
  $('#json').textContent = JSON.stringify(result, null, 2);
}

function escapeHtml(value) { const node = document.createElement('span'); node.textContent = value || ''; return node.innerHTML; }

$('#controls').addEventListener('submit', async event => {
  event.preventDefault();
  if (!audioFile) { statusNode.textContent = 'Choose or record audio first.'; return; }
  const button = $('#run'); button.disabled = true; statusNode.textContent = 'Decoding locally… first load may take a moment.';
  const body = new FormData(event.target); body.set('file', audioFile);
  body.set('word_timestamps', event.target.word_timestamps.checked ? 'true' : 'false');
  try {
    const response = await api('/api/transcribe', { method: 'POST', body });
    const data = await response.json();
    results = data.results;
    const tabs = $('#result-tabs'); tabs.textContent = '';
    Object.keys(results).forEach((name, index) => {
      const tab = document.createElement('button'); tab.type = 'button'; tab.dataset.name = name; tab.textContent = name.replace('_', ' ');
      tab.addEventListener('click', () => showResult(name)); tabs.append(tab);
      if (index === 0) activeResult = name;
    });
    $('#results').classList.remove('hidden'); showResult(activeResult); statusNode.textContent = 'Complete'; health();
  } catch (error) { statusNode.textContent = error.message; }
  finally { button.disabled = false; }
});

$('#model-toggle').addEventListener('click', async () => {
  const button = $('#model-toggle');
  button.disabled = true;
  activeBase = '';
  $('#secondary-toggle').textContent = 'Check Secondary Load';
  statusNode.textContent = localLoaded ? 'Unloading UI-local GPU model…' : 'Loading UI-local GPU model…';
  try {
    const data = await (await localApi(localLoaded ? '/api/unload' : '/api/load', { method: 'POST' })).json();
    statusNode.textContent = data.loaded ? 'UI-local GPU model loaded' : 'UI-local GPU model unloaded';
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    await health();
  }
});

$('#secondary-toggle').addEventListener('click', async () => {
  if (activeBase) {
    activeBase = '';
    statusNode.textContent = 'Using the UI-local model path.';
    await health();
    return;
  }
  const button = $('#secondary-toggle');
  button.disabled = true;
  statusNode.textContent = 'Checking the secondary HTTP model…';
  try {
    const candidate = secondaryBase();
    const state = await (await apiAt(candidate, '/api/health')).json();
    if (!state.loaded) throw new Error('Secondary service is running, but its model is unloaded.');
    activeBase = candidate;
    statusNode.textContent = 'Attached to the already-loaded secondary HTTP model.';
  } catch (error) {
    statusNode.textContent = `Secondary load unavailable: ${error.message}`;
  } finally {
    button.disabled = false;
    await health();
  }
});

$('#copy').addEventListener('click', () => navigator.clipboard.writeText(results[activeResult]?.text || ''));

async function initialize() {
  try { uiConfig = await (await localApi('/api/ui-config')).json(); } catch {}
  await health();
}

initialize();
