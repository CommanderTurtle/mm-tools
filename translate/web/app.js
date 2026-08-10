const $ = selector => document.querySelector(selector);
let useExternal = false;
let localLoaded = false;
let languages = {};

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(message);
  }
  return response;
}

function setStatus(message) { $('#status').textContent = message; }

function renderRuntime(state, label) {
  $('#runtime').textContent = `${label} · ${state.runtime} · ${state.loaded ? 'loaded' : 'unloaded'} · cloud ${state.cloud ? 'on' : 'off'}`;
}

async function health() {
  try {
    const state = await (await request('/api/health')).json();
    localLoaded = state.loaded;
    $('#model-toggle').textContent = localLoaded ? 'Unload local model' : 'Load local model';
    $('#secondary-toggle').textContent = useExternal ? 'Use UI-local model' : 'Check External Load';
    if (useExternal) {
      try {
        const external = await (await request('/api/external-health')).json();
        renderRuntime(external, 'External HTTP');
        return;
      } catch (error) {
        setStatus(`External service disconnected; no fallback was run: ${error.message}`);
        $('#runtime').textContent = `External HTTP · unavailable · no fallback`;
        return;
      }
    }
    renderRuntime(state, 'UI-local');
  } catch (error) {
    $('#runtime').textContent = `Runtime unavailable · ${error.message}`;
  }
}

function fillLanguages() {
  const source = $('#source-language');
  const target = $('#target-language');
  const entries = Object.entries(languages).sort((a, b) => a[1].localeCompare(b[1]));
  entries.forEach(([code, name]) => {
    source.add(new Option(`${name} (${code})`, code));
    target.add(new Option(`${name} (${code})`, name));
  });
  target.value = 'English';
}

async function initialize() {
  try {
    languages = (await (await request('/api/languages')).json()).languages;
    fillLanguages();
  } catch (error) { setStatus(`Language list unavailable: ${error.message}`); }
  await health();
}

$('#source').addEventListener('input', () => { $('#source-count').textContent = $('#source').value.length; });
$('#paste').addEventListener('click', async () => {
  try { $('#source').value = await navigator.clipboard.readText(); $('#source').dispatchEvent(new Event('input')); }
  catch (error) { setStatus(`Clipboard unavailable: ${error.message}`); }
});
$('#copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#output').value);
  setStatus('Translation copied');
});

$('#swap').addEventListener('click', () => {
  const output = $('#output').value;
  if (!output) return;
  const targetName = $('#target-language').value;
  const targetCode = Object.keys(languages).find(code => languages[code] === targetName) || 'auto';
  const previousSource = $('#source-language').value;
  $('#source').value = output;
  $('#source').dispatchEvent(new Event('input'));
  $('#source-language').value = targetCode;
  if (previousSource !== 'auto' && languages[previousSource]) $('#target-language').value = languages[previousSource];
  $('#output').value = '';
  $('#meta').textContent = 'Directions swapped';
});

$('#translate').addEventListener('click', async () => {
  const text = $('#source').value.trim();
  if (!text) { setStatus('Enter text to translate.'); return; }
  const button = $('#translate');
  button.disabled = true;
  setStatus(useExternal ? 'Translating with the explicitly attached service…' : 'Translating with the UI-local model…');
  try {
    const payload = await (await request('/api/translate', {
      method: 'POST',
      body: JSON.stringify({
        text,
        source_language: $('#source-language').value,
        target_language: $('#target-language').value,
        max_tokens: Number($('#max-tokens').value),
        backend: useExternal ? 'external' : 'local',
      }),
    })).json();
    $('#output').value = payload.translation;
    const source = languages[payload.source_language] || payload.source_language || 'automatic';
    const confidence = payload.source_confidence == null ? '' : ` · ${(Number(payload.source_confidence) * 100).toFixed(1)}% confidence`;
    $('#meta').textContent = `${payload.backend === 'external' ? 'External' : 'UI-local'} · source ${source}${confidence}`;
    setStatus('Complete');
  } catch (error) { setStatus(error.message); }
  finally { button.disabled = false; await health(); }
});

$('#model-toggle').addEventListener('click', async () => {
  useExternal = false;
  setStatus(localLoaded ? 'Unloading UI-local model…' : 'Loading UI-local model…');
  try { await request(localLoaded ? '/api/unload' : '/api/load', { method: 'POST' }); }
  catch (error) { setStatus(error.message); }
  await health();
});

$('#secondary-toggle').addEventListener('click', async () => {
  if (useExternal) {
    useExternal = false;
    setStatus('Using the UI-local model path.');
    await health();
    return;
  }
  setStatus('Checking the external HTTP model…');
  try {
    const state = await (await request('/api/external-health')).json();
    if (!state.loaded) throw new Error('External service is running, but its model is unloaded.');
    useExternal = true;
    setStatus('Attached to the already-loaded external model.');
  } catch (error) { setStatus(`External load unavailable: ${error.message}`); }
  await health();
});

document.addEventListener('keydown', event => {
  if (event.ctrlKey && event.key === 'Enter') { event.preventDefault(); $('#translate').click(); }
});

initialize();
