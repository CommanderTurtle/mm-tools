const signal = document.querySelector('#signal');
const title = document.querySelector('#state-title');
const detail = document.querySelector('#state-detail');
const variant = document.querySelector('#variant');
const loadButton = document.querySelector('#load');
const unloadButton = document.querySelector('#unload');

async function request(endpoint, body) {
  const response = await fetch(endpoint, { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

async function refresh() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const modelsReady = Object.values(data.models).every(Boolean);
    signal.className = data.loaded ? 'loaded' : modelsReady ? 'ready' : 'error';
    title.textContent = data.loaded ? `${data.variant} loaded` : modelsReady ? 'Models ready · currently unloaded' : 'Model tree incomplete';
    detail.textContent = `${data.device.toUpperCase()} · private local process`;
    if (data.loaded) variant.value = data.variant;
  } catch (error) {
    signal.className = 'error';
    title.textContent = 'Studio unavailable';
    detail.textContent = error.message;
  }
}

loadButton.addEventListener('click', async () => {
  loadButton.disabled = true;
  title.textContent = `Loading ${variant.value}…`;
  const body = new FormData();
  body.set('variant', variant.value);
  try { await request('/api/load', body); } catch (error) { title.textContent = error.message; }
  finally { loadButton.disabled = false; refresh(); }
});

unloadButton.addEventListener('click', async () => {
  unloadButton.disabled = true;
  try { await request('/api/unload'); } catch (error) { title.textContent = error.message; }
  finally { unloadButton.disabled = false; refresh(); }
});

document.querySelector('#prompt-audio').addEventListener('change', (event) => {
  document.querySelector('#audio-title').textContent = event.target.files[0]?.name || 'Choose clean prompt singing';
});

const form = document.querySelector('#generate-form');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('.render');
  const status = document.querySelector('#status');
  const result = document.querySelector('#result');
  button.disabled = true;
  result.hidden = true;
  status.textContent = 'Encoding the reference and rendering…';
  try {
    const body = new FormData(form);
    body.set('variant', variant.value);
    body.set('lyrics_only', form.elements.lyrics_only.checked ? 'true' : 'false');
    const data = await request('/api/generate', body);
    const player = document.querySelector('#player');
    const download = document.querySelector('#download');
    player.src = `${data.url}?v=${Date.now()}`;
    download.href = data.url;
    download.download = data.name;
    result.hidden = false;
    status.textContent = `Complete — ${data.duration.toFixed(2)} seconds.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    refresh();
  }
});

refresh();
setInterval(refresh, 5000);
