const signal = document.querySelector('#signal');
const runtimeTitle = document.querySelector('#runtime-title');
const runtimeDetail = document.querySelector('#runtime-detail');

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const ready = Object.values(data.models).every(Boolean);
    signal.className = `signal ${ready ? 'ready' : 'error'}`;
    runtimeTitle.textContent = data.busy ? 'Generation in progress' : ready ? 'All four checkpoints ready' : 'Checkpoint set incomplete';
    runtimeDetail.textContent = data.busy ? 'One isolated model worker is active.' : 'The studio holds no model in memory while idle.';
  } catch (error) {
    signal.className = 'signal error';
    runtimeTitle.textContent = 'Studio unavailable';
    runtimeDetail.textContent = error.message;
  }
}

function renderResults(target, artifacts) {
  target.replaceChildren(...artifacts.map((artifact) => {
    const link = document.createElement('a');
    link.href = artifact.url;
    link.textContent = `Download ${artifact.name}`;
    link.download = artifact.name;
    return link;
  }));
}

async function submit(form, endpoint, status, results) {
  status.textContent = 'Model worker starting…';
  results.replaceChildren();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    const body = new FormData(form);
    for (const checkbox of form.querySelectorAll('input[type="checkbox"]')) {
      body.set(checkbox.name, checkbox.checked ? (checkbox.name === 'register_decay' ? '1' : 'true') : (checkbox.name === 'register_decay' ? '0' : 'false'));
    }
    const response = await fetch(endpoint, { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Generation failed');
    renderResults(results, data.artifacts);
    status.textContent = `Complete — ${data.artifacts.length} MIDI file${data.artifacts.length === 1 ? '' : 's'} ready.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    refreshStatus();
  }
}

const harmonyForm = document.querySelector('#harmony-form');
harmonyForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submit(harmonyForm, '/api/generate-harmony', document.querySelector('#harmony-status'), document.querySelector('#harmony-results'));
});

const orchestrateForm = document.querySelector('#orchestrate-form');
orchestrateForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submit(orchestrateForm, '/api/orchestrate', document.querySelector('#orchestrate-status'), document.querySelector('#orchestrate-results'));
});

document.querySelector('#midi').addEventListener('change', (event) => {
  document.querySelector('#file-title').textContent = event.target.files[0]?.name || 'Choose or drop a MIDI file';
});

refreshStatus();
setInterval(refreshStatus, 5000);
