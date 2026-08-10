const $ = selector => document.querySelector(selector);
let sourceFile = null;
let job = null;
let timer = null;
let cursor = 0;
let config = null;
let artifacts = [];
let currentView = 'original';
let editorDoc = null;
let selectedLayerId = null;
let editorImages = new Map();
let undoStack = [];
let redoStack = [];
let interaction = null;
let modelTimer = null;

async function loadConfig() {
  config = await fetch('/api/config').then(r => r.json());
  $('#gpu-note').textContent = config.gpu_note || '';
  const issues = [];
  if (!config.venv_present) issues.push('runtime not installed');
  if (!config.model?.model_present) issues.push('Diffusers checkpoint missing');
  $('#readiness').textContent = issues.length ? issues.join(' · ') : 'Direct local Diffusers lane configured';
  renderModel(config.model);
  clearInterval(modelTimer);
  modelTimer = setInterval(pollModel, 1200);
  updateRun();
}

function updateRun() {
  $('#run').disabled = !sourceFile || !config?.venv_present || !config?.model?.ready || ['queued','running','stopping'].includes(job?.status);
}

function renderModel(model) {
  config.model = model;
  const state = model?.state || 'unavailable';
  $('#model-state').textContent = state[0].toUpperCase() + state.slice(1);
  $('#model-path').textContent = model?.model || '';
  $('#model-load').disabled = ['loading','ready','running','unloading'].includes(state) || !model?.model_present;
  $('#model-unload').disabled = !['ready','error'].includes(state);
  if (model?.error) setStatus('failed', model.error);
  updateRun();
}

async function pollModel() {
  const response = await fetch('/api/model');
  if (response.ok) renderModel(await response.json());
}

async function modelAction(action) {
  const response = await fetch(`/api/model/${action}`, {method:'POST'});
  const data = await response.json();
  if (!response.ok) { setStatus('failed', data.detail || `Could not ${action} model`); return; }
  renderModel(data);
  setStatus(data.state, action === 'load' ? 'Loading the local Diffusers checkpoint…' : 'Releasing model memory…');
}
$('#model-load').addEventListener('click', () => modelAction('load'));
$('#model-unload').addEventListener('click', () => modelAction('unload'));

function choose(file) {
  sourceFile = file;
  editorDoc = null; selectedLayerId = null; editorImages.clear(); undoStack = []; redoStack = [];
  $('#source-info').textContent = `${file.name} · ${(file.size / 1048576).toFixed(2)} MiB`;
  $('#canvas-image').src = URL.createObjectURL(file);
  $('#canvas').classList.remove('empty');
  currentView = 'original'; selectViewButton(); updateRun();
}

const drop = $('#dropzone');
$('#image').addEventListener('change', e => e.target.files[0] && choose(e.target.files[0]));
for (const event of ['dragenter','dragover']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('drag'); });
for (const event of ['dragleave','drop']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('drag'); });
drop.addEventListener('drop', e => e.dataTransfer.files[0] && choose(e.dataTransfer.files[0]));

$('#run').addEventListener('click', async () => {
  const body = new FormData();
  body.set('image', sourceFile);
  body.set('layers', $('#layers').value);
  body.set('steps', $('#steps').value);
  body.set('resolution', $('#resolution').value);
  body.set('cfg', $('#cfg').value);
  body.set('seed', $('#seed').value);
  $('#run').disabled = true; setStatus('queued', 'Preparing direct Diffusers run…');
  const response = await fetch('/api/jobs', {method:'POST', body});
  const data = await response.json();
  if (!response.ok) { setStatus('failed', data.detail || 'Could not start'); updateRun(); return; }
  job = data; cursor = 0; $('#console').textContent = ''; $('#stop').disabled = false;
  clearInterval(timer); timer = setInterval(poll, 900); poll();
});

$('#stop').addEventListener('click', async () => {
  if (!job) return;
  await fetch(`/api/jobs/${job.id}/stop`, {method:'POST'}); poll();
});

async function poll() {
  if (!job) return;
  const response = await fetch(`/api/jobs/${job.id}?after=${cursor}`);
  if (!response.ok) return;
  job = await response.json();
  cursor = job.log_cursor;
  if (job.logs.length) {
    const output = $('#console'); output.textContent += `${output.textContent ? '\n' : ''}${job.logs.join('\n')}`;
    output.scrollTop = output.scrollHeight;
  }
  artifacts = job.artifacts || [];
  renderArtifacts();
  setStatus(job.status, job.status === 'running' ? 'Qwen-Image-Layered is rebuilding editable structure…' : `Process ${job.returncode ?? ''}`);
  if (!['queued','running'].includes(job.status)) {
    clearInterval(timer); $('#stop').disabled = true; updateRun(); await renderLayers();
    const preferred = artifactFor('reconstructed.png') || artifactFor('reconstructed_bordered.png');
    if (preferred) showArtifact(preferred);
  }
}

function setStatus(state, note) {
  const timeline = document.querySelector('.timeline'); timeline.className = `timeline ${state}`;
  $('#job-status').textContent = state[0].toUpperCase() + state.slice(1);
  $('#job-note').textContent = note;
}

function artifactFor(name) { return artifacts.find(item => item.name === name); }
function artifactUrl(item) { return `/api/jobs/${job.id}/artifact?path=${encodeURIComponent(item.path)}`; }
function showArtifact(item) { $('#editor-canvas').classList.add('hidden'); $('#canvas-image').classList.remove('hidden'); $('#canvas-image').src = artifactUrl(item); $('#canvas').classList.remove('empty'); }

function renderArtifacts() {
  const panel = $('#artifacts-panel');
  panel.innerHTML = artifacts.length ? '' : '<div class="placeholder">No artifacts yet.</div>';
  artifacts.forEach(item => {
    const row = document.createElement('div'); row.className = 'artifact';
    row.innerHTML = `<span>${item.name}</span><small>${item.kind} · ${(item.bytes/1024).toFixed(0)} KiB</small>`;
    row.addEventListener('click', () => item.kind === 'image' ? showArtifact(item) : window.open(artifactUrl(item), '_blank'));
    panel.append(row);
  });
}

async function renderLayers() {
  if (!job) return;
  const data = await fetch(`/api/jobs/${job.id}/layers`).then(r => r.json());
  const panel = $('#layer-list'); panel.innerHTML = '';
  if (!data.layers?.length) { panel.innerHTML = '<div class="placeholder">No parse tree was produced.</div>'; return; }
  editorDoc = data; selectedLayerId = data.layers.at(-1)?.id || null;
  editorImages.clear(); undoStack = []; redoStack = []; updateHistoryButtons();
  await Promise.all(data.layers.filter(layer => layer.asset_url).map(layer => new Promise(resolve => {
    const image = new Image(); image.onload = image.onerror = resolve; image.src = layer.asset_url; editorImages.set(layer.id, image);
  })));
  renderLayerPanel(); renderProperties(); $('#save-editor').disabled = false;
}

document.querySelectorAll('.view-switch button').forEach(button => button.addEventListener('click', () => {
  currentView = button.dataset.view; selectViewButton();
  if (!job && currentView !== 'original') return;
  if (currentView === 'edit') { showEditor(); return; }
  $('#editor-canvas').classList.add('hidden'); $('#canvas-image').classList.remove('hidden');
  if (currentView === 'original') $('#canvas-image').src = sourceFile ? URL.createObjectURL(sourceFile) : '';
  else {
    const names = {reconstructed:'reconstructed.png', bordered:'reconstructed_bordered.png'};
    const found = artifactFor(names[currentView]);
    if (found) showArtifact(found);
  }
}));
function selectViewButton(){ document.querySelectorAll('.view-switch button').forEach(b=>b.classList.toggle('active',b.dataset.view===currentView)); }

document.querySelectorAll('.panel-tabs button').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.panel-tabs button').forEach(b => b.classList.toggle('active', b===button));
  $('#layers-panel').classList.toggle('hidden', button.dataset.panel !== 'layers');
  $('#artifacts-panel').classList.toggle('hidden', button.dataset.panel !== 'artifacts');
}));
$('#clear-log').addEventListener('click', () => $('#console').textContent = '');
function escapeHtml(value){const node=document.createElement('span');node.textContent=value;return node.innerHTML;}
function cloneEditor(){ return JSON.parse(JSON.stringify(editorDoc)); }
function remember(){ if (!editorDoc) return; undoStack.push(cloneEditor()); if (undoStack.length > 80) undoStack.shift(); redoStack = []; updateHistoryButtons(); }
function updateHistoryButtons(){ $('#undo').disabled = !undoStack.length; $('#redo').disabled = !redoStack.length; }
function selectedLayer(){ return editorDoc?.layers.find(layer => layer.id === selectedLayerId); }
function normalizeZ(){ [...editorDoc.layers].sort((a,b)=>a.z-b.z).forEach((layer,index)=>layer.z=index); }

function renderLayerPanel(){
  const panel = $('#layer-list'); panel.innerHTML = '';
  if (!editorDoc?.layers?.length) { panel.innerHTML = '<div class="placeholder">No recovered layers.</div>'; return; }
  [...editorDoc.layers].sort((a,b)=>b.z-a.z).forEach(layer => {
    const row = document.createElement('div'); row.className = `layer${layer.id===selectedLayerId?' selected':''}`;
    row.innerHTML = `<button class="layer-eye${layer.visible?'':' off'}" title="Toggle visibility">${layer.visible?'●':'○'}</button><span><strong>${escapeHtml(layer.name)}</strong><br><small>${escapeHtml(layer.type)} · z${layer.z}</small></span><span class="layer-actions"><button data-move="up" title="Bring forward">↑</button><button data-move="down" title="Send backward">↓</button></span>`;
    row.addEventListener('click', event => {
      if (event.target.closest('.layer-eye')) { remember(); layer.visible=!layer.visible; renderEditor(); renderLayerPanel(); return; }
      const move = event.target.dataset.move;
      if (move) { remember(); layer.z += move==='up' ? 1.5 : -1.5; normalizeZ(); renderEditor(); renderLayerPanel(); return; }
      selectedLayerId=layer.id; renderLayerPanel(); renderProperties(); renderEditor();
    });
    panel.append(row);
  });
}

function renderProperties(){
  const box=$('#properties'), layer=selectedLayer();
  if (!layer) { box.className='hidden'; box.innerHTML=''; return; }
  box.className='properties';
  const text = layer.type === 'text' ? `<label>Text<textarea data-text="content">${escapeHtml(layer.text.content||'')}</textarea></label><div class="grid"><label>Font size<input type="number" min="1" data-text="font_size" value="${layer.text.font_size}"></label><label>Color<input type="color" data-text="color" value="${/^#[0-9a-f]{6}$/i.test(layer.text.color)?layer.text.color:'#ffffff'}"></label></div>` : '';
  box.innerHTML=`<h3>${escapeHtml(layer.name)}</h3><label>Name<input data-field="name" value="${escapeHtml(layer.name)}"></label><div class="grid"><label>X<input type="number" data-field="x" value="${Math.round(layer.x)}"></label><label>Y<input type="number" data-field="y" value="${Math.round(layer.y)}"></label><label>Width<input type="number" min="1" data-field="width" value="${Math.round(layer.width)}"></label><label>Height<input type="number" min="1" data-field="height" value="${Math.round(layer.height)}"></label><label>Rotation<input type="number" data-field="rotation" value="${layer.rotation}"></label><label>Opacity<input type="number" min="0" max="1" step="0.05" data-field="opacity" value="${layer.opacity}"></label></div>${text}<div class="property-actions"><button id="duplicate-layer" class="quiet">Duplicate</button><button id="delete-layer" class="danger">Delete</button></div>`;
  box.querySelectorAll('[data-field],[data-text]').forEach(input=>input.addEventListener('change',()=>{
    remember(); const numeric=['x','y','width','height','rotation','opacity'].includes(input.dataset.field);
    if (input.dataset.field) layer[input.dataset.field]=numeric?Number(input.value):input.value;
    else layer.text[input.dataset.text]=input.dataset.text==='font_size'?Number(input.value):input.value;
    renderEditor(); renderLayerPanel(); renderProperties();
  }));
  $('#duplicate-layer').onclick=()=>{ remember(); const copy=structuredClone(layer); copy.id=`${layer.id}-copy-${Date.now().toString(36)}`; copy.name=`${layer.name} copy`; copy.x+=12; copy.y+=12; copy.z=Math.max(...editorDoc.layers.map(x=>x.z))+1; editorImages.set(copy.id,editorImages.get(layer.id)); editorDoc.layers.push(copy); selectedLayerId=copy.id; renderLayerPanel(); renderProperties(); renderEditor(); };
  $('#delete-layer').onclick=()=>{ remember(); editorDoc.layers=editorDoc.layers.filter(x=>x.id!==layer.id); selectedLayerId=editorDoc.layers.at(-1)?.id||null; normalizeZ(); renderLayerPanel(); renderProperties(); renderEditor(); };
}

function showEditor(){
  if (!editorDoc) return;
  $('#canvas-image').classList.add('hidden'); $('#editor-canvas').classList.remove('hidden'); $('#canvas').classList.remove('empty'); renderEditor();
}

function renderEditor(){
  if (!editorDoc) return;
  const canvas=$('#editor-canvas'), ctx=canvas.getContext('2d');
  canvas.width=editorDoc.canvas.width; canvas.height=editorDoc.canvas.height; ctx.clearRect(0,0,canvas.width,canvas.height);
  [...editorDoc.layers].sort((a,b)=>a.z-b.z).forEach(layer=>{
    if (!layer.visible) return; ctx.save(); ctx.globalAlpha=layer.opacity;
    ctx.translate(layer.x+layer.width/2,layer.y+layer.height/2); ctx.rotate((layer.rotation||0)*Math.PI/180);
    if (layer.type==='text') {
      const size=layer.text.font_size||16; ctx.font=`${layer.text.italic?'italic ':''}${layer.text.bold?'bold ':''}${size}px ${layer.text.font_family||'sans-serif'}`; ctx.fillStyle=layer.text.color||'#fff'; ctx.textBaseline='top';
      String(layer.text.content||'').split('\n').forEach((line,index)=>ctx.fillText(line,-layer.width/2,-layer.height/2+index*size*1.2,layer.width));
    } else { const image=editorImages.get(layer.id); if (image?.complete && image.naturalWidth) ctx.drawImage(image,-layer.width/2,-layer.height/2,layer.width,layer.height); }
    ctx.restore();
  });
  const layer=selectedLayer(); if (!layer || !layer.visible) return;
  ctx.save(); ctx.strokeStyle='#8b78ff'; ctx.lineWidth=Math.max(1,canvas.width/900); ctx.setLineDash([8,5]); ctx.strokeRect(layer.x,layer.y,layer.width,layer.height); ctx.setLineDash([]); ctx.fillStyle='#8b78ff'; const handle=Math.max(10,canvas.width/90); ctx.fillRect(layer.x+layer.width-handle/2,layer.y+layer.height-handle/2,handle,handle); ctx.restore();
}

function canvasPoint(event){ const canvas=$('#editor-canvas'), rect=canvas.getBoundingClientRect(); return {x:(event.clientX-rect.left)*canvas.width/rect.width,y:(event.clientY-rect.top)*canvas.height/rect.height}; }
$('#editor-canvas').addEventListener('pointerdown',event=>{
  if (!editorDoc) return; const p=canvasPoint(event), sorted=[...editorDoc.layers].filter(x=>x.visible&&!x.locked).sort((a,b)=>b.z-a.z); let target=selectedLayer(); const handle=Math.max(14,editorDoc.canvas.width/70);
  const onHandle=target&&Math.abs(p.x-(target.x+target.width))<handle&&Math.abs(p.y-(target.y+target.height))<handle;
  if (!onHandle) target=sorted.find(layer=>p.x>=layer.x&&p.x<=layer.x+layer.width&&p.y>=layer.y&&p.y<=layer.y+layer.height);
  if (!target) return; selectedLayerId=target.id; remember(); interaction={mode:onHandle?'resize':'move',start:p,x:target.x,y:target.y,width:target.width,height:target.height}; event.currentTarget.setPointerCapture(event.pointerId); event.currentTarget.classList.add('dragging'); renderLayerPanel(); renderProperties(); renderEditor();
});
$('#editor-canvas').addEventListener('pointermove',event=>{
  if (!interaction) return; const p=canvasPoint(event), layer=selectedLayer(), dx=p.x-interaction.start.x,dy=p.y-interaction.start.y;
  if(interaction.mode==='move'){layer.x=interaction.x+dx;layer.y=interaction.y+dy}else{layer.width=Math.max(1,interaction.width+dx);layer.height=Math.max(1,interaction.height+dy)} renderEditor();
});
function finishInteraction(event){ if(!interaction)return; interaction=null; event.currentTarget.classList.remove('dragging'); renderProperties(); }
$('#editor-canvas').addEventListener('pointerup',finishInteraction); $('#editor-canvas').addEventListener('pointercancel',finishInteraction);

$('#undo').addEventListener('click',()=>{ if(!undoStack.length)return; redoStack.push(cloneEditor()); editorDoc=undoStack.pop(); selectedLayerId=editorDoc.layers.find(x=>x.id===selectedLayerId)?.id||editorDoc.layers.at(-1)?.id; updateHistoryButtons(); renderLayerPanel();renderProperties();renderEditor(); });
$('#redo').addEventListener('click',()=>{ if(!redoStack.length)return; undoStack.push(cloneEditor()); editorDoc=redoStack.pop(); selectedLayerId=editorDoc.layers.find(x=>x.id===selectedLayerId)?.id||editorDoc.layers.at(-1)?.id; updateHistoryButtons(); renderLayerPanel();renderProperties();renderEditor(); });
$('#save-editor').addEventListener('click',async()=>{
  if(!editorDoc||!job)return; const button=$('#save-editor'); button.disabled=true; button.textContent='Exporting…';
  const response=await fetch(`/api/jobs/${job.id}/editor`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(editorDoc)}); const data=await response.json(); button.textContent='Export layers'; button.disabled=false;
  if(!response.ok){setStatus('failed',data.detail||'Layer export failed');return;} artifacts=data.artifacts||artifacts; renderArtifacts(); setStatus('complete','Editable JSON, aligned PNG layers, composite, and ZIP are ready.');
});
loadConfig();
