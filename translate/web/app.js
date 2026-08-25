const $ = selector => document.querySelector(selector);
let useExternal = false;
let localLoaded = false;
let languages = {};
let showMarkdownPreview = false;

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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

function safeMarkdownUrl(value) {
  const url = String(value || '').trim();
  if (!url) return '#';
  if (/^(?:https?:|mailto:|tel:)/i.test(url)) return url;
  if (/^data:image\/(?:avif|gif|jpe?g|png|webp);base64,/i.test(url)) return url;
  if (/^(?:#|\/|\.\/|\.\.\/)/.test(url) || !/^[a-z][a-z\d+.-]*:/i.test(url)) return url;
  return '#';
}

function renderInlineMarkdown(value) {
  return value
    .replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;([^&]*)&quot;)?\)/g, (_match, alt, url, title = '') => (
      `<img src="${safeMarkdownUrl(url)}" alt="${alt}"${title ? ` title="${title}"` : ''}>`
    ))
    .replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;([^&]*)&quot;)?\)/g, (_match, label, url, title = '') => (
      `<a href="${safeMarkdownUrl(url)}"${title ? ` title="${title}"` : ''} target="_blank" rel="noopener noreferrer">${label}</a>`
    ))
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, '<span class="wikilink" title="$1">$2</span>')
    .replace(/\[\[([^\]]+)\]\]/g, '<span class="wikilink">$1</span>')
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/___(.+?)___/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/~~(.+?)~~/g, '<del>$1</del>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`\n]+)`/g, '<code>$1</code>');
}

function renderMarkdown(markdown) {
  let html = escapeHtml(markdown.replace(/\r\n?/g, '\n'));
  const blocks = [];
  const hold = block => {
    const index = blocks.push(block) - 1;
    return `@@TRANSLATE_MARKDOWN_BLOCK_${index}@@`;
  };

  html = html.replace(/^---\n([\s\S]*?)\n---(?:\n|$)/, (_match, body) => (
    hold(`<details class="markdown-frontmatter"><summary>Document metadata</summary><pre><code>${body}</code></pre></details>`)
  ));
  html = html.replace(/```([^\n]*)\n([\s\S]*?)```/g, (_match, language, code) => (
    hold(`<pre><code${language.trim() ? ` data-language="${language.trim()}"` : ''}>${code.replace(/\n$/, '')}</code></pre>`)
  ));

  html = html
    .replace(/^######\s+(.+)$/gm, '<h6>$1</h6>')
    .replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>')
    .replace(/^####\s+(.+)$/gm, '<h4>$1</h4>')
    .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
    .replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

  html = renderInlineMarkdown(html)
    .replace(/^&gt;\s?\[!([^\]]+)\][+-]?\s*\n((?:&gt;.*(?:\n|$))*)/gm, (_match, kind, body) => (
      hold(`<aside class="markdown-callout"><strong>${kind}</strong><div>${body.replace(/^&gt;\s?/gm, '').trim().replace(/\n/g, '<br>')}</div></aside>`)
    ))
    .replace(/^(?:&gt;\s?.+(?:\n|$))+/gm, quote => (
      hold(`<blockquote>${quote.replace(/^&gt;\s?/gm, '').trim().replace(/\n/g, '<br>')}</blockquote>`)
    ))
    .replace(/^ {0,3}(?:---+|___+|\*\*\*+)\s*$/gm, '<hr>');

  html = html.replace(
    /^(\|[^\n]+\|)\n(\|[-:|\s]+\|)\n((?:\|[^\n]+\|(?:\n|$))+)/gm,
    (_match, heading, _separator, body) => {
      const cells = row => row.split('|').slice(1, -1).map(cell => cell.trim());
      const head = cells(heading).map(cell => `<th>${cell}</th>`).join('');
      const rows = body.trim().split('\n').map(row => (
        `<tr>${cells(row).map(cell => `<td>${cell}</td>`).join('')}</tr>`
      )).join('');
      return hold(`<div class="markdown-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`);
    },
  );

  html = html.split(/\n{2,}/).map(part => {
    const value = part.trim();
    if (!value) return '';
    if (/^@@TRANSLATE_MARKDOWN_BLOCK_\d+@@$/.test(value)) return value;
    if (/^<(?:h[1-6]|hr)/.test(value)) return value;
    const lines = value.split('\n');
    if (lines.every(line => /^\s*[-*+]\s+/.test(line))) {
      return `<ul>${lines.map(line => {
        const item = line.replace(/^\s*[-*+]\s+/, '');
        const task = item.match(/^\[([ xX])\]\s*(.*)$/);
        return task
          ? `<li class="task"><input type="checkbox" disabled ${task[1].toLowerCase() === 'x' ? 'checked' : ''}>${task[2]}</li>`
          : `<li>${item}</li>`;
      }).join('')}</ul>`;
    }
    if (lines.every(line => /^\s*\d+[.)]\s+/.test(line))) {
      return `<ol>${lines.map(line => `<li>${line.replace(/^\s*\d+[.)]\s+/, '')}</li>`).join('')}</ol>`;
    }
    return `<p>${value.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  blocks.forEach((block, index) => {
    html = html.replace(`@@TRANSLATE_MARKDOWN_BLOCK_${index}@@`, block);
  });
  return html;
}

function syncOutputView() {
  const output = $('#output');
  const preview = $('#markdown-preview');
  const toggle = $('#preview-toggle');
  const hasOutput = Boolean(output.value);
  if (!hasOutput) showMarkdownPreview = false;
  toggle.disabled = !hasOutput;
  toggle.textContent = showMarkdownPreview ? 'View raw text' : 'View pretty Markdown';
  toggle.setAttribute('aria-pressed', String(showMarkdownPreview));
  toggle.classList.toggle('active', showMarkdownPreview);
  output.hidden = showMarkdownPreview;
  preview.hidden = !showMarkdownPreview;
  if (showMarkdownPreview) preview.innerHTML = renderMarkdown(output.value);
  else preview.replaceChildren();
}

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
$('#preview-toggle').addEventListener('click', () => {
  if (!$('#output').value) return;
  showMarkdownPreview = !showMarkdownPreview;
  syncOutputView();
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
  syncOutputView();
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
    syncOutputView();
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
