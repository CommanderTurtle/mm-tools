import { marked } from "/vendor/marked/marked.esm.js";
import DOMPurify from "/vendor/purify/purify.es.mjs";

const $ = (id) => document.getElementById(id);
const fields = ["mode", "instruction", "max_page_tokens", "max_length", "dpi"];
let current = null,
  order = [],
  page = 0,
  epoch = 0,
  busy = false,
  pending = false;
let streamOffset = 0,
  raw = "",
  result = null,
  activeKey = null,
  view = "rendered";
const running = (status) => ["running", "preparing"].includes(status);
const colors = {
  text: "#28795f",
  title: "#7159a3",
  table: "#a07122",
  equation: "#416fa7",
  image: "#b05c43",
  chart: "#b05c43",
  figure: "#b05c43",
};
const color = (label) =>
  colors[label.replace(/^(cross_|append_)/, "")] || "#617c78";
const files = (key, folder, name) =>
  `/files/${key}/${folder}/${encodeURIComponent(name)}`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || response.statusText);
  return data;
}
function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function status(message) {
  $("status").textContent = message;
}
function settings() {
  return Object.fromEntries(
    fields.map((id) => [
      id,
      ["mode", "instruction"].includes(id) ? $(id).value : Number($(id).value),
    ]),
  );
}
function applySettings(value) {
  for (const id of fields) $(id).value = value[id];
  $("max_page_tokens").dispatchEvent(new Event("input"));
  $("mode").dispatchEvent(new Event("change"));
}
function controls() {
  $("settingsFields").disabled = busy || pending;
  $("parse").disabled = busy || pending || !order.length;
  $("example").disabled = busy || pending;
  $("cancel").hidden = !busy;
  $("cancel").disabled = pending || !activeKey;
  for (const id of ["moveLeft", "moveRight", "removePage"])
    $(id).disabled = busy || pending || !order.length;
  $("prev").disabled = page <= 0;
  $("next").disabled = page >= order.length - 1;
  document.body.classList.toggle("busy", busy);
}
async function action(fn) {
  if (pending) return;
  pending = true;
  notice();
  controls();
  try {
    await fn();
  } catch (error) {
    notice(error.message);
  } finally {
    pending = false;
    controls();
  }
}
function selectView(name) {
  view = name;
  for (const target of ["rendered", "raw", "blocks"])
    $(target).hidden = target !== name;
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.classList.toggle("selected", b.dataset.view === name));
}
function showPage() {
  page = Math.max(0, Math.min(page, order.length - 1));
  const item = current?.pages.find((p) => p.id === order[page]);
  $("pageCount").textContent = order.length
    ? `${page + 1} / ${order.length}`
    : "0 / 0";
  $("pageName").textContent = item?.title || "Drop a document to begin";
  $("dimensions").textContent = item ? `${item.width} × ${item.height}` : "";
  $("pageImage").hidden = !item;
  $("sourceSheet").hidden = !item;
  $("empty").hidden = !!item;
  if (item) $("pageImage").src = files(current.id, "pages", item.id);
  document
    .querySelectorAll(".thumb")
    .forEach((b, i) => b.classList.toggle("active", i === page));
  drawBoxes();
  controls();
}
function pageRail() {
  $("pages").replaceChildren();
  for (const [i, id] of order.entries()) {
    const item = current.pages.find((p) => p.id === id);
    const button = document.createElement("button");
    button.className = "thumb";
    button.title = item.title;
    const img = document.createElement("img");
    img.src = files(current.id, "thumbs", id);
    img.alt = "";
    img.loading = "lazy";
    const label = document.createElement("span");
    label.textContent = String(i + 1).padStart(2, "0");
    button.append(img, label);
    button.onclick = () => {
      page = i;
      showPage();
    };
    $("pages").append(button);
  }
  showPage();
}
function drawBoxes() {
  const svg = $("overlay");
  svg.replaceChildren();
  svg.hidden = !$("boxes").checked;
  if (!current) return;
  const sourcePage = (result?.pages || current.order).indexOf(order[page]) + 1;
  (result?.blocks || current.progress.blocks || []).forEach((block, i) => {
    if (block.page !== sourcePage || !block.bbox) return;
    const [x, y, x2, y2] = block.bbox;
    if (x2 <= x || y2 <= y) return;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    for (const [key, value] of Object.entries({
      x,
      y,
      width: x2 - x,
      height: y2 - y,
    }))
      rect.setAttribute(key, value);
    rect.style.setProperty("--region", color(block.label));
    const title = document.createElementNS(svg.namespaceURI, "title");
    title.textContent = block.label;
    rect.append(title);
    rect.onclick = () => {
      selectView("blocks");
      const target = $("block-" + i);
      target?.scrollIntoView({ block: "center", behavior: "smooth" });
      document
        .querySelectorAll(".focused")
        .forEach((b) => b.classList.remove("focused"));
      target?.classList.add("focused");
      rect.classList.add("focused");
    };
    svg.append(rect);
  });
}
function safeMarkup(node, text) {
  const math = [];
  const prefix = "OCRMath" + crypto.randomUUID().replaceAll("-", "") + "X";
  const protectedText = text.replace(
    /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<![\\$])\$([^$\n]+)\$/g,
    (raw, display, brackets, inline, dollars) => {
      math.push({
        raw,
        tex: display ?? brackets ?? inline ?? dollars,
        display: display !== undefined || brackets !== undefined,
      });
      return prefix + (math.length - 1) + "Z";
    },
  );
  node.innerHTML = DOMPurify.sanitize(marked.parse(protectedText), {
    FORBID_TAGS: ["style", "iframe", "form", "input", "button"],
    FORBID_ATTR: ["style"],
  });
  node.querySelectorAll("img").forEach((img) => {
    const name = img.getAttribute("src");
    if (/^page\d+_\d+_\d+_\d+_\d+\.jpg$/.test(name))
      img.src = files(current.id, "output", name);
    else img.remove();
  });
  node.querySelectorAll("a").forEach((a) => {
    a.rel = "noreferrer noopener";
    a.target = "_blank";
  });
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const pattern = new RegExp(prefix + "(\\d+)Z", "g");
  for (const leaf of nodes) {
    const parts = [...leaf.textContent.matchAll(pattern)];
    if (!parts.length) continue;
    const fragment = document.createDocumentFragment();
    let offset = 0;
    for (const match of parts) {
      fragment.append(leaf.textContent.slice(offset, match.index));
      const entry = math[Number(match[1])];
      const span = document.createElement("span");
      if (leaf.parentElement.closest("code,pre") || !window.katex)
        span.textContent = entry.raw;
      else
        window.katex.render(entry.tex, span, {
          displayMode: entry.display,
          throwOnError: false,
          trust: false,
        });
      fragment.append(span);
      offset = match.index + match[0].length;
    }
    fragment.append(leaf.textContent.slice(offset));
    leaf.replaceWith(fragment);
  }
}
function renderResult() {
  $("rendered").replaceChildren();
  const paper = document.createElement("article");
  paper.className = "paper";
  let displayText = "\n\n" + result.markdown + "\n\n";
  for (const block of result.blocks) {
    if (block.label.replace(/^(cross_|append_)/, "") !== "equation") continue;
    const tex = block.content.trim();
    if (tex && !/^(\$|\\[[(])/.test(tex))
      displayText = displayText.replaceAll(
        "\n\n" + tex + "\n\n",
        () => "\n\n$$" + tex + "$$\n\n",
      );
  }
  safeMarkup(paper, displayText);
  $("rendered").append(paper);
  $("blocks").replaceChildren();
  for (const [i, block] of result.blocks.entries()) {
    const article = document.createElement("article");
    article.className = "block";
    article.id = "block-" + i;
    article.style.setProperty("--region", color(block.label));
    const label = document.createElement("header");
    label.textContent = `PAGE ${block.page} / ${block.label.replaceAll("_", " ")}`;
    const content = document.createElement("div");
    content.className = "content";
    if (block.label.replace(/^(cross_|append_)/, "") === "table")
      safeMarkup(content, block.content);
    else if (
      block.label.replace(/^(cross_|append_)/, "") === "equation" &&
      window.katex
    ) {
      const tex = block.content
        .trim()
        .replace(/^\$\$|\$\$$/g, "")
        .replace(/^\\\[|\\\]$/g, "")
        .replace(/^\\\(|\\\)$/g, "");
      window.katex.render(tex, content, {
        displayMode: true,
        throwOnError: false,
        trust: false,
      });
    } else {
      content.textContent = block.content;
      if (window.renderMathInElement)
        window.renderMathInElement(content, {
          throwOnError: false,
          trust: false,
        });
    }
    article.append(label, content);
    article.onclick = () => {
      const index = order.indexOf(result.pages[block.page - 1]);
      if (index >= 0) {
        page = index;
        showPage();
      }
    };
    $("blocks").append(article);
  }
  $("copy").disabled = false;
  for (const [id, name] of [
    ["markdown", "document.md"],
    ["layout", "layout.json"],
  ]) {
    $(id).href = files(current.id, "output", name) + "?download=1";
    $(id).hidden = false;
  }
  $("zip").href = `/files/${current.id}/document.zip`;
  $("zip").hidden = false;
  $("metrics").textContent =
    `${result.pages_out} pages · ${result.tokens.toLocaleString()} tokens · ${result.elapsed}s`;
  if (
    result.finish_reason === "length" ||
    result.pages_out < result.expected_pages ||
    result.capped_pages?.length
  )
    notice(
      "Possible partial document: a context/page-token budget was reached or fewer pages were returned. Output is saved; raise the relevant budget or select fewer pages." +
        (result.capped_pages?.length
          ? ` Capped pages: ${result.capped_pages.join(", ")}.`
          : ""),
    );
  drawBoxes();
}
function resetOutput() {
  result = null;
  raw = "";
  streamOffset = 0;
  $("raw").textContent = "";
  $("blocks").replaceChildren();
  $("rendered").textContent = "Recognized content appears here.";
  $("metrics").textContent = "";
  $("copy").disabled = true;
  for (const id of ["markdown", "layout", "zip"]) $(id).hidden = true;
}
async function stream(key, stamp) {
  let chunk;
  do {
    chunk = await api(`/api/documents/${key}/stream?offset=${streamOffset}`);
    if (stamp !== epoch) return;
    streamOffset = chunk.offset;
    raw += chunk.text;
    // The viewport remains bounded; the full raw document is retained on disk.
    if (raw.length > 1_000_000) raw = raw.slice(-1_000_000);
    const el = $("raw");
    const follow = el.scrollTop + el.clientHeight >= el.scrollHeight - 50;
    el.textContent =
      (streamOffset > 1_000_000
        ? "[Recent output — full raw text in the bundle]\n"
        : "") + raw;
    if (follow) el.scrollTop = el.scrollHeight;
  } while (chunk.more && stamp === epoch);
}
async function monitor(key, stamp) {
  if (stamp !== epoch) return;
  try {
    const data = await api(`/api/documents/${key}`);
    if (stamp !== epoch) return;
    current = data;
    busy = running(data.status);
    activeKey = busy ? key : null;
    status(data.progress.text || data.status);
    await stream(key, stamp);
    if (stamp !== epoch) return;
    controls();
    if (busy) {
      if (data.progress.blocks?.length) {
        drawBoxes();
        if (view === "rendered") {
          const paper = document.createElement("article");
          paper.className = "paper";
          const label = document.createElement("small");
          label.textContent =
            "LIVE PAGE · document merges when parsing completes";
          paper.append(label);
          for (const block of data.progress.blocks) {
            const region = document.createElement("div");
            region.textContent = block.content;
            paper.append(region);
          }
          $("rendered").replaceChildren(paper);
        }
      }
      setTimeout(() => monitor(key, stamp), 500);
      return;
    }
    if (!order.length && data.pages.length) {
      order = data.order.length ? data.order : data.pages.map((p) => p.id);
      pageRail();
    }
    if (data.status === "complete") {
      const response = await fetch(files(key, "output", "layout.json"));
      if (!response.ok) throw Error("Could not read document layout");
      const loaded = await response.json();
      if (stamp !== epoch) return;
      result = loaded;
      renderResult();
      status(
        result.finish_reason === "length" ||
          result.capped_pages?.length ||
          result.pages_out < result.expected_pages
          ? "Document saved · review limits"
          : "Document complete",
      );
    } else if (data.error) notice(data.error);
    else
      status(
        data.status === "ready" ? "Arrange pages, then parse." : data.status,
      );
    await refreshHistory();
  } catch (error) {
    if (stamp === epoch) {
      notice(error.message);
      status("Connection interrupted; reopen Documents to reconnect");
      controls();
    }
  }
}
function adopt(data) {
  epoch++;
  current = data;
  order = data.order.length ? [...data.order] : data.pages.map((p) => p.id);
  page = 0;
  resetOutput();
  notice();
  applySettings(data.settings);
  $("title").textContent = data.title;
  busy = running(data.status);
  activeKey = busy ? data.id : null;
  pageRail();
  monitor(data.id, epoch);
}
async function upload(selected) {
  if (busy || !selected.length) return;
  const form = new FormData();
  for (const file of selected) form.append("files", file);
  form.append("settings", JSON.stringify(settings()));
  status("Uploading document…");
  adopt(await api("/api/documents", { method: "POST", body: form }));
}
async function refreshHistory() {
  const rows = await api("/api/documents");
  $("historyCount").textContent = rows.length;
  $("history").replaceChildren();
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "historyRow";
    const open = document.createElement("button");
    open.textContent = row.title;
    const sub = document.createElement("small");
    sub.textContent = `${new Date(row.created * 1000).toLocaleString()} · ${row.status}`;
    open.append(sub);
    open.onclick = () =>
      action(async () => {
        const data = await api(`/api/documents/${row.id}`);
        const cfg = await api("/api/config");
        if (cfg.active && cfg.active !== row.id)
          throw Error(
            "Stop the active operation before opening another document",
          );
        adopt(data);
        $("historyDialog").close();
      });
    const remove = document.createElement("button");
    remove.textContent = "Delete";
    remove.disabled = running(row.status);
    remove.onclick = () =>
      action(async () => {
        if (!confirm("Delete this document and its saved outputs?")) return;
        await api(`/api/documents/${row.id}`, { method: "DELETE" });
        if (current?.id === row.id) {
          epoch++;
          current = null;
          order = [];
          resetOutput();
          pageRail();
          $("title").textContent = "No document open";
        }
        await refreshHistory();
      });
    line.append(open, remove);
    $("history").append(line);
  }
}
$("files").onchange = () => action(() => upload([...$("files").files]));
$("settingsForm").onsubmit = (event) => {
  event.preventDefault();
  action(async () => {
    if (!current) return;
    const key = current.id;
    const body = JSON.stringify({ settings: settings(), order });
    adopt(
      await api(`/api/documents/${key}/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      }),
    );
  });
};
$("cancel").onclick = () =>
  action(async () => {
    if (activeKey)
      await api(`/api/documents/${activeKey}/cancel`, { method: "POST" });
    status("Stopping…");
  });
$("prev").onclick = () => {
  page--;
  showPage();
};
$("next").onclick = () => {
  page++;
  showPage();
};
$("fit").onclick = () => {
  const actual = $("sourceSheet").classList.toggle("actual");
  $("fit").textContent = actual ? "Fit" : "100%";
};
$("boxes").onchange = drawBoxes;
function move(delta) {
  if (busy) return;
  const next = page + delta;
  if (next < 0 || next >= order.length) return;
  [order[page], order[next]] = [order[next], order[page]];
  page = next;
  pageRail();
}
$("moveLeft").onclick = () => move(-1);
$("moveRight").onclick = () => move(1);
$("removePage").onclick = () => {
  if (!busy) {
    order.splice(page, 1);
    pageRail();
  }
};
$("max_page_tokens").oninput = () => {
  $("tokenValue").textContent = Number(
    $("max_page_tokens").value,
  ).toLocaleString();
};
$("mode").onchange = () => {
  $("modeHint").textContent = {
    auto: "Auto tiles single pages; documents use one global view per page.",
    gundam: "Dynamic tiles on every page. More detail, more image tokens.",
    base: "One global view per page. Preserves cross-page context.",
  }[$("mode").value];
};
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => selectView(b.dataset.view)));
$("historyToggle").onclick = () =>
  action(async () => {
    await refreshHistory();
    $("historyDialog").showModal();
  });
$("closeHistory").onclick = () => $("historyDialog").close();
$("showLog").onclick = () =>
  action(async () => {
    if (!current) return;
    const response = await fetch(`/api/documents/${current.id}/log`);
    if (!response.ok) throw Error("Could not read log");
    $("log").textContent = await response.text();
    $("logDialog").showModal();
  });
$("closeLog").onclick = () => $("logDialog").close();
$("copy").onclick = () =>
  action(async () => {
    await navigator.clipboard.writeText(result.markdown);
    status("Markdown copied");
  });
$("example").onclick = () =>
  action(async () => adopt(await api("/api/example", { method: "POST" })));
document.addEventListener("dragover", (event) => {
  event.preventDefault();
  document.body.classList.add("dragging");
});
document.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) document.body.classList.remove("dragging");
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  document.body.classList.remove("dragging");
  action(() => upload([...event.dataTransfer.files]));
});
action(async () => {
  const cfg = await api("/api/config");
  applySettings(cfg.defaults);
  await refreshHistory();
  if (cfg.active) adopt(await api(`/api/documents/${cfg.active}`));
  else controls();
});
