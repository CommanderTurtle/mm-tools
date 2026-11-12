const $ = (id) => document.getElementById(id);
const colors = [
  "#ff5c7c",
  "#2fd47b",
  "#409cff",
  "#ffb830",
  "#b05cff",
  "#00ccc4",
];
const numeric = [
  "max_new_tokens",
  "temperature",
  "top_p",
  "top_k",
  "repetition_penalty",
];
let config,
  task = "referring",
  source = null,
  inputImage = null,
  boxes = [],
  drag = null;
let current = null,
  result = null,
  resultImage = null,
  busy = false,
  epoch = 0;
let timer,
  renderFrame,
  renderVersion = 0,
  replayFrame,
  replayLimit = Infinity,
  cameraStream;
const maskImages = new Map();
let cameraRequest = 0;
const status = (text, error = false) => {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
};
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function image(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Unable to read image"));
    img.src = url;
  });
}
function element(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function locked(value) {
  busy = value;
  $("inputs").disabled = value;
  document.body.classList.toggle("busy", value);
  $("stop").hidden = !value;
  $("stop").disabled = !current;
  $("history-button").disabled = value;
  document
    .querySelectorAll("#tasks button, #examples button")
    .forEach((b) => (b.disabled = value));
}
function raw(text) {
  $("raw").replaceChildren(
    ...text.split(/(<\d+>|<\/?[A-Za-z_|]+>)/g).map((part) => {
      if (!part.startsWith("<")) return document.createTextNode(part);
      return element(
        "span",
        part,
        /^<\d+>$/.test(part) ? "token-coordinate" : "token-marker",
      );
    }),
  );
}
function stopReplay() {
  cancelAnimationFrame(replayFrame);
  replayLimit = Infinity;
  if (result) raw(result.response);
}
function clearResult() {
  stopReplay();
  result = resultImage = null;
  maskImages.clear();
  ++renderVersion;
  $("result").hidden = true;
  $("result-empty").hidden = false;
  $("stats").replaceChildren();
  $("instances").replaceChildren();
  $("instance").replaceChildren(new Option("All objects", "all"));
  $("exports").hidden = true;
  $("sent-prompt").hidden = true;
  $("prompt-text").textContent = "";
  $("log-button").hidden = true;
  for (const id of ["download-view", "replay", "copy"]) $(id).disabled = true;
  raw("Awaiting perception.");
}
function selectTask(value) {
  task = value;
  document
    .querySelectorAll("#tasks button")
    .forEach((b) => b.setAttribute("aria-pressed", b.dataset.task === task));
  const boxMode = ["box_matting", "visual_prompt"].includes(task);
  $("clear-boxes").hidden = $("box-hint").hidden = !boxMode;
  $("box-hint").textContent =
    task === "box_matting"
      ? "Draw one target box"
      : "Draw up to 32 example boxes";
  if (task === "box_matting" && boxes.length > 1) boxes = boxes.slice(0, 1);
  $("caption-label").hidden = task !== "freeform";
  $("query-label").textContent =
    task === "detection"
      ? "Categories · one per line"
      : task === "ocr_grounding"
        ? "Text to locate"
        : task === "freeform"
          ? "Question / caption instruction"
          : task === "box_matting"
            ? "Object label (optional)"
            : "Grounding expression";
  $("query").placeholder =
    task === "detection"
      ? "chair\ndog\nbush"
      : task === "ocr_grounding"
        ? "COFFEE"
        : task === "freeform"
          ? "Describe this image in detail."
          : "person";
  $("query-label").hidden = $("query").hidden = task === "ocr";
  drawInput();
}
function drawInput() {
  if (!inputImage) return;
  const canvas = $("input");
  const scale = Math.min(
    1,
    1000 / Math.max(inputImage.width, inputImage.height),
  );
  canvas.width = Math.round(inputImage.width * scale);
  canvas.height = Math.round(inputImage.height * scale);
  // Explicit CSS dimensions preserve the same rectangle for drawing and input.
  const fit = Math.min(
    1,
    ($("drop").clientWidth - 2) / canvas.width,
    ($("drop").clientHeight - 2) / canvas.height,
  );
  canvas.style.width = `${canvas.width * fit}px`;
  canvas.style.height = `${canvas.height * fit}px`;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(inputImage, 0, 0, canvas.width, canvas.height);
  if (["box_matting", "visual_prompt"].includes(task)) {
    const all = drag ? [...boxes, rectangle(drag.start, drag.end)] : boxes;
    for (const [i, box] of all.entries()) {
      const [x, y, r, b] = box;
      ctx.strokeStyle = colors[i % colors.length];
      ctx.fillStyle = `${colors[i % colors.length]}25`;
      ctx.lineWidth = 2 / fit;
      ctx.fillRect(
        x * canvas.width,
        y * canvas.height,
        (r - x) * canvas.width,
        (b - y) * canvas.height,
      );
      ctx.strokeRect(
        x * canvas.width,
        y * canvas.height,
        (r - x) * canvas.width,
        (b - y) * canvas.height,
      );
    }
  }
  canvas.hidden = false;
  $("upload-hint").hidden = true;
  $("dimensions").textContent = `${inputImage.width} × ${inputImage.height}`;
}
function point(event) {
  const rect = $("input").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  ];
}
function rectangle(a, b) {
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[0], b[0]),
    Math.max(a[1], b[1]),
  ];
}
$("input").addEventListener("pointerdown", (event) => {
  if (busy || !inputImage || !["box_matting", "visual_prompt"].includes(task))
    return;
  if (task === "box_matting") boxes = [];
  if (boxes.length >= 32) return status("Clear boxes before drawing more.");
  drag = { start: point(event), end: point(event) };
  $("input").setPointerCapture(event.pointerId);
  drawInput();
});
$("input").addEventListener("pointermove", (event) => {
  if (drag) {
    drag.end = point(event);
    drawInput();
  }
});
$("input").addEventListener("pointerup", () => {
  if (!drag) return;
  const rect = rectangle(drag.start, drag.end);
  if (
    (rect[2] - rect[0]) * inputImage.width >= 2 &&
    (rect[3] - rect[1]) * inputImage.height >= 2
  )
    boxes.push(rect);
  drag = null;
  drawInput();
});
$("input").addEventListener("pointercancel", () => {
  drag = null;
  drawInput();
});
$("clear-boxes").onclick = () => {
  boxes = [];
  drawInput();
};
new ResizeObserver(drawInput).observe($("drop"));
async function chooseSource(next, url) {
  if (busy) return;
  const version = ++epoch;
  clearTimeout(timer);
  current = null;
  source = null;
  clearResult();
  status("Reading image…");
  try {
    const decoded = await image(url);
    if (version !== epoch) return;
    inputImage = decoded;
    source = next;
    boxes = [];
    drawInput();
    status("Ready.");
  } catch (error) {
    if (version === epoch) status(error.message, true);
  }
}
async function chooseFile(file) {
  if (!file || busy) return;
  const url = URL.createObjectURL(file);
  try {
    await chooseSource({ file }, url);
  } finally {
    URL.revokeObjectURL(url);
  }
}
$("choose").onclick = () => $("upload").click();
$("upload").onchange = () => {
  void chooseFile($("upload").files[0]);
  $("upload").value = "";
};
$("drop").ondragover = (event) => {
  event.preventDefault();
  if (!busy) $("drop").classList.add("over");
};
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (event) => {
  event.preventDefault();
  $("drop").classList.remove("over");
  void chooseFile(event.dataTransfer.files[0]);
};
document.addEventListener("paste", (event) => {
  const file = [...event.clipboardData.items]
    .find((item) => item.type.startsWith("image/"))
    ?.getAsFile();
  if (file && !busy) {
    event.preventDefault();
    void chooseFile(file);
  }
});
function settings() {
  return {
    task,
    query: $("query").value,
    prompt: $("prompt").value,
    boxes: structuredClone(boxes),
    ...Object.fromEntries(numeric.map((key) => [key, Number($(key).value)])),
  };
}
function restoreSettings(value) {
  boxes = structuredClone(value.boxes || []);
  for (const key of numeric) $(key).value = value[key];
  $("tokens-value").value = value.max_new_tokens;
  $("temp-value").value = value.temperature;
  $("query").value = value.query;
  $("prompt").value = value.prompt;
  $("caption").value = [...$("caption").options].some(
    (o) => o.value === value.query,
  )
    ? value.query
    : "";
  selectTask(value.task);
}
$("max_new_tokens").oninput = () =>
  ($("tokens-value").value = $("max_new_tokens").value);
$("temperature").oninput = () =>
  ($("temp-value").value = $("temperature").value);
$("caption").onchange = () => {
  if ($("caption").value) $("query").value = $("caption").value;
};
function colorFor(label, controls) {
  if (controls.palette !== "auto")
    return controls.palette === "custom" ? controls.color : controls.palette;
  let hash = 0;
  for (const char of label) hash = (hash * 31 + char.codePointAt(0)) >>> 0;
  return colors[hash % colors.length];
}
function controls() {
  return {
    display: $("display").value,
    instance: $("instance").value,
    palette: $("palette").value,
    color: $("color").value,
    boxes: $("show-boxes").checked,
    labels: $("show-labels").checked,
    masks: $("show-masks").checked,
    width: +$("line-width").value,
    font: +$("font-size").value,
    opacity: +$("opacity").value,
    threshold: +$("threshold").value,
  };
}
function mask(key, name) {
  const url = `/files/${key}/output/${name}`;
  let pending = maskImages.get(url);
  if (!pending)
    pending = image(url).catch((error) => {
      if (maskImages.get(url) === pending) maskImages.delete(url);
      throw error;
    });
  maskImages.delete(url);
  maskImages.set(url, pending);
  while (maskImages.size > 4) maskImages.delete(maskImages.keys().next().value);
  return pending;
}
async function compose(
  manifest,
  original,
  key,
  opts,
  size,
  limit = Infinity,
  cancelled = () => false,
) {
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, size / Math.max(manifest.width, manifest.height));
  canvas.width = Math.max(1, Math.round(manifest.width * scale));
  canvas.height = Math.max(1, Math.round(manifest.height * scale));
  const { width: w, height: h } = canvas;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(original, 0, 0, w, h);
  const matte = ["matting", "box_matting"].includes(manifest.task);
  const mode =
    opts.display === "auto" ? (matte ? "green" : "overlay") : opts.display;
  const selected = manifest.instances
    .map((item, index) => ({ ...item, index }))
    .filter(
      (item) =>
        item.index < limit &&
        (opts.instance === "all" || item.index === +opts.instance),
    );
  const masked = selected.filter((item) => item.mask);
  if (
    mode !== "original" &&
    masked.length &&
    (opts.masks || mode !== "overlay")
  ) {
    const pixels = ctx.getImageData(0, 0, w, h);
    const alpha = new Uint8Array(w * h);
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    const sc = scratch.getContext("2d", { willReadFrequently: true });
    for (const item of masked) {
      const decoded = await mask(key, item.mask);
      if (cancelled()) return null;
      sc.clearRect(0, 0, w, h);
      sc.drawImage(decoded, 0, 0, w, h);
      const values = sc.getImageData(0, 0, w, h).data;
      const rgb = colorFor(item.label, opts)
        .slice(1)
        .match(/../g)
        .map((n) => parseInt(n, 16));
      for (let i = 0, p = 0; i < alpha.length; i++, p += 4) {
        const a = matte
          ? values[p]
          : values[p] >= opts.threshold * 255
            ? 255
            : 0;
        alpha[i] = Math.max(alpha[i], a);
        if (mode === "overlay") {
          const weight = (a / 255) * opts.opacity;
          for (let c = 0; c < 3; c++)
            pixels.data[p + c] =
              pixels.data[p + c] * (1 - weight) + rgb[c] * weight;
        }
      }
    }
    if (mode !== "overlay") {
      for (let i = 0, p = 0; i < alpha.length; i++, p += 4) {
        if (mode === "alpha") {
          pixels.data[p] = pixels.data[p + 1] = pixels.data[p + 2] = alpha[i];
        } else if (mode === "cutout") pixels.data[p + 3] = alpha[i];
        else if (mode === "green") {
          const a = alpha[i] / 255;
          pixels.data[p] *= a;
          pixels.data[p + 1] = pixels.data[p + 1] * a + 214 * (1 - a);
          pixels.data[p + 2] = pixels.data[p + 2] * a + 90 * (1 - a);
        }
      }
    }
    ctx.putImageData(pixels, 0, 0);
  }
  if (mode === "overlay") {
    const visualScale = Math.max(0.6, w / 900);
    ctx.lineWidth = opts.width * visualScale;
    ctx.font = `${opts.font * visualScale}px system-ui`;
    for (const item of selected) {
      const c = item.coords.map((n) => n * scale);
      const color = colorFor(item.label, opts);
      ctx.strokeStyle = ctx.fillStyle = color;
      if (opts.boxes) {
        if (item.type === "point") {
          ctx.beginPath();
          ctx.arc(c[0], c[1], 5 * visualScale, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.stroke();
        } else ctx.strokeRect(c[0], c[1], c[2] - c[0], c[3] - c[1]);
      }
      if (opts.labels) {
        const height = (opts.font + 8) * visualScale;
        const width = Math.min(
          w,
          ctx.measureText(item.label).width + 12 * visualScale,
        );
        const x = Math.max(0, Math.min(w - width, c[0]));
        const y = Math.max(height, Math.min(h, c[1]));
        ctx.fillStyle = "#071518dd";
        ctx.fillRect(x, y - height, width, height);
        ctx.fillStyle = color;
        ctx.fillText(
          item.label,
          x + 6 * visualScale,
          y - 6 * visualScale,
          width - 12 * visualScale,
        );
      }
    }
  }
  return canvas;
}
function redraw() {
  cancelAnimationFrame(renderFrame);
  const version = ++renderVersion;
  if (!result || !resultImage) return;
  const manifest = result,
    original = resultImage,
    key = current;
  const opts = controls(),
    limit = replayLimit;
  renderFrame = requestAnimationFrame(async () => {
    try {
      const composed = await compose(
        manifest,
        original,
        key,
        opts,
        1600,
        limit,
        () => version !== renderVersion,
      );
      if (version !== renderVersion) return;
      const canvas = $("result");
      canvas.width = composed.width;
      canvas.height = composed.height;
      canvas.getContext("2d").drawImage(composed, 0, 0);
      canvas.hidden = false;
      $("result-empty").hidden = true;
    } catch (error) {
      if (version === renderVersion) status(error.message, true);
    }
  });
}
function exportsFor() {
  const links = [
    ["Bundle", "result.zip"],
    ["Annotations", "output/annotations.json"],
    ["Response", "output/response.txt"],
    ["Native overlay", "output/overlay.png"],
  ];
  if (result.masks)
    links.push(["Alpha", "output/alpha.png"], ["Cutout", "output/cutout.png"]);
  const selected = result.instances[+$("instance").value];
  if ($("instance").value !== "all" && selected?.mask)
    links.push(["Selected mask", `output/${selected.mask}`]);
  $("exports").replaceChildren(
    ...links.map(([title, path]) => {
      const link = element("a", `${title} ↗`);
      link.href = `/files/${current}/${path}?download=1`;
      link.download = path.split("/").at(-1);
      return link;
    }),
  );
  $("exports").hidden = false;
}
function chooseInstance(value) {
  stopReplay();
  $("instance").value = value;
  document
    .querySelectorAll("#instances button")
    .forEach((b) => b.setAttribute("aria-pressed", b.dataset.index === value));
  exportsFor();
  redraw();
}
async function showResult(data, version) {
  const decoded = await image(`/files/${data.id}/input.png`);
  if (version !== epoch) return;
  inputImage = resultImage = decoded;
  result = data.result;
  $("sent-prompt").hidden = false;
  $("prompt-text").textContent = result.prompt;
  drawInput();
  raw(result.response);
  const stats = [
    ["Instances", result.instances.length],
    ["Masks", result.masks],
    ["Inference", `${result.latency_s.toFixed(2)} s`],
  ];
  if (result.count !== null) stats.push(["Count", result.count]);
  if (result.supplied_box) stats.push(["Box", "User supplied"]);
  $("stats").replaceChildren(
    ...stats.map(([label, value]) => {
      const el = element("span", label, "stat");
      el.append(element("b", value));
      return el;
    }),
  );
  $("instance").replaceChildren(
    new Option("All objects", "all"),
    ...result.instances.map(
      (item, i) => new Option(`${i + 1} · ${item.label}`, i),
    ),
  );
  $("instances").replaceChildren(
    ...result.instances.map((item, i) => {
      const button = element("button");
      const dot = element("i");
      dot.style.background = colorFor(item.label, { palette: "auto" });
      button.append(dot, document.createTextNode(`${i + 1} ${item.label}`));
      button.dataset.index = String(i);
      button.setAttribute("aria-pressed", "false");
      button.onclick = () =>
        chooseInstance($("instance").value === String(i) ? "all" : String(i));
      return button;
    }),
  );
  for (const id of ["download-view", "replay", "copy"]) $(id).disabled = false;
  exportsFor();
  redraw();
}
async function acceptRun(data, version) {
  if (version !== epoch) return;
  current = data.id;
  source = { reuse: data.id };
  try {
    localStorage.setItem("pixel-wizard-run", data.id);
  } catch {
    /* Storage can be disabled. */
  }
  locked(data.status === "running");
  $("log-button").hidden = false;
  status(
    data.status === "running"
      ? data.progress.stage || "Starting worker…"
      : data.error || data.status,
    data.status === "failed" || data.status === "interrupted",
  );
  if (data.result && data.status === "complete") {
    await showResult(data, version);
  } else if (data.image && !inputImage) {
    const decoded = await image(`/files/${data.id}/input.png`);
    if (version === epoch) {
      inputImage = decoded;
      drawInput();
    }
  }
  if (data.status === "running" && version === epoch)
    timer = setTimeout(() => poll(data.id, version), 700);
}
async function poll(key, version) {
  if (version !== epoch) return;
  try {
    await acceptRun(await api(`/api/runs/${key}`), version);
  } catch (error) {
    if (version !== epoch) return;
    status(`${error.message}. Reconnecting…`, true);
    timer = setTimeout(() => poll(key, version), 3000);
  }
}
async function openRun(key) {
  if (busy) return;
  const version = ++epoch;
  clearTimeout(timer);
  clearResult();
  locked(true);
  source = current = null;
  inputImage = null;
  $("input").hidden = true;
  $("upload-hint").hidden = false;
  try {
    const latest = await api("/api/config");
    if (version !== epoch) return;
    const data = await api(`/api/runs/${latest.active || key}`);
    if (version !== epoch) return;
    restoreSettings(data.settings);
    await acceptRun(data, version);
    if (!data.image && data.status !== "running")
      status(`${data.status} · input retained; run again to prepare it.`);
  } catch (error) {
    if (version === epoch) {
      locked(false);
      status(error.message, true);
    }
  }
}
$("run").onclick = async () => {
  if (busy) return;
  if (!source) return status("Choose an image first.", true);
  if (["box_matting", "visual_prompt"].includes(task) && !boxes.length)
    return status("Draw a reference box on the input image.", true);
  const form = new FormData();
  form.append("settings", JSON.stringify(settings()));
  if (source.file) form.append("image", source.file);
  else if (source.example) form.append("example", source.example);
  else form.append("reuse", source.reuse);
  const version = ++epoch;
  clearTimeout(timer);
  clearResult();
  current = null;
  locked(true);
  status("Starting worker…");
  try {
    await acceptRun(
      await api("/api/runs", { method: "POST", body: form }),
      version,
    );
  } catch (error) {
    if (version === epoch) {
      locked(false);
      status(error.message, true);
    }
  }
};
$("stop").onclick = async () => {
  if (!current) return;
  try {
    await api(`/api/runs/${current}/cancel`, { method: "POST" });
    status("Stopping worker and releasing model…");
  } catch (error) {
    status(error.message, true);
  }
};
$("instance").onchange = () => chooseInstance($("instance").value);
for (const id of [
  "display",
  "palette",
  "color",
  "show-boxes",
  "show-labels",
  "show-masks",
  "line-width",
  "font-size",
  "opacity",
  "threshold",
]) {
  $(id).addEventListener("input", () => {
    stopReplay();
    redraw();
  });
}
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(result.response);
    status("Response copied.");
  } catch {
    status("Clipboard unavailable; select the response to copy it.", true);
  }
};
$("download-view").onclick = async () => {
  if (!result) return;
  const manifest = result,
    original = resultImage,
    key = current,
    opts = controls();
  $("download-view").disabled = true;
  try {
    const canvas = await compose(manifest, original, key, opts, Infinity);
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
    if (!blob) throw new Error("PNG export failed");
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = `pixel-${key.slice(0, 8)}-${opts.display}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 20000);
  } catch (error) {
    status(error.message, true);
  } finally {
    $("download-view").disabled = !result;
  }
};
$("replay").onclick = () => {
  if (!result) return;
  stopReplay();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return redraw();
  const started = performance.now(),
    manifest = result;
  let previous = -1;
  function tick(now) {
    if (result !== manifest) return;
    const portion = Math.min(1, (now - started) / 1800);
    const limit = Math.ceil(portion * manifest.instances.length);
    raw(
      manifest.response.slice(0, Math.ceil(portion * manifest.response.length)),
    );
    if (limit !== previous) {
      replayLimit = limit;
      previous = limit;
      redraw();
    }
    if (portion < 1) replayFrame = requestAnimationFrame(tick);
    else {
      stopReplay();
      redraw();
      status("Replay complete · saved response.");
    }
  }
  status("Replaying saved geometry and response…");
  replayFrame = requestAnimationFrame(tick);
};
async function history() {
  const rows = await api("/api/runs");
  $("history-list").replaceChildren(
    ...rows.map((row) => {
      const wrapper = element("div", undefined, "history-row");
      const open = element("button", row.title);
      open.append(
        element(
          "small",
          `${config.tasks[row.settings.task]} · ${row.status} · ${new Date(row.created * 1000).toLocaleString()}`,
        ),
      );
      open.onclick = () => {
        $("history").close();
        void openRun(row.id);
      };
      const remove = element("button", "Delete");
      remove.disabled = row.status === "running";
      remove.onclick = async () => {
        if (
          !confirm(`Delete this run and its saved image/results?\n${row.title}`)
        )
          return;
        try {
          await api(`/api/runs/${row.id}`, { method: "DELETE" });
          if (current === row.id) {
            ++epoch;
            clearTimeout(timer);
            clearResult();
            current = source = null;
          }
          await history();
        } catch (error) {
          status(error.message, true);
        }
      };
      wrapper.append(open, remove);
      return wrapper;
    }),
  );
  if (!rows.length) $("history-list").append(element("p", "No saved runs."));
}
$("history-button").onclick = async () => {
  if (busy) return;
  $("history-list").replaceChildren(element("p", "Reading saved runs…"));
  $("history").showModal();
  try {
    await history();
  } catch (error) {
    status(error.message, true);
  }
};
$("close-history").onclick = () => $("history").close();
$("log-button").onclick = async () => {
  if (!current) return;
  try {
    const response = await fetch(`/api/runs/${current}/log`);
    if (!response.ok) throw new Error("Log unavailable");
    $("log-text").textContent = await response.text();
    $("logs").showModal();
  } catch (error) {
    status(error.message, true);
  }
};
$("close-log").onclick = () => $("logs").close();
function stopCamera() {
  ++cameraRequest;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  $("camera-video").srcObject = null;
}
$("camera-button").onclick = async () => {
  const request = ++cameraRequest;
  try {
    $("camera").showModal();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    if (!$("camera").open || request !== cameraRequest) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = stream;
    $("camera-video").srcObject = stream;
  } catch (error) {
    if (request !== cameraRequest) return;
    $("camera").close();
    status(error.message, true);
  }
};
$("close-camera").onclick = () => $("camera").close();
$("camera").addEventListener("close", stopCamera);
$("capture").onclick = () => {
  const version = epoch;
  const video = $("camera-video");
  if (!video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (blob && version === epoch)
      void chooseFile(new File([blob], "camera.png", { type: "image/png" }));
  }, "image/png");
  $("camera").close();
};
window.addEventListener("pagehide", stopCamera);
async function initialize() {
  locked(true);
  try {
    config = await api("/api/config");
    $("tasks").replaceChildren(
      ...Object.entries(config.tasks).map(([key, title]) => {
        const button = element("button", title);
        button.dataset.task = key;
        button.onclick = () => selectTask(key);
        return button;
      }),
    );
    $("examples").replaceChildren(
      ...config.examples.map((example) => {
        const button = element("button");
        const img = element("img");
        img.src = `/examples/${example.image}`;
        img.alt = example.title;
        img.loading = "lazy";
        button.append(
          img,
          element("strong", example.title),
          element("small", config.tasks[example.task]),
        );
        button.onclick = async () => {
          restoreSettings({
            ...config.defaults,
            task: example.task,
            query: example.query,
          });
          await chooseSource({ example: example.image }, img.src);
        };
        return button;
      }),
    );
    restoreSettings(config.defaults);
    $("revision").textContent =
      `WeVisOne ${config.source.revision.slice(0, 7)}`;
    locked(false);
    let saved;
    try {
      saved = localStorage.getItem("pixel-wizard-run");
    } catch {
      /* Optional saved selection. */
    }
    if (config.active || /^[a-f0-9]{32}$/.test(saved || ""))
      await openRun(config.active || saved);
  } catch (error) {
    status(error.message, true);
    locked(true);
    $("stop").hidden = true;
  }
}
void initialize();
