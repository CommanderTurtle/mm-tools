import { AssetView } from "./view.js";
const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
let cases = [],
  current = null,
  selected = null,
  viewMode = "source",
  loaded = "",
  busy = null,
  pendingPhase = "",
  timer = 0,
  refreshing = false,
  toastTimer = 0;
const labels = {
  plan: "Plan & scaffold",
  native: "Generate",
  refine: "Refine & simulate",
  package: "Package",
};
const filename = (key, path) =>
  `/files/${key}/${path.split("/").map(encodeURIComponent).join("/")}`;
const element = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
};
function toast(text) {
  $("#toast").textContent = text;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 15000);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok)
    throw Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
const view = new AssetView($("#stage"), {
  status: (text) => ($("#viewer-status").textContent = text),
  clips: (clips) => {
    const select = $("#clip");
    select.replaceChildren();
    clips.forEach((c, i) =>
      select.add(new Option(c.name || `Action ${i + 1}`, i)),
    );
    if (!clips.length) select.add(new Option("Static asset", ""));
    select.disabled = !clips.length;
    $("#play").disabled = !clips.length;
    $("#scrub").disabled = !clips.length;
  },
  time: (time, duration) => {
    $("#scrub").max = String(duration || 1);
    $("#scrub").value = String(time);
    $("#time").textContent = `${time.toFixed(2)} s`;
  },
  play: (playing) => {
    $("#play").textContent = playing ? "Ⅱ" : "▶";
    $("#play").setAttribute(
      "aria-label",
      playing ? "Pause animation" : "Play animation",
    );
  },
  joints: (joints) => {
    const panel = $("#joints");
    panel.replaceChildren();
    $("#parts-note").textContent = joints.length
      ? "Drag a bound part horizontally, or use its slider. Geometric controls; physics is recorded separately."
      : "No direct joint bindings in this asset.";
    joints.forEach((j, i) => {
      const row = element("div", undefined, "joint"),
        head = element("div", undefined, "joint-head"),
        button = element("button", j.label),
        value = element("output", `${j.value.toFixed(1)} ${j.unit}`),
        input = document.createElement("input");
      input.type = "range";
      input.min = j.min;
      input.max = j.max;
      input.step = (j.max - j.min) / 1000;
      input.value = j.value;
      input.setAttribute("aria-label", j.label);
      button.onclick = () => view.selectJoint(i);
      input.oninput = () => view.setJoint(i, Number(input.value));
      head.append(button, value);
      row.append(head, input);
      panel.append(row);
    });
  },
  joint: (index, value) => {
    const row = $("#joints").children[index];
    if (row) {
      row.querySelector("input").value = value;
      row.querySelector("output").textContent =
        `${value.toFixed(1)} ${view.joints[index].unit}`;
    }
  },
  selected: (index) =>
    [...$("#joints").children].forEach((row, i) =>
      row.classList.toggle("active", index === i),
    ),
});
function config() {
  const result = {};
  for (const name of [
    "device",
    "seed",
    "resolution",
    "texture",
    "steps",
    "min_free_gb",
    "minutes",
  ])
    result[name] = Number($("#" + name).value);
  for (const name of ["mode", "model", "blender", "feedback"])
    result[name] = $("#" + name).value;
  result.resume = $("#resume").checked;
  return result;
}
function saveSettings() {
  try {
    const c = config();
    delete c.feedback;
    localStorage.setItem("physag3d-settings", JSON.stringify(c));
  } catch {}
}
function restoreSettings() {
  try {
    const data = JSON.parse(localStorage.getItem("physag3d-settings") || "{}");
    for (const [name, value] of Object.entries(data)) {
      const input = $("#" + name);
      if (!input || name === "model") continue;
      if (input.type === "checkbox") input.checked = value;
      else input.value = value;
    }
    $("#model").dataset.saved = data.model || "";
  } catch {}
}
function library() {
  const filter = $("#search").value.toLowerCase();
  $("#count").textContent = cases.length;
  $("#cases").replaceChildren();
  for (const row of cases.filter((c) =>
    c.title.toLowerCase().includes(filter),
  )) {
    const button = element("button", undefined, "case");
    button.classList.toggle("selected", current?.id === row.id);
    const img = document.createElement("img");
    img.src = filename(row.id, "input/source.png");
    img.alt = "";
    img.loading = "lazy";
    const text = element("span", row.title),
      last = row.attempts.at(-1);
    text.append(
      element(
        "small",
        last ? `${labels[last.phase]} · ${last.status}` : "Image ready",
      ),
    );
    button.append(img, text);
    button.onclick = () => choose(row.id);
    $("#cases").append(button);
  }
}
function choose(id) {
  current = cases.find((c) => c.id === id) || null;
  selected = current?.attempts.at(-1)?.id || null;
  viewMode = "source";
  loaded = "";
  render();
  library();
}
function attempt() {
  return current?.attempts.find((a) => a.id === selected) || null;
}
function nativeAttempt() {
  const row = attempt();
  if (row?.phase === "native" && row.native_model) return row;
  return (
    current?.attempts.find((a) => a.id === row?.native && a.native_model) ||
    current?.attempts.findLast((a) => a.native_model)
  );
}
function refinedAttempt() {
  const row = attempt();
  return row?.delivery ? row : null;
}
function downloads(row) {
  const panel = $("#downloads");
  panel.replaceChildren();
  $("#limitations").replaceChildren();
  $("#video").hidden = true;
  $("#video").pause();
  $("#video").removeAttribute("src");
  $("#delivery-status").textContent = "—";
  $("#evidence").textContent = "";
  if (!current || !row) return;
  const prefix = `runs/${row.id}/`,
    data = row.delivery?.delivery;
  const link = (name, path, cls) => {
    const a = element("a", name, cls);
    a.href = filename(current.id, prefix + path) + "?download=1";
    a.download = "";
    panel.append(a);
  };
  if (row.zip) link("Download asset bundle", "asset.zip", "bundle");
  if (row.native_model) {
    link("Native GLB", "output/candidate_yup.glb");
    link("Native geometry", "output/native_raw.npz");
    link("Generation record", "output/run.json");
  }
  if (!data) return;
  $("#delivery-status").textContent = data.status.replaceAll("_", " ");
  for (const [key, label] of Object.entries({
    model: "GLB",
    blend: "Blender",
    simulator: "MJCF",
    physics_report: "Physics report",
    native_preservation_evidence: "Preservation",
  }))
    if (data[key]) link(label, "work/" + data[key]);
  link("Delivery record", "work/delivery.json");
  $("#evidence").textContent =
    `Physics: ${data.physics_status.replaceAll("_", " ")} · ${row.delivery.verification}`;
  for (const text of data.limitations || [])
    $("#limitations").append(element("li", text));
  if (data.blocker) $("#limitations").append(element("li", data.blocker));
  if (data.video) {
    $("#video").src = filename(current.id, prefix + "work/" + data.video);
    $("#video").hidden = false;
  }
}
function compare(row) {
  const panel = $("#comparison");
  panel.replaceChildren();
  if (!current) return;
  const values = [["Source", "input/source.png"]];
  const data = row?.delivery?.delivery;
  if (data?.native_render)
    values.push(["Native", `runs/${row.id}/work/${data.native_render}`]);
  if (data?.refined_render)
    values.push(["Refined", `runs/${row.id}/work/${data.refined_render}`]);
  for (const [label, path] of values) {
    const figure = element("figure"),
      img = document.createElement("img");
    img.src = filename(current.id, path);
    img.alt = label;
    img.loading = "lazy";
    figure.append(img, element("figcaption", label));
    panel.append(figure);
  }
}
function showView() {
  const native = nativeAttempt(),
    refined = refinedAttempt();
  if (
    (viewMode === "native" && !native) ||
    (viewMode === "refined" && !refined?.delivery?.delivery.model)
  )
    viewMode = "source";
  $$("#views button").forEach((b) => {
    b.classList.toggle("selected", b.dataset.view === viewMode);
    b.disabled =
      b.dataset.view === "native"
        ? !native
        : b.dataset.view === "refined"
          ? !refined?.delivery?.delivery.model
          : !current;
  });
  $("#source").hidden = !current || viewMode !== "source";
  $("#empty").hidden = !!current;
  $("#stage").style.visibility = viewMode === "source" ? "hidden" : "visible";
  $$(".viewport-tools button").forEach(
    (b) => (b.disabled = viewMode === "source"),
  );
  $("#capture").disabled = viewMode === "source";
  if (current) $("#source").src = filename(current.id, "input/source.png");
  let url = "",
    bindings = { joints: [] };
  if (viewMode === "native")
    url = filename(current.id, `runs/${native.id}/output/candidate_yup.glb`);
  if (viewMode === "refined") {
    url = filename(
      current.id,
      `runs/${refined.id}/work/${refined.delivery.delivery.model}`,
    );
    bindings = refined.delivery.interaction;
  }
  $("#motion-kind").textContent =
    viewMode === "refined"
      ? refined.delivery.delivery.motion_kind.replaceAll("_", " ")
      : viewMode === "native"
        ? "Static neural candidate"
        : current
          ? "Original observation"
          : "Awaiting an image";
  if (url !== loaded) {
    loaded = url;
    $("#wire").setAttribute("aria-pressed", "false");
    $("#orbit").setAttribute("aria-pressed", "false");
    if (url) view.load(url, bindings);
    else {
      view.clear();
      $("#viewer-status").textContent = "";
    }
  }
}
function render() {
  const row = attempt();
  $("#title").textContent = current?.title || "An object, in motion.";
  const select = $("#attempt");
  select.replaceChildren();
  for (const a of current?.attempts || []) {
    select.add(
      new Option(
        `${labels[a.phase]} · ${new Date(a.created * 1000).toLocaleTimeString()} · ${a.status}`,
        a.id,
      ),
    );
  }
  if (!select.options.length) select.add(new Option("No attempts yet", ""));
  select.value = selected || "";
  for (const b of $$(".stages button")) {
    const phase = b.dataset.phase;
    b.disabled = !current || !!busy;
    b.classList.toggle(
      "done",
      !!current?.attempts.some(
        (a) => a.phase === phase && ["complete", "partial"].includes(a.status),
      ),
    );
    b.classList.toggle(
      "running",
      current?.id === busy?.case &&
        row?.id === busy?.attempt &&
        row?.phase === phase,
    );
  }
  $("#cancel").hidden = !busy;
  $("#activity").textContent = busy
    ? "Stage running · files saved locally"
    : "Ready";
  $("#stage-status").textContent = row
    ? `${row.status}${row.error ? " · " + row.error : ""}`
    : "";
  downloads(row);
  compare(row);
  showView();
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const rows = await api("/api/cases");
    cases = rows;
    busy = null;
    for (const c of rows) {
      const a = c.attempts.find((a) => a.status === "running");
      if (a) busy = { case: c.id, attempt: a.id };
    }
    if (current) {
      const next = rows.find((c) => c.id === current.id);
      const previous = attempt()?.status;
      current = next || null;
      const updated = attempt();
      if (previous !== updated?.status) {
        loaded = "";
        render();
      } else {
        $("#stage-status").textContent = updated
          ? `${updated.status} · ${updated.progress.message || updated.error || ""}`
          : "";
        $("#cancel").hidden = !busy;
        $(".stages")
          .querySelectorAll("button")
          .forEach((b) => (b.disabled = !current || !!busy));
        $("#activity").textContent = busy
          ? updated?.progress.message || "Stage running"
          : "Ready";
      }
    } else if (rows.length) choose(rows[0].id);
    library();
    if (current && selected && $("#log-panel").open) {
      const response = await fetch(
        `/api/cases/${current.id}/runs/${selected}/log`,
      );
      if (response.ok) {
        const text = await response.text();
        const log = $("#log");
        const bottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
        log.textContent = text;
        if (bottom) log.scrollTop = log.scrollHeight;
      }
    }
  } catch (error) {
    toast(error.message);
  } finally {
    refreshing = false;
    clearTimeout(timer);
    timer = setTimeout(refresh, document.hidden ? 10000 : busy ? 1500 : 6000);
  }
}
async function upload(file) {
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  $("#upload").disabled = true;
  try {
    const row = await api("/api/cases", { method: "POST", body: form });
    cases.unshift(row);
    choose(row.id);
  } catch (error) {
    toast(error.message);
  } finally {
    $("#upload").disabled = false;
    $("#upload").value = "";
  }
}
$("#upload").onchange = () => upload($("#upload").files[0]);
const drop = $("#drop");
drop.ondragover = (e) => {
  e.preventDefault();
  drop.classList.add("drag");
};
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  upload(e.dataTransfer.files[0]);
};
$("#search").oninput = library;
$("#attempt").onchange = () => {
  selected = $("#attempt").value;
  loaded = "";
  render();
};
$("#mode").onchange = () => {
  $("#nvidia-ready-label").hidden = $("#mode").value !== "nvidia";
  $("#model").replaceChildren(new Option("Select after refresh", ""));
  saveSettings();
};
$("#discover").onclick = async () => {
  const button = $("#discover");
  button.disabled = true;
  button.textContent = "Reading OMP…";
  try {
    const result = await post("/api/models", {
      mode: $("#mode").value,
      ready: $("#nvidia-ready").checked,
    });
    const select = $("#model"),
      previous = select.value || select.dataset.saved;
    select.replaceChildren(new Option("Select model", ""));
    for (const m of result.models)
      select.add(new Option(`${m.name || m.id} · ${m.provider}`, m.selector));
    if (result.models.some((m) => m.selector === previous))
      select.value = previous;
    $("#model-note").textContent =
      result.message ||
      `${result.models.length} image-capable models · exact provider/id selection`;
    if (result.message) toast(result.message);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Refresh models";
  }
};
for (const input of $$(
  ".inspector input,.inspector select,.setup input,.setup select",
))
  input.addEventListener("change", saveSettings);
for (const button of $$(".stages button"))
  button.onclick = () => {
    pendingPhase = button.dataset.phase;
    if (pendingPhase === "package") {
      start();
      return;
    }
    $("#handoff-title").textContent = labels[pendingPhase];
    $("#handoff-text").textContent =
      pendingPhase === "native"
        ? "Unload the external agent model to free the selected GPU. TRELLIS runs once and releases its allocations when this stage ends."
        : "Load the model selected in Runtime. Generation is stopped before this stage. OMP will read the image and saved artifacts, then use Blender and Python for this stage.";
    $("#agent-warning").hidden = pendingPhase === "native";
    $("#ready").checked = false;
    $("#confirm").disabled = true;
    $("#handoff").showModal();
  };
$("#ready").onchange = () => ($("#confirm").disabled = !$("#ready").checked);
$("#handoff").addEventListener("close", () => {
  if ($("#handoff").returnValue === "run") start();
});
async function start() {
  if (!current) return;
  try {
    saveSettings();
    const row = await post(`/api/cases/${current.id}/run`, {
      phase: pendingPhase,
      attempt: selected,
      settings: config(),
      ready: true,
    });
    current = row;
    selected = row.attempts.at(-1).id;
    busy = { case: row.id, attempt: selected };
    $("#log-panel").open = true;
    loaded = "";
    render();
  } catch (error) {
    toast(error.message);
  }
}
$("#cancel").onclick = async () => {
  try {
    await post("/api/cancel", {});
    $("#activity").textContent = "Stopping owned processes…";
  } catch (error) {
    toast(error.message);
  }
};
for (const button of $$("#views button"))
  button.onclick = () => {
    viewMode = button.dataset.view;
    showView();
  };
$("#home").onclick = () => view.home();
$("#reset-joints").onclick = () => view.resetJoints();
for (const name of ["wire", "orbit"])
  $("#" + name).onclick = () => {
    const enabled = $("#" + name).getAttribute("aria-pressed") !== "true";
    $("#" + name).setAttribute("aria-pressed", String(enabled));
    view[name](enabled);
  };
$("#play").onclick = () => view.play();
$("#clip").onchange = () => view.selectClip(Number($("#clip").value));
$("#scrub").oninput = () => view.seek(Number($("#scrub").value));
$("#speed").onchange = () => (view.speed = Number($("#speed").value));
$("#loop").onchange = () => view.setLoop($("#loop").checked);
$("#capture").onclick = () => {
  try {
    const a = document.createElement("a");
    a.download = (current?.title || "asset") + ".png";
    a.href = view.snapshot();
    a.click();
  } catch (error) {
    toast(error.message);
  }
};
$("#log-panel").ontoggle = () => {
  if ($("#log-panel").open) {
    clearTimeout(timer);
    refresh();
  }
};
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    clearTimeout(timer);
    refresh();
  }
});
restoreSettings();
$("#nvidia-ready-label").hidden = $("#mode").value !== "nvidia";
render();
refresh();
