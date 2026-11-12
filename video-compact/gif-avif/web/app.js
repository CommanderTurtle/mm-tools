const $ = (selector) => document.querySelector(selector);

let file = null;
let job = null;
let timer = null;
let format = "gif";
let sourceUrl = null;

const formatBytes = (value) => {
  if (value == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index ? 2 : 0)} ${units[index]}`;
};

fetch("/api/capabilities", { cache: "no-store" })
  .then((response) => response.json())
  .then((capabilities) => {
    $("#runtime").textContent = `GIF${capabilities.gif_cuda ? " + RTX prep" : ""} · AVIF · WebM${capabilities.webm_nvenc ? " + RTX" : ""} · local only`;
    if (!capabilities.gif_cuda) {
      $("#gif-engine option[value='cuda']").disabled = true;
    }
    if (!capabilities.avif_nvenc) {
      $("#avif-engine option[value='nvenc']").disabled = true;
    }
    if (!capabilities.webm_nvenc) {
      $("#webm-engine option[value='nvenc']").disabled = true;
      $("#webm-engine").value = "software";
    }
  });

function resetOutput() {
  $("#download").removeAttribute("href");
  $("#download").classList.add("disabled");
  $("#preview").removeAttribute("src");
  $("#preview").style.display = "none";
  $("#video-preview").removeAttribute("src");
  $("#video-preview").style.display = "none";
}

function choose(selected) {
  file = selected;
  resetOutput();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(selected);
  $("#source").src = sourceUrl;
  $("#source").style.display = "block";
  $("#source-meta").textContent = `${selected.name} · ${formatBytes(selected.size)}`;
  $("#run").disabled = false;
  $("#status").textContent = "Ready to convert.";
  $("#status").classList.remove("error");
}

$("#file").addEventListener("change", (event) => event.target.files[0] && choose(event.target.files[0]));
const drop = $("#drop");
for (const name of ["dragenter", "dragover"]) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("drag");
  });
}
for (const name of ["dragleave", "drop"]) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
  });
}
drop.addEventListener("drop", (event) => event.dataTransfer.files[0] && choose(event.dataTransfer.files[0]));

$("#use-start").addEventListener("click", () => $("#start").value = $("#source").currentTime.toFixed(2));
$("#use-end").addEventListener("click", () => $("#end").value = $("#source").currentTime.toFixed(2));

document.querySelectorAll("[data-format]").forEach((button) => button.addEventListener("click", () => {
  format = button.dataset.format;
  document.querySelectorAll("[data-format]").forEach((item) => item.classList.toggle("active", item === button));
  $("#gif-options").classList.toggle("hidden", format !== "gif");
  $("#avif-options").classList.toggle("hidden", format !== "avif");
  $("#webm-options").classList.toggle("hidden", format !== "webm");
}));

$("#avif-quality").addEventListener("input", (event) => $("#quality-out").textContent = event.target.value);
$("#webm-quality").addEventListener("input", (event) => $("#webm-quality-out").textContent = event.target.value);

$("#run").addEventListener("click", async () => {
  if (!file) return;
  resetOutput();
  const data = new FormData();
  data.set("video", file);
  data.set("format", format);
  const ids = [
    "start", "end", "width", "fps", "speed", "crop-x", "crop-y",
    "crop-width", "crop-height", "rotate", "flip", "loop", "gif-colors",
    "gif-dither", "gif-palette", "gif-alpha", "gif-engine",
    "avif-quality", "avif-effort", "avif-engine", "webm-quality", "webm-engine",
  ];
  ids.forEach((id) => {
    const key = id.replaceAll("-", "_").replace("gif_alpha", "gif_alpha_threshold");
    data.set(key, $("#" + id).value);
  });
  data.set("avif_10bit", $("#avif-10bit").checked ? "true" : "false");
  data.set("webm_audio", $("#webm-audio").checked ? "true" : "false");

  $("#run").disabled = true;
  $("#stop").disabled = false;
  $("#status").textContent = "Streaming to local workspace…";
  $("#status").classList.remove("error");
  $("#bar").style.width = "1%";
  const response = await fetch("/api/jobs", { method: "POST", body: data });
  const payload = await response.json();
  if (!response.ok) {
    fail(payload.detail || "Could not start");
    return;
  }
  job = payload;
  clearInterval(timer);
  timer = setInterval(poll, 700);
  poll();
});

async function poll() {
  if (!job) return;
  const response = await fetch(`/api/jobs/${job.id}`, { cache: "no-store" });
  if (!response.ok) return;
  job = await response.json();
  const jobFormat = job.options.format;
  $("#bar").style.width = `${Math.max(1, job.progress * 100)}%`;
  $("#status").textContent = {
    queued: "Queued locally…",
    running: `Encoding ${jobFormat.toUpperCase()}…`,
    complete: "Complete.",
    failed: job.error,
    stopped: "Stopped.",
  }[job.status] || job.status;
  $("#status").classList.toggle("error", job.status === "failed");
  $("#log").textContent = job.logs.length ? job.logs.join("\n") : "FFmpeg is running quietly.";
  if (job.before) $("#before").textContent = formatBytes(job.before.bytes);
  if (job.after) {
    $("#after").textContent = formatBytes(job.after.bytes);
    $("#dimensions").textContent = `${job.after.width}×${job.after.height}`;
    $("#actual-fps").textContent = job.after.fps.toFixed(1);
  }
  if (job.output_ready) {
    const url = `/api/jobs/${job.id}/output`;
    if (jobFormat === "webm") {
      $("#video-preview").src = url;
      $("#video-preview").style.display = "block";
    } else {
      $("#preview").src = url;
      $("#preview").style.display = "block";
    }
    $("#download").textContent = `Download ${jobFormat.toUpperCase()}`;
    $("#download").href = `${url}?download=true`;
    $("#download").classList.remove("disabled");
  }
  if (!["queued", "running"].includes(job.status)) {
    clearInterval(timer);
    $("#stop").disabled = true;
    $("#run").disabled = false;
  }
}

$("#stop").addEventListener("click", async () => {
  if (job) await fetch(`/api/jobs/${job.id}/stop`, { method: "POST" });
});

function fail(message) {
  $("#status").textContent = message;
  $("#status").classList.add("error");
  $("#stop").disabled = true;
  $("#run").disabled = false;
}
