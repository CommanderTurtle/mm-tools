import { AssetView } from "./view.js";
const $ = (s) => document.querySelector(s);
const url = (path) => "../" + path.split("/").map(encodeURIComponent).join("/");
const view = new AssetView($("#stage"), {
  status: (text) => ($("#status").textContent = text),
  clips: (clips) => {
    const select = $("#clips");
    select.replaceChildren();
    clips.forEach((c, i) =>
      select.add(new Option(c.name || `Action ${i + 1}`, i)),
    );
    $("#play").disabled = !clips.length;
    $("#scrub").disabled = !clips.length;
  },
  play: (playing) => ($("#play").textContent = playing ? "Pause" : "Play"),
  time: (time, duration) => {
    $("#scrub").max = duration || 1;
    $("#scrub").value = time;
    $("#time").textContent = time.toFixed(2) + " s";
  },
  joints: (joints) => {
    const panel = $("#joints");
    panel.replaceChildren();
    joints.forEach((j, i) => {
      const row = document.createElement("div");
      row.className = "joint";
      const label = document.createElement("label"),
        output = document.createElement("output"),
        input = document.createElement("input");
      label.textContent = j.label;
      output.textContent = j.value.toFixed(1) + " " + j.unit;
      label.append(output);
      input.type = "range";
      input.min = j.min;
      input.max = j.max;
      input.step = (j.max - j.min) / 1000;
      input.value = j.value;
      input.setAttribute("aria-label", j.label);
      input.oninput = () => view.setJoint(i, Number(input.value));
      row.append(label, input);
      panel.append(row);
    });
  },
  joint: (i, value) => {
    const row = $("#joints").children[i];
    if (row) {
      row.querySelector("input").value = value;
      row.querySelector("output").textContent =
        value.toFixed(1) + " " + view.joints[i].unit;
    }
  },
});
$("#play").onclick = () => view.play();
$("#home").onclick = () => view.home();
let wire = false;
$("#wire").onclick = () => {
  wire = !wire;
  view.wire(wire);
};
$("#clips").onchange = () => view.selectClip(Number($("#clips").value));
$("#scrub").oninput = () => view.seek(Number($("#scrub").value));
try {
  const response = await fetch("./scene.json");
  if (!response.ok) throw Error("Missing scene.json");
  const { delivery: data, interaction } = await response.json();
  $("#title").textContent = data.title || "Object";
  $("#kind").textContent =
    `${data.motion_kind.replaceAll("_", " ")} · physics: ${data.physics_status}`;
  $("#record").textContent = JSON.stringify(data, null, 2);
  $("#limitations").textContent = (data.limitations || []).join("\n");
  if (data.input) $("#source").src = url(data.input);
  if (data.model) {
    $("#download").href = url(data.model);
    $("#download").download = "";
    await view.load(url(data.model), interaction);
  } else {
    $("#download").hidden = true;
    $("#status").textContent = "Partial delivery · mesh unavailable";
  }
  if (data.video) {
    $("#video").src = url(data.video);
    $("#video").hidden = false;
  }
  for (const [key, label] of [
    ["native_render", "Native"],
    ["refined_render", "Refined"],
  ]) {
    if (!data[key]) continue;
    const figure = document.createElement("figure"),
      img = document.createElement("img"),
      caption = document.createElement("figcaption");
    img.src = url(data[key]);
    img.alt = label;
    caption.textContent = label;
    figure.append(img, caption);
    $("#compare").append(figure);
  }
} catch (error) {
  $("#status").textContent =
    error.message + " · Serve this extracted folder over localhost.";
}
