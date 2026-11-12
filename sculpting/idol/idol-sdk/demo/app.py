import json
import random
from pathlib import Path

import gradio as gr

try:
    import spaces  # Hugging Face ZeroGPU; imported before the model so its CUDA placement is deferred

    gpu = spaces.GPU(duration=120)
except ImportError:
    def gpu(fn):
        return fn

import model  # noqa: E402

# The splat viewer opens every object at orbit radius 5; this factor zooms it in on load (5.5/10 zoom instead of 3/10).
VIEWER_ZOOM = 5.5 / 3


NAME = "Idol"
TAGLINE = "<strong>One image in, a 3D object out.</strong> Idol turns a single picture of an object into " \
          "3D Gaussians splat. Pick an example below, or <em>upload your own.</em>"
LINKS = {"Code": "https://github.com/speridlabs/idol-sdk",
         "Weights": "https://huggingface.co/speridlabs/idol"}

ROOT = Path(__file__).parent
PRESETS = json.loads((ROOT / "examples/presets.json").read_text())
MODE_LABEL = {mode: label for label, mode in model.MODES.items()}
# the example shown when the page opens, with its splat and model input generated beforehand with its own settings
DEFAULT_EXAMPLE = next(p for p in PRESETS if p["image"] == "examples/obj-backpack.webp")
DEFAULT_PLY = ROOT / "examples/default-obj-backpack.ply"
GAUSSIAN_LABEL = {count: label for label, count in model.GAUSSIANS.items()}


@gpu
def infer(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians):
    if image is None:
        raise gr.Error("Upload an image or pick an example first.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    ply, prepared = model.generate(image, model.MODES[mode], int(steps), cfg_scale, rescale_t, int(seed),
                                   model.GAUSSIANS[gaussians])
    return ply, prepared, seed


def meta(gaussians, steps):
    return f'<div class="sl-meta"><span>{gaussians} Gaussians · {int(steps)} steps</span></div>'


def run(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians):
    ply, _, seed = infer(image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians)
    return ply, seed, meta(gaussians, steps)


def show_default_example():
    if not DEFAULT_PLY.exists():  # the .ply ships with the Space only; without it the viewer starts empty
        return None, ""
    p = DEFAULT_EXAMPLE
    return str(DEFAULT_PLY), meta(GAUSSIAN_LABEL[p["num_gaussians"]], p["steps"])


def shuffle_examples():
    # every visitor sees the examples in a new random order; the order is kept so a click maps back to its preset
    order = random.sample(range(len(PRESETS)), len(PRESETS))
    return [str(ROOT / PRESETS[i]["image"]) for i in order], order


def pick(order, evt: gr.SelectData):
    p = PRESETS[order[evt.index]]
    return (str(ROOT / p["image"]), MODE_LABEL[p["mode"]], p["steps"], p["cfg"], p["rescale_t"], p["seed"], False,
            GAUSSIAN_LABEL[p["num_gaussians"]])


FONT = [gr.themes.GoogleFont("Inter", weights=(400, 500, 600, 700)), "ui-sans-serif", "system-ui", "sans-serif"]
MONO = [gr.themes.GoogleFont("IBM Plex Mono", weights=(400, 500, 600, 700)), "ui-monospace", "monospace"]
theme = gr.themes.Base(font=FONT, font_mono=MONO, radius_size=gr.themes.sizes.radius_none,
                       text_size=gr.themes.sizes.text_lg).set(
    body_background_fill="#ffffff", body_background_fill_dark="#000000",
    body_text_color="#000000", body_text_color_dark="#ffffff",
    block_background_fill="#ffffff", block_background_fill_dark="#000000",
    background_fill_primary="#ffffff", background_fill_primary_dark="#000000",
    background_fill_secondary="#f7f7f7", background_fill_secondary_dark="#111111",
    border_color_primary="#e3e3e3", border_color_primary_dark="rgba(255,255,255,0.15)",
    block_border_color="#e3e3e3", block_border_color_dark="rgba(255,255,255,0.15)", block_shadow="none",
    input_background_fill="#ffffff", input_background_fill_dark="#000000",
    input_border_color="#d4d4d4", input_border_color_dark="rgba(255,255,255,0.25)",
    input_border_color_focus="#000000", input_border_color_focus_dark="#ffffff",
    button_primary_background_fill="#000000", button_primary_background_fill_dark="#ffffff",
    button_primary_background_fill_hover="#d94a1f", button_primary_background_fill_hover_dark="#d94a1f",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#000000",
    color_accent="#d94a1f", color_accent_soft="#fbe7e0", slider_color="#d94a1f", slider_color_dark="#d94a1f",
    checkbox_background_color_selected="#000000", checkbox_background_color_selected_dark="#ffffff",
)
CSS = """
:root { --sl-fg: #000; --sl-fg-2: #444; --sl-fg-3: #6b6b6b; --sl-rule: #e3e3e3; }
.dark { --sl-fg: #fff; --sl-fg-2: rgba(255,255,255,.75); --sl-fg-3: rgba(255,255,255,.55);
  --sl-rule: rgba(255,255,255,.15); }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 32px !important; }
footer { display: none !important; }
.sl-head { padding: 40px 0 28px; border-bottom: 1px solid var(--sl-rule); margin-bottom: 8px; }
.sl-label, .sl-meta, .sl-links a, .sl-links button, .sl-section, .block-label, .label-wrap span, label > span {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important; text-transform: uppercase;
  letter-spacing: .12em; font-size: 13px !important; font-weight: 600 !important; color: var(--sl-fg-3) !important; }
.gradio-container { font-weight: 500; }
.sl-head h1 { margin: 14px 0 0; font-size: clamp(56px, 6.4vw, 104px); font-weight: 700; line-height: .86;
  letter-spacing: -0.055em; color: var(--sl-fg); }
.sl-row { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px 40px; }
.sl-head p { margin: 18px 0 0; max-width: 60ch; font-size: 19px; font-weight: 500; line-height: 1.5; color: var(--sl-fg-2); }
.sl-head p strong { color: var(--sl-fg); font-weight: 600; }
.sl-head p em { color: #d94a1f; font-style: normal; font-weight: 600; }
.sl-links { display: flex; gap: 10px; }
.sl-links a, .sl-links button { display: inline-flex; align-items: center; height: 40px; padding: 0 20px;
  border: 1px solid var(--sl-fg); background: transparent; color: var(--sl-fg) !important; text-decoration: none;
  cursor: pointer; transition: background .15s, color .15s; }
.sl-links a:hover, .sl-links button:hover { background: var(--sl-fg); color: var(--body-background-fill) !important; }
#sl-theme { width: 44px; padding: 0; justify-content: center; font-size: 19px !important; letter-spacing: 0; }
#sl-theme::before { content: "☾"; }
.dark #sl-theme::before { content: "☀"; }
.sl-meta { display: flex; gap: 16px; white-space: nowrap; padding-top: 6px; font-size: 12px !important;
  font-weight: 400 !important; color: var(--sl-fg-2) !important; }
div.sl-section { font-weight: 700 !important; }
.sl-section { border-top: 1px solid var(--sl-rule); padding-top: 28px !important; margin-top: 28px; }
#generate { height: 52px; font-size: 17px; font-weight: 600; letter-spacing: .01em; }
#result { background: #0c0c0c; }
#result img { object-fit: contain; }
#presets .thumbnail-item { border: 0 !important; box-shadow: none !important; overflow: hidden; }
#presets .thumbnail-item img { transition: transform .6s ease-out; }
#presets .thumbnail-item:hover img { transform: scale(1.04); }
#presets .caption-label { display: none; }
#presets .grid-wrap { max-height: none !important; height: auto !important; overflow: visible !important; }
#sl-settings > .label-wrap span { color: var(--sl-fg) !important; font-weight: 600; }
#presets .thumbnail-item.selected { outline: 2px solid #d94a1f; outline-offset: -2px; }
/* only the input image stretches, so the input column ends where the viewer ends; the viewer keeps its height */
#sl-main > .column > *, #sl-main > .column > .form > * { flex-grow: 0 !important; }
#sl-main > .column:first-child > #sl-input { flex: 1 1 0 !important; height: auto !important; min-height: 360px; }
#sl-input .image-container, #sl-input .upload-container { min-height: 0; }
/* icons: larger and drawn with a thicker stroke */
.icon-button > div.small, .icon-button > div.x-small { width: 20px; height: 20px; }
#result .icon-button-wrapper button > div { width: 24px; height: 24px; }
.source-selection button svg, .icon-button svg { stroke-width: 2.5; }
.source-selection button { width: 28px; height: 28px; }
/* sliders: same width for the min/max labels and the number boxes, so all tracks start and end at the same x */
#sl-settings .slider_input_container .min_value, #sl-settings .slider_input_container .max_value {
  flex: 0 0 2.6em; }
#sl-settings .slider_input_container .min_value { text-align: left; }
#sl-settings .slider_input_container .max_value { text-align: right; }
#sl-settings .tab-like-container input[type="number"] { width: 64px; flex: 0 0 64px; }
@media (max-width: 760px) {
  .sl-head { padding-top: 24px; }
  #presets .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
  #sl-meta-spacer { display: none !important; }
}
"""
# Runs once when the page loads. Three jobs: (1) file requests carry the session cookies: the splat viewer fetches
# the .ply with `credentials: "omit"`, which a private Space answers with a 404, so the generated object never
# appeared; (2) each loaded splat opens zoomed in by VIEWER_ZOOM; (3) dark by default; the header toggle flips
# gradio's `dark` class and remembers the choice.
THEME_JS = """() => {
  const VIEWER_ZOOM = VIEWER_ZOOM_VALUE;
  const originalFetch = window.fetch.bind(window);
  window.fetch = (resource, options) => {
    const url = typeof resource === 'string' ? resource : (resource && resource.url) || '';
    if (url.includes('/gradio_api/file=')) options = Object.assign({}, options, { credentials: 'same-origin' });
    // the splat viewer fetches the .ply right after creating its orbit camera (radius 5): zoom that camera in with
    // one wheel step. Its wheel handler does radius += deltaY * 0.025 * (0.1 + 0.9 * (radius - 0.1) / 29.9).
    const canvas = document.querySelector('#result canvas');
    if (canvas && url.split('?')[0].endsWith('.ply')) {
      const startRadius = 5, targetRadius = startRadius / VIEWER_ZOOM;
      const wheelScale = 0.025 * (0.1 + 0.9 * (startRadius - 0.1) / 29.9);
      canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: (targetRadius - startRadius) / wheelScale, cancelable: true }));
    }
    return originalFetch(resource, options);
  };
  const set = (dark) => { document.body.classList.toggle('dark', dark); localStorage.setItem('sl-theme', dark ? 'dark' : 'light'); };
  set(localStorage.getItem('sl-theme') !== 'light');
  document.addEventListener('click', (e) => { if (e.target.closest('#sl-theme')) set(!document.body.classList.contains('dark')); });
}""".replace("VIEWER_ZOOM_VALUE", f"{VIEWER_ZOOM:.6f}")
# Page padding on phones. It lives in <head> because gradio scopes every rule inside the custom CSS under the page
# container, so a rule that targets the container itself (or the <main> around it) never matches from there.
PHONE_HEAD = """<style>@media (max-width: 760px) {
  .gradio-container { padding: 0 16px !important; }
  main.fillable.app { padding-left: 0 !important; padding-right: 0 !important; }
}</style>"""
LINK_HTML = "".join(f'<a href="{u}" target="_blank">{t}</a>' for t, u in LINKS.items())
HEAD = f"""
<div class="sl-head">
  <span class="sl-label">Speridlabs · Research</span>
  <div class="sl-row">
    <h1>{NAME}</h1>
    <div class="sl-links">{LINK_HTML}<button id="sl-theme" type="button" aria-label="Toggle light/dark"></button></div>
  </div>
  <p>{TAGLINE}</p>
</div>
"""

first = DEFAULT_EXAMPLE
with gr.Blocks(title=f"{NAME} · Speridlabs", theme=theme, css=CSS, js=THEME_JS, head=PHONE_HEAD) as demo:
    gr.HTML(HEAD)
    with gr.Row(equal_height=True, elem_id="sl-main"):
        with gr.Column(scale=5):
            image = gr.Image(value=str(ROOT / first["image"]), type="pil", image_mode="RGBA", label="Input image",
                             sources=["upload", "clipboard"], height=360, elem_id="sl-input")
            mode = gr.Radio(list(model.MODES), value=MODE_LABEL[first["mode"]], label="Input type")
            run_btn = gr.Button("Generate", variant="primary", elem_id="generate")
            with gr.Accordion("Settings", open=False, elem_id="sl-settings"):
                gaussians = gr.Dropdown(list(model.GAUSSIANS), value=GAUSSIAN_LABEL[first["num_gaussians"]],
                                        label="Gaussians")
                steps = gr.Slider(10, 100, value=first["steps"], step=1, label="Steps")
                cfg_scale = gr.Slider(0.0, 10.0, value=first["cfg"], step=0.1, label="Guidance scale")
                rescale_t = gr.Slider(1.0, 6.0, value=first["rescale_t"], step=0.1, label="Time shift")
                seed = gr.Number(value=first["seed"], precision=0, label="Seed")
                randomize = gr.Checkbox(value=False, label="Random seed")
        with gr.Column(scale=7):
            result = gr.Model3D(show_label=False, elem_id="result", height="min(640px, 92vw)",
                                clear_color=(0.05, 0.05, 0.05, 1.0))
    with gr.Row():
        gr.Column(scale=5, min_width=0, elem_id="sl-meta-spacer")
        with gr.Column(scale=7):
            info = gr.HTML("")
    gr.HTML('<div class="sl-section">Examples — click to load the image and its settings</div>')
    presets = gr.Gallery(None, columns=6, rows=4, height="auto",
                         object_fit="contain", allow_preview=False, show_label=False, show_share_button=False,
                         show_download_button=False, elem_id="presets")
    order = gr.State()
    inputs = [image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians]
    run_btn.click(run, inputs, [result, seed, info], show_progress_on=result)
    presets.select(pick, order, [image, mode, steps, cfg_scale, rescale_t, seed, randomize, gaussians])
    demo.load(shuffle_examples, None, [presets, order], show_progress="hidden")
    # loaded after the page script, so the viewer's .ply request already carries the cookies and gets the zoom
    demo.load(show_default_example, None, [result, info], show_progress="hidden")

if __name__ == "__main__":
    demo.queue().launch()
