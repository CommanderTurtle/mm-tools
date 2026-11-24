"""A web demo in the browser (Gradio): text-to-image, image-to-image and dense prediction with one loaded model.

    pip install gradio -c requirements.txt     # keeps the pinned versions
    python app.py                     # weights from the Hugging Face Hub
    python app.py --weights weights/jeveration.pt --decoder weights/decoder512.pt --source-decoder weights/source_decoder512.pt
    python app.py --share             # also print a temporary public link

The time shown under a result is the model's time for that request on this machine. Each tab lists examples to try
(pictures in examples/).
"""
import argparse
import os
import time

import numpy as np
import torch

try:
    import spaces                      # Hugging Face ZeroGPU: a GPU is attached only while a decorated function runs
except ImportError:
    spaces = None

from jeveration import config
from jeveration.dense import depth_view, normal_view, seg_view

WEIGHTS_REPO = "linlinlinlinsdada/Jeveration"
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
T2I_EXAMPLES = [                                            # each looks good at every seed tried
    "An oil painting of sunflowers in a blue ceramic vase, thick brush strokes.",
    "An owl with bright yellow eyes perched on a mossy branch at twilight, warm golden backlight, detailed feathers.",
    "An impressionist oil painting of a bouquet of peonies in a vase.",
    "A stone cottage with a thatched roof and a flower garden in the countryside.",
    "A watercolor painting of a sailboat on a calm sea at sunset.",
    "A lone pine tree on a small island in a calm lake at dawn.",
    "A watercolor painting of a snowy village at dusk with warm lit windows.",
    "A burger with melted cheese, lettuce and tomato on a wooden board, close-up.",
    "A painting in the style of Van Gogh of a starry night over a quiet village with a swirling sky.",
    "A small white chapel on a green alpine meadow with snowy peaks behind.",
    "A spirit fox made of glowing blue flames sitting in a dark forest, magical atmosphere.",
    "A river winding through a green valley with snowy mountains in the distance.",
    "An oil painting of a mountain lake in autumn with golden trees, visible brush strokes.",
    "A bouquet of tulips in a glass vase on a windowsill, soft morning light.",
    "A lion's face with a golden mane at sunset, savanna grass, cinematic lighting.",
    "A waterfall cascading into an emerald pool in a tropical jungle, sun rays through the mist.",
    "A Chinese ink wash painting of misty mountains and a lone boat on a river.",
    "A stack of fluffy pancakes with blueberries and maple syrup dripping down, butter on top, morning light.",
    "A snowy mountain cabin with warm glowing windows at blue hour, surrounded by pine trees.",
    "A close-up portrait of a woman wearing a flower crown in a meadow, dreamy pastel tones.",
]
I2I_EXAMPLES = [                                            # (file in examples/, task, caption, the file is a photo)
    ("hed_puppy.png", "hed", "A fluffy white Samoyed puppy smiling in a field of yellow flowers on a sunny afternoon.", False),
    ("hed_tulips.png", "hed", "A bouquet of tulips in a glass vase on a windowsill, soft morning light.", False),
    ("photo_owl.jpg", "hed", "An owl with bright yellow eyes perched on a mossy branch at twilight, warm golden backlight, "
                             "detailed feathers.", True),
    ("depth_pancakes.png", "depth",
     "A stack of fluffy pancakes with blueberries and maple syrup dripping down, butter on top, morning light.", False),
    ("photo_croissant.jpg", "depth", "A golden croissant on a marble table next to a cup of coffee, soft morning light.", True),
    ("gray_burger.jpg", "colorize", "", False), ("gray_pancakes.jpg", "colorize", "", False),
    ("photo_boathouse.jpg", "anime", "", False), ("photo_waterfall.jpg", "anime", "", False),
    ("photo_pine.jpg", "ghibli", "", False), ("photo_valley.jpg", "ghibli", "", False),
    ("photo_chapel.jpg", "watercolor", "", False), ("photo_snowcabin.jpg", "watercolor", "", False),
]
DENSE_EXAMPLES = ["room_1.png", "room_2.png", "room_3.png", "room_4.png"]
I2I_HELP = {
    "hed": "Input: a soft edge map (white edges on black), or a photo with the box below ticked. Add a caption.",
    "depth": "Input: a depth map (bright = near), or a photo with the box below ticked. Add a caption.",
    "colorize": "Input: a photo. It is read as a grey picture and colorized.",
    "anime": "Input: a photo.", "ghibli": "Input: a photo.", "watercolor": "Input: a photo.",
}


def timed(pipe, call):
    """run call() -> (its result, 'NN ms on <device name>')"""
    cuda = pipe.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(pipe.device)
    t = time.perf_counter()
    out = call()
    if cuda:
        torch.cuda.synchronize(pipe.device)
    ms = (time.perf_counter() - t) * 1000
    return out, f"{ms:.0f} ms on {torch.cuda.get_device_name(pipe.device) if cuda else 'the CPU'}"


def text_to_image(pipe, prompt, seed):
    """-> (uint8 [512, 512, 3], time note)"""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a prompt.")
    images, note = timed(pipe, lambda: pipe.text_to_image([prompt], pipe.generator(int(seed))))
    return images[0], note


def image_to_image(pipe, image, task, caption, from_photo, seed):
    """image: a PIL image -> (the prepared 512-px input, the result, time note), both uint8 [512, 512, 3]"""
    from jeveration.images import load_square, to_gray
    if image is None:
        raise ValueError("Upload an image.")
    source = load_square(image, config.IMAGE_SIZE)
    if task == "colorize":
        source = to_gray(source)
    elif from_photo and task in ("hed", "depth"):
        from jeveration.annotators import structure_map
        source = structure_map(source, task, pipe.device)
    images, note = timed(pipe, lambda: pipe.image_to_image(source[None], task, [(caption or "").strip()], 0,
                                                        pipe.generator(int(seed))))
    return source, images[0], note


def dense_predict(pipe, image):
    """image: a PIL image -> (segmentation in colour, depth view with nearer brighter, normals as RGB, note)"""
    if image is None:
        raise ValueError("Upload a picture.")
    picture = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pred, note = timed(pipe, lambda: pipe.dense_predict(picture[None]))
    pred = {t: v[0].cpu().numpy() for t, v in pred.items()}
    seg = pred["seg"].astype(np.uint8)
    d = pred["depth"].astype(np.float32)
    classes = ", ".join(config.NYU13[i] for i in np.unique(seg))
    return (seg_view(seg, picture), depth_view(d), normal_view(pred["normal"].astype(np.float32)),
            f"{note}; depth {d.min():.2f} to {d.max():.2f} m; classes: {classes}")


def build(pipe):
    """the Gradio interface around a loaded pipeline; results are served as PNG (Gradio's default is lossy WebP)"""
    import gradio as gr

    def example(name):
        return os.path.join(EXAMPLES, name)

    def guarded(fn):
        def run(*args):
            try:
                return fn(pipe, *args)
            except ValueError as e:
                raise gr.Error(str(e))
        # ZeroGPU checks the reserved seconds against the visitor's remaining quota; a request needs 1 to 7 s, the
        # first one from a photo also loads its edge or depth model
        return spaces.GPU(duration=60)(run) if spaces else run

    with gr.Blocks(title="Jeveration") as demo:
        gr.Markdown("# Jeveration\nA frozen vision-language model returns a whole 512 x 512 image as one decision, in a "
                    "single forward pass. One checkpoint: text-to-image, image-to-image, dense prediction.")
        with gr.Tab("Text-to-image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", lines=3)
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    go = gr.Button("Generate", variant="primary")
                with gr.Column():
                    out = gr.Image(label="Image", type="numpy", format="png")
                    note = gr.Markdown()
            gr.Examples(T2I_EXAMPLES, inputs=prompt, examples_per_page=len(T2I_EXAMPLES))
            go.click(guarded(text_to_image), [prompt, seed], [out, note])
            prompt.submit(guarded(text_to_image), [prompt, seed], [out, note])
        with gr.Tab("Image-to-image"):
            with gr.Row():
                with gr.Column():
                    src = gr.Image(label="Input", type="pil")
                    task = gr.Radio(list(config.I2I_TASKS), value="hed", label="Task")
                    help_ = gr.Markdown(I2I_HELP["hed"])
                    caption = gr.Textbox(label="Caption of the wanted image (hed / depth)")
                    from_photo = gr.Checkbox(label="The input is a photo: compute its edge / depth map first (hed / depth)")
                    seed2 = gr.Number(label="Seed", value=0, precision=0)
                    go2 = gr.Button("Generate", variant="primary")
                with gr.Column():
                    prepared = gr.Image(label="Input as the model reads it (512 x 512)", type="numpy", format="png")
                    out2 = gr.Image(label="Image", type="numpy", format="png")
                    note2 = gr.Markdown()
            gr.Examples([[example(f), t, c, p] for f, t, c, p in I2I_EXAMPLES], inputs=[src, task, caption, from_photo],
                        examples_per_page=len(I2I_EXAMPLES))
            task.change(lambda t: I2I_HELP[t], task, help_)
            go2.click(guarded(image_to_image), [src, task, caption, from_photo, seed2], [prepared, out2, note2])
        with gr.Tab("Dense prediction"):
            gr.Markdown("Segmentation (13 classes), depth and surface normals of a picture. Trained on NYUv2: indoor "
                        "scenes, 4:3 pictures.")
            with gr.Row():
                with gr.Column():
                    pic = gr.Image(label="Picture", type="pil")
                    gr.Examples([example(f) for f in DENSE_EXAMPLES], inputs=pic, label="Examples (choose one, then Predict)")
                    go3 = gr.Button("Predict", variant="primary")
                    note3 = gr.Markdown()
                with gr.Column():
                    seg = gr.Image(label="Segmentation", type="numpy", format="png")
                    depth = gr.Image(label="Depth (nearer is brighter)", type="numpy", format="png")
                    normal = gr.Image(label="Surface normals", type="numpy", format="png")
            go3.click(guarded(dense_predict), pic, [seg, depth, normal, note3])
    return demo


def load(a):
    from jeveration import JeverationPipeline
    paths = {"jeveration.pt": a.weights, "decoder512.pt": a.decoder, "source_decoder512.pt": a.source_decoder}
    if not all(paths.values()):
        from huggingface_hub import hf_hub_download
        paths = {name: p or hf_hub_download(WEIGHTS_REPO, name) for name, p in paths.items()}
    pipe = JeverationPipeline(paths["jeveration.pt"], decoder=paths["decoder512.pt"],
                              source_decoder=paths["source_decoder512.pt"], backbone=a.backbone, tatitok=a.tatitok,
                              clip=a.clip, device=a.device)
    if spaces is None:                                      # compile the kernels before the first visitor's request
        for _ in range(2):
            text_to_image(pipe, T2I_EXAMPLES[0], 0)
    return pipe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="the Jeveration checkpoint (default: downloaded from the Hub)")
    ap.add_argument("--decoder", default=None, help="the 512-px decoder checkpoint (default: downloaded from the Hub)")
    ap.add_argument("--source-decoder", default=None, help="the source-conditioned 512-px decoder checkpoint (default: the Hub)")
    ap.add_argument("--backbone", default=None, help="local copy of Qwen3.5-4B (default: the Hugging Face Hub)")
    ap.add_argument("--tatitok", default=None, help="local copy of the TA-TiTok VQ-128 tokenizer (default: the Hub)")
    ap.add_argument("--clip", default=None, help="local folder with CLIP ViT-L/14-336's open_clip_model.safetensors (default: the Hub)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default=None, help="address to listen on (default 127.0.0.1; 0.0.0.0: reachable from other machines)")
    ap.add_argument("--port", type=int, default=None, help="default 7860")
    ap.add_argument("--share", action="store_true", help="also create a temporary public link (gradio.live)")
    a = ap.parse_args()
    build(load(a)).queue().launch(server_name=a.host, server_port=a.port, share=a.share, allowed_paths=[EXAMPLES])


if __name__ == "__main__":
    main()
