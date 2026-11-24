"""Build the files of a Budget3D model repository on the Hugging Face Hub from a training run (nothing is uploaded).

    # weights: the EMA of a training checkpoint as .safetensors (+ training_args.json); no optimizer state
    python scripts/export_hf.py weights --ckpt runs/budget3d_idol_turbo/checkpoint-final.pth \
        --out stage/Budget3D-Idol-Turbo/budget3d_idol_turbo.safetensors --base_model speridlabs/idol-turbo

    # labelled budget clips (input + 100 % ... 40 % with the measured speed-up and peak memory) from a gallery
    python scripts/export_hf.py clips --gallery gallery --out stage/Budget3D-Idol/assets --title "Budget3D-Idol"

    # the gallery's own grid videos (input + every budget) of every n-th object, an unselected sample
    python scripts/export_hf.py grids --gallery gallery --out stage/Budget3D-Idol-Turbo/assets/step20000 --every 17

A released file holds exactly the tensors ``budget3d.models.load.load_trained`` expects: the EMA weights of every trained
parameter in fp32, with the training step, base model and variant in the safetensors metadata.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

# training arguments that are local paths, not part of the recipe
LOCAL_ARGS = ("data", "out", "weights", "lora", "config")
# the clips' default objects, in this order, and the ones that make up the reel
EXAMPLES = ["fig-knight", "food-avocado", "fig-axolotl", "fig-robot-cute", "fur-chesterfield", "obj-power-drill",
            "fig-dog-corgi", "food-gingerbread-house", "toy-train", "fig-viking", "nb-castle-iso", "food-macarons",
            "obj-roller-skate", "fig-ghost", "fur-egg-chair", "food-donut", "toy-submarine", "fig-fox",
            "fur-bar-stool", "obj-stand-mixer"]
REEL = ["fig-knight", "food-avocado", "fig-axolotl", "fig-robot-cute", "food-donut", "fig-fox", "fig-dog-corgi",
        "food-gingerbread-house"]
CLIP_BUDGETS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def export_weights(ckpt, out, base_model, variant=None, args_out=None):
    """The EMA weights of a training checkpoint -> ``out`` (.safetensors) and its training arguments -> ``args_out``
    (default: ``training_args.json`` next to ``out``; local paths are left out).

    Returns (step, number of tensors, number of parameters).  The optimizer state and the raw weights stay behind.
    """
    import torch
    from safetensors.torch import save_file
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    ema = {k: v.detach().clone().float().contiguous() for k, v in ck["ema"].items()}
    step = int(ck["step"])
    meta = {"format": "pt", "step": str(step), "weights": "ema", "dtype": "float32", "base_model": base_model}
    if variant:
        meta["variant"] = variant
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_file(ema, out, metadata=meta)
    args = {k: v for k, v in ck.get("args", {}).items() if k not in LOCAL_ARGS}
    with open(args_out or os.path.join(os.path.dirname(os.path.abspath(out)), "training_args.json"), "w") as f:
        json.dump({"step": step, "base_model": base_model, "train_args": args}, f, indent=1)
    return step, len(ema), sum(v.numel() for v in ema.values())


def label_font(size):
    """A font that has the "×" and "→" glyphs of the labels: $BUDGET3D_FONT, else DejaVu Sans from the system or from
    matplotlib, else None (the caller then writes plain ASCII labels in Pillow's default font)."""
    from PIL import ImageFont
    candidates = [os.environ.get("BUDGET3D_FONT"), "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    try:
        import matplotlib
        candidates.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"))
    except ImportError:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
    return None


def perf_by_budget(gallery):
    """{budget: (speed-up over 100 %, peak sampling memory in GB)} from the gallery's perf_report.json."""
    with open(os.path.join(gallery, "perf_report.json")) as f:
        rep = json.load(f)
    return {round(x["budget"], 2): (x["speedup_paired"]["mean"], x["sample_peak_alloc_gb"]["mean"])
            for x in rep["per_budget"]}


def compose_clip(gallery, name, perf, title, size=384, budgets=CLIP_BUDGETS, label100="1.00× (all tokens)"):
    """[T, H, W, 3] uint8: the input image and one orbit per budget in a 2 x 4 grid, each panel labelled with its
    token count, measured speed-up and peak sampling memory."""
    import numpy as np
    from PIL import Image, ImageDraw
    from to_mp4 import font as default_font, read_frames
    unicode_ok = label_font(12) is not None
    font = label_font if unicode_ok else default_font
    plain = (lambda t: t) if unicode_ok else (lambda t: t.replace("×", "x").replace("→", "->").replace("·", "-"))
    with open(os.path.join(gallery, "objects", f"{name}.json")) as f:
        o = json.load(f)
    runs = {round(r["budget"], 2): r for r in o["runs"]}
    clips = []
    for b in budgets:
        webm = os.path.join(gallery, runs[b]["video"])
        mp4 = os.path.splitext(webm)[0] + ".mp4"
        clips.append(read_frames(mp4 if os.path.exists(mp4) else webm, size))
    n = min(len(c) for c in clips)
    cols, lab, head = 4, 58, 56
    W, H = cols * size, head + 2 * (size + lab)
    base = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(base)
    d.text((16, 12), plain(title), fill=(25, 25, 25), font=font(26))
    inp = Image.new("RGB", (size, size), "white")
    im = Image.open(os.path.join(gallery, o["input"])).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    inp.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    base.paste(inp, (0, head))
    pos = [((k % cols) * size, head + (k // cols) * (size + lab)) for k in range(len(budgets) + 1)]
    d.text((pos[0][0] + 12, pos[0][1] + size + 8), "input image", fill=(25, 25, 25), font=font(26))
    for (x, y), b in zip(pos[1:], budgets):
        d.text((x + 12, y + size + 6), plain(f"{round(b * 100)}%  ·  {runs[b]['tokens']:,} tokens"), fill=(25, 25, 25), font=font(22))
        if b in perf:
            speedup, peak = perf[b]
            text = (f"{speedup:.2f}× faster" if b < 1 else label100) + f"  ·  {peak:.1f} GB peak"
            d.text((x + 12, y + size + 31), plain(text), fill=(20, 95, 200) if b < 1 else (110, 110, 110), font=font(20))
    canvas = np.asarray(base).copy()
    out = np.empty((n, H, W, 3), np.uint8)
    for i in range(n):
        for (x, y), c in zip(pos[1:], clips):
            canvas[y:y + size, x:x + size] = c[i]
        out[i] = canvas
    return out


def write_mp4(frames, path, fps=24):
    from to_mp4 import FFMPEG, X264
    _, H, W, _ = frames.shape
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", *X264, path], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


def build_clips(gallery, out, title, examples=EXAMPLES, reel=REEL, max_examples=16, write_examples=True):
    """<out>/<object>.mp4 for the first ``max_examples`` available examples and <out>/reel.mp4 (the reel objects back
    to back).  Returns (names written, objects in the reel)."""
    import numpy as np
    os.makedirs(out, exist_ok=True)
    perf = perf_by_budget(gallery)
    done = lambda n: os.path.exists(os.path.join(gallery, "objects", f"{n}.json"))
    chosen = [n for n in examples if done(n)][:max_examples] if write_examples else []
    written, reel_clips = [], []
    for name in dict.fromkeys(chosen + [n for n in reel if done(n)]):
        clip = compose_clip(gallery, name, perf, title)
        if name in chosen:
            write_mp4(clip, os.path.join(out, f"{name}.mp4"))
            written.append(name)
        if name in reel:
            reel_clips.append(clip)
    if reel_clips:
        write_mp4(np.concatenate(reel_clips), os.path.join(out, "reel.mp4"))
    return written, len(reel_clips)


def copy_grids(gallery, out, every=17, offset=0, names=None):
    """Copy the gallery's grid videos (grids/<object>.mp4: input + every budget) of every ``every``-th object in the
    order of the gallery manifest -- a sample fixed by position, not by how the result looks -- or of ``names``."""
    with open(os.path.join(gallery, "manifest.json")) as f:
        order = [o["name"] for o in json.load(f)["objects"]]
    names = names or order[offset::every]
    os.makedirs(out, exist_ok=True)
    for n in names:
        shutil.copyfile(os.path.join(gallery, "grids", f"{n}.mp4"), os.path.join(out, f"{n}.mp4"))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights", help="EMA weights of a training checkpoint -> .safetensors")
    w.add_argument("--ckpt", required=True)
    w.add_argument("--out", required=True, help="the .safetensors file to write")
    w.add_argument("--base_model", required=True, help="speridlabs/idol or speridlabs/idol-turbo")
    w.add_argument("--args_out", default=None, help="where the training arguments go (default: training_args.json next to --out)")
    w.add_argument("--variant", default=None, help="free-text tag stored in the metadata (e.g. step-10000)")
    c = sub.add_parser("clips", help="labelled budget clips and a reel from a gallery with a perf report")
    c.add_argument("--gallery", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--title", required=True, help="the line drawn above every clip")
    c.add_argument("--max_examples", type=int, default=16)
    c.add_argument("--names", nargs="*", default=None, help="the objects to make clips of (default: the built-in list)")
    c.add_argument("--reel_only", action="store_true")
    g = sub.add_parser("grids", help="copy the grid videos of every n-th object of a gallery")
    g.add_argument("--gallery", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--every", type=int, default=17)
    g.add_argument("--offset", type=int, default=0)
    g.add_argument("--names", nargs="*", default=None, help="explicit object names instead of every n-th")
    a = ap.parse_args()
    if a.cmd == "weights":
        step, n, params = export_weights(a.ckpt, a.out, a.base_model, a.variant, a.args_out)
        print(f"[export] {a.out}: {n} EMA tensors, {params / 1e6:.2f} M parameters, step {step}")
    elif a.cmd == "clips":
        names, n_reel = build_clips(a.gallery, a.out, a.title, examples=a.names or EXAMPLES,
                                    reel=[] if a.names else REEL, max_examples=a.max_examples,
                                    write_examples=not a.reel_only)
        print(f"[export] {a.out}: {len(names)} clips {names} and a reel of {n_reel} objects")
    else:
        print(f"[export] {a.out}: {copy_grids(a.gallery, a.out, a.every, a.offset, a.names)}")


if __name__ == "__main__":
    main()
