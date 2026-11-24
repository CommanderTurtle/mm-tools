"""Budget-sweep gallery: generate each input image at 100% ... 10% of the 8192 tokens with
no dense warm-up, render a looping 360-degree orbit of every result, and write an HTML page.

    python scripts/gallery.py --ckpt weights/budget3d_idol.safetensors --out gallery/ \
        --images third_party/idol-sdk/demo/examples/*.webp [--turbo] [--fast]

Controlled comparison: every budget of one object starts from the same noise (same seed) and is
rendered along the same orbit (the camera predicted at 100%), so the only variable is the budget.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

import cv2
import numpy as np
import torch
from PIL import Image

from budget3d.models.load import decode_gaussians as decode  # noqa: F401  (re-exported for bench_infer and the tests)
from budget3d.models.load import idol_weights_dir, load_budget3d, load_decoders
from budget3d.render.splat import Splats, orbit_positions, render
from budget3d.sampling.sampler import ImagePreprocessor as Preprocessor
from budget3d.sampling.sampler import sample


class MemProbe:
    """Peak and time-averaged GPU memory of the block it wraps.

    peak_*: torch's own high-water marks (exact).  avg_alloc: mean of memory_allocated() polled
    every ``dt`` s by a side thread (time-weighted, since the polls are evenly spaced)."""
    MB = 2 ** 20

    def __init__(self, dt=0.02):
        self.dt = dt

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.base = torch.cuda.memory_allocated()
        self.samples, self.stop = [], threading.Event()
        def poll():
            while not self.stop.is_set():
                self.samples.append(torch.cuda.memory_allocated())
                time.sleep(self.dt)
        self.th = threading.Thread(target=poll, daemon=True)
        self.th.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        self.stop.set()
        self.th.join()
        s = self.samples or [torch.cuda.memory_allocated()]
        self.stats = {"peak_alloc_mb": round(torch.cuda.max_memory_allocated() / self.MB, 1),
                      "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / self.MB, 1),
                      "avg_alloc_mb": round(sum(s) / len(s) / self.MB, 1),
                      "base_alloc_mb": round(self.base / self.MB, 1),   # weights etc. resident before
                      "n_polls": len(s)}
        return False


BUDGETS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def load_models(ckpt_path, dev, ema=True, turbo=False, lora=None):
    """The model and Idol's decoders -> (student, dec, dec_gs, config, training step, weights dir).

    ``turbo``: the Idol-turbo variant (phase-LoRA experts injected first, then the checkpoint on top).
    ``ckpt_path`` None / "none" gives the untrained start: the released Idol, or Idol-turbo with ``turbo``.
    The model is returned in fp32 end to end (no bf16 noise in a quality comparison); ``--fast`` switches bf16 on."""
    weights = idol_weights_dir()
    student, step = load_budget3d(None if ckpt_path in (None, "none") else ckpt_path, turbo=turbo, lora=lora,
                                  device=dev, ema=ema, weights_dir=weights)
    for m in student.modules():
        if hasattr(m, "bf16_blocks"):
            m.bf16_blocks = False
    dec, dec_gs, cfg_json = load_decoders(weights, dev)
    return student, dec, dec_gs, cfg_json, step, weights


def orbit_camera(cam_latent):
    from idol.camera import decode_camera
    c = decode_camera(cam_latent)
    if c["radius"] > 50:           # orthographic prediction: show it through a long lens
        c = dict(c, radius=4.0, fov=2 * math.degrees(math.atan(0.62 / 4.0)))
    return c


def write_webm(frames, path, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))
    assert vw.isOpened(), "VP9 writer unavailable"
    try:
        vw.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    except Exception:
        pass
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()


def object_noise(student, seed, dev):
    """The same starting noise for every object and every budget (independent of batching)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(1, student.q_token_length, student.in_channels, device=dev, generator=gen)
    cam = torch.randn(1, 1, student.cam_channels, device=dev, generator=gen)
    return z, cam


def done_names(out):
    names = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, "objects"))} \
        if os.path.isdir(os.path.join(out, "objects")) else set()
    mp = os.path.join(out, "manifest.json")
    if os.path.exists(mp):
        try:
            names |= {o["name"] for o in json.load(open(mp))["objects"]}
        except Exception:
            pass
    return names


def rebuild_index(out, settings=None):
    """manifest.json / index.html / gallery.md from every finished object on disk, so several
    jobs (or a requeued one) can fill the same folder without overwriting each other."""
    mp = os.path.join(out, "manifest.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    objs = {o["name"]: o for o in old.get("objects", [])}
    od = os.path.join(out, "objects")
    if os.path.isdir(od):
        for f in sorted(os.listdir(od), key=lambda f: os.path.getmtime(os.path.join(od, f))):
            if f.endswith(".json"):
                o = json.load(open(os.path.join(od, f)))
                objs[o["name"]] = o
    m = {k: v for k, v in old.items() if k != "objects"}
    m.update(settings or {})
    # the first six (the original showcase) stay on top, the rest alphabetically
    first = ["fig-knight", "food-cupcake", "fur-chaise", "toy-robot-tin", "obj-stand-mixer", "nb-bird-wood"]
    m["objects"] = sorted(objs.values(), key=lambda o: (first.index(o["name"]) if o["name"] in first else 99,
                                                        o["name"].startswith("dome-"), o["name"]))
    tmp = f"{mp}.{os.getpid()}.tmp"            # unique: another process may be rebuilding too
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, mp)
    write_html(out, m)
    write_markdown(out, m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False)
    ap.add_argument("--images", nargs="+", default=[])
    ap.add_argument("--exclude", nargs="*", default=[], help="object names to leave out (e.g. done elsewhere)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rebuild_only", action="store_true", help="just regenerate manifest/html/md from disk")
    ap.add_argument("--batch", type=int, default=8, help="objects sampled together per budget")
    ap.add_argument("--shard", type=int, default=0, help="this process's slice of --images (parallel GPUs)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default 50, or 5 with --turbo)")
    ap.add_argument("--turbo", action="store_true",
                    help="idol-turbo variant: phase-LoRA experts, guidance-free sampler (cfg ignored); --ckpt none = original idol-turbo")
    ap.add_argument("--lora", default=None, help="idol-turbo LoRA file (default: speridlabs/idol-turbo)")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--save_ply", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="optimised inference (fast.py): bf16 blocks, TF32, fused attention, torch.compile")
    ap.add_argument("--grid_tile", type=int, default=256, help="tile size of the per-object GIF grid")
    ap.add_argument("--fov", type=float, default=40.0, help="orbit vertical field of view (degrees)")
    ap.add_argument("--fill", type=float, default=1.0, help="bounding-sphere margin (1 = sphere touches the frame)")
    a = ap.parse_args()
    if a.rebuild_only:
        m = rebuild_index(a.out)
        print(f"[gallery] rebuilt index: {len(m['objects'])} objects")
        return
    dev = "cuda"
    for d in ("videos", "inputs", "posters", "latents", "grids", "objects") + (("ply",) if a.save_ply else ()):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    name_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    uniq, seen = [], set()
    for p in a.images:                      # dedupe first, then slice: a stable partition across
        if name_of(p) not in seen:          # shards, independent of what has finished already
            seen.add(name_of(p))
            uniq.append(p)
    done = done_names(a.out)
    todo = [p for p in uniq[a.shard::a.num_shards] if name_of(p) not in done and name_of(p) not in a.exclude]
    print(f"[gallery] {len(todo)} objects to do ({len(done)} already in {a.out})", flush=True)
    if not todo:
        rebuild_index(a.out)
        return

    if a.steps is None:
        a.steps = 5 if a.turbo else 50
    if a.turbo:
        a.cfg = 0.0
    student, dec, dec_gs, cfg_json, step, weights = load_models(a.ckpt, dev, turbo=a.turbo, lora=a.lora)
    gpu = torch.cuda.get_device_name()
    precision = "fp32"
    if a.fast:
        from budget3d.sampling.fast import describe, optimize_for_inference
        optimize_for_inference(student)
        precision = describe()
    print(f"[gallery] student at step {step} (EMA), {precision}, GPU {gpu}", flush=True)
    settings = {"checkpoint": os.path.abspath(a.ckpt) if a.ckpt not in (None, "none") else ("speridlabs/idol-turbo" if a.turbo else "speridlabs/idol"),
                "turbo": a.turbo, "train_step": step, "warmup_steps": a.warmup_steps,
                "steps": a.steps, "cfg": a.cfg, "rescale_t": a.rescale_t, "seed": a.seed,
                "num_gaussians": a.num_gaussians, "frames": a.frames, "fps": a.fps, "size": a.size,
                "budgets": a.budgets, "gpu": gpu, "precision": precision, "fast": a.fast}
    prep = Preprocessor(weights, dev, a.mode)
    z1, c1 = object_noise(student, a.seed, dev)

    for c0 in range(0, len(todo), a.batch):
        chunk = todo[c0:c0 + a.batch]
        B = len(chunk)
        tc = time.time()
        names, prepared, conds = [], [], []
        for p in chunk:
            im, cond = prep(p)
            names.append(name_of(p)); prepared.append(im); conds.append(cond)
            im.save(os.path.join(a.out, "inputs", f"{name_of(p)}.png"))
        cond = torch.cat(conds, dim=0)
        noise = (z1.expand(B, -1, -1).clone(), c1.expand(B, -1, -1).clone())
        entries = [{"name": n, "input": f"inputs/{n}.png", "runs": []} for n in names]
        orbits, thumbs = [None] * B, [[] for _ in range(B)]
        for budget in sorted(a.budgets, reverse=True):
            with MemProbe() as mp:
                if a.turbo:
                    from budget3d.sampling.turbo import sample_turbo
                    latent, camera, layout = sample_turbo(student, cond, budget, a.steps, a.rescale_t, seed=a.seed,
                                                          device=dev, noise=noise)
                else:
                    latent, camera, layout = sample(student, cond, budget, a.steps, a.cfg, a.rescale_t, seed=a.seed,
                                                    device=dev, warmup_steps=a.warmup_steps, noise=noise)
            t_sample = mp.seconds
            t_render = 0.0
            for i, name in enumerate(names):
                cam = orbit_camera(camera[i, 0])
                with MemProbe() as md:
                    g = decode(latent[i:i + 1], dec, dec_gs, cfg_json, a.num_gaussians)
                    sp = Splats(g)
                if orbits[i] is None:
                    # One orbit per object, fixed at the largest budget: the predicted yaw/pitch
                    # (the turn starts and ends at the input view), with the distance fitted to
                    # the object's bounding sphere -- the predicted radius/fov describe the
                    # original photo, not the tightly re-cropped input.
                    r_obj = float(torch.quantile(sp.xyz.norm(dim=1), 0.995))
                    orbits[i] = dict(cam, fov=a.fov, radius=a.fill * r_obj / math.sin(math.radians(a.fov) / 2))
                orbit = orbits[i]
                tag = f"{name}_b{round(budget * 100):03d}"
                torch.save({"latent": latent[i:i + 1].cpu(), "camera": camera[i:i + 1].cpu(), "orbit": orbit},
                           os.path.join(a.out, "latents", f"{tag}.pt"))
                with MemProbe() as mr:
                    frames = [(render(sp, c, orbit["fov"], size=a.size).cpu().numpy() * 255).round().astype(np.uint8)
                              for c in orbit_positions(orbit["yaw"], orbit["pitch"], orbit["radius"], a.frames)]
                t_render += mr.seconds
                write_webm(frames, os.path.join(a.out, "videos", f"{tag}.webm"), a.fps)
                Image.fromarray(frames[0]).save(os.path.join(a.out, "posters", f"{tag}.jpg"), quality=92)
                if a.save_ply:
                    g.save_ply(os.path.join(a.out, "ply", f"{tag}.ply"))
                thumbs[i].append((f"{round(budget * 100)}%  ·  {layout.n_tokens} tok",
                                  [Image.fromarray(f).resize((a.grid_tile, a.grid_tile), Image.LANCZOS) for f in frames]))
                entries[i]["runs"].append({
                    "budget": budget, "tokens": layout.n_tokens, "k4": layout.k4, "k2": layout.k2,
                    "n4": layout.n4, "n2": layout.n2, "n1": layout.n1,
                    "sample_s": round(t_sample / B, 2), "batch": B, "gaussians": sp.n, "camera": cam,
                    "orbit": orbit, "video": f"videos/{tag}.webm", "poster": f"posters/{tag}.jpg",
                    "perf": {"sample_batch_s": round(t_sample, 3), "sample_mem": mp.stats,
                             "decode_s": round(md.seconds, 3), "decode_mem": md.stats,
                             "render_s": round(mr.seconds, 3), "render_fps": round(a.frames / mr.seconds, 2),
                             "render_mem": mr.stats, "steps": a.steps, "batch": B, "gpu": gpu,
                             "batch_index": c0 // a.batch, "shard": a.shard}})
                del g, sp, frames
            print(f"[gallery] batch {c0 // a.batch + 1}/{math.ceil(len(todo) / a.batch)} ({B} obj)  {budget:4.0%}  "
                  f"N={layout.n_tokens:5d}  sample {t_sample:6.1f}s ({t_sample / B:5.1f}s/obj)  "
                  f"render {t_render:5.1f}s  peak {mp.stats['peak_alloc_mb'] / 1024:5.1f}G "
                  f"avg {mp.stats['avg_alloc_mb'] / 1024:5.1f}G", flush=True)
            del latent, camera
        for i, name in enumerate(names):
            entries[i]["grid_gif"] = f"grids/{name}.gif"
            write_grid_gif(prepared[i], thumbs[i], os.path.join(a.out, entries[i]["grid_gif"]), a.fps,
                           a.grid_tile, name)
            tmp = os.path.join(a.out, "objects", f"{name}.json.tmp")
            json.dump(entries[i], open(tmp, "w"), indent=1)
            os.replace(tmp, os.path.join(a.out, "objects", f"{name}.json"))
        m = rebuild_index(a.out, settings)
        print(f"[gallery] done: {', '.join(names)}  ({time.time() - tc:.0f}s for the batch; "
              f"{len(m['objects'])} objects in the gallery)", flush=True)
    print("[gallery] wrote", os.path.join(a.out, "index.html"), flush=True)


def write_grid_gif(prepared, thumbs, path, fps, tile, title):
    """One animated GIF per object: input + every budget, 2 rows x 6, all tiles in lock-step."""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=18)
        tfont = ImageFont.load_default(size=20)
    except TypeError:
        font = tfont = ImageFont.load_default()
    cols, lab, head = 6, 30, 34
    rows = math.ceil((len(thumbs) + 1) / cols)
    W, H = cols * tile, head + rows * (tile + lab)
    inp = Image.new("RGB", (tile, tile), "white")
    im = prepared.convert("RGBA")
    im.thumbnail((tile, tile), Image.LANCZOS)
    inp.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2), im)
    base = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(base)
    d.text((10, 7), f"{title}   (input, then 100% -> 10% of 8192 tokens, no warm-up)", fill=(30, 30, 30), font=tfont)
    cells = [("input", None)] + [(lbl, fr) for lbl, fr in thumbs]
    for k, (lbl, _) in enumerate(cells):
        x, y = (k % cols) * tile, head + (k // cols) * (tile + lab)
        d.text((x + 8, y + tile + 5), lbl, fill=(30, 30, 30), font=font)
    base.paste(inp, (0, head))
    n = len(thumbs[0][1])
    out = []
    for i in range(n):
        f = base.copy()
        for k, (_, fr) in enumerate(cells[1:], start=1):
            f.paste(fr[i], ((k % cols) * tile, head + (k // cols) * (tile + lab)))
        out.append(f.quantize(colors=255, method=Image.Quantize.MEDIANCUT))
    out[0].save(path, save_all=True, append_images=out[1:], duration=round(1000 / fps), loop=0, disposal=1)


def write_markdown(out_dir, m):
    """gallery.md: open with VS Code's Markdown preview (Ctrl+Shift+V) -- the GIFs animate there."""
    L = ["# Budget3D budget sweep (no warm-up)", "",
         f"Checkpoint step {m['train_step']} (EMA) · warm-up steps: {m['warmup_steps']} · {m['steps']} Euler steps · "
         f"CFG {m['cfg']} · seed {m['seed']} · same seed and same orbit for every budget of an object.", "",
         "Each GIF: the input image, then 100% -> 10% of Idol's 8192 latent tokens. "
         "A full 360-degree turn that starts and ends at the input view.", ""]
    for o in m["objects"]:
        L += [f"## {o['name']}", "", f"![{o['name']}]({o['grid_gif']})", "",
              f"[grid MP4](grids/{o['name']}.mp4) · per-budget MP4: " + " · ".join(
                  f"[{round(r['budget'] * 100)}%]({r['video'][:-5]}.mp4)" for r in o["runs"]), "",
              "| budget | tokens | s=4 cubes | s=2 cells | s=1 points | sample (s) |",
              "|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {round(r['budget'] * 100)}% | {r['tokens']} | {r['n4']} | {r['n2']} | {r['n1']} | {r['sample_s']} |"
              for r in o["runs"]]
        L += [""]
    tmp = os.path.join(out_dir, f"gallery.md.{os.getpid()}.tmp")
    open(tmp, "w").write("\n".join(L))
    os.replace(tmp, os.path.join(out_dir, "gallery.md"))


def write_html(out_dir, m):
    """index.html + gallery.js + gallery_data.js.  No inline script: VS Code's HTML Preview (and any
    page under a strict Content-Security-Policy) only runs scripts loaded from files."""
    head, rest = HTML_TEMPLATE.split("<script>", 1)
    js, tail = rest.split("</script>", 1)
    js = js.replace("const M = __DATA__;", "const M = window.GALLERY;")
    for name, text in (("gallery_data.js", "window.GALLERY = " + json.dumps(m) + ";\n"), ("gallery.js", js)):
        tmp = os.path.join(out_dir, f"{name}.{os.getpid()}.tmp")
        open(tmp, "w").write(text)
        os.replace(tmp, os.path.join(out_dir, name))
    html = head + '<script src="gallery_data.js"></script>\n<script src="gallery.js"></script>' + tail
    tmp = os.path.join(out_dir, f"index.html.{os.getpid()}.tmp")
    open(tmp, "w").write(html)
    os.replace(tmp, os.path.join(out_dir, "index.html"))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Budget3D Budget Sweep</title>
<style>
:root{--bg:#f6f6f4;--panel:#ffffff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2f6fdf;--chip:#eef2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9b9a94;--line:#2e2e2b;--accent:#7aa5f5;--chip:#22293a}}
:root[data-theme="dark"]{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9b9a94;--line:#2e2e2b;--accent:#7aa5f5;--chip:#22293a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:24px 16px 8px;max-width:1700px;margin:0 auto}
h1{font-size:22px;margin:0 0 6px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);max-width:900px}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 4px}
.meta span{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:12px}
.bar{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);
     border-bottom:1px solid var(--line);padding:8px 16px}
.bar .in{max-width:1700px;margin:0 auto;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button,select{font:inherit;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
label{color:var(--muted);font-size:13px}
main{max-width:1700px;margin:0 auto;padding:8px 16px 40px}
.obj{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;margin:14px 0}
.obj h2{font-size:15px;margin:0 0 10px;font-weight:600;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
.links{font-size:12px;font-weight:400;color:var(--muted)}
a{color:var(--accent)}
.row{display:grid;grid-template-columns:repeat(11,minmax(0,1fr));gap:8px}
.cell{min-width:0}
.cell video,.cell img{width:100%;aspect-ratio:1;display:block;border-radius:8px;background:#fff;border:1px solid var(--line);object-fit:cover}
.cell.input img{object-fit:contain}
.lab{font-size:12px;margin-top:4px;display:flex;justify-content:space-between;gap:4px}
.lab b{font-weight:600}
.lab i{font-style:normal;color:var(--muted);white-space:nowrap}
.cell video{cursor:zoom-in}
@media (max-width:1100px){.row{grid-template-columns:repeat(6,minmax(0,1fr))}}
@media (max-width:640px){.row{grid-template-columns:repeat(3,minmax(0,1fr))}}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;align-items:center;justify-content:center;z-index:10;padding:16px}
#lb.on{display:flex}
#lb .box{display:flex;flex-direction:column;gap:8px;max-width:min(92vw,820px);width:100%}
#lb video{width:100%;border-radius:10px;background:#fff}
#lb .cap{color:#eee;display:flex;justify-content:space-between;align-items:center;gap:10px}
#lb input[type=range]{flex:1}
table{border-collapse:collapse;font-size:12px;margin-top:10px;width:100%}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}
th:first-child,td:first-child{text-align:left}
details{margin-top:8px;color:var(--muted)}
</style></head><body>
<header>
  <h1>Budget3D: budget sweep, no warm-up</h1>
  <div class="sub">Each row is one input image (only rows on screen play), generated at 100% down to 10% of Idol's 8192 latent tokens. The
  token budget is enforced from the very first Euler step: the layout is built from pure noise, with no dense
  warm-up steps. Every budget of an object uses the same seed and is rendered along the same 360&deg; orbit, which
  starts and ends at the predicted input view, so the only thing that changes across a row is the budget.</div>
  <div class="meta" id="meta"></div>
</header>
<div class="bar"><div class="in">
  <button id="pp">Pause</button>
  <button id="rs">Restart all</button>
  <label>Speed <select id="sp"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>
  <label><input type="checkbox" id="sync" checked> keep each row in sync</label>
  <label>Jump to <select id="jump"><option value="">object…</option></select></label>
</div></div>
<main id="main"></main>
<div id="lb"><div class="box">
  <video id="lbv" muted loop playsinline></video>
  <div class="cap"><span id="lbt"></span><input type="range" id="lbr" min="0" max="9" step="1"><span id="lbb"></span></div>
</div></div>
<script>
const M = __DATA__;
const meta = document.getElementById('meta');
[`checkpoint step ${M.train_step}`, `warm-up steps: ${M.warmup_steps}`, `${M.steps} Euler steps`, `CFG ${M.cfg}`,
 `seed ${M.seed}`, `${M.num_gaussians.toLocaleString()} Gaussians`, `${M.size}px · ${M.frames} frames @ ${M.fps} fps`, M.gpu]
 .forEach(t => { const s = document.createElement('span'); s.textContent = t; meta.appendChild(s); });
meta.insertAdjacentHTML('afterbegin', `<span><b>${M.objects.length} objects</b></span>`);
const main = document.getElementById('main');
const jump = document.getElementById('jump');
const rows = [];
M.objects.forEach((o, i) => {
  const sec = document.createElement('section'); sec.className = 'obj'; sec.id = 'o-' + o.name; sec.dataset.i = i;
  const h = document.createElement('h2'); h.textContent = `${i + 1}. ${o.name}`;
  const gl = document.createElement('span'); gl.className = 'links';
  gl.innerHTML = `<a href="grids/${o.name}.mp4" target="_blank">grid MP4</a> · <a href="${o.grid_gif || `grids/${o.name}.gif`}" target="_blank">grid GIF</a>`;
  h.appendChild(gl); sec.appendChild(h);
  const row = document.createElement('div'); row.className = 'row';
  const ci = document.createElement('div'); ci.className = 'cell input';
  ci.innerHTML = `<img loading="lazy" src="${o.input}" alt="input"><div class="lab"><b>input</b><i>image</i></div>`;
  row.appendChild(ci);
  const vids = [];
  o.runs.forEach((r, k) => {
    const c = document.createElement('div'); c.className = 'cell';
    const v = document.createElement('video');
    Object.assign(v, {poster: r.poster, muted: true, loop: true, playsInline: true, preload: 'none'});
    v.dataset.src = r.video;
    v.addEventListener('click', () => openLB(o, k, vids[0]));
    vids.push(v); c.appendChild(v);
    const l = document.createElement('div'); l.className = 'lab';
    l.innerHTML = `<b>${Math.round(r.budget * 100)}%</b><i>${r.tokens} tok</i>`;
    c.appendChild(l); row.appendChild(c);
  });
  sec.appendChild(row);
  const d = document.createElement('details');
  d.innerHTML = '<summary>layout and timing</summary><table><tr><th>budget</th><th>tokens</th><th>s=4 cubes</th>' +
    '<th>s=2 cells</th><th>s=1 points</th><th>sample s/object</th><th>Gaussians kept</th></tr>' +
    o.runs.map(r => `<tr><td>${Math.round(r.budget*100)}%</td><td>${r.tokens}</td><td>${r.n4}</td><td>${r.n2}</td>` +
      `<td>${r.n1}</td><td>${r.sample_s}</td><td>${r.gaussians.toLocaleString()}</td></tr>`).join('') + '</table>';
  sec.appendChild(d);
  main.appendChild(sec);
  rows.push({sec, vids, on: false});
  const opt = document.createElement('option'); opt.value = sec.id; opt.textContent = `${i + 1}. ${o.name}`; jump.appendChild(opt);
});
jump.onchange = () => { const el = document.getElementById(jump.value); if (el) el.scrollIntoView({behavior: 'smooth'}); };
let playing = true, rate = 1;
// only rows near the viewport hold decoders and play; the rest are unloaded (posters stay)
function activate(r) {
  r.on = true;
  r.vids.forEach(v => {
    if (!v.firstChild) {
      v.insertAdjacentHTML('beforeend', `<source src="${v.dataset.src.replace(/\.webm$/, '.mp4')}" type="video/mp4">` +
                                        `<source src="${v.dataset.src}" type="video/webm">`);
      v.load();
    }
    v.playbackRate = rate; });
  let ready = 0;
  const go = () => { r.vids.forEach(v => { v.currentTime = 0; if (playing && r.on) v.play().catch(() => {}); }); };
  r.vids.forEach(v => {
    if (v.readyState >= 3) { if (++ready === r.vids.length) go(); }
    else v.addEventListener('canplay', function once() { v.removeEventListener('canplay', once); if (++ready === r.vids.length) go(); });
  });
}
function deactivate(r) { r.on = false; r.vids.forEach(v => { v.pause(); v.replaceChildren(); v.load(); }); }
const io = new IntersectionObserver(es => es.forEach(e => {
  const r = rows[+e.target.dataset.i]; if (e.isIntersecting) { if (!r.on) activate(r); } else if (r.on) deactivate(r);
}), {rootMargin: '300px 0px'});
rows.forEach(r => io.observe(r.sec));
const live = () => rows.filter(r => r.on);
document.getElementById('pp').onclick = e => {
  playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
  live().forEach(r => r.vids.forEach(v => playing ? v.play().catch(() => {}) : v.pause())); };
document.getElementById('rs').onclick = () => live().forEach(r => r.vids.forEach(v => { v.currentTime = 0; }));
document.getElementById('sp').onchange = e => { rate = +e.target.value; live().forEach(r => r.vids.forEach(v => v.playbackRate = rate)); };
setInterval(() => {
  if (!document.getElementById('sync').checked || !playing) return;
  live().forEach(r => { const m = r.vids[0], t = m.currentTime;
    r.vids.forEach(v => { if (v !== m && Math.abs(v.currentTime - t) > 0.08) v.currentTime = t; }); });
}, 500);
const lb = document.getElementById('lb'), lbv = document.getElementById('lbv'), lbr = document.getElementById('lbr');
let cur = null;
function showLB(k) {
  const r = cur.runs[k], t = lbv.currentTime || 0;
  lbv.onerror = () => { if (!lbv.src.endsWith('.webm')) { lbv.src = r.video; lbv.currentTime = t; lbv.play().catch(() => {}); } };
  lbv.src = r.video.replace(/\.webm$/, '.mp4'); lbv.currentTime = t; lbv.play().catch(() => {});
  document.getElementById('lbt').textContent = cur.name;
  document.getElementById('lbb').textContent = `${Math.round(r.budget*100)}% · ${r.tokens} tokens`;
  lbr.value = k;
}
function openLB(o, k, m) { cur = o; lbr.max = o.runs.length - 1; lb.classList.add('on'); lbv.currentTime = m.currentTime; showLB(k); }
lbr.oninput = () => showLB(+lbr.value);
lb.addEventListener('click', e => { if (e.target === lb) { lb.classList.remove('on'); lbv.pause(); } });
document.addEventListener('keydown', e => {
  if (!lb.classList.contains('on')) return;
  if (e.key === 'Escape') { lb.classList.remove('on'); lbv.pause(); }
  if (e.key === 'ArrowRight' && +lbr.value < +lbr.max) showLB(+lbr.value + 1);
  if (e.key === 'ArrowLeft' && +lbr.value > 0) showLB(+lbr.value - 1);
});
</script></body></html>
"""

if __name__ == "__main__":
    main()
