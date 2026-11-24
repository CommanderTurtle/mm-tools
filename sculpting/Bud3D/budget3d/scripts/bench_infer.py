"""Inference speed / memory / accuracy benchmark for the Budget3D-Idol sampler.

    python scripts/bench_infer.py --ckpt weights/budget3d_idol.safetensors --out bench/h100 \
        --images idol-sdk/demo/examples/fig-knight.webp ... [--variants fp32,bf16,bf16_cudnn,bf16_cudnn_compile]

For every variant and budget it times the sampling loop (CFG, ``--steps`` Euler steps, the same noise and
conditioning), records peak / mean GPU memory, and compares the final latent with the fp32 reference
(relative L2 error).  Timings are per sampling call on a batch of ``--batch`` objects, measured after a
warm-up call, so compile and autotuning are excluded.  Writes <out>/bench.json and a markdown table.

Variants (cumulative):
  fp32                 everything in fp32, PyTorch SDPA (the gallery setting so far)
  bf16                 bf16 matmuls inside the transformer blocks (as in training), TF32 for the rest,
                       SDPA picks the FlashAttention kernel (tiled attention)
  bf16_cudnn           + cuDNN fused attention (Idol's own optimised backend; fastest on Hopper)
  bf16_cudnn_compile   + torch.compile of the transformer blocks (one graph per token count)
  bf16_cudnn_compileall + torch.compile of the Budget3D refiners and mixers as well
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

import torch

from gallery import MemProbe, Preprocessor, decode, load_models, object_noise, orbit_camera
from budget3d.render.splat import Splats, orbit_positions, render
from budget3d.sampling.sampler import sample

VARIANTS = ["fp32", "bf16", "bf16_cudnn", "bf16_cudnn_compile", "bf16_cudnn_compileall"]


def set_variant(student, name):
    """fp32 | bf16 | bf16_cudnn | bf16_cudnn_compile | bf16_cudnn_compileall | ..._autotune | ..._cudagraphs"""
    from budget3d.upstream import idol_layers
    bf16 = name != "fp32"
    for m in student.modules():
        if hasattr(m, "bf16_blocks"):
            m.bf16_blocks = bf16
    torch.backends.cuda.matmul.allow_tf32 = bf16
    torch.backends.cudnn.allow_tf32 = bf16
    idol_layers.ATTN_BACKEND = "sdpa_cudnn" if "cudnn" in name else "sdpa"
    if "compile" in name:
        from budget3d.sampling.fast import optimize_for_inference
        mode = "max-autotune-no-cudagraphs" if "autotune" in name else (
            "reduce-overhead" if "cudagraphs" in name else None)
        optimize_for_inference(student, attention=idol_layers.ATTN_BACKEND, compile_blocks=True,
                               compile_adapters="compileall" in name, mode=mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--renders", type=int, default=4, help="render one view of the first N objects per run")
    ap.add_argument("--profile", action="store_true", help="torch.profiler breakdown at 100%% and 40%%")
    ap.add_argument("--ref_gallery", default=None,
                    help="fp32 gallery whose latents/<name>_bNNN.pt (same seed and noise) serve as the reference")
    ap.add_argument("--warmup_n", type=int, default=6, help="Euler steps of the untimed warm-up call")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    student, dec, dec_gs, cfg_json, step, weights = load_models(a.ckpt, dev)
    prep = Preprocessor(weights, dev, "synthetic")
    conds = [prep(p)[1] for p in a.images[: a.batch]]
    while len(conds) < a.batch:
        conds += conds[: a.batch - len(conds)]
    cond = torch.cat(conds[: a.batch], dim=0)
    # the preprocessor (BiRefNet + DINOv3 + FLUX.2 VAE) stays resident, as in the gallery and any real
    # pipeline, so the memory numbers here and there have the same baseline
    z1, c1 = object_noise(student, 42, dev)
    noise = (z1.expand(a.batch, -1, -1).clone(), c1.expand(a.batch, -1, -1).clone())
    gpu = torch.cuda.get_device_name()

    names = [os.path.splitext(os.path.basename(p))[0] for p in a.images[: a.batch]]
    names = (names * a.batch)[: a.batch]

    def view_tiles(lat, cam):
        """one view, 90 degrees round from the input view, of the first objects: a visual check"""
        import math
        import numpy as np
        tiles = []
        for i in range(min(a.renders, a.batch)):
            cam_i = orbit_camera(cam[i, 0])
            sp = Splats(decode(lat[i:i + 1].float(), dec, dec_gs, cfg_json, 262144))
            r_obj = float(torch.quantile(sp.xyz.norm(dim=1), 0.995))
            pos = orbit_positions(cam_i["yaw"], cam_i["pitch"], r_obj / math.sin(math.radians(40) / 2), 4)[1]
            tiles.append((render(sp, pos, 40, size=320).cpu().numpy() * 255).round().astype(np.uint8))
        return np.concatenate(tiles, 1)

    def gallery_ref(b):
        if not a.ref_gallery:
            return None
        fs = [os.path.join(a.ref_gallery, "latents", f"{n}_b{round(b * 100):03d}.pt") for n in names]
        if not all(os.path.exists(f) for f in fs):
            return None
        d = [torch.load(f, map_location=dev) for f in fs]
        return torch.cat([x["latent"] for x in d]).float(), torch.cat([x["camera"] for x in d]).float()

    from PIL import Image
    ref, rows = {}, []
    if a.ref_gallery and a.renders:
        for b in a.budgets:
            g = gallery_ref(b)
            if g is not None:
                Image.fromarray(view_tiles(*g)).save(os.path.join(a.out, f"view_fp32ref_b{round(b * 100):03d}.png"))
    for var in a.variants.split(","):
        set_variant(student, var)
        for b in a.budgets:
            run = lambda: sample(student, cond, b, a.steps, a.cfg, 3.0, seed=42, device=dev, warmup_steps=0,
                                 noise=(noise[0].clone(), noise[1].clone()))
            # untimed warm-up with a few steps: compile, autotune and cuDNN plans for this shape
            sample(student, cond, b, min(a.steps, a.warmup_n), a.cfg, 3.0, seed=42, device=dev, warmup_steps=0,
                   noise=(noise[0].clone(), noise[1].clone()))
            times, stats = [], None
            for _ in range(a.repeats):
                with MemProbe() as mp:
                    lat, cam, layout = run()
                times.append(mp.seconds)
                stats = mp.stats
            key = round(b, 2)
            if a.renders:
                Image.fromarray(view_tiles(lat, cam)).save(os.path.join(a.out, f"view_{var}_b{round(b * 100):03d}.png"))
            if var == "fp32":
                ref[key] = lat.float()
            r_lat = ref.get(key)
            if r_lat is None:
                g = gallery_ref(b)
                r_lat = None if g is None else g[0]
            err = None if r_lat is None else float((lat.float() - r_lat).norm() / r_lat.norm())
            r = {"variant": var, "budget": b, "tokens": layout.n_tokens, "batch": a.batch, "steps": a.steps,
                 "s_per_call": st.median(times), "s_per_obj": st.median(times) / a.batch, "mem": stats,
                 "rel_l2_vs_fp32": err, "gpu": gpu}
            rows.append(r)
            print(f"[bench] {var:20s} {b:4.0%} N={layout.n_tokens:5d}  {r['s_per_obj']:6.2f} s/obj  "
                  f"peak {stats['peak_alloc_mb'] / 1024:5.1f}G avg {stats['avg_alloc_mb'] / 1024:5.1f}G  "
                  f"err {err if err is None else f'{err:.2e}'}", flush=True)
            json.dump({"gpu": gpu, "ckpt": a.ckpt, "rows": rows}, open(os.path.join(a.out, "bench.json"), "w"), indent=1)

    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        for b in (1.0, 0.4):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                sample(student, cond, b, 10, a.cfg, 3.0, seed=42, device=dev, warmup_steps=0,
                       noise=(noise[0].clone(), noise[1].clone()))
                torch.cuda.synchronize()
            tab = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
            open(os.path.join(a.out, f"profile_{round(b * 100)}.txt"), "w").write(tab)
            print(f"[bench] profile at {b:.0%}:\n{tab}", flush=True)

    # markdown summary: speedup relative to the same variant at 100 %, and to fp32 at 100 %
    L = [f"GPU: {gpu}, batch {a.batch}, {a.steps} steps, CFG {a.cfg}", "",
         "| variant | budget | tokens | s/obj | speedup vs same variant @100% | speedup vs fp32 @100% | peak GB | avg GB | rel. L2 vs fp32 |",
         "|---|---|---|---|---|---|---|---|---|"]
    base = {r["variant"]: r["s_per_obj"] for r in rows if r["budget"] == 1.0}
    f100 = base.get("fp32")
    if f100 is None and a.ref_gallery and os.path.exists(os.path.join(a.ref_gallery, "perf_report.json")):
        # the fp32 sweep's own measurement (same GPU type, batch, steps, CFG), so fp32 need not be re-run
        pr = json.load(open(os.path.join(a.ref_gallery, "perf_report.json")))
        f100 = next((x["sample_s_per_obj"]["mean"] for x in pr["per_budget"] if x["budget"] == 1.0), None)
        L[0] += f"; fp32 reference: {a.ref_gallery}/perf_report.json ({f100:.2f} s/obj at 100%)" if f100 else ""
    for r in rows:
        L.append(f"| {r['variant']} | {r['budget']:.0%} | {r['tokens']} | {r['s_per_obj']:.2f} | "
                 f"{base[r['variant']] / r['s_per_obj']:.2f}× | {(f100 / r['s_per_obj']) if f100 else float('nan'):.2f}× | "
                 f"{r['mem']['peak_alloc_mb'] / 1024:.1f} | {r['mem']['avg_alloc_mb'] / 1024:.1f} | "
                 f"{'' if r['rel_l2_vs_fp32'] is None else f'{r['rel_l2_vs_fp32']:.1e}'} |")
    open(os.path.join(a.out, "bench.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
