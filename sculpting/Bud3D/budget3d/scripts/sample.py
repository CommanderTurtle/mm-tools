"""Sample Budget3D at a token budget: one image in, a .ply of 3D Gaussians out.

    python scripts/sample.py --ckpt weights/budget3d_idol.safetensors --image photo.jpg --budget 0.5 --out photo.ply
    python scripts/sample.py --ckpt weights/budget3d_idol_turbo.safetensors --turbo --image photo.jpg --budget 0.5

Budget3D-Idol runs 50 Euler steps with classifier-free guidance 3; with ``--turbo`` the checkpoint is a
Budget3D-Idol-Turbo one and runs 5 guidance-free steps.  ``--fast`` switches on bf16 blocks, fused attention and
torch.compile (worth it from a handful of objects on: the first call compiles).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

import torch

from budget3d.models.load import decode_gaussians, idol_weights_dir, load_budget3d, load_decoders
from budget3d.sampling.sampler import ImagePreprocessor, sample


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a released .safetensors file or a training checkpoint")
    ap.add_argument("--image", required=True)
    ap.add_argument("--budget", type=float, default=1.0, help="fraction of the 8,192 dense tokens (0.0625 to 1)")
    ap.add_argument("--turbo", action="store_true", help="a Budget3D-Idol-Turbo checkpoint: Idol-turbo's phase-LoRA "
                                                          "experts, 5 guidance-free steps (--cfg is ignored)")
    ap.add_argument("--lora", default=None, help="Idol-turbo's LoRA file (default: downloaded from speridlabs/idol-turbo)")
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default 50, or 5 with --turbo)")
    ap.add_argument("--cfg", type=float, default=3.0, help="classifier-free guidance strength")
    ap.add_argument("--rescale_t", type=float, default=3.0, help="Idol's time shift")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"],
                    help="Idol's crop: synthetic for renders and generated images, real for photographs")
    ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_steps", type=int, default=0, help="initial dense steps before the budget applies")
    ap.add_argument("--raw", action="store_true", help="use the raw instead of the EMA weights of a training checkpoint")
    ap.add_argument("--fast", action="store_true", help="bf16 blocks, fused attention and torch.compile")
    ap.add_argument("--out", default="sample.ply")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    weights = idol_weights_dir()
    model, step = load_budget3d(a.ckpt, turbo=a.turbo, lora=a.lora, device=dev, ema=not a.raw, weights_dir=weights)
    if a.fast:
        from budget3d.sampling.fast import optimize_for_inference
        optimize_for_inference(model)
    print(f"[sample] loaded {a.ckpt} (step {step}, {'raw' if a.raw else 'EMA'} weights)")

    prepared, cond = ImagePreprocessor(weights, dev, a.mode)(a.image)       # Idol's own framing and frozen encoders
    if a.turbo:
        from budget3d.sampling.turbo import sample_turbo
        latent, camera, layout = sample_turbo(model, cond, a.budget, a.steps or 5, a.rescale_t, seed=a.seed, device=dev)
    else:
        latent, camera, layout = sample(model, cond, a.budget, a.steps or 50, a.cfg, a.rescale_t,
                                        seed=a.seed, device=dev, warmup_steps=a.warmup_steps)
    print(f"[sample] budget {a.budget:.0%} -> {layout}")

    # Idol's frozen decoders, unchanged: the model's output is always the full 8,192-token latent
    dec, dec_gs, cfg_json = load_decoders(weights, dev)
    decode_gaussians(latent, dec, dec_gs, cfg_json, a.num_gaussians).save_ply(a.out)
    prepared.save(os.path.splitext(a.out)[0] + "_input.png")
    print("[sample] wrote", a.out)


if __name__ == "__main__":
    main()
