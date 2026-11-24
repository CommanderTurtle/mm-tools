#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The videos of the project page and of the qualitative figures (Figs. 3 and 4).

    python scripts/generate_project_page.py --model ff-s --output-dir outputs/project_page/ff-s
    python scripts/generate_project_page.py --model ff-l --output-dir outputs/project_page/ff-l
    # Causal Forcing++: the same generator with an extra DiT pass for the KV cache and the Wan VAE
    python scripts/generate_project_page.py --model ff-s --context dit --decoder vae \
        --output-dir outputs/project_page/cfpp

By default the script renders the 22 prompts of the project page, in page order. Prompt ``i`` of
``--prompt-file`` (default ``data/prompts/demos.txt``) uses seed ``(seed + 1000003 * i) mod 2**32``.
For each prompt it writes ``p<iiii>.mp4``, optional PNG stills at the ``--stills`` times
(``stills/p<iiii>_<t>s.png``) and one line of ``records.jsonl``, which includes the SHA-256 of the
BF16 latents.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.io import write_png

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastframes.pipeline import CONTEXT_MODES, MODELS, FastFrames, configure_backends, to_uint8  # noqa: E402
from fastframes.video import save_video  # noqa: E402
from fastframes.wan.vae import VideoVAE  # noqa: E402

FPS = 16
SEED_STRIDE = 1000003
PROJECT_PAGE = [92, 48, 31, 90, 94, 103, 73, 16, 17, 20, 24, 5, 13, 30, 33, 36, 44, 61, 75, 85, 106, 108]
# Transition-noise conventions of the project-page and figure videos: for 1.3B, a second generator
# seeded like the initial noise; FastFrames-L, its release convention.
PAGE_NOISE = {"ff-s": "reseeded", "ff-l": "pregenerated"}


def parse_indices(text, count):
    if text is None:
        return PROJECT_PAGE
    if text == "all":
        return list(range(count))
    indices = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        indices.extend(range(int(first), int(last or first) + 1))
    return indices


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def make_decoder(args, model, wan_dir):
    """``decode(latents, features, modulation) -> [81, 3, 480, 832]`` in ``[-1, 1]``."""
    if args.decoder == "pixel":
        return lambda latents, features, modulation: model.decode(features, modulation)[0]
    vae = VideoVAE(Path(wan_dir) / "Wan2.1_VAE.pth").to(model.device, torch.bfloat16)
    return lambda latents, features, modulation: vae.decode_to_pixel(latents)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--context", choices=CONTEXT_MODES, default="refiner",
                        help="how finished frames enter the KV cache (see FastFrames.rollout)")
    parser.add_argument("--decoder", choices=("pixel", "vae"), default="pixel")
    parser.add_argument("--noise", choices=("sequential", "pregenerated", "reseeded"),
                        help="transition-noise convention (default: that of the project page)")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "data" / "prompts" / "demos.txt")
    parser.add_argument("--indices", help="prompt indices, e.g. 0,4,10-12, or 'all' "
                        "(default: the project page)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stills", default="", help="comma-separated times in seconds, e.g. 0,5 (the figures)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = parse_indices(args.indices, len(prompts))[args.shard::args.num_shards]
    stills = [float(t) for t in args.stills.split(",") if t]
    spec, root = MODELS[args.model], args.checkpoint_dir
    config = dataclasses.replace(spec.config, noise=args.noise or PAGE_NOISE[args.model])
    configure_backends(config)
    model = FastFrames.from_pretrained(config, root / spec.wan_dir, root / spec.dit, root / spec.refiner,
                                       root / spec.decoder)
    decode = make_decoder(args, model, root / spec.wan_dir)
    out = args.output_dir
    (out / "stills" if stills else out).mkdir(parents=True, exist_ok=True)
    for index in indices:
        start = time.perf_counter()
        prompt, seed = prompts[index], (args.seed + SEED_STRIDE * index) % 2**32
        latents, features, modulation = model.rollout(model.encode_prompt(prompt), seed, context=args.context)
        with torch.inference_mode():
            frames = to_uint8(decode(latents, features, modulation)).cpu()
        save_video(out / f"p{index:04d}.mp4", frames, fps=FPS)
        for t in stills:
            frame = min(round(t * FPS), len(frames) - 1)
            write_png(frames[frame].permute(2, 0, 1).contiguous(), str(out / "stills" / f"p{index:04d}_{t:g}s.png"))
        record = {"index": index, "seed": seed, "prompt": prompt, "model": args.model, "context": args.context,
                  "decoder": args.decoder, "noise": config.noise, "latent_sha256": tensor_sha256(latents),
                  "frames_sha256": hashlib.sha256(frames.numpy().tobytes()).hexdigest()}
        with open(out / "records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"p{index:04d} seed={seed} {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
