#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Steady-state latency of FastFrames-S/-L, Causal Forcing++ and Self-Forcing (Table 1).

    python evaluation/latency/benchmark.py ff-s --checkpoint-dir checkpoints            # 1 GPU
    python evaluation/latency/benchmark.py ff-s --checkpoint-dir checkpoints --gpus 8   # context parallel
    python evaluation/latency/benchmark.py cfpp --checkpoint-dir checkpoints --upstream upstream/Causal-Forcing

Every component of one generation step is timed in its own process (``profile_components.py``,
or ``baselines.py`` for the upstream baseline DiTs; ``torchrun`` for ``--gpus 8``) with a full
KV cache of 21 latent frames. The step latency is the sum of the component medians:

* FastFrames: ``DiT(t=1000) + DiT(t=833) + KV refinement + pixel decoder`` (one latent frame);
* Causal Forcing++: ``3 x DiT(1 latent) + VAE(1 latent)`` (one latent frame);
* Self-Forcing: ``5 x DiT(3 latents) + 3 x VAE(1 latent)`` (three latent frames).

The component records are written to ``--output-dir``; ``summarize.py`` reads them as well.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fastframes.pipeline import MODELS  # noqa: E402

PAPER_MS = {("ff-s", 1): 36, ("ff-s", 8): 14, ("ff-l", 1): 158, ("ff-l", 8): 45,
            ("cfpp", 1): 272, ("self-forcing", 1): 717}
UPSTREAM = {"cfpp": "Causal-Forcing", "self-forcing": "Self-Forcing"}
NAMES = {"ff-s": "FastFrames-S", "ff-l": "FastFrames-L", "cfpp": "Causal Forcing++", "self-forcing": "Self-Forcing"}


def components(args):
    """``(label, count, script, arguments, record name)`` of every term of the step latency."""
    root = args.checkpoint_dir
    vae = ["vae", "--vae", str(root / MODELS["ff-s"].wan_dir / "Wan2.1_VAE.pth")]
    if args.method in UPSTREAM:
        upstream = str((args.upstream or Path("upstream") / UPSTREAM[args.method]).resolve())
        if args.method == "cfpp":
            return [("DiT evaluation, 1 latent frame", 3, "baselines.py", ["--upstream", upstream, "--chunk", "1"],
                     "dit"),
                    ("VAE decode, 1 latent frame", 1, "profile_components.py", vae, "vae")]
        return [("DiT evaluation, 3 latent frames", 5, "baselines.py", ["--upstream", upstream, "--chunk", "3"],
                 "dit"),
                ("VAE decode, 1 latent frame", 3, "profile_components.py", vae, "vae")]
    spec = MODELS[args.method]
    model = {"ff-s": "1.3b", "ff-l": "14b"}[args.method]
    dit = ["--model", model, "--dit", str(root / spec.dit)]
    if args.gpus > 1:
        kv = ["--shard", "tokens" if model == "1.3b" else "layers"]
    else:
        kv = ["--streams", "6"] if model == "1.3b" else []
    return [("DiT evaluation, t=1000", 1, "profile_components.py", ["dit", *dit, "--timestep", "1000"], "dit_t1000"),
            ("DiT evaluation, t=833", 1, "profile_components.py", ["dit", *dit], "dit_t833"),
            ("KV refinement", 1, "profile_components.py",
             ["kv-refine", *dit, "--refiner", str(root / spec.refiner), *kv], "kv_refine"),
            ("pixel decoder", 1, "profile_components.py",
             ["pixel-decoder", "--model", model, "--decoder", str(root / spec.decoder)], "pixel_decoder")]


def run(script, arguments, output, gpus):
    launch = [sys.executable]
    if gpus > 1:
        launch += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(gpus)]
    subprocess.run([*launch, script, *arguments, "--output", str(output)], cwd=HERE, check=True)
    return json.loads(output.read_text())["latency_ms"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=sorted(NAMES))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="released weights and Wan2.1 base models (see the top-level README)")
    p.add_argument("--gpus", type=int, choices=(1, 8), default=1)
    p.add_argument("--upstream", type=Path, help="clone of the baseline's repository "
                   "(default: upstream/Causal-Forcing or upstream/Self-Forcing, see docs/installation.md#baselines)")
    p.add_argument("--output-dir", type=Path, help="default: outputs/latency/<method>_<gpus>gpu")
    args = p.parse_args()
    if args.gpus > 1 and args.method in UPSTREAM:
        p.error("the baselines are measured on one GPU")
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    out = (args.output_dir or Path("outputs") / "latency" / f"{args.method}_{args.gpus}gpu").resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, count, script, arguments, name in components(args):
        rows.append((label, count, run(script, arguments, out / f"{name}.json", args.gpus)))
    total = sum(count * ms for _, count, ms in rows)
    paper = PAPER_MS.get((args.method, args.gpus))
    (out / "latency.json").write_text(json.dumps(
        {"method": args.method, "gpus": args.gpus, "resolution": "480p", "total_ms": total, "paper_ms": paper,
         "components": [{"component": label, "count": count, "ms": ms} for label, count, ms in rows]},
        indent=2) + "\n")

    frames = "three latent frames (12 video frames)" if args.method == "self-forcing" else \
        "one latent frame (4 video frames)"
    print(f"\n{NAMES[args.method]}, {args.gpus} GPU{'s' if args.gpus > 1 else ''}, 480p: "
          f"latency of one step, {frames}")
    for label, count, ms in rows:
        print(f"  {count} x {label:32s} {ms:7.2f} ms  {count * ms:8.2f} ms")
    print(f"  {'total':47s} {total:8.2f} ms   (paper: {paper} ms)")


if __name__ == "__main__":
    main()
