#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generate the 6,220-video VBench corpus with a released FastFrames model.

Sampling and filenames follow ``protocol.py``; valid existing videos are skipped.
See docs/inference.md#evaluation for sharding commands and the scoring protocol.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="released model name, e.g. ff-s or ff-l")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    protocol.add_suite_arguments(parser)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")

    protocol.require_encoder()  # OpenH264 for the MP4s
    suite = protocol.load_suite(args.protocol, args.full_info, args.original_prompts, args.extended_prompts)
    jobs = protocol.build_sample_specs(suite)[args.shard::args.num_shards]
    pending = [job for job in jobs if not protocol.is_valid_video(args.output_dir / job.filename)]
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} jobs, {len(pending)} to generate", flush=True)
    if not pending:
        return 0

    from scripts.generate import load_model

    model = load_model(args.model, args.checkpoint_dir)
    for job in pending:
        start = time.perf_counter()
        frames = model.generate(job.conditioning_prompt, job.seed)
        protocol.write_faststart_video(args.output_dir / job.filename, frames)
        print(f"{job.filename}  seed={job.seed}  {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
