#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch one training stage.

    torchrun --nproc_per_node 8 scripts/train.py --config configs/<stage>.yaml [key=value ...]
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastframes.training.config import load_config  # noqa: E402
from fastframes.wan.attention import set_cross_attention_backend  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    set_cross_attention_backend(config.get("model", {}).get("cross_attention", "cudnn"))
    stage = importlib.import_module(f"fastframes.training.{config['stage']}")
    stage.train(config)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
