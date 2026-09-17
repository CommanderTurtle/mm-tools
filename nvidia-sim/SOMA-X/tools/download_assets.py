# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def download_assets(target_dir=None, revision="main"):
    """Download SOMA assets from HuggingFace.

    Args:
        target_dir: If provided, used as the HuggingFace cache directory.
            The actual assets will be stored in a subdirectory managed by
            huggingface_hub.  If None, uses the default HF cache
            (``~/.cache/huggingface/hub/``).
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the downloaded assets directory.
    """
    from soma.assets import get_assets_dir

    path = get_assets_dir(revision=revision, cache_dir=target_dir)
    logger.info(f"Assets downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SOMA assets from HuggingFace")
    parser.add_argument(
        "--target-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface/hub/)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision to download (default: main)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)
    download_assets(target_dir=args.target_dir, revision=args.revision)
