"""Create ``matting`` annotations for P3M-500-NP, AM-2K-test and PPM-100.

The script links each release folder ``--raw_root/<FOLDER>`` to
``--eval_root/<FOLDER>``, matches images and alpha mattes by file name, and
writes ``--eval_root/_annotations/matting/<DATASET>.jsonl``. Each image is
queried with a single expression (``person`` or ``animal``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from evaluation.prepare.common import IMAGE_SUFFIXES, link_directory, write_jsonl

# dataset: (release folder, image directory, alpha directory, expression).
# Directories are relative to EVAL_ROOT.
DATASETS = {
    "P3M-500-NP": ("P3M-10k", "P3M-10k/validation/P3M-500-NP/original_image",
                   "P3M-10k/validation/P3M-500-NP/mask", "person"),
    "AM-2K-test": ("AM-2K", "AM-2K/validation/original", "AM-2K/validation/mask", "animal"),
    "PPM-100": ("PPM-100", "PPM-100/image", "PPM-100/matte", "person"),
}


def _by_stem(directory: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in files:
                raise ValueError(f"duplicate stem {path.stem!r} in {directory}")
            files[path.stem] = path
    return files


def convert(dataset: str, raw: Path, eval_root: Path) -> Iterable[dict]:
    folder, image_dir, alpha_dir, expression = DATASETS[dataset]
    link_directory(eval_root / folder, raw / folder)
    images, alphas = _by_stem(eval_root / image_dir), _by_stem(eval_root / alpha_dir)
    if set(images) != set(alphas):
        raise ValueError(f"{dataset}: unpaired files: no alpha {sorted(set(images) - set(alphas))[:5]}, "
                         f"no image {sorted(set(alphas) - set(images))[:5]}")
    for stem in sorted(images):
        yield {
            "sample_id": stem,
            "dataset": dataset,
            "image_path": f"{image_dir}/{images[stem].name}",
            "alpha_path": f"{alpha_dir}/{alphas[stem].name}",
            "expression": expression,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="folder holding the extracted releases")
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        output = args.eval_root / "_annotations" / "matting" / f"{dataset}.jsonl"
        write_jsonl(output, convert(dataset, args.raw_root, args.eval_root))


if __name__ == "__main__":
    main()
