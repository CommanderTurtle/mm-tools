from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .core import remove_object


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove a masked object without a cloud service")
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path, help="White/alpha marks the removal area")
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("telea", "navier-stokes"), default="telea")
    parser.add_argument("--radius", type=float, default=5)
    parser.add_argument("--grow", type=int, default=5)
    parser.add_argument("--feather", type=int, default=4)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise SystemExit("Could not read the image or mask.")
    result = remove_object(image, mask, method=args.method, radius=args.radius, grow=args.grow, feather=args.feather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise SystemExit(f"Could not save {args.output}")


if __name__ == "__main__":
    main()
