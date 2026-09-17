from __future__ import annotations

"""Validate a frozen scene-geometry manifest against its source dataset."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.build_manifests import (  # noqa: E402
    build_manifest,
)
from benchmarks.scene_reconstruction.geometry_protocol import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--mask-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = load_manifest(args.manifest.resolve())
    rebuilt = build_manifest(
        fire3d_test_root=args.fire3d_test_root.resolve(),
        dataset=expected["dataset"],
        video_id=int(expected["video_id"]),
        min_visible_pixels=int(expected["object_set"]["min_visible_pixels"]),
        mask_workers=args.mask_workers,
    )
    if rebuilt != expected:
        expected_scenes = {scene["scene_id"]: scene for scene in expected["scenes"]}
        rebuilt_scenes = {scene["scene_id"]: scene for scene in rebuilt["scenes"]}
        summary = {
            "manifest": str(args.manifest),
            "valid": False,
            "expected_totals": expected.get("totals"),
            "rebuilt_totals": rebuilt.get("totals"),
            "missing_scenes": sorted(set(expected_scenes) - set(rebuilt_scenes)),
            "extra_scenes": sorted(set(rebuilt_scenes) - set(expected_scenes)),
            "changed_scenes": sorted(
                scene_id
                for scene_id in set(expected_scenes) & set(rebuilt_scenes)
                if expected_scenes[scene_id] != rebuilt_scenes[scene_id]
            ),
        }
        print(json.dumps(summary, indent=2))
        raise SystemExit(1)
    print(json.dumps({"manifest": str(args.manifest), "valid": True, **expected["totals"]}, indent=2))


if __name__ == "__main__":
    main()
