#!/usr/bin/env python3
"""Unified two-protocol renderer for percept+recon results, all datasets.

Exactly two figure protocols, applied over dataset-default or YAML-supplied
cameras:

  geometry  instance-color mesh rendering: golden-ratio diverse palette, the
            reconstructed room painted flat gray (or excluded when
            render.background is false).
  texture   textured mesh rendering under the Aug-28 inference-rgb protocol:
            Blender 4.5.1 BLENDER_EEVEE_NEXT, canonical_pbr recipe, 32 samples,
            two-sided camera+shadow rendering, verified-opaque BLENDED->DITHERED
            material promotion (eval/rendering/blender_glb_material_policy.py).

Camera pose, intrinsics, and image resolution come from a YAML config
(--config); anything omitted falls back to the dataset's own cameras:

    protocols: [geometry, texture]
    resolution: {width: 960, height: 540}     # optional; K is rescaled to fit
    intrinsics: {K: [[fx,0,cx],[0,fy,cy],[0,0,1]]}   # optional override
    cameras:
      source: dataset            # dataset | explicit
      num_views: 8
      seed: 20260901
      frames:                    # only for source: explicit
        - {eye: [x,y,z], lookat: [x,y,z], up: [x,y,z]}
    render:
      samples: 32
      background: true           # false = exclude the room instance entirely
      gray_rgba: [0.5, 0.5, 0.5, 1.0]

Input: a run root with reconstruction/<scene_id> (unified layout) or
reconstruction/<dataset>/<scene_id> (legacy). Output: per-scene per-protocol
sheets under <run-root>/visualization plus render_index.json with absolute
figure paths -- the same contract for every dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = Path(sys.executable)
BLENDER = REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"
BLENDER_ENTRY = REPO_ROOT / "eval/rendering/scene_comparison_blender.py"
PROTOCOLS = ("geometry", "texture")
DATASETS = ("ithor", "imaginarium", "scannetpp", "single_image")

from eval.unified_protocol import (  # noqa: E402
    load_protocol,
    native_dataset_environment,
    protocol_identity,
    protocol_render_config,
    validate_scope,
)
from eval.render_protocol import (  # noqa: E402
    load_render_protocol,
    render_protocol_config,
    render_protocol_identity,
    validate_render_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--source",
        choices=("recon", "gt", "comparison"),
        default="recon",
        help=(
            "recon (default): render a reconstruction run root. gt: render the "
            "dataset's ground-truth scene (objects + layout, dataset-world "
            "transforms). comparison: one GT-texture / ours-geometry / "
            "ours-texture row per camera -- ithor/imaginarium only."
        ),
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="YAML render config")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Frozen inference protocol name or JSON path. Its render block is "
            "used unless --render-protocol supplies an independent override."
        ),
    )
    parser.add_argument(
        "--render-protocol",
        default=None,
        help=(
            "Frozen render-only protocol name or JSON path controlling raster, "
            "lighting, material, shading, and output semantics."
        ),
    )
    parser.add_argument(
        "--fire3d-test-root", type=Path, default=REPO_ROOT / "data"
    )
    parser.add_argument(
        "--background",
        choices=("config", "on", "off", "both"),
        default="config",
        help=(
            "Override render.background from the YAML. 'both' renders each "
            "protocol twice and suffixes the outputs _bg / _nobg."
        ),
    )
    parser.add_argument(
        "--background-instance",
        choices=("prior", "largest", "zero"),
        default="prior",
        help=(
            "How to identify the room instance. 'prior' uses the recorded "
            "room-box prior, or the explicit layout_<scene> object in oracle "
            "GT-perception runs, and yields None when neither exists. 'zero' "
            "takes instance id 0. "
            "'largest' takes the instance with the most faces -- correct for "
            "imported baselines like Gen3DSR whose background mesh dominates, "
            "and deliberately NOT the default, since on a run without a room it "
            "would strip the biggest foreground object."
        ),
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=("normal", *sorted(BEAUTY_PRESETS)),
        help=(
            "Presentation style for the TEXTURE protocol; the geometry protocol "
            "always renders 'normal'. Default depends on --source (user "
            "directive 2026-09-03): our reconstructions (--source recon, either "
            "perception mode) default to 'beauty_v0', while ground-truth "
            "renders (--source gt) stay 'normal'. Pass --style normal to get "
            "the measurement-faithful Aug-28 look for a recon render too. The "
            "resolved style and its parameters are recorded in the render index."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.scene_id:
        parser.error("Pass --scene-id (repeatable) or --all")
    if args.source in ("gt", "comparison"):
        if args.dataset not in ("ithor", "imaginarium"):
            parser.error(
                f"--source {args.source} requires GT scene assets (ithor, imaginarium)"
            )
        if args.all:
            parser.error(f"--source {args.source} requires explicit --scene-id")
        if args.output_root is None:
            parser.error(f"--source {args.source} requires --output-root")
    if args.source in ("recon", "comparison"):
        if args.run_root is None:
            parser.error(f"--run-root is required for --source {args.source}")
        if args.output_root is None:
            args.output_root = args.run_root / "visualization"
    return args


def load_config(
    path: Path | None,
    protocol: dict | None = None,
    render_protocol: dict | None = None,
) -> dict:
    config: dict = {}
    if path is not None:
        import yaml

        config = yaml.safe_load(path.read_text()) or {}
    if protocol is not None:
        frozen = protocol_render_config(protocol)
        config["protocols"] = frozen["protocols"]
        config.setdefault("render", {}).update(frozen["render"])
    if render_protocol is not None:
        frozen = render_protocol_config(render_protocol)
        config["protocols"] = frozen["protocols"]
        config.setdefault("render", {}).update(frozen["render"])
    config.setdefault("protocols", list(PROTOCOLS))
    cameras = config.setdefault("cameras", {})
    cameras.setdefault("source", "dataset")
    cameras.setdefault("num_views", 8)
    cameras.setdefault("seed", 20260901)
    render = config.setdefault("render", {})
    render.setdefault("engine", "BLENDER_EEVEE_NEXT")
    render.setdefault("recipe", "canonical_pbr")
    render.setdefault("render_profile", "inference-rgb")
    render.setdefault("samples", 32)
    render.setdefault("background", True)
    render.setdefault("gray_rgba", [0.5, 0.5, 0.5, 1.0])
    render.setdefault("geometry_palette", "diverse")
    render.setdefault("gt_geometry_background", render["background"])
    render.setdefault("gt_texture_background", render["background"])
    render.setdefault("reconstruction_geometry_background", render["background"])
    render.setdefault("reconstruction_texture_background", render["background"])
    render.setdefault("style_applies_to", ["texture"])
    render.setdefault(
        "camera",
        {"type": "PERSP", "intrinsics": "task_exact", "clip_start": 0.01,
         "clip_end": 1000.0},
    )
    render.setdefault(
        "output",
        {"file_format": "PNG", "color_mode": "RGB", "color_depth": "8",
         "film_transparent": False, "empty_background_rgb": [255, 255, 255],
         "use_file_extension": True},
    )
    render.setdefault(
        "empty_background_rgb", render["output"]["empty_background_rgb"]
    )
    unknown = set(config["protocols"]) - set(PROTOCOLS)
    if unknown:
        raise SystemExit(f"unknown protocols {sorted(unknown)}; valid: {PROTOCOLS}")
    return config


def find_recon_dir(run_root: Path, dataset: str, scene_id: str) -> Path | None:
    for candidate in (
        run_root / "reconstruction" / scene_id,
        run_root / "reconstruction" / dataset / scene_id,
    ):
        if (candidate / "inference_summary.json").is_file():
            return candidate
    return None


def list_scenes(run_root: Path, dataset: str) -> list[str]:
    scenes: set[str] = set()
    for base in (run_root / "reconstruction", run_root / "reconstruction" / dataset):
        if base.is_dir():
            for child in base.iterdir():
                if (child / "inference_summary.json").is_file():
                    scenes.add(child.name)
    return sorted(scenes)


def frames_from_c2ws(c2ws: np.ndarray, indices: list[int]) -> list[dict]:
    """OpenCV c2w poses -> eye/lookat/up view frames."""
    frames = []
    for index in indices:
        pose = np.asarray(c2ws[index], dtype=np.float64)
        eye = pose[:3, 3]
        forward = pose[:3, 2]
        up = -pose[:3, 1]
        frames.append(
            {
                "eye": eye.tolist(),
                "lookat": (eye + forward).tolist(),
                "up": up.tolist(),
                "forward": forward.tolist(),
            }
        )
    return frames


def pick_indices(total: int, num_views: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if total <= num_views:
        return list(range(total))
    return sorted(rng.choice(total, size=num_views, replace=False).tolist())


# ---------------------------------------------------------------------------
# Dataset camera emitters: (K 3x3 list, width, height, frames) in the frame of
# the composed world GLB.
# ---------------------------------------------------------------------------

def cameras_scannetpp(args, cameras_cfg: dict, scene_id: str):
    os.environ.setdefault("FF_SCANNETPP_WHITELIST", "0")
    from eval.rendering.render_scannetpp_recon_views import camera_frames, scene_alignment

    _, intrinsics, c2ws, center, angle, _ = scene_alignment(scene_id)
    indices = pick_indices(len(c2ws), cameras_cfg["num_views"], cameras_cfg["seed"])
    frames = camera_frames(c2ws, center, angle, indices)
    K = np.asarray(intrinsics[0], dtype=np.float64)
    # native scannetpp resolution from the intrinsics' principal point context
    from utils.data_scannetpp import load_scannetpp_data, load_scene_frames

    entry = next(d for d in load_scannetpp_data() if d["scene_id"] == scene_id)
    rgbs, *_ = load_scene_frames(entry, max_num_frames=1)
    height, width = int(rgbs.shape[1]), int(rgbs.shape[2])
    return K.tolist(), width, height, frames


def cameras_fire3d(args, cameras_cfg: dict, scene_id: str):
    """ithor / imaginarium: dataset cameras; recon world == dataset world."""
    from utils.read_frames import read_cameras

    subdir = "Imaginarium" if args.dataset == "imaginarium" else "ithor"
    scene_dir = args.fire3d_test_root / subdir / "renders" / scene_id
    candidates = sorted(scene_dir.glob("*.json"))
    if not candidates:
        raise RuntimeError(f"no camera file under {scene_dir}")
    camera_path = candidates[0]  # fire3d layout: <video_index>.json, e.g. 0.json
    intrinsics, c2ws, (height, width) = read_cameras(str(camera_path))
    indices = pick_indices(len(c2ws), cameras_cfg["num_views"], cameras_cfg["seed"])
    return (
        np.asarray(intrinsics[0], dtype=np.float64).tolist(),
        int(width),
        int(height),
        frames_from_c2ws(np.asarray(c2ws), indices),
    )


def cameras_single_image(args, cameras_cfg: dict, scene_id: str):
    from utils.data_single_image import scene_camera

    cam = scene_camera(scene_id, native_resolution=True)
    return cam["K"], cam["width"], cam["height"], [dict(cam["frame"], forward=cam["forward"])]


CAMERA_EMITTERS = {
    "scannetpp": cameras_scannetpp,
    "ithor": cameras_fire3d,
    "imaginarium": cameras_fire3d,
    "single_image": cameras_single_image,
}


def resolve_cameras(args, config: dict, scene_id: str):
    cameras_cfg = config["cameras"]
    if cameras_cfg["source"] == "explicit":
        frames = [
            dict(frame, forward=(np.asarray(frame["lookat"]) - np.asarray(frame["eye"])).tolist())
            for frame in cameras_cfg["frames"]
        ]
        K = config["intrinsics"]["K"]
        width = config["resolution"]["width"]
        height = config["resolution"]["height"]
        return K, width, height, frames
    K, width, height, frames = CAMERA_EMITTERS[args.dataset](args, cameras_cfg, scene_id)
    if "intrinsics" in config:
        K = config["intrinsics"]["K"]
    if "resolution" in config:
        # rescale K to the requested resolution
        new_w = int(config["resolution"]["width"])
        new_h = int(config["resolution"]["height"])
        K = np.asarray(K, dtype=np.float64)
        K[0] *= new_w / width
        K[1] *= new_h / height
        K = K.tolist()
        width, height = new_w, new_h
    return K, width, height, frames


def background_variants(args, config: dict) -> list[tuple[str, bool]]:
    """(label, flag) pairs to render. 'both' writes two labelled subfolders."""

    choice = getattr(args, "background", "config")
    if choice == "both":
        return [("bg", True), ("nobg", False)]
    if choice == "on":
        return [("", True)]
    if choice == "off":
        return [("", False)]
    return [("", bool(config["render"]["background"]))]


def apply_background_variant(
    config: dict, enabled: bool, *, override_protocol_defaults: bool
) -> None:
    """Apply one variant, including protocol-specific flags when overridden."""

    render = config["render"]
    render["background"] = bool(enabled)
    if not override_protocol_defaults:
        return
    for source in ("gt", "reconstruction"):
        for protocol in PROTOCOLS:
            render[f"{source}_{protocol}_background"] = bool(enabled)


def protocol_background(config: dict, source: str, protocol: str) -> bool:
    render = config["render"]
    key = f"{source}_{protocol}_background"
    return bool(render.get(key, render["background"]))


def task_view_index(frame: dict, sequence_index: int) -> int:
    return int(frame.get("source_frame_index", sequence_index))


def should_invoke_renderer(expected: list[Path], skip_existing: bool) -> bool:
    """Reuse render files only when the caller explicitly requests it."""

    return not skip_existing or not all(path.is_file() for path in expected)


def task_aspect_policy(config: dict, width: int, height: int) -> dict:
    diagnostics = (config.get("sampling") or {}).get("diagnostics") or {}
    native = diagnostics.get("native_resolution")
    if native:
        return {
            "name": "preserve_vertical_fov_expand_horizontal",
            "source_size": [int(native[1]), int(native[0])],
            "target_size": [int(width), int(height)],
        }
    return {
        "name": "native_dataset_camera",
        "source_size": [int(height), int(width)],
        "target_size": [int(width), int(height)],
    }


def room_instance_id(recon_dir: Path, strategy: str = "prior") -> int | None:
    """Background instance id under the requested resolution strategy.

    `prior` first reads the recorded room-box prior. Oracle GT-perception runs
    instead identify their room explicitly as `layout_<scene_id>`; that exact
    name is accepted only under the oracle reconstruction scope. External
    baselines and foreground-only GT runs therefore still resolve to None.
    There is deliberately no automatic largest-node fallback: on a run without
    a room it would strip the biggest foreground object. `largest` and `zero`
    remain opt-in strategies for imported baselines.
    """
    summary = json.loads((recon_dir / "inference_summary.json").read_text())
    if strategy == "prior":
        prior = (summary.get("input_audit") or {}).get("background_room_box_prior") or {}
        room_id = prior.get("background_local_instance_id")
        if room_id is None:
            scope = (summary.get("settings") or {}).get("scope")
            expected_name = f"layout_{summary.get('scene_id', '')}"
            candidates = [
                record
                for record in summary.get("objects") or []
                if scope == "geometry_and_appearance_oracle_id_and_pose"
                and record.get("object_name") == expected_name
                and (record.get("object_id") is not None or record.get("instance_id") is not None)
            ]
            if not candidates:
                return None
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one oracle layout object named {expected_name!r}, "
                    f"found {len(candidates)}"
                )
            room_id = candidates[0].get("object_id", candidates[0].get("instance_id"))
        room_id = int(room_id)
        # The room-box prior is authoritative for predicted-perception runs.
        # Do not remap it through `background_position_0000` or the appearance
        # background flag: the composer labels source instance 0 that way even
        # when perception identified a nonzero instance as the room. We inspect
        # composed metadata only to verify that the prior instance survived.
        composed = recon_dir / "appearance/appearance_summary.json"
        if composed.is_file():
            payload = json.loads(composed.read_text())
            records = payload.get("composed_world_scene", {}).get("objects", [])
            # The prior names an instance perception found, but reconstruction
            # can still drop it: at a coarse feature stride a small instance may
            # receive no conditioning points. Absent from the composition means
            # the same thing as no reconstructed room.
            present = {
                int(o["instance_id"])
                for o in records
                if o.get("instance_id") is not None
            }
            if present and room_id not in present:
                print(
                    f"[render] room instance {room_id} is not in the composed scene "
                    f"({sorted(present)}); treating the scene as having no room",
                    flush=True,
                )
                return None
        return room_id
    records = summary.get("objects") or []
    if not records:
        return None
    if strategy == "zero":
        return 0 if any(
            int(r.get("instance_id", r.get("object_id", -1))) == 0 for r in records
        ) else None
    if strategy == "largest":
        sized = [r for r in records if r.get("faces")]
        if not sized:
            return None
        return int(max(sized, key=lambda r: int(r["faces"]))["instance_id"])
    raise ValueError(f"Unknown background-instance strategy: {strategy}")


# Presentation-only styles applied on top of the canonical recipe. Each knob is
# consumed by apply_beauty_style in render_scene_comparison_blender.py and the
# chosen preset is recorded per scene in render_index.json.
BEAUTY_PRESETS: dict[str, dict] = {
    "b1_agx": {"view_transform": "AgX"},
    "b2_agx_punchy": {"view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3},
    "b3_softkey": {"soft_key": True},
    "b4_agx_softkey": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3,
        "soft_key": True,
    },
    "b5_agx_sky": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3,
        "sky_world": True, "sky_strength": 0.5,
    },
    "b6_agx_softkey_smooth": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3,
        "soft_key": True, "auto_smooth_degrees": 40,
    },
    "b7_lift": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3,
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.10, "roughness_floor": 0.35,
    },
    # Round 2: round 1 showed every AgX arm underexposed (+0.3 EV does not
    # offset AgX midtone compression against a recipe tuned for Standard) and
    # the b8 vignette veiling the whole frame. These re-expose at +1.3 EV and
    # drop the compositor.
    "b9_agx_bright": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 1.3,
    },
    "b10_agx_softkey_bright": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 1.3,
        "soft_key": True, "auto_smooth_degrees": 40,
    },
    "b11_lift_bright": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 1.3,
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.10, "roughness_floor": 0.35,
    },
    "b12_softkey_smooth_lift": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.10, "roughness_floor": 0.35,
    },
    # beauty_v0, locked 2026-09-03 to round-8 `p4_repair_aggressive` (user
    # directive). Lineage: c5_combo (round 3) -> d2_dim (round 4, sun_scale
    # fixed the bright-scene blowout) -> p4 (round 8, weighted normals + the
    # most aggressive dark-region repair, max_size 150 k with fill blur).
    # Every bake-off arm survives as its own named preset, so this lock is
    # reversible and the comparison runs stay reproducible.
    #
    # Recorded caveat: at max_size 150 k the repair fills regions with no valid
    # colour nearby, inventing it from distant borders -- on computer_room_03
    # that exposes bright patches under the desk which p3 (max_size 20 k) leaves
    # dark. This is a presentation style, never a measurement figure; the
    # texture-only restriction and the index provenance block enforce that.
    "beauty_v0": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic",
        "weighted_normal": True,
        "despeckle_components": True,
        "despeckle_max_size": 150000,
        "despeckle_contrast": 0.20,
        "despeckle_blur": True,
    },
    # Material-side smoothing (round 5). "matte" isolates the material knobs on
    # the plain canonical lighting; "beauty_v0_matte" stacks them on beauty_v0.
    # Cubic texture interpolation smooths texel noise and chart-seam gradient
    # breaks on our upscaled 512 atlases; metallic_scale tames noisy predicted
    # metallic; low specular kills the plastic sheen of flat-shaded surfaces.
    "matte": {
        "texture_interpolation": "Cubic",
        "metallic_scale": 0.2,
        "specular_level": 0.2,
        "sheen_weight": 0.1,
        "roughness_floor": 0.35,
    },
    "beauty_v0_matte": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic",
        "metallic_scale": 0.2,
        "specular_level": 0.2,
        "sheen_weight": 0.1,
    },
    # Round 8: exact-component dark-region repair, replacing the v1 opening.
    # Measured: every dark component sits in a much brighter border (gap
    # 0.28-0.72), so they are all holes, not real dark surfaces -- size now
    # only governs fill quality. p1 conservative, p2/p3 catch the desk smudges,
    # p4 aggressive, p5 adds geometry+material smoothing on top.
    "p1_repair_small": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle_components": True,
        "despeckle_max_size": 2000,
        "despeckle_contrast": 0.15,
    },
    "p2_repair_medium": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle_components": True,
        "despeckle_max_size": 20000,
        "despeckle_contrast": 0.15,
    },
    "p3_repair_normal": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle_components": True,
        "despeckle_max_size": 20000,
        "despeckle_contrast": 0.15,
        "weighted_normal": True,
    },
    "p4_repair_aggressive": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle_components": True,
        "despeckle_max_size": 150000,
        "despeckle_contrast": 0.2,
        "despeckle_blur": True,
        "weighted_normal": True,
    },
    "p5_repair_full": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle_components": True,
        "despeckle_max_size": 20000,
        "despeckle_contrast": 0.15,
        "weighted_normal": True,
        "smooth_factor": 0.4,
        "smooth_iterations": 2,
        "metallic_scale": 0.0,
        "specular_level": 0.15,
        "sheen_weight": 0.15,
    },
    # Round 7: normal smoothing and dark-speck removal. All on the chosen d2
    # lighting + Cubic, with dark_lift REPLACED by the size-selective despeckle
    # where noted -- our atlases hold both tiny specks (median 4-17 texels, the
    # visible dots) and huge dark regions (up to 116 k texels, unfilled atlas
    # and real undersides), and a blanket black-point lift grays out the latter.
    "n1_weighted_normal": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "dark_lift": 0.14,
        "weighted_normal": True,
    },
    "n2_geo_smooth": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "dark_lift": 0.14,
        "smooth_factor": 0.5,
        "smooth_iterations": 2,
    },
    "n3_despeckle": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "despeckle": True,
    },
    "n4_normal_despeckle": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "weighted_normal": True,
        "despeckle": True,
    },
    "n5_max": {
        "soft_key": True,
        "auto_smooth_degrees": 40,
        "roughness_floor": 0.35,
        "sun_scale": 0.25,
        "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82),
        "world_color": (0.92, 0.9, 0.87),
        "world_strength": 0.4,
        "grade_saturation": 1.1,
        "grade_contrast": 0.03,
        "raytracing": True,
        "samples": 64,
        "texture_interpolation": 'Cubic',
        "weighted_normal": True,
        "smooth_factor": 0.4,
        "smooth_iterations": 2,
        "despeckle": True,
        "metallic_scale": 0.0,
        "specular_level": 0.15,
        "sheen_weight": 0.15,
    },
    # Round 6: material candidates, all on the chosen beauty_v0 (d2) lighting so
    # only the material treatment varies. m1 isolates interpolation; m2 kills
    # metallic entirely; m3 pushes toward velvet/fabric; m4 adds a touch of
    # subsurface; m5 goes the opposite way -- clean controlled gloss.
    "m1_cubic": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic",
    },
    "m2_nometal": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic", "metallic_scale": 0.0,
    },
    "m3_velvet": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.45,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic", "metallic_scale": 0.0,
        "specular_level": 0.1, "sheen_weight": 0.25,
    },
    "m4_subsurface": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic", "metallic_scale": 0.2,
        "specular_level": 0.2, "subsurface_weight": 0.04,
    },
    "m5_gloss": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.25,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
        "texture_interpolation": "Cubic", "metallic_scale": 0.2,
        "specular_level": 0.4,
    },
    # Round 4: v0 overexposed bright scenes because its key/world/GI stacked on
    # the untouched canonical sun. These scale the sun down and lower the key.
    "d1_sunscale": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.35, "key_energy": 250.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.45,
        "grade_saturation": 1.10, "grade_contrast": 0.04,
        "raytracing": True, "samples": 64,
    },
    "d2_dim": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.25, "key_energy": 200.0,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.40,
        "grade_saturation": 1.10, "grade_contrast": 0.03,
        "raytracing": True, "samples": 64,
    },
    "d3_neutral": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "sun_scale": 0.35, "key_energy": 250.0,
        "key_color": (1.0, 0.97, 0.92), "world_color": (0.94, 0.93, 0.92),
        "world_strength": 0.45,
        "grade_saturation": 1.10, "grade_contrast": 0.04,
        "raytracing": True, "samples": 64,
    },
    # Round 3, all on the Standard-transform winner (b12): Filmic middle
    # ground, compositor grade instead of a transform change, warm key + lifted
    # warm world, and Eevee raytraced GI.
    "c1_filmic": {
        "view_transform": "Filmic", "look": "Medium High Contrast",
        "exposure": 0.55,
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.12, "roughness_floor": 0.35,
    },
    "c2_grade": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.12, "roughness_floor": 0.35,
        "grade_saturation": 1.15, "grade_contrast": 0.05,
    },
    "c3_warm": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.12, "roughness_floor": 0.35,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.5,
    },
    "c4_rt": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.12, "roughness_floor": 0.35,
        "raytracing": True, "samples": 64,
    },
    "c5_combo": {
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.14, "roughness_floor": 0.35,
        "key_color": (1.0, 0.93, 0.82), "world_color": (0.92, 0.90, 0.87),
        "world_strength": 0.5,
        "grade_saturation": 1.12, "grade_contrast": 0.04,
        "raytracing": True, "samples": 64,
    },
    "b8_full": {
        "view_transform": "AgX", "look": "AgX - Punchy", "exposure": 0.3,
        "soft_key": True, "auto_smooth_degrees": 40,
        "dark_lift": 0.10, "roughness_floor": 0.35,
        "bloom": True, "vignette": True, "samples": 64,
    },
}

MANIFEST_DIR = REPO_ROOT / "benchmarks/scene_reconstruction/manifests"


def resolved_beauty_style(args, render_cfg: dict) -> dict | None:
    """Resolve JSON-owned style parameters before legacy named presets."""

    if "style_parameters" in render_cfg:
        parameters = dict(render_cfg["style_parameters"])
        return parameters or None
    return BEAUTY_PRESETS.get(getattr(args, "style", "normal"))


def render_task_contract(args, render_cfg: dict) -> dict:
    """Fields shared by GT and reconstruction Blender tasks."""

    camera = render_cfg.get("camera") or {
        "type": "PERSP",
        "intrinsics": "task_exact",
        "clip_start": 0.01,
        "clip_end": 1000.0,
    }
    output = render_cfg.get("output") or {
        "file_format": "PNG",
        "color_mode": "RGB",
        "color_depth": "8",
        "film_transparent": False,
        "empty_background_rgb": [255, 255, 255],
        "use_file_extension": True,
    }
    return {
        "engine": render_cfg.get("engine", "BLENDER_EEVEE_NEXT"),
        "samples": int(render_cfg["samples"]),
        "recipe": render_cfg.get("recipe", "canonical_pbr"),
        "render_profile": render_cfg.get("render_profile", "inference-rgb"),
        "camera_settings": dict(camera),
        "output_settings": dict(output),
        "empty_background_rgb": list(
            render_cfg.get("empty_background_rgb", output["empty_background_rgb"])
        ),
        "beauty_style": resolved_beauty_style(args, render_cfg),
        "beauty_style_applies_to": list(
            render_cfg.get("style_applies_to", ["texture"])
        ),
    }


def bind_index_protocol_identity(index: dict, key: str, identity: dict) -> None:
    """Bind one immutable protocol revision to an output index."""

    existing = index.get(key)
    if existing is None:
        index[key] = identity
        return
    existing_fingerprint = (
        existing.get("resolved_name"),
        existing.get("resolved_sha256") or existing.get("source_sha256"),
    )
    requested_fingerprint = (
        identity.get("resolved_name"),
        identity.get("resolved_sha256") or identity.get("source_sha256"),
    )
    if existing_fingerprint != requested_fingerprint:
        raise SystemExit(
            f"Output index is already bound to a different {key}: "
            f"{existing_fingerprint!r} != {requested_fingerprint!r}. "
            "Use a new output root."
        )


def gt_layout_view_exclusions(config: dict, view_indices: set[int]) -> dict[str, list[str]]:
    """Validate an explicit open-wall rule for the native GT layout only."""

    presentation = config.get("presentation") or {}
    raw = presentation.get("gt_layout_exclude_nodes_by_view") or {}
    if not isinstance(raw, dict):
        raise ValueError("gt_layout_exclude_nodes_by_view must be an object")
    normalized: dict[str, list[str]] = {}
    for raw_index, raw_names in raw.items():
        try:
            view_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid GT layout exclusion view {raw_index!r}") from error
        if view_index not in view_indices:
            raise ValueError(
                f"GT layout exclusion targets missing view {view_index}; "
                f"available views are {sorted(view_indices)}"
            )
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(
                f"GT layout exclusion for view {view_index} must be a nonempty list"
            )
        names = []
        for name in raw_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("GT layout exclusion node names must be nonempty strings")
            names.append(name.strip())
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate GT layout exclusion node for view {view_index}")
        normalized[str(view_index)] = names
    return normalized


def build_gt_task(
    args, config: dict, scene_id: str, K, width, height, frames, out_dir: Path
) -> Path:
    """Render the ground-truth scene under the same two protocols.

    Reuses the benchmark GT asset contract (scene_render_contract): per-object
    GLBs + the layout mesh with dataset-world transforms. The layout plays the
    background role: gray in the geometry protocol, omitted entirely when
    render.background is false.
    """
    from eval.rendering.scene_assets import (
        full_gt_scene_assets,
        load_scene_transforms,
    )

    render_cfg = config["render"]
    manifest = json.loads(
        (MANIFEST_DIR / f"{args.dataset}_v1.json").read_text(encoding="utf-8")
    )
    scene = next(
        (entry for entry in manifest["scenes"] if entry["scene_id"] == scene_id), None
    )
    if scene is None:
        raise RuntimeError(f"{scene_id} is not in {args.dataset}_v1.json")
    subdir = "Imaginarium" if args.dataset == "imaginarium" else "ithor"
    dataset_root = args.fire3d_test_root / subdir
    transforms = load_scene_transforms(dataset_root / scene["transforms_path"])
    objects, layout = full_gt_scene_assets(dataset_root, scene, transforms)

    view_indices = {task_view_index(frame, i) for i, frame in enumerate(frames)}
    layout_view_exclusions = gt_layout_view_exclusions(config, view_indices)
    methods, policy = [], {}
    for protocol in config["protocols"]:
        background = protocol_background(config, "gt", protocol)
        assets = []
        for asset in objects:
            asset = dict(asset)
            if protocol == "geometry":
                asset["material_mode"] = "instance_color"
                asset["instance_palette"] = render_cfg["geometry_palette"]
            assets.append(asset)
        if background:
            layout_asset = dict(layout)
            if layout_view_exclusions:
                layout_asset["exclude_nodes_by_view"] = layout_view_exclusions
            if protocol == "geometry":
                layout_asset["material_mode"] = "instance_color"
                layout_asset["object_id"] = 0
                layout_asset["override_rgba"] = list(render_cfg["gray_rgba"])
            assets.append(layout_asset)
        key = f"gt_{protocol}"
        methods.append({"key": key, "label": f"GT {protocol}", "assets": assets})
        policy[key] = "none"

    views = [
        {
            "view_index": task_view_index(frame, i),
            "frame": {k: frame[k] for k in ("eye", "lookat", "up")},
            "forward": frame["forward"],
            "visible_gt_objects": [],
            "visible_gt_pixels": {},
        }
        for i, frame in enumerate(frames)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.json").write_text(
        json.dumps({"K": K, "width": width, "height": height,
                    "frames": [v["frame"] for v in views]}, indent=1)
    )
    task = {
        "schema": "ff_efm3d_shaper_pipeline_comparison_task_v2",
        "comparison_mode": f"{args.dataset}_unified_two_protocol_gt",
        "dataset": args.dataset,
        "scene_id": scene_id,
        "coordinate_contract": "dataset_world_without_posthoc_alignment",
        "scope": "protocol_specific_gt_background",
        "excluded_object_ids": [], "explicit_excluded_object_ids": [],
        "auto_excluded_room_slab_ids": [],
        "evaluation_object_ids": [int(a["object_id"]) for a in objects],
        "rendered_gt_object_ids": [int(a["object_id"]) for a in objects],
        "width": width, "height": height, "K": K,
        "aspect_policy": task_aspect_policy(config, width, height),
        **render_task_contract(args, render_cfg),
        "views": views,
        "context_assets": [],
        "background_policy": policy,
        "gt_layout_exclude_nodes_by_view": layout_view_exclusions,
        "methods": methods,
        "output_dir": str(out_dir),
        "camera_path": str(out_dir / "cameras.json"),
    }
    (out_dir / "render_task.json").write_text(json.dumps(task, indent=1))
    return out_dir / "render_task.json"


def build_task(
    args, config: dict, scene_id: str, recon_dir: Path,
    K, width, height, frames, out_dir: Path,
) -> Path:
    render_cfg = config["render"]
    glb = recon_dir / "appearance/predicted_textured_world_scene.glb"
    if not glb.is_file():
        raise RuntimeError(f"missing composed world GLB: {glb}")
    room_id = room_instance_id(recon_dir, getattr(args, "background_instance", "prior"))

    methods, policy = [], {}
    for protocol in config["protocols"]:
        background = protocol_background(config, "reconstruction", protocol)
        asset = {
            "name": f"ours_{protocol}",
            "scene_role": "ours_scene_composite",
            "kind": "world_scene_glb",
            "path": str(glb),
            "force_opaque_materials": [],
        }
        if protocol == "geometry":
            asset["material_mode"] = "instance_color_by_instance"
            asset["instance_palette"] = render_cfg["geometry_palette"]
            if background and room_id is not None:
                asset["gray_instance_id"] = int(room_id)
                asset["gray_rgba"] = list(render_cfg["gray_rgba"])
        if not background and room_id is not None:
            asset["exclude_instance_id"] = int(room_id)
        methods.append({"key": f"ours_{protocol}", "label": f"Ours {protocol}", "assets": [asset]})
        policy[f"ours_{protocol}"] = (
            "ours_reconstructed_textured" if background else "none"
        )

    views = [
        {
            "view_index": task_view_index(frame, i),
            "frame": {k: frame[k] for k in ("eye", "lookat", "up")},
            "forward": frame["forward"],
            "visible_gt_objects": [],
            "visible_gt_pixels": {},
        }
        for i, frame in enumerate(frames)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.json").write_text(
        json.dumps({"K": K, "width": width, "height": height,
                    "frames": [v["frame"] for v in views]}, indent=1)
    )
    task = {
        "schema": "ff_efm3d_shaper_pipeline_comparison_task_v2",
        "comparison_mode": f"{args.dataset}_unified_two_protocol",
        "dataset": args.dataset,
        "scene_id": scene_id,
        "coordinate_contract": "recon_world_without_posthoc_alignment",
        "scope": "method_specific_textured_backgrounds",
        "excluded_object_ids": [], "explicit_excluded_object_ids": [],
        "auto_excluded_room_slab_ids": [], "evaluation_object_ids": [],
        "rendered_gt_object_ids": [],
        "width": width, "height": height, "K": K,
        "aspect_policy": task_aspect_policy(config, width, height),
        **render_task_contract(args, render_cfg),
        "views": views,
        "context_assets": [],
        "background_policy": policy,
        "methods": methods,
        "output_dir": str(out_dir),
        "camera_path": str(out_dir / "cameras.json"),
    }
    (out_dir / "render_task.json").write_text(json.dumps(task, indent=1))
    return out_dir / "render_task.json"


def build_comparison_task(
    args,
    config: dict,
    scene_id: str,
    recon_dir: Path,
    K,
    width: int,
    height: int,
    frames: list[dict],
    out_dir: Path,
) -> Path:
    """Build the frozen GT texture / ours geometry / ours texture task."""

    gt_task_path = build_gt_task(
        args, config, scene_id, K, width, height, frames, out_dir / "task_sources/gt"
    )
    ours_task_path = build_task(
        args,
        config,
        scene_id,
        recon_dir,
        K,
        width,
        height,
        frames,
        out_dir / "task_sources/reconstruction",
    )
    gt_task = json.loads(gt_task_path.read_text(encoding="utf-8"))
    ours_task = json.loads(ours_task_path.read_text(encoding="utf-8"))
    by_key = {
        method["key"]: method
        for method in [*gt_task["methods"], *ours_task["methods"]]
    }
    method_keys = ("gt_texture", "ours_geometry", "ours_texture")
    missing = [key for key in method_keys if key not in by_key]
    if missing:
        raise RuntimeError(f"Comparison task is missing methods: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "cameras.json"
    camera_path.write_text(
        json.dumps(
            {
                "K": K,
                "width": width,
                "height": height,
                "frames": [
                    {key: frame[key] for key in ("eye", "lookat", "up")}
                    for frame in frames
                ],
            },
            indent=1,
        )
    )
    task = dict(ours_task)
    task.update(
        {
            "comparison_mode": f"{args.dataset}_unified_reproduction_comparison",
            "coordinate_contract": "dataset_world_without_posthoc_alignment",
            "scope": "gt_texture_ours_geometry_ours_texture",
            "evaluation_object_ids": gt_task.get("evaluation_object_ids", []),
            "rendered_gt_object_ids": gt_task.get("rendered_gt_object_ids", []),
            "methods": [by_key[key] for key in method_keys],
            "background_policy": {
                **gt_task.get("background_policy", {}),
                **ours_task.get("background_policy", {}),
            },
            "output_dir": str(out_dir),
            "camera_path": str(camera_path),
        }
    )
    task_path = out_dir / "render_task.json"
    task_path.write_text(json.dumps(task, indent=1))
    return task_path


def assemble_comparison(
    scene_id: str,
    out_dir: Path,
    sheet_dir: Path,
    frames: list[dict],
    width: int,
    height: int,
) -> dict[str, str]:
    from PIL import Image
    from eval.rendering.render_common import (
        beauty_on_gray,
        labeled,
    )

    methods = (
        ("gt_texture", "Ground truth", "GT textured room background"),
        (
            "ours_geometry",
            "Ours geometry",
            "Predicted perception + reconstruction | instance colors",
        ),
        (
            "ours_texture",
            "Ours textured",
            "Perception + teacher12 CFG 3 | reconstructed textured background",
        ),
    )
    canvas = Image.new("RGB", (len(methods) * width, len(frames) * height), (232, 235, 239))
    for row, frame in enumerate(frames):
        view_index = task_view_index(frame, row)
        for column, (method, title, subtitle) in enumerate(methods):
            source = out_dir / "views" / f"{method}_v{view_index:04d}_beauty.png"
            if not source.is_file():
                raise RuntimeError(f"missing comparison render: {source}")
            tile = labeled(
                beauty_on_gray(source),
                f"{title} | view {view_index}",
                subtitle,
            )
            if tile.size != (width, height):
                tile = tile.resize((width, height), Image.LANCZOS)
            canvas.paste(tile, (column * width, row * height))
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet = sheet_dir / f"{scene_id}_predicted_inputs.png"
    canvas.save(sheet)
    convenience = sheet_dir / "predicted_inputs.png"
    if convenience != sheet:
        canvas.save(convenience)
    return {
        "comparison": str(sheet.resolve()),
        "comparison_convenience": str(convenience.resolve()),
    }


def assemble_sheets(
    config: dict, scene_id: str, out_dir: Path, sheet_dir: Path,
    num_views: int, width: int, height: int, prefix: str = "ours",
) -> dict[str, str]:
    from PIL import Image

    figures = {}
    columns = min(4, max(1, num_views))
    rows = math.ceil(num_views / columns)
    for protocol in config["protocols"]:
        canvas = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
        found = 0
        frames = config.get("cameras", {}).get("frames", [])
        for i in range(num_views):
            frame = frames[i] if i < len(frames) else {}
            view_index = task_view_index(frame, i)
            beauty = out_dir / "views" / f"{prefix}_{protocol}_v{view_index:04d}_beauty.png"
            if not beauty.is_file():
                continue
            tile = Image.open(beauty).convert("RGB")
            if tile.size != (width, height):
                tile = tile.resize((width, height), Image.LANCZOS)
            canvas.paste(tile, ((i % columns) * width, (i // columns) * height))
            found += 1
        if found == 0:
            raise RuntimeError(f"no rendered views found for protocol {protocol} in {out_dir}")
        sheet_dir.mkdir(parents=True, exist_ok=True)
        sheet = sheet_dir / (
            f"{scene_id}_{protocol}.png"
            if prefix == "ours"
            else f"{scene_id}_{prefix}_{protocol}.png"
        )
        canvas.save(sheet)
        figures[f"{protocol}"] = str(sheet.resolve())
    return figures


def completed_requested_renders(
    scenes: dict[str, dict], requested_keys: list[str]
) -> int:
    return sum(
        1 for key in requested_keys
        if scenes.get(key, {}).get("status") == "complete"
    )


def main() -> None:
    args = parse_args()
    protocol = None
    protocol_source = None
    render_protocol = None
    render_protocol_source = None
    if args.protocol is not None:
        try:
            protocol, protocol_source = load_protocol(args.protocol)
            validate_scope(protocol, args.dataset, list(args.scene_id))
            os.environ.update(native_dataset_environment(protocol, args.dataset))
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if args.render_protocol is not None:
        try:
            render_protocol, render_protocol_source = load_render_protocol(
                args.render_protocol
            )
            validate_render_scope(render_protocol, args.dataset, args.source)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    config = load_config(args.config, protocol, render_protocol)
    protocol_style = (config.get("render") or {}).get("style")
    if args.style is None:
        args.style = (
            protocol_style
            or ("normal" if args.source == "gt" else "beauty_v0")
        )
    elif protocol_style and args.style != protocol_style:
        raise SystemExit(
            f"Protocol requires render style {protocol_style!r}; received "
            f"{args.style!r}"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.source in ("gt", "comparison"):
        scene_ids = args.scene_id
    else:
        scene_ids = (
            list_scenes(args.run_root, args.dataset) if args.all else args.scene_id
        )
        if not scene_ids:
            raise SystemExit(f"no reconstructed scenes under {args.run_root}")

    index_path = args.output_root / "render_index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    index.setdefault("schema", "ff_unified_render_index_v2")
    index.setdefault("dataset", args.dataset)
    index.setdefault("protocols", config["protocols"])
    index.setdefault("source", args.source)
    index.setdefault("render_profile", config["render"]["render_profile"])
    index.setdefault("style", getattr(args, "style", "normal"))
    index.setdefault("background_override", args.background)
    style_parameters = resolved_beauty_style(args, config["render"])
    if style_parameters:
        index["beauty_style_parameters"] = style_parameters
        index["beauty_style_applies_to"] = config["render"]["style_applies_to"]
    index.setdefault(
        "resolved_render_settings",
        {
            key: config["render"][key]
            for key in (
                "engine",
                "recipe",
                "render_profile",
                "samples",
                "geometry_palette",
                "gray_rgba",
                "camera",
                "output",
            )
        },
    )
    if config.get("sampling"):
        index.setdefault("view_sampling", config["sampling"])
    if protocol is not None:
        bind_index_protocol_identity(
            index,
            "inference_protocol",
            protocol_identity(args.protocol, protocol, protocol_source),
        )
    if render_protocol is not None:
        bind_index_protocol_identity(
            index,
            "render_protocol",
            render_protocol_identity(
                args.render_protocol, render_protocol, render_protocol_source
            ),
        )
    scenes = index.setdefault("scenes", {})

    requested_keys: list[str] = []
    for n, scene_id in enumerate(scene_ids, start=1):
        variants = (
            [("", bool(config["render"]["background"]))]
            if args.source == "comparison"
            else background_variants(args, config)
        )
        variant_keys = [
            f"{scene_id}/{label}" if label else scene_id
            for label, _ in variants
        ]
        requested_keys.extend(variant_keys)
        if args.skip_existing and all(
            scenes.get(k, {}).get("status") == "complete" for k in variant_keys
        ):
            print(f"[render] ({n}/{len(scene_ids)}) {scene_id}: exists", flush=True)
            continue
        if args.source in ("recon", "comparison"):
            recon_dir = find_recon_dir(args.run_root, args.dataset, scene_id)
            if recon_dir is None:
                scenes[scene_id] = {"status": "no_reconstruction"}
                continue
        started = time.time()
        for label, flag in variants:
            apply_background_variant(
                config,
                flag,
                override_protocol_defaults=args.background != "config",
            )
            variant_root = args.output_root / label if label else args.output_root
            key = f"{scene_id}/{label}" if label else scene_id
            try:
                K, width, height, frames = resolve_cameras(args, config, scene_id)
                out_dir = variant_root / "sources" / scene_id
                if args.source == "gt":
                    task = build_gt_task(
                        args, config, scene_id, K, width, height, frames, out_dir
                    )
                elif args.source == "comparison":
                    task = build_comparison_task(
                        args,
                        config,
                        scene_id,
                        recon_dir,
                        K,
                        width,
                        height,
                        frames,
                        out_dir,
                    )
                else:
                    task = build_task(
                        args, config, scene_id, recon_dir, K, width, height, frames,
                        out_dir,
                    )
                prefix = "gt" if args.source == "gt" else "ours"
                methods = (
                    ("gt_texture", "ours_geometry", "ours_texture")
                    if args.source == "comparison"
                    else tuple(f"{prefix}_{name}" for name in config["protocols"])
                )
                expected = [
                    out_dir
                    / "views"
                    / f"{method}_v{task_view_index(frame, i):04d}_beauty.png"
                    for method in methods
                    for i, frame in enumerate(frames)
                ]
                if should_invoke_renderer(expected, args.skip_existing):
                    log = variant_root / "logs" / f"{scene_id}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("w") as handle:
                        code = subprocess.call(
                            [str(BLENDER), "-b", "--factory-startup", "--python",
                             str(BLENDER_ENTRY), "--", "--task", str(task), "--overwrite"],
                            stdout=handle, stderr=subprocess.STDOUT, env=env,
                            cwd=str(REPO_ROOT),
                        )
                    if code != 0:
                        raise RuntimeError(f"blender exited {code}; see {log}")
                if args.source == "comparison":
                    figures = assemble_comparison(
                        scene_id, out_dir, variant_root, frames, width, height
                    )
                else:
                    figures = assemble_sheets(
                        config,
                        scene_id,
                        out_dir,
                        variant_root,
                        len(frames),
                        width,
                        height,
                        prefix=prefix,
                    )
                scenes[key] = {
                    "status": "complete",
                    "figures": figures,
                    "num_views": len(frames),
                    "background": flag,
                    "background_instance_strategy": getattr(
                        args, "background_instance", "prior"
                    ),
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            except Exception as error:
                scenes[key] = {"status": "failed", "error": str(error)}
            print(f"[render] ({n}/{len(scene_ids)}) {key}: {scenes[key]['status']}",
                  flush=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    done = completed_requested_renders(scenes, requested_keys)
    print(
        f"[render] {done}/{len(requested_keys)} complete; index at {index_path}",
        flush=True,
    )
    if done != len(requested_keys):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
