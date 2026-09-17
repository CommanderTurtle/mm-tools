#!/usr/bin/env python3
"""Rerender benchmark RGB frames at the dataset's exact source cameras."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pickle
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_SCRIPT = REPO_ROOT / "eval/rendering/blender_rerender_imaginarium_rgb.py"
DATASET_DEFAULT_SCENES = {
    "imaginarium": ("bedroom_06", "bedroom_14"),
    "ithor": ("iTHOR_FloorPlan415_physics", "iTHOR_FloorPlan6_physics"),
}
LEGACY_ZERO_ALPHA_OPAQUE_PREFIXES = ("glass_detailed_white",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DEFAULT_SCENES),
        default="imaginarium",
    )
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument(
        "--source-fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--overlay-fire3d-test-root", type=Path, required=True)
    parser.add_argument(
        "--blender",
        default=str(REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--recipe", default="canonical_pbr")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        default=[],
        help="Render only selected source-frame indices; repeat as needed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def link_exact(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(f"Conflicting symlink: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_overlay(
    source_fire3d_root: Path,
    overlay_fire3d_root: Path,
    scene_ids: list[str],
    dataset_subdir: str = "Imaginarium",
) -> tuple[Path, Path]:
    source_dataset = source_fire3d_root.resolve() / dataset_subdir
    overlay_dataset = overlay_fire3d_root.resolve() / dataset_subdir
    source_renders = source_dataset / "renders"
    overlay_renders = overlay_dataset / "renders"
    overlay_renders.mkdir(parents=True, exist_ok=True)
    selected = set(scene_ids)

    source_scene_dirs = sorted(path for path in source_renders.iterdir() if path.is_dir())
    available = {path.name for path in source_scene_dirs}
    missing = sorted(selected.difference(available))
    if missing:
        raise FileNotFoundError(f"Missing source scene directories: {missing}")

    for source_scene in source_scene_dirs:
        destination_scene = overlay_renders / source_scene.name
        if source_scene.name not in selected:
            link_exact(source_scene, destination_scene)
            continue
        destination_scene.mkdir(parents=True, exist_ok=True)
        for child in sorted(source_scene.iterdir()):
            if child.name == "0_frames":
                continue
            link_exact(child, destination_scene / child.name)
        (destination_scene / "0_frames").mkdir(parents=True, exist_ok=True)

    for child in sorted(source_dataset.iterdir()):
        if child.name == "renders":
            continue
        link_exact(child, overlay_dataset / child.name)
    return source_dataset, overlay_dataset


def load_manifest_scenes(
    dataset: str, scene_ids: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = (
        REPO_ROOT
        / "benchmarks/scene_reconstruction/manifests"
        / f"{dataset}_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    missing = [scene_id for scene_id in scene_ids if scene_id not in by_name]
    if missing:
        raise KeyError(f"Scenes are absent from {manifest_path}: {missing}")
    return manifest, [by_name[scene_id] for scene_id in scene_ids]


def compose_matrix(record: dict[str, Any]) -> np.ndarray:
    import trimesh

    scale = float(record["scale"])
    return trimesh.transformations.compose_matrix(
        scale=[scale, scale, scale],
        angles=np.asarray(record["angles"], dtype=np.float64),
        translate=np.asarray(record["trans"], dtype=np.float64),
    )


def read_glb_document(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        magic, version, total_length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2 or total_length != path.stat().st_size:
            raise ValueError(f"Invalid GLB header: {path}")
        document = None
        binary = b""
        while handle.tell() < total_length:
            chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
            payload = handle.read(chunk_length)
            if chunk_type == 0x4E4F534A:
                document = json.loads(payload.rstrip(b" \t\r\n\0"))
            elif chunk_type == 0x004E4942:
                binary = payload
    if document is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    return document, binary


def gltf_image_bytes(
    document: dict[str, Any], binary: bytes, image_index: int, glb_path: Path
) -> bytes:
    image = document["images"][image_index]
    if "bufferView" in image:
        view = document["bufferViews"][int(image["bufferView"])]
        if int(view.get("buffer", 0)) != 0:
            raise ValueError(f"Unsupported nonzero GLB image buffer: {glb_path}")
        start = int(view.get("byteOffset", 0))
        end = start + int(view["byteLength"])
        return binary[start:end]
    uri = image.get("uri")
    if not uri:
        raise ValueError(f"GLB image has no bufferView or URI: {glb_path}")
    if uri.startswith("data:"):
        return base64.b64decode(uri.partition(",")[2])
    return (glb_path.parent / uri).read_bytes()


def fully_opaque_blend_material_names(path: Path) -> list[str]:
    """Return BLEND materials that must remain visible in inference-RGB renders.

    In addition to materials with provably opaque effective alpha, this handles a
    legacy Unreal export contract where white glass tabletops were emitted with a
    zero alpha factor even though the source renderer displayed them as opaque.
    """
    document, binary = read_glb_document(path)
    opaque = []
    for material in document.get("materials", []):
        if material.get("alphaMode", "OPAQUE") != "BLEND" or not material.get("name"):
            continue
        pbr = material.get("pbrMetallicRoughness", {})
        factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        legacy_zero_alpha_opaque = (
            float(factor[3]) <= 1e-6
            and pbr.get("baseColorTexture") is None
            and material["name"].casefold().startswith(
                LEGACY_ZERO_ALPHA_OPAQUE_PREFIXES
            )
        )
        if legacy_zero_alpha_opaque:
            opaque.append(material["name"])
            continue
        if float(factor[3]) < 1.0 - 1e-6:
            continue
        texture_info = pbr.get("baseColorTexture")
        if texture_info is None:
            opaque.append(material["name"])
            continue
        texture = document["textures"][int(texture_info["index"])]
        image_index = texture.get("source")
        if image_index is None:
            image_index = texture.get("extensions", {}).get(
                "KHR_texture_basisu", {}
            ).get("source")
        if image_index is None:
            continue
        payload = gltf_image_bytes(document, binary, int(image_index), path)
        with Image.open(io.BytesIO(payload)) as image:
            alpha_extrema = (
                image.getchannel("A").getextrema()
                if "A" in image.getbands()
                else (255, 255)
            )
        if alpha_extrema[0] == 255:
            opaque.append(material["name"])
    return sorted(set(opaque))


def build_render_task(
    *,
    dataset: str,
    scene: dict[str, Any],
    source_dataset: Path,
    overlay_dataset: Path,
    samples: int,
    recipe: str,
    jpeg_quality: int,
    frame_indices: list[int],
    output_scene_dir: Path | None = None,
) -> Path:
    camera_path = source_dataset / scene["camera_path"]
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    if len(camera["frames"]) != int(scene["num_frames"]):
        raise ValueError(
            f"Camera frame count mismatch for {scene['scene_id']}: "
            f"{len(camera['frames'])} versus {scene['num_frames']}"
        )
    transform_path = source_dataset / scene["transforms_path"]
    with transform_path.open("rb") as handle:
        transforms = pickle.load(handle)
    layout_keys = sorted(key for key in transforms if key.startswith("layout_"))
    if len(layout_keys) != 1:
        raise ValueError(f"Expected one layout transform in {transform_path}")
    object_keys = [
        f"object_{int(object_id):04d}" for object_id in scene["gt_object_ids"]
    ]
    assets = []
    for key in [layout_keys[0], *object_keys]:
        mesh_path = source_dataset / scene["mesh_dir"] / f"{key}.glb"
        if key not in transforms or not mesh_path.is_file():
            raise FileNotFoundError(f"Missing GT asset or transform: {mesh_path}")
        assets.append(
            {
                "name": key,
                "path": str(mesh_path.resolve()),
                "object_to_world": compose_matrix(transforms[key]).tolist(),
                "force_opaque_materials": fully_opaque_blend_material_names(
                    mesh_path
                ),
            }
        )

    output_dir = (
        output_scene_dir / Path(scene["frames_dir"]).name
        if output_scene_dir is not None
        else overlay_dataset / scene["frames_dir"]
    )
    selected_indices = sorted(set(frame_indices)) or list(range(len(camera["frames"])))
    invalid = [
        index for index in selected_indices if not 0 <= index < len(camera["frames"])
    ]
    if invalid:
        raise IndexError(f"Invalid frame indices for {scene['scene_id']}: {invalid}")
    task = {
        "schema": "ff_exact_camera_rgb_rerender_task_v2",
        "dataset": dataset,
        "scene_id": scene["scene_id"],
        "camera_path": str(camera_path.resolve()),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "K": camera["K"],
        "samples": int(samples),
        "recipe": recipe,
        "jpeg_quality": int(jpeg_quality),
        "gt_assets": assets,
        "frames": [
            {
                "frame_index": index,
                "camera": frame,
                "output": str((output_dir / f"frame_{index:04d}.jpg").resolve()),
            }
            for index, frame in enumerate(camera["frames"])
            if index in selected_indices
        ],
        "output_dir": str(output_dir.resolve()),
    }
    task_path = output_dir.parent / "exact_camera_rgb_rerender_task.json"
    task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return task_path


def run_blender(args: argparse.Namespace, task_path: Path) -> None:
    blender = Path(shutil.which(args.blender) or args.blender).resolve()
    command = [
        str(blender),
        "-b",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--task",
        str(task_path),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_contact_sheet(frame_paths: list[Path], output: Path) -> None:
    tile_size = 192
    columns = 10
    rows = (len(frame_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(frame_paths):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + 50, y + 20), fill=(0, 0, 0))
        draw.text((x + 5, y + 4), f"{index:02d}", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def make_comparison_sheets(
    source_frames: list[Path],
    rerendered_frames: list[Path],
    output_dir: Path,
    scene_id: str,
) -> dict[str, str]:
    if len(source_frames) != len(rerendered_frames):
        raise ValueError(
            f"RGB comparison count mismatch: {len(source_frames)} versus "
            f"{len(rerendered_frames)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_sheet = output_dir / f"{scene_id}_existing_rgb60.jpg"
    updated_sheet = output_dir / f"{scene_id}_updated_rgb60.jpg"
    overview = output_dir / f"{scene_id}_existing_vs_updated_rgb60.jpg"
    details = output_dir / f"{scene_id}_existing_vs_updated_selected8.jpg"
    make_contact_sheet(source_frames, existing_sheet)
    make_contact_sheet(rerendered_frames, updated_sheet)

    with Image.open(existing_sheet) as existing_image, Image.open(updated_sheet) as updated_image:
        width, height = existing_image.size
        if updated_image.size != (width, height):
            raise RuntimeError("Existing and updated contact-sheet dimensions differ")
        header = 44
        combined = Image.new("RGB", (width * 2, height + header), (20, 23, 29))
        combined.paste(existing_image.convert("RGB"), (0, header))
        combined.paste(updated_image.convert("RGB"), (width, header))
        draw = ImageDraw.Draw(combined)
        draw.text((16, 14), "Existing test RGBs", fill=(255, 255, 255))
        draw.text((width + 16, 14), "Updated exact-camera RGBs", fill=(255, 255, 255))
        combined.save(overview, quality=95)

    selected = np.linspace(0, len(source_frames) - 1, 8, dtype=np.int64).tolist()
    image_size = 512
    label_height = 34
    pair_width = image_size * 2
    pair_height = image_size + label_height
    detail_sheet = Image.new("RGB", (pair_width * 2, pair_height * 4), (20, 23, 29))
    draw = ImageDraw.Draw(detail_sheet)
    for slot, frame_index in enumerate(selected):
        x = (slot % 2) * pair_width
        y = (slot // 2) * pair_height
        draw.text((x + 8, y + 10), f"Frame {frame_index:02d}: existing", fill=(255, 255, 255))
        draw.text(
            (x + image_size + 8, y + 10),
            f"Frame {frame_index:02d}: updated",
            fill=(255, 255, 255),
        )
        with Image.open(source_frames[frame_index]) as existing_image:
            detail_sheet.paste(existing_image.convert("RGB"), (x, y + label_height))
        with Image.open(rerendered_frames[frame_index]) as updated_image:
            detail_sheet.paste(
                updated_image.convert("RGB"),
                (x + image_size, y + label_height),
            )
    detail_sheet.save(details, quality=95)
    return {
        "existing_contact_sheet": str(existing_sheet.resolve()),
        "updated_contact_sheet": str(updated_sheet.resolve()),
        "overview_comparison": str(overview.resolve()),
        "selected8_comparison": str(details.resolve()),
    }


def validate_scene(
    scene: dict[str, Any], source_dataset: Path, overlay_dataset: Path
) -> dict[str, Any]:
    source_camera = source_dataset / scene["camera_path"]
    overlay_camera = overlay_dataset / scene["camera_path"]
    source_frame_paths = sorted(
        (source_dataset / scene["frames_dir"]).glob("frame_*.jpg")
    )
    frame_paths = sorted((overlay_dataset / scene["frames_dir"]).glob("frame_*.jpg"))
    if len(frame_paths) != int(scene["num_frames"]):
        raise RuntimeError(
            f"Expected {scene['num_frames']} rendered frames for {scene['scene_id']}, "
            f"found {len(frame_paths)}"
        )
    camera = json.loads(source_camera.read_text(encoding="utf-8"))
    means = []
    for path in frame_paths:
        with Image.open(path) as image:
            if image.size != (int(camera["width"]), int(camera["height"])):
                raise RuntimeError(f"Unexpected image size {image.size}: {path}")
            means.append(float(np.asarray(image.convert("RGB"), dtype=np.float32).mean()))
    if min(means) <= 1.0:
        raise RuntimeError(f"Blank or nearly blank rerender in {scene['scene_id']}")
    for key in ("depths_dir", "masks_dir", "transforms_path"):
        overlay_path = overlay_dataset / scene[key]
        source_path = source_dataset / scene[key]
        if overlay_path.resolve() != source_path.resolve():
            raise RuntimeError(f"Overlay {key} does not resolve to source: {overlay_path}")
    if overlay_camera.resolve() != source_camera.resolve():
        raise RuntimeError(f"Overlay camera does not resolve to source: {overlay_camera}")

    comparisons = make_comparison_sheets(
        source_frame_paths,
        frame_paths,
        overlay_dataset / "rerender_visualizations",
        scene["scene_id"],
    )
    return {
        "scene_id": scene["scene_id"],
        "num_frames": len(frame_paths),
        "resolution": [int(camera["width"]), int(camera["height"])],
        "K": camera["K"],
        "camera_sha256": sha256(source_camera),
        "rgb_mean_range": [min(means), max(means)],
        "frames_dir": str((overlay_dataset / scene["frames_dir"]).resolve()),
        **comparisons,
    }


def main() -> None:
    args = parse_args()
    scene_ids = args.scene or list(DATASET_DEFAULT_SCENES[args.dataset])
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError(f"Duplicate --scene values: {scene_ids}")
    manifest, scenes = load_manifest_scenes(args.dataset, scene_ids)
    source_dataset, overlay_dataset = prepare_overlay(
        args.source_fire3d_test_root,
        args.overlay_fire3d_test_root,
        scene_ids,
        manifest["dataset_subdir"],
    )
    records = []
    for scene in scenes:
        task_path = build_render_task(
            dataset=args.dataset,
            scene=scene,
            source_dataset=source_dataset,
            overlay_dataset=overlay_dataset,
            samples=args.samples,
            recipe=args.recipe,
            jpeg_quality=args.jpeg_quality,
            frame_indices=args.frame_index,
        )
        run_blender(args, task_path)
        records.append(validate_scene(scene, source_dataset, overlay_dataset))

    source_names = sorted(path.name for path in (source_dataset / "renders").iterdir() if path.is_dir())
    overlay_names = sorted(path.name for path in (overlay_dataset / "renders").iterdir() if path.is_dir())
    if source_names != overlay_names:
        raise RuntimeError("Overlay scene ordering differs from the source dataset")
    summary = {
        "schema": "ff_exact_camera_rgb_rerender_v2",
        "dataset": args.dataset,
        "source_dataset": str(source_dataset),
        "overlay_dataset": str(overlay_dataset),
        "render_engine": "BLENDER_EEVEE_NEXT",
        "recipe": args.recipe,
        "samples": args.samples,
        "jpeg_quality": args.jpeg_quality,
        "scene_count_preserved": len(source_names),
        "records": records,
    }
    summary_path = overlay_dataset / "exact_camera_rgb_rerender_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
