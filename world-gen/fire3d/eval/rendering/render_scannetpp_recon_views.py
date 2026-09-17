#!/usr/bin/env python3
"""Render a ScanNet++ reconstruction from real dataset cameras.

Picks N poses at random from the frames the loader used, transforms them into
the configured reconstruction frame (wall-aligned or unchanged), and renders
instance-colour geometry and textured mesh, each with and without the
reconstructed room instance.

Cameras are used exactly as calibrated: the scene's own K at an integer
multiple of native resolution, so the field of view is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = Path(sys.executable)
BLENDER = REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--recon-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--resolution-scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scene_alignment(scene_id: str):
    from utils.data_scannetpp import (load_scannetpp_data, load_scene_frames,
                                      scene_wall_alignment)

    data_list = load_scannetpp_data()
    entry = next((d for d in data_list if d["scene_id"] == scene_id), None)
    if entry is None:
        raise SystemExit(f"scene {scene_id} is not loadable")
    rgbs, depths, intrinsics, c2ws, _ = load_scene_frames(entry)
    _, center, angle, diag = scene_wall_alignment(rgbs, depths, intrinsics, c2ws)
    return entry, intrinsics, c2ws, center, angle, diag


def camera_frames(c2ws, center, angle, indices):
    """Dataset poses moved into the reconstruction (rotated) frame."""
    rot = rotation_matrix(angle)
    pivot = np.array([center[0], center[1], 0.0])
    frames = []
    for i in indices:
        pose = np.asarray(c2ws[i], dtype=np.float64)
        R, t = pose[:3, :3], pose[:3, 3]
        eye = rot @ (t - pivot) + pivot
        forward = rot @ R[:, 2]          # OpenCV c2w: +z forward
        up = rot @ (-R[:, 1])            # image up is -y
        frames.append({"eye": eye.tolist(),
                       "lookat": (eye + forward).tolist(),
                       "up": up.tolist(),
                       "forward": forward.tolist()})
    return frames


def original_frame_files(entry, view_indices, max_num_frames: int | None = None):
    """Map subsampled view indices back to the original RGB files.

    load_scene_frames() caps the scene at `max_num_frames` via a linspace over
    the full frame list, so view index i refers to that subsample, not to the
    original frame numbering.
    """
    if max_num_frames is None:
        from utils.data_scannetpp import default_max_frames

        max_num_frames = default_max_frames()
    files = sorted(f for f in os.listdir(entry["frames_dir"]) if f.endswith(".png"))
    n = len(files)
    if n > max_num_frames:
        selected = np.rint(np.linspace(0, n - 1, num=max_num_frames)).astype(np.int64)
        selected = np.clip(selected, 0, n - 1)
    else:
        selected = np.arange(n, dtype=np.int64)
    paths = [Path(entry["frames_dir"]) / files[selected[i]] for i in view_indices]
    return paths, [int(selected[i]) for i in view_indices]


def room_instance(recon_scene: Path) -> tuple[int, dict]:
    """Largest-volume composed node: the reconstructed room."""
    appearance = recon_scene / "appearance"
    summary = json.loads((appearance / "appearance_summary.json").read_text())
    by_node = {r["node_name"]: r for r in summary["composed_world_scene"]["objects"]}
    glb = trimesh.load(appearance / "predicted_textured_world_scene.glb", process=False)
    volumes = {}
    for node in glb.graph.nodes_geometry:
        transform, name = glb.graph[node]
        v = np.asarray(glb.geometry[name].vertices)
        w = (transform[:3, :3] @ v.T + transform[:3, 3:4]).T
        volumes[node] = float(np.prod(w.max(0) - w.min(0)))
    order = sorted(volumes, key=lambda k: -volumes[k])
    room = order[0]
    ratio = volumes[room] / volumes[order[1]] if len(order) > 1 else float("inf")
    return int(by_node[room]["instance_id"]), {
        "selected_node": room, "volume": round(volumes[room], 3),
        "volume_ratio_to_runner_up": round(ratio, 2), "num_nodes": len(order)}


def build_task(args, scene_id, frames, K, width, height, glb, room_id, out_dir):
    methods, policy = [], {}
    for tag, exclude in (("bg", None), ("nobg", room_id)):
        for kind, material in (("geometry", "instance_color_by_instance"),
                               ("textured", None)):
            key = f"ours_{kind}_{tag}"
            asset = {"name": key, "scene_role": "ours_scene_composite",
                     "kind": "world_scene_glb", "path": str(glb),
                     "force_opaque_materials": []}
            if exclude is not None:
                asset["exclude_instance_id"] = exclude
            if material is not None:
                asset["material_mode"] = material
            methods.append({"key": key, "label": key, "assets": [asset]})
            policy[key] = "ours_reconstructed_textured" if tag == "bg" else "none"

    views = [{"view_index": i, "frame": {k: f[k] for k in ("eye", "lookat", "up")},
              "forward": f["forward"], "visible_gt_objects": [], "visible_gt_pixels": {}}
             for i, f in enumerate(frames)]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.json").write_text(json.dumps(
        {"K": K, "width": width, "height": height,
         "frames": [v["frame"] for v in views]}, indent=1))
    task = {
        "schema": "ff_efm3d_shaper_pipeline_comparison_task_v2",
        "comparison_mode": "scannetpp_percept_recon_only",
        "dataset": "scannetpp", "scene_id": scene_id,
        "coordinate_contract": "wall_aligned_world_without_posthoc_alignment",
        "scope": "method_specific_textured_backgrounds",
        "excluded_object_ids": [], "explicit_excluded_object_ids": [],
        "auto_excluded_room_slab_ids": [], "evaluation_object_ids": [],
        "rendered_gt_object_ids": [],
        "width": width, "height": height, "K": K,
        "aspect_policy": {"name": "native_dataset_camera",
                          "source_size": [height, width],
                          "target_size": [width, height]},
        "samples": 32, "recipe": "canonical_pbr", "render_profile": "inference-rgb",
        "views": views, "context_assets": [], "background_policy": policy,
        "empty_background_rgb": [255, 255, 255],
        "methods": methods, "output_dir": str(out_dir),
        "camera_path": str(out_dir / "cameras.json"),
    }
    (out_dir / "render_task.json").write_text(json.dumps(task, indent=1))
    return out_dir / "render_task.json"


def assemble(scene_id, out_dir, vis_dir, frames, width, height, view_indices,
             rgb_paths, source_frames):
    from PIL import Image, ImageDraw, ImageFont

    HDR = 46
    def font(size):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(path).is_file():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()
    F_T, F_S = font(26), font(17)

    vis_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for tag, title in (("bg", "with reconstructed room"),
                       ("nobg", "room instance excluded")):
        cols = [(None, "Input RGB (reference)"),
                ("geometry", "Ours geometry | instance colors"),
                ("textured", "Ours textured")]
        fig = Image.new("RGB", (width * len(cols), (height + HDR) * len(frames)),
                        (17, 19, 24))
        draw = ImageDraw.Draw(fig)
        for row in range(len(frames)):
            for col, (kind, label) in enumerate(cols):
                if kind is None:
                    panel = Image.open(rgb_paths[row]).convert("RGB")
                    if panel.size != (width, height):
                        panel = panel.resize((width, height), Image.LANCZOS)
                    subtitle = f"source frame {source_frames[row]}"
                else:
                    panel = Image.open(
                        out_dir / "views" / f"ours_{kind}_{tag}_v{row:04d}_beauty.png"
                    ).convert("RGB")
                    subtitle = f"view {view_indices[row]} | {tag}"
                x, y = col * width, row * (height + HDR)
                draw.text((x + 12, y + 6), f"{label}", font=F_T, fill=(255, 255, 255))
                draw.text((x + 12, y + 28), subtitle, font=F_S, fill=(170, 176, 190))
                fig.paste(panel, (x, y + HDR))
        path = vis_dir / f"scannetpp_{scene_id}_recon_8views_{tag}.png"
        fig.save(path)
        written[tag] = {"path": str(path), "size": list(fig.size), "note": title}
        print(f"  wrote {path.name}  {fig.size}  ({title})", flush=True)
    return written


def main() -> None:
    args = parse_args()
    entry, intrinsics, c2ws, center, angle, diag = scene_alignment(args.scene_id)
    rng = np.random.default_rng(args.seed)
    view_indices = sorted(rng.choice(len(c2ws), size=args.num_views, replace=False).tolist())
    frames = camera_frames(c2ws, center, angle, view_indices)
    print(f"[render] {args.scene_id}: rotation {np.degrees(angle):+.2f} deg, "
          f"frames {view_indices}", flush=True)

    K0 = np.asarray(intrinsics[0], dtype=np.float64)
    if float(np.abs(np.asarray(intrinsics) - K0).max()) > 1e-6:
        raise SystemExit("per-frame intrinsics differ; the renderer takes a single K")
    scale = int(args.resolution_scale)
    import json as _json
    cam_json = _json.loads(Path(entry["camera_path"]).read_text())
    width = int(cam_json["width"]) * scale
    height = int(cam_json["height"]) * scale
    K = (K0 * scale)
    K[2, 2] = 1.0
    K = K.tolist()

    recon_scene = args.recon_root / args.scene_id
    glb = recon_scene / "appearance/predicted_textured_world_scene.glb"
    room_id, room_audit = room_instance(recon_scene)
    print(f"[render] room instance {room_id} "
          f"(volume ratio {room_audit['volume_ratio_to_runner_up']})", flush=True)

    src_dir = args.output_root / "sources" / args.scene_id
    task_path = build_task(args, args.scene_id, frames, K, width, height,
                           glb, room_id, src_dir)

    if not args.skip_render:
        args.log_root.mkdir(parents=True, exist_ok=True)
        log = args.log_root / f"render_{args.scene_id}.log"
        cmd = [str(BLENDER), "-b", "--factory-startup", "--python",
               str(REPO_ROOT / "eval/rendering/scene_comparison_blender.py"),
               "--", "--task", str(task_path), "--overwrite"]
        print(f"[render] blender -> {log}", flush=True)
        with log.open("w") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                  cwd=str(REPO_ROOT),
                                  env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
        if proc.returncode != 0:
            raise SystemExit(f"blender failed; see {log}")

    rgb_paths, source_frames = original_frame_files(entry, view_indices)
    print(f"[render] source RGB frames: {source_frames}", flush=True)
    written = assemble(args.scene_id, src_dir, args.output_root / "visualization",
                       frames, width, height, view_indices, rgb_paths, source_frames)
    summary_path = args.output_root / f"{args.scene_id}_render_summary.json"
    summary_path.write_text(json.dumps(
        {"scene_id": args.scene_id, "view_indices": view_indices,
         "rotation_deg": float(np.degrees(angle)), "K": K,
         "width": width, "height": height,
         "aspect": round(width / height, 4),
         "source_frames": source_frames,
         "room_instance_id": room_id, "room_audit": room_audit,
         "figures": written}, indent=2) + "\n")
    print(f"[render] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
