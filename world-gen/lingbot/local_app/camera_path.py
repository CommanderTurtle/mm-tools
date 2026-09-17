from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """OpenCV camera-to-world rotation (+x right, +y down, +z forward)."""

    y, p, r = map(math.radians, (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def _pose(position: list[float], yaw: float, pitch: float, roll: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = _rotation(yaw, pitch, roll)
    value[:3, 3] = np.asarray(position, dtype=np.float32)
    return value


def _keyframes(controls: dict[str, Any], frames: int) -> np.ndarray:
    raw = controls.get("keyframes") or []
    if not raw:
        raw = [
            {"frame": 0, "x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "roll": 0},
            {"frame": frames - 1, "x": 0, "y": 0, "z": 1, "yaw": 0, "pitch": 0, "roll": 0},
        ]
    points = sorted(raw, key=lambda item: int(item.get("frame", 0)))
    points[0] = {**points[0], "frame": 0}
    points[-1] = {**points[-1], "frame": frames - 1}
    channels = ("x", "y", "z", "yaw", "pitch", "roll")
    source = np.asarray([max(0, min(frames - 1, int(item.get("frame", 0)))) for item in points])
    target = np.arange(frames)
    values = {
        key: np.interp(target, source, [float(item.get(key, 0.0)) for item in points])
        for key in channels
    }
    return np.stack(
        [_pose([values["x"][i], values["y"][i], values["z"][i]], values["yaw"][i], values["pitch"][i], values["roll"][i]) for i in range(frames)]
    )


def design(controls: dict[str, Any], destination: Path) -> dict[str, Any]:
    frames = int(controls.get("frame_num", 81))
    frames = max(5, ((frames - 1) // 4) * 4 + 1)
    motion = str(controls.get("camera_motion", "dolly_forward"))
    distance = float(controls.get("travel", 1.0))
    yaw = float(controls.get("yaw", 45.0))
    pitch = float(controls.get("pitch", 0.0))
    roll = float(controls.get("roll", 0.0))
    curve = float(controls.get("ease", 0.0))
    linear = np.linspace(0.0, 1.0, frames)
    t = linear if curve <= 0 else (1.0 - np.cos(np.pi * linear)) * 0.5

    if motion == "custom":
        poses = _keyframes(controls, frames)
    else:
        poses = []
        for amount in t:
            position = [0.0, 0.0, 0.0]
            heading, tilt, bank = 0.0, pitch * amount, roll * amount
            if motion == "dolly_forward":
                position[2] = distance * amount
            elif motion == "dolly_back":
                position[2] = -distance * amount
            elif motion == "truck_left":
                position[0] = -distance * amount
            elif motion == "truck_right":
                position[0] = distance * amount
            elif motion == "crane_up":
                position[1] = -distance * amount
            elif motion == "crane_down":
                position[1] = distance * amount
            elif motion == "pan":
                heading = yaw * amount
            elif motion == "orbit":
                angle = math.radians(yaw * amount)
                position = [distance * math.sin(angle), 0.0, distance * (1.0 - math.cos(angle))]
                heading = -yaw * amount
            elif motion == "handheld":
                position = [
                    distance * 0.04 * math.sin(amount * math.pi * 7),
                    distance * 0.025 * math.sin(amount * math.pi * 11),
                    distance * amount,
                ]
                heading = yaw * 0.035 * math.sin(amount * math.pi * 5)
                tilt += pitch * 0.04 * math.sin(amount * math.pi * 9)
            elif motion != "locked":
                raise ValueError(f"Unsupported camera motion: {motion}")
            poses.append(_pose(position, heading, tilt, bank))
        poses = np.stack(poses)

    focal = float(controls.get("focal_length", 700.0))
    cx = float(controls.get("principal_x", 416.0))
    cy = float(controls.get("principal_y", 240.0))
    intrinsics = np.tile(np.asarray([[focal, focal, cx, cy]], dtype=np.float32), (frames, 1))
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "poses.npy", poses.astype(np.float32))
    np.save(destination / "intrinsics.npy", intrinsics)
    metadata = {
        "schema": "mmtools_lingbot_camera_v1",
        "coordinate_system": "OpenCV camera-to-world (+x right, +y down, +z forward)",
        "frames": frames,
        "motion": motion,
        "travel": distance,
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "roll_degrees": roll,
        "intrinsics_832x480": [focal, focal, cx, cy],
        "first_pose": poses[0].tolist(),
        "last_pose": poses[-1].tolist(),
    }
    (destination / "camera.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_svg(destination / "camera-path.svg", poses, motion)
    return metadata


def _safe_member(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Unsafe path in action bundle: {name}")
    return target


def import_bundle(source: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".npz":
        archive = np.load(source, allow_pickle=False)
        if not {"poses", "intrinsics"}.issubset(archive.files):
            raise ValueError("Action NPZ must contain poses and intrinsics arrays.")
        np.save(destination / "poses.npy", archive["poses"])
        np.save(destination / "intrinsics.npy", archive["intrinsics"])
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as bundle:
            candidates = {Path(name).name: name for name in bundle.namelist() if not name.endswith("/")}
            for expected in ("poses.npy", "intrinsics.npy"):
                if expected not in candidates:
                    raise ValueError(f"Action ZIP is missing {expected}.")
                target = _safe_member(destination, expected)
                target.write_bytes(bundle.read(candidates[expected]))
    else:
        raise ValueError("Action bundle must be .npz or .zip.")
    poses = np.load(destination / "poses.npy", allow_pickle=False)
    intrinsics = np.load(destination / "intrinsics.npy", allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 5:
        raise ValueError("poses.npy must have shape [frames, 4, 4] with at least five frames.")
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError("intrinsics.npy must have shape [frames, 4] (fx, fy, cx, cy).")
    metadata = {
        "schema": "mmtools_lingbot_camera_v1",
        "source": source.name,
        "frames": int(len(poses)),
        "intrinsics_rows": int(len(intrinsics)),
        "coordinate_system": "OpenCV camera-to-world",
    }
    (destination / "camera.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_svg(destination / "camera-path.svg", poses, "imported")
    return metadata


def package(source: Path, destination: Path) -> Path:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("poses.npy", "intrinsics.npy", "camera.json", "camera-path.svg"):
            path = source / name
            if path.is_file():
                archive.write(path, name)
    return destination


def _write_svg(path: Path, poses: np.ndarray, title: str) -> None:
    width, height, pad = 960, 420, 48
    xyz = np.asarray(poses)[:, :3, 3]

    def scale(values: np.ndarray, lo: float, hi: float) -> list[float]:
        minimum, maximum = float(values.min()), float(values.max())
        if abs(maximum - minimum) < 1e-8:
            return [(lo + hi) / 2] * len(values)
        return [lo + (float(value) - minimum) / (maximum - minimum) * (hi - lo) for value in values]

    x = scale(xyz[:, 0], pad, width - pad)
    z = scale(xyz[:, 2], height - pad, pad)
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, z))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="28" fill="#09121d"/><g stroke="#1b3445" opacity=".65">''' + "".join(
        f'<line x1="{i}" y1="{pad}" x2="{i}" y2="{height-pad}"/>' for i in range(pad, width, 64)
    ) + "".join(f'<line x1="{pad}" y1="{i}" x2="{width-pad}" y2="{i}"/>' for i in range(pad, height, 64)) + f'''</g>
<polyline points="{points}" fill="none" stroke="#65f0be" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="{x[0]:.1f}" cy="{z[0]:.1f}" r="10" fill="#d6a678"/><circle cx="{x[-1]:.1f}" cy="{z[-1]:.1f}" r="10" fill="#65f0be"/>
<text x="{pad}" y="30" fill="#e9f6f2" font-family="ui-monospace,monospace" font-size="16">{title} · top view · {len(poses)} source poses</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
