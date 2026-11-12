"""
Overlay WASD + arrow-key indicators on camera-control inference videos.

For each video file under ``--video-dir`` that matches the camera-bench
filename schema (``CC-<id>-<bucket>-<idx>__...mp4``), this script:

    1. Looks up the matching trajectory NPZ via ``--suite-json``
       (``camera_trajectory_spec.ground_truth_npz`` field).
    2. Builds a ``pose_dict`` and discretises the per-frame relative c2w
       motion into 4T4R action labels using
       :func:`hymm.models.camera.geometry.discretize_poses.discretize_poses_old`
       with ``downsample_ratio=1`` so every original-FPS frame gets one label.
    3. Decodes each label into 8 binary key states:
        - **Movement (WASD)**: W = forward, A = left, S = backward, D = right
        - **View (arrow keys)**: ↑ = pitch up, ↓ = pitch down,
                                 ← = yaw left, → = yaw right
    4. Re-encodes the video, overlaying:
        - WASD panel in the **bottom-left** corner
        - Arrow-key panel in the **bottom-right** corner
       Active keys are drawn with a bright blue fill (matching the WASD
       overlay style in the reference UI screenshot); inactive keys stay
       dim translucent grey.

Frame-count mismatches between the NPZ (trajectory was generated at
``fps=24``) and the rendered mp4 (e.g. ``inference.video_fps=16``) are
handled by **nearest-neighbor mapping** ``i_traj = round(i_video *
(N_traj - 1) / (N_video - 1))``, so overlays line up correctly even when
the model decodes at a different temporal stride than the trajectory.

Usage:
    PYTHONPATH=. python3 hymm/models/camera/geometry/overlay_camera_keys_on_video.py \\
        --video-dir /path/to/videos \\
        --suite-json /path/to/suite.json \\
        --output-dir /path/to/overlaid_videos

    # filter to a single test_id (good for spot-checking):
    PYTHONPATH=. python3 hymm/models/camera/geometry/overlay_camera_keys_on_video.py \\
        --video-dir /path/to/videos \\
        --suite-json /path/to/suite.json \\
        --output-dir ./_overlay_check \\
        --filter CC-C11-XL-050
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hymm.models.camera.geometry.discretize_poses import discretize_poses_old  # noqa: E402
from hymm.models.camera.geometry.discretize_utils import LABEL_TO_ONEHOT_4T4R  # noqa: E402

import imageio  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — visual style for the overlay
# ---------------------------------------------------------------------------

KEY_SIZE_DEFAULT = 36  # px, square key — keep small so the panel
# stays well under ~20% of frame height for
# typical 352p / 480p inference outputs.
KEY_GAP_DEFAULT = 5  # px between adjacent keys in a panel
PANEL_MARGIN_DEFAULT = 14  # px from frame edge

# Colors are RGBA. Inactive keys are translucent so they don't block content.
COLOR_INACTIVE_FILL = (30, 30, 40, 170)
COLOR_INACTIVE_BORDER = (200, 200, 210, 200)
COLOR_INACTIVE_TEXT = (220, 220, 225, 230)
COLOR_ACTIVE_FILL = (76, 139, 245, 235)  # Material/Tailwind blue
COLOR_ACTIVE_BORDER = (255, 255, 255, 255)
COLOR_ACTIVE_TEXT = (255, 255, 255, 255)
SHADOW_COLOR = (0, 0, 0, 130)

KEY_RADIUS = 6  # rounded corner radius (scales with key_size)
SHADOW_OFFSET = (2, 2)
BORDER_WIDTH = 2

# Test-id pattern: CC-<local>-<bucket>-<idx>
# Examples: CC-A01-S-001, CC-C11-XL-050, CC-DM01-XL-001
TEST_ID_RE = re.compile(r"^(CC-[A-Z0-9]+-[A-Z]+-\d+)")


# Common font search list (first existing wins). Bold sans-serif renders
# the arrow glyphs and the WASD letters cleanly.
FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Suite indexing
# ---------------------------------------------------------------------------


@dataclass
class SuiteEntry:
    test_id: str
    npz_path: Path  # absolute, after resolution against suite_json's parent
    num_frames_pixel: int


def _load_suite_index(suite_json: Path) -> Dict[str, SuiteEntry]:
    """Parse suite.json -> {test_id: SuiteEntry}.

    Uses each entry's ``camera_trajectory_spec.ground_truth_npz`` field
    (relative path) and resolves it against the suite.json's parent dir.
    Entries without a valid NPZ field are skipped with a warning.
    """
    suite_dir = suite_json.resolve().parent
    with open(suite_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{suite_json} does not contain a JSON list at the top level.")

    index: Dict[str, SuiteEntry] = {}
    for it in data:
        tid = it.get("test_id")
        spec = it.get("camera_trajectory_spec", {})
        if not tid or not isinstance(spec, dict):
            continue
        npz_rel = spec.get("ground_truth_npz")
        if not npz_rel:
            continue
        npz_abs = (suite_dir / npz_rel).resolve()
        index[tid] = SuiteEntry(
            test_id=tid,
            npz_path=npz_abs,
            num_frames_pixel=int(spec.get("num_frames_pixel", 0)),
        )
    return index


def _extract_test_id(filename: str) -> Optional[str]:
    m = TEST_ID_RE.match(filename)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Trajectory discretisation
# ---------------------------------------------------------------------------


def _coerce_intrinsics_Tx3x3(intrinsics: np.ndarray, num_frames: int) -> np.ndarray:
    """Normalize NPZ intrinsics to ``[T, 3, 3]``.

    Accepts constructed-bench ``[T,3,3]`` and vipe-export formats:
    shared ``[fx,fy,cx,cy]`` / ``[1,4]`` / ``[T,4]``.
    """
    from hymm.models.camera.geometry.normalization import _intrinsics_4_to_3x3

    intr = np.asarray(intrinsics, dtype=np.float64)
    if intr.ndim == 1:
        intr = intr.reshape(1, -1)
    if intr.ndim == 2 and intr.shape[-1] == 4:
        intr = _intrinsics_4_to_3x3(intr)
    elif not (intr.ndim == 3 and intr.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Unsupported intrinsics shape {tuple(intr.shape)}; "
            f"expected [T,3,3] or [fx,fy,cx,cy]/[T,4]."
        )
    if intr.shape[0] == 1 and num_frames > 1:
        intr = np.broadcast_to(intr, (num_frames,) + intr.shape[1:]).copy()
    elif intr.shape[0] != num_frames:
        raise ValueError(
            f"c2ws/intrinsics length mismatch: c2ws={num_frames} "
            f"intrinsics={intr.shape[0]}"
        )
    return intr


def _npz_to_pose_dict(
    c2ws: np.ndarray, intrinsics: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convert NPZ arrays into the dict layout expected by discretize_poses_old."""
    n = c2ws.shape[0]
    intrinsics = _coerce_intrinsics_Tx3x3(intrinsics, n)
    pose_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for i in range(n):
        pose_dict[str(i)] = {
            "extrinsic": c2ws[i].astype(np.float64),
            "intrinsic": intrinsics[i].astype(np.float64),
        }
    return pose_dict


def _decode_action_label_4t4r(
    label: int,
) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
    """Return (W, A, S, D, up, down, left, right) booleans for a 4T4R label."""
    trans_label = int(label) // 9
    rot_label = int(label) % 9
    trans_one_hot = LABEL_TO_ONEHOT_4T4R.get(trans_label, (0, 0, 0, 0))
    rot_one_hot = LABEL_TO_ONEHOT_4T4R.get(rot_label, (0, 0, 0, 0))
    fwd, bwd, right, left = (bool(x) for x in trans_one_hot)
    yaw_r, yaw_l, pitch_up, pitch_down = (bool(x) for x in rot_one_hot)
    return (
        fwd,  # W = forward
        left,  # A = left
        bwd,  # S = backward
        right,  # D = right
        pitch_up,  # ↑
        pitch_down,  # ↓
        yaw_l,  # ←
        yaw_r,  # →
    )


def _smooth_keys(
    key_seq: List[Tuple[bool, ...]],
    window: int,
) -> List[Tuple[bool, ...]]:
    """Median-filter each binary channel over a centred window of length ``window``.

    Suppresses single-frame flicker without softening real direction changes
    (because each channel is binary and the median picks the dominant value
    in the window).
    """
    if window <= 1 or len(key_seq) <= 1:
        return key_seq
    n = len(key_seq)
    n_ch = len(key_seq[0])
    arr = np.array([list(s) for s in key_seq], dtype=np.uint8)  # [N, n_ch]
    half = window // 2
    out = np.empty_like(arr)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sub = arr[lo:hi]
        out[i] = (sub.sum(axis=0) * 2 >= sub.shape[0]).astype(np.uint8)
    return [tuple(bool(x) for x in row) for row in out]


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


@dataclass
class OverlayStyle:
    key_size: int = KEY_SIZE_DEFAULT
    key_gap: int = KEY_GAP_DEFAULT
    margin: int = PANEL_MARGIN_DEFAULT
    font_scale: float = 0.55  # text size relative to key_size

    def font_size(self) -> int:
        return max(10, int(round(self.key_size * self.font_scale)))


def _draw_one_key(
    canvas: Image.Image,
    top_left: Tuple[int, int],
    size: int,
    label: str,
    active: bool,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x, y = top_left
    box = (x, y, x + size, y + size)

    sx, sy = SHADOW_OFFSET
    shadow_box = (x + sx, y + sy, x + size + sx, y + size + sy)
    draw.rounded_rectangle(shadow_box, radius=KEY_RADIUS, fill=SHADOW_COLOR)

    fill = COLOR_ACTIVE_FILL if active else COLOR_INACTIVE_FILL
    border = COLOR_ACTIVE_BORDER if active else COLOR_INACTIVE_BORDER
    text_color = COLOR_ACTIVE_TEXT if active else COLOR_INACTIVE_TEXT
    draw.rounded_rectangle(
        box, radius=KEY_RADIUS, fill=fill, outline=border, width=BORDER_WIDTH
    )

    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        offx = bbox[0]
        offy = bbox[1]
    except AttributeError:
        tw, th = draw.textsize(label, font=font)  # type: ignore[attr-defined]
        offx = offy = 0
    tx = x + (size - tw) // 2 - offx
    ty = y + (size - th) // 2 - offy
    draw.text((tx, ty), label, font=font, fill=text_color)


def _panel_size(style: OverlayStyle) -> Tuple[int, int]:
    """Width, height of a 3x2 panel (top row 1 key centred, bottom row 3 keys)."""
    w = style.key_size * 3 + style.key_gap * 2
    h = style.key_size * 2 + style.key_gap
    return w, h


def _draw_panel(
    canvas: Image.Image,
    panel_origin: Tuple[int, int],
    style: OverlayStyle,
    top_label: str,
    bottom_labels: Tuple[str, str, str],
    active_top: bool,
    active_bottom: Tuple[bool, bool, bool],
    font: ImageFont.ImageFont,
) -> None:
    """Draw one keyboard panel (top-row centred + 3-key bottom row)."""
    px, py = panel_origin
    s = style.key_size
    gap = style.key_gap

    top_x = px + s + gap
    top_y = py
    _draw_one_key(canvas, (top_x, top_y), s, top_label, active_top, font)

    bot_y = py + s + gap
    for i, (lbl, act) in enumerate(zip(bottom_labels, active_bottom)):
        bx = px + i * (s + gap)
        _draw_one_key(canvas, (bx, bot_y), s, lbl, act, font)


def _render_overlay(
    frame: np.ndarray,
    keys: Tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    style: OverlayStyle,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    """Composite WASD (left) + arrow (right) panels onto an RGB frame."""
    h, w = frame.shape[:2]
    base = Image.fromarray(frame, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    pw, ph = _panel_size(style)
    left_origin = (style.margin, h - ph - style.margin)
    right_origin = (w - pw - style.margin, h - ph - style.margin)

    W, A, S, D, UP, DOWN, LEFT, RIGHT = keys
    _draw_panel(
        overlay,
        left_origin,
        style,
        top_label="W",
        bottom_labels=("A", "S", "D"),
        active_top=W,
        active_bottom=(A, S, D),
        font=font,
    )
    _draw_panel(
        overlay,
        right_origin,
        style,
        top_label="\u2191",  # ↑
        bottom_labels=("\u2190", "\u2193", "\u2192"),  # ← ↓ →
        active_top=UP,
        active_bottom=(LEFT, DOWN, RIGHT),
        font=font,
    )

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composite)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def _gt_keys_to_panel_tuple(
    gt_keys_row: np.ndarray,
    gt_mouse_row: Optional[np.ndarray],
    mouse_thresh: float,
) -> Tuple[bool, ...]:
    """Convert one row of `gt_keys` (channel order W/A/S/D/↑/↓/←/→) plus
    optional mouse deltas into the 8-tuple expected by `_render_overlay`
    (W, A, S, D, UP, DOWN, LEFT, RIGHT).

    A mouse delta with |dx| or |dy| ≥ `mouse_thresh` lights up the
    corresponding arrow key, so a "mouse-only" segment still shows
    something in the arrow panel.
    """
    w = bool(gt_keys_row[0])
    a = bool(gt_keys_row[1])
    s = bool(gt_keys_row[2])
    d = bool(gt_keys_row[3])
    up = bool(gt_keys_row[4])
    dn = bool(gt_keys_row[5])
    lf = bool(gt_keys_row[6])
    rt = bool(gt_keys_row[7])
    if gt_mouse_row is not None:
        mdx, mdy = float(gt_mouse_row[0]), float(gt_mouse_row[1])
        if mdx >= mouse_thresh:
            rt = True
        elif mdx <= -mouse_thresh:
            lf = True
        if mdy >= mouse_thresh:
            dn = True  # mouse_dy > 0 = move-down = pitch down
        elif mdy <= -mouse_thresh:
            up = True
    return (w, a, s, d, up, dn, lf, rt)


def _precomputed_demo_keys(data, smooth_window):
    """Map stored inferred labels to each original turn, then hold its tail
    over explicit right-padding. Never difference normalized camera matrices.
    Smoothing is turn-local, so control cannot bleed across turn boundaries.
    """
    from hymm.models.camera.geometry.demo_precomputed import load_precomputed

    boundaries = data["segment_boundaries_pixel"].tolist()
    arrays = load_precomputed(
        data,
        boundaries[-1],
        boundaries,
        json.loads(str(data["normalization_json"].item())),
    )
    offsets = arrays["turn_camera_offsets"]
    keys = []
    for i, count in enumerate(data["turn_num_frames"].tolist()):
        start, end = offsets[i : i + 2]
        labels = arrays["action_labels"][start:end]
        if np.any(labels >= 81):
            raise ValueError("Precomputed 4T4R labels must be in [0, 80]")
        indices = np.rint(np.linspace(0, len(labels) - 1, count)).astype(int)
        turn_keys = [_decode_action_label_4t4r(int(labels[j])) for j in indices]
        if smooth_window > 1:
            turn_keys = _smooth_keys(turn_keys, window=smooth_window)
        if not arrays["turn_has_camera"][i]:
            turn_keys = [(False,) * 8] * count
        padding = boundaries[i + 1] - boundaries[i] - count
        keys.extend(turn_keys)
        keys.extend([turn_keys[-1]] * padding)
    return keys, "precomputed_demo_inferred"


def _compute_traj_keys(
    npz_path: Path,
    mode: str,
    smooth_window: int,
    mouse_thresh: float = 0.5,
    abs_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[bool, ...]], str]:
    """Load NPZ and return (per-pixel-frame key tuples, source_label).

    Source-label is one of:
      - "gt"          : NPZ contains a `gt_keys` array (built by
                        `tools_tmp/build_camera_bench_keyboard.py`).
                        Used directly — no discretisation involved.
      - "discretised" : Fallback path; per-frame relative c2w motion is
                        run through `discretize_poses_old` 4T4R and decoded.
    """
    data = np.load(npz_path)
    if "camera_format" in data.files:
        try:
            return _precomputed_demo_keys(data, smooth_window)
        finally:
            data.close()
    if "c2ws" not in data.files or "intrinsics" not in data.files:
        raise ValueError(
            f"{npz_path} missing 'c2ws' or 'intrinsics' (found: {data.files})"
        )

    # ---- Preferred path: ground-truth keys baked into the NPZ ----
    if "gt_keys" in data.files:
        gt_keys = data["gt_keys"]
        if gt_keys.ndim != 2 or gt_keys.shape[1] != 8:
            raise ValueError(f"{npz_path} gt_keys must be [N,8], got {gt_keys.shape}")
        gt_mouse = data["gt_mouse"] if "gt_mouse" in data.files else None
        if gt_mouse is not None and (
            gt_mouse.ndim != 2 or gt_mouse.shape != (gt_keys.shape[0], 2)
        ):
            # Don't fail hard — just ignore malformed mouse, we can still draw keys.
            gt_mouse = None
        keys: List[Tuple[bool, ...]] = []
        for i in range(gt_keys.shape[0]):
            row = gt_keys[i]
            mrow = gt_mouse[i] if gt_mouse is not None else None
            keys.append(_gt_keys_to_panel_tuple(row, mrow, mouse_thresh))
        if smooth_window > 1:
            keys = _smooth_keys(keys, window=smooth_window)
        return keys, "gt"

    # ---- Fallback: discretise the c2w sequence ----
    c2ws = data["c2ws"].astype(np.float64)
    intr = data["intrinsics"].astype(np.float64)
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"{npz_path} c2ws must be [N,4,4], got {c2ws.shape}")

    pose_dict = _npz_to_pose_dict(c2ws, intr)
    _, _, action_labels = discretize_poses_old(
        pose_dict,
        tps=False,
        mode=mode,
        downsample_ratio=1,
        normalize_displacement=True,
        absolute_threshold=abs_threshold,
    )
    keys = [_decode_action_label_4t4r(int(lbl)) for lbl in action_labels]
    if smooth_window > 1:
        keys = _smooth_keys(keys, window=smooth_window)
    return keys, "discretised"


def _build_video_frame_camera_mask(
    n_video: int,
    segment_frame_boundaries: Optional[List[int]],
    segment_has_camera: Optional[List[bool]],
) -> Optional[List[bool]]:
    """Map segment-level camera flags onto rendered video frames."""
    if segment_frame_boundaries is None or segment_has_camera is None:
        return None

    boundaries = [int(value) for value in segment_frame_boundaries]
    flags = [bool(value) for value in segment_has_camera]
    if (
        n_video <= 0
        or len(boundaries) != len(flags) + 1
        or not boundaries
        or boundaries[0] != 0
        or boundaries[-1] <= 0
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    ):
        return None

    timeline_length = boundaries[-1]
    mask: List[bool] = []
    segment_index = 0
    for frame_index in range(n_video):
        timeline_frame = min(
            timeline_length - 1,
            frame_index * timeline_length // n_video,
        )
        while (
            segment_index + 1 < len(flags)
            and timeline_frame >= boundaries[segment_index + 1]
        ):
            segment_index += 1
        mask.append(flags[segment_index])
    return mask


def _process_one_video(
    video_path: Path,
    npz_path: Path,
    output_path: Path,
    mode: str,
    smooth_window: int,
    style: OverlayStyle,
    h264_quality: int,
    abs_threshold: Optional[float] = None,
    segment_frame_boundaries: Optional[List[int]] = None,
    segment_has_camera: Optional[List[bool]] = None,
    video_frame_offset: int = 0,
) -> Dict[str, Any]:
    """Render an annotated copy of ``video_path`` to ``output_path``."""
    keys_traj, key_source = _compute_traj_keys(
        npz_path,
        mode=mode,
        smooth_window=smooth_window,
        abs_threshold=abs_threshold,
    )
    n_traj = len(keys_traj)
    if n_traj < 2:
        raise ValueError(f"trajectory at {npz_path} has only {n_traj} frames")

    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}
    fps = float(meta.get("fps", 24))

    frames = [np.asarray(f) for f in reader]
    reader.close()
    n_video = len(frames)
    if n_video == 0:
        raise ValueError(f"{video_path} has 0 readable frames")
    if video_frame_offset < 0:
        raise ValueError("video_frame_offset must be nonnegative")
    full_video_frames = n_video + video_frame_offset
    if video_frame_offset and (
        segment_frame_boundaries is None
        or segment_frame_boundaries[-1] != full_video_frames
    ):
        raise ValueError(
            "Cropped overlay requires the full generation frame boundaries"
        )
    precomputed_demo = key_source == "precomputed_demo_inferred"
    if precomputed_demo:
        if full_video_frames != n_traj:
            raise ValueError(
                f"Precomputed demo overlay requires {n_traj} full frames, got {full_video_frames}; "
                "refusing whole-video time stretching."
            )
        with np.load(npz_path, allow_pickle=False) as data:
            stored_boundaries = data["segment_boundaries_pixel"].tolist()
            stored_flags = data["turn_has_camera"].tolist()
        if (
            segment_frame_boundaries is not None
            and list(segment_frame_boundaries) != stored_boundaries
        ):
            raise ValueError("Overlay segment boundaries differ from precomputed NPZ")
        if segment_has_camera is not None and list(segment_has_camera) != stored_flags:
            raise ValueError("Overlay camera gates differ from precomputed NPZ")
        segment_frame_boundaries = stored_boundaries
        segment_has_camera = stored_flags
    video_frame_camera_mask = _build_video_frame_camera_mask(
        n_video=full_video_frames,
        segment_frame_boundaries=segment_frame_boundaries,
        segment_has_camera=segment_has_camera,
    )

    font = _load_font(style.font_size())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=h264_quality,
        pixelformat="yuv420p",
        macro_block_size=1,
        # Overlay encoding replaces the video, so it must also keep its index first.
        output_params=["-movflags", "+faststart"],
    )
    try:
        for i_v, frame in enumerate(frames):
            full_index = i_v + video_frame_offset
            i_t = (
                full_index
                if precomputed_demo
                else int(
                    round(full_index * (n_traj - 1) / max(1, full_video_frames - 1))
                )
            )
            i_t = max(0, min(n_traj - 1, i_t))
            keys = keys_traj[i_t]
            if (
                video_frame_camera_mask is not None
                and not video_frame_camera_mask[full_index]
            ):
                if precomputed_demo:
                    # No camera panel on initialization/prompt-only turns.
                    writer.append_data(frame)
                    continue
                keys = tuple(False for _ in keys)
            annotated = _render_overlay(frame, keys, style, font)
            writer.append_data(annotated)
    finally:
        writer.close()

    return {"n_video": n_video, "n_traj": n_traj, "key_source": key_source}


# ---------------------------------------------------------------------------
# Public single-file API (used by hymm/inference.py to overlay keys
# right after a video is rendered, without going through the CLI / suite-json
# discovery path).
# ---------------------------------------------------------------------------


def overlay_one_video(
    video_path,
    npz_path,
    output_path,
    *,
    mode: str = "4t4r_robust",
    smooth_window: int = 5,
    key_size: int = KEY_SIZE_DEFAULT,
    key_gap: int = KEY_GAP_DEFAULT,
    margin: int = PANEL_MARGIN_DEFAULT,
    font_scale: float = 0.55,
    h264_quality: int = 8,
    abs_threshold: Optional[float] = None,
    segment_frame_boundaries: Optional[List[int]] = None,
    segment_has_camera: Optional[List[bool]] = None,
    video_frame_offset: int = 0,
) -> Dict[str, Any]:
    """Overlay WASD + arrow-key indicators on a single rendered video.

    Designed to be called from `hymm/inference.py` immediately after a
    video is written to disk. Accepts ``str`` or ``Path`` for all paths.

    Marked precomputed demo NPZs use stored inferred action labels with
    turn-local timing/smoothing and explicit tail-padding; no normalization or
    camera differencing is repeated. These indicators are not recorded keys.
    Prompt-only turns have no panel, and frame-count mismatches are rejected.
    ``video_frame_offset`` is a prefix removed ONLY from the rendered video.
    Supply the full generation boundaries/gates; overlay samples that original
    timeline starting at the given offset, without stretching it.

    If the NPZ already contains ``gt_keys`` / ``gt_mouse`` (the format
    produced by ``tools_tmp/build_camera_bench_keyboard.py``), those
    ground-truth signals are used directly; otherwise the per-frame keys
    are inferred from c2w deltas via 4T4R discretisation.

    Returns
    -------
    dict
        ``key_source`` is "gt", "discretised", or "precomputed_demo_inferred".
    """
    style = OverlayStyle(
        key_size=int(key_size),
        key_gap=int(key_gap),
        margin=int(margin),
        font_scale=float(font_scale),
    )
    return _process_one_video(
        video_path=Path(video_path),
        npz_path=Path(npz_path),
        output_path=Path(output_path),
        mode=mode,
        smooth_window=int(smooth_window),
        style=style,
        h264_quality=int(h264_quality),
        abs_threshold=(float(abs_threshold) if abs_threshold is not None else None),
        segment_frame_boundaries=segment_frame_boundaries,
        segment_has_camera=segment_has_camera,
        video_frame_offset=int(video_frame_offset),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _list_videos(video_dir: Path, ext: str, filter_pat: Optional[str]) -> List[Path]:
    pat = ext if ext.startswith(".") else f".{ext}"
    files = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == pat
    )
    if filter_pat:
        files = [p for p in files if filter_pat in p.name]
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Overlay WASD + arrow-key indicators (camera trajectory) on inference videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory of input mp4 videos (CC-*-*-*__... naming).",
    )
    parser.add_argument(
        "--suite-json",
        type=str,
        required=True,
        help="suite.json that contains test_id -> ground_truth_npz mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./output_tmp under the current working directory).",
    )
    parser.add_argument(
        "--ext", type=str, default="mp4", help="Video file extension to scan."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter for video filenames (e.g. 'CC-C11-XL').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="4t4r_robust",
        choices=["4t4r", "4t4r_robust"],
        help="Discretisation mode passed to discretize_poses_old.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Median-filter window over per-frame key flags (1 = off).",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=None,
        help="Override discretize_4t4r_robust absolute per-frame displacement "
        "threshold (world units). Lower (e.g. 0.002) to capture slow steady "
        "lateral motion such as truck_left/right. None = discretiser default (0.02).",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=KEY_SIZE_DEFAULT,
        help="Square key side length in px.",
    )
    parser.add_argument(
        "--key-gap",
        type=int,
        default=KEY_GAP_DEFAULT,
        help="Gap between adjacent keys in px.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=PANEL_MARGIN_DEFAULT,
        help="Margin from frame edge in px.",
    )
    parser.add_argument(
        "--font-scale", type=float, default=0.55, help="Font size relative to key_size."
    )
    parser.add_argument(
        "--h264-quality",
        type=int,
        default=8,
        help="imageio libx264 quality (0..10, higher = better).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose annotated output already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many videos (debug).",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    suite_json = Path(args.suite_json).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "output_tmp").resolve()
    )

    if not video_dir.is_dir():
        sys.exit(f"--video-dir not a directory: {video_dir}")
    if not suite_json.is_file():
        sys.exit(f"--suite-json not a file: {suite_json}")

    print(f"video_dir : {video_dir}")
    print(f"suite_json: {suite_json}")
    print(f"output_dir: {output_dir}")

    suite_index = _load_suite_index(suite_json)
    print(f"suite_index: {len(suite_index)} entries with NPZ paths")

    videos = _list_videos(video_dir, args.ext, args.filter)
    if args.limit is not None:
        videos = videos[: args.limit]
    print(f"videos    : {len(videos)} matching files")
    if not videos:
        return

    style = OverlayStyle(
        key_size=args.key_size,
        key_gap=args.key_gap,
        margin=args.margin,
        font_scale=args.font_scale,
    )

    n_done = n_skip = n_miss = n_err = 0
    for i, vp in enumerate(videos, 1):
        tid = _extract_test_id(vp.name)
        out_path = output_dir / vp.name
        prefix = f"[{i}/{len(videos)}] {vp.name}"

        if not tid:
            print(f"{prefix}  ! no test_id parseable, skip")
            n_miss += 1
            continue
        entry = suite_index.get(tid)
        if entry is None:
            print(f"{prefix}  ! test_id={tid} not in suite, skip")
            n_miss += 1
            continue
        if not entry.npz_path.is_file():
            print(f"{prefix}  ! NPZ missing: {entry.npz_path}, skip")
            n_miss += 1
            continue
        if args.skip_existing and out_path.is_file():
            print(f"{prefix}  - exists, skip")
            n_skip += 1
            continue

        try:
            stats = _process_one_video(
                video_path=vp,
                npz_path=entry.npz_path,
                output_path=out_path,
                mode=args.mode,
                smooth_window=args.smooth_window,
                style=style,
                h264_quality=args.h264_quality,
                abs_threshold=args.abs_threshold,
            )
        except Exception as e:
            print(f"{prefix}  ! error: {e}")
            n_err += 1
            continue
        print(
            f"{prefix}  ok  n_video={stats['n_video']} n_traj={stats['n_traj']} src={stats['key_source']} -> {out_path.name}"
        )
        n_done += 1

    print("\n" + "=" * 70)
    print(f"done={n_done}  skip_existing={n_skip}  missing={n_miss}  errors={n_err}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
