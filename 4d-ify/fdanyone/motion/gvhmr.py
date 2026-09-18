"""Classic GVHMR inference used by 4DAnyone."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import itertools
import json
import os
import sys
import traceback
import types
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fdanyone.errors import AssetError, VideoContractError
from fdanyone.motion.result import SMPL_PARAMETER_NAMES, MotionResult
from fdanyone.vendor.pytorch3d_compat import install_if_needed as install_pytorch3d_compat
from fdanyone.video import CanonicalClip

GVHMR_ASSETS = (
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "inputs/checkpoints/yolo/yolov8x.pt",
    "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
)

# Preserve the integrated source revision in cache identities.
PINNED_GVHMR_REVISION = "6ec3ca39336c50492c0fae65fba2fb831fc7d866"


def validate_gvhmr(root: str | Path, *, require_dpvo: bool = False) -> tuple[Path, str]:
    """Locate the GVHMR checkout and files consumed by inference."""

    path = Path(root).expanduser().resolve()
    required = ("hmr4d/__init__.py", *GVHMR_ASSETS)
    if require_dpvo:
        required = (*required, "inputs/checkpoints/dpvo/dpvo.pth", "dpvo/config/default.yaml")
    missing = [relative for relative in required if not (path / relative).is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise AssetError(
            f"The integrated GVHMR runtime is incomplete under {path}. "
            f"Run this project's setup and model downloader; missing:\n"
            f"  - {formatted}"
        )
    revision = PINNED_GVHMR_REVISION
    if len(revision) != 40:
        raise AssetError(f"Cannot identify the GVHMR revision at {path}.")
    return path, revision


def hydra_override(name: str, value: str | Path) -> str:
    """Quote a path for the internal GVHMR Hydra config."""

    if not name.isidentifier():
        raise ValueError(f"Invalid Hydra field name: {name!r}.")
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


def _full_frame_bbox_xyxy(width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Video dimensions must be positive, got {width}x{height}.")
    return 0.0, 0.0, float(width), float(height)


def _is_empty_tracker_error(error: BaseException) -> bool:
    """Recognize only the pinned GVHMR tracker's empty-result failure."""

    if not isinstance(error, IndexError) or str(error) != "list index out of range":
        return False
    frames = traceback.extract_tb(error.__traceback__)
    return any(
        frame.name == "get_one_track" and Path(frame.filename).parts[-4:] == ("hmr4d", "utils", "preproc", "tracker.py")
        for frame in frames
    )


@contextmanager
def gvhmr_imports(root: Path) -> Iterator[None]:
    """Temporarily import GVHMR as if its checkout were the working tree."""

    old_cwd = Path.cwd()
    root_text = str(root)
    already_present = root_text in sys.path
    if not already_present:
        sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if not already_present:
            with contextlib.suppress(ValueError):
                sys.path.remove(root_text)


@contextmanager
def _legacy_checkpoint_loading():
    """Restore pre-2.6 ``torch.load`` behavior for trusted GVHMR assets."""

    name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                category=UserWarning,
            )
            yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _install_optional_import_stubs() -> None:
    """Avoid the visualization dependency that inference never calls."""

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Wis3D visualization is not part of 4DAnyone inference.")

    if importlib.util.find_spec("wis3d") is None:
        module = types.ModuleType("hmr4d.utils.wis3d_utils")
        module.make_wis3d = unavailable
        module.add_motion_as_lines = unavailable
        sys.modules[module.__name__] = module

def _register_inference_store() -> None:
    """Register only the Hydra groups referenced by GVHMR's demo config."""

    _install_optional_import_stubs()
    for module in (
        "hmr4d.model.gvhmr.gvhmr_pl_demo",
        "hmr4d.model.gvhmr.utils.endecoder",
        "hmr4d.network.gvhmr.relative_transformer",
    ):
        importlib.import_module(module)

    # GVHMR installs a colored handler on the root logger. Remove only that
    # duplicate because the public pipeline already owns a handler.
    logger_module = sys.modules.get("hmr4d.utils.pylogger")
    logger = getattr(logger_module, "Log", None)
    handler = getattr(logger_module, "ch", None)
    if (
        logger is not None
        and handler in logger.handlers
        and any(candidate is not handler for candidate in logger.handlers)
    ):
        logger.removeHandler(handler)


def _run_preprocess(cfg) -> None:
    """Run tracker, ViTPose, and image-feature extraction."""

    import torch
    from hmr4d.utils.geo.hmr_cam import convert_K_to_K4, estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.preproc.tracker import Tracker
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.video_io_utils import get_video_lwh
    from tqdm import tqdm

    Log.info("[Preprocess] Start!")
    started = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        try:
            bbx_xyxy = tracker.get_one_track(video_path).float()
        except IndexError as error:
            if not _is_empty_tracker_error(error):
                raise
            frame_count, width, height = get_video_lwh(video_path)
            bbx_xyxy = torch.tensor(
                _full_frame_bbox_xyxy(width, height),
                dtype=torch.float32,
            ).repeat(frame_count, 1)
            Log.warning(
                "GVHMR tracker produced no usable person track; using a full-frame bbox for "
                f"all {frame_count} frames of {video_path}."
            )
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, weights_only=True)["bbx_xys"]
        Log.info("[Preprocess] bbx (xyxy, xys) from %s", paths.bbx)

    if not Path(paths.vitpose).exists():
        extractor = VitPoseExtractor()
        torch.save(extractor.extract(video_path, bbx_xys), paths.vitpose)
        del extractor
    else:
        Log.info("[Preprocess] vitpose from %s", paths.vitpose)

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        torch.save(extractor.extract_video_features(video_path, bbx_xys), paths.vit_features)
        del extractor
    else:
        Log.info("[Preprocess] vit_features from %s", paths.vit_features)

    if not bool(cfg.static_cam):
        if not Path(paths.slam).exists():
            if bool(cfg.use_dpvo):
                from hmr4d.utils.preproc.slam import SLAMModel

                frame_count, width, height = get_video_lwh(video_path)
                intrinsics = convert_K_to_K4(estimate_K(width, height))
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                progress = tqdm(total=frame_count, desc="DPVO")
                while slam.track():
                    progress.update()
                progress.close()
                torch.save(slam.process(), paths.slam)
            else:
                from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO

                trajectory = SimpleVO(
                    video_path,
                    scale=0.5,
                    step=8,
                    method="sift",
                    f_mm=cfg.f_mm,
                ).compute()
                torch.save(trajectory, paths.slam)
        else:
            Log.info("[Preprocess] camera trajectory from %s", paths.slam)

    Log.info("[Preprocess] End. Time elapsed: %.2fs", Log.time() - started)


def _load_data(cfg):
    """Build the camera and pose tensors consumed by GVHMR."""

    import torch
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor, estimate_K
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.video_io_utils import get_video_lwh
    from pytorch3d.transforms import quaternion_to_matrix

    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if bool(cfg.static_cam):
        rotation_world_to_camera = torch.eye(3).repeat(length, 1, 1)
    else:
        trajectory = torch.load(paths.slam, weights_only=False)
        if bool(cfg.use_dpvo):
            quaternion = torch.as_tensor(trajectory[:, [6, 3, 4, 5]])
            rotation_world_to_camera = quaternion_to_matrix(quaternion).mT
        else:
            rotation_world_to_camera = torch.as_tensor(trajectory[:, :3, :3])
        if len(rotation_world_to_camera) != length:
            raise RuntimeError(
                f"Camera recovery produced {len(rotation_world_to_camera)} frames, expected {length}."
            )
    if cfg.f_mm is None:
        intrinsics = estimate_K(width, height).repeat(length, 1, 1)
    else:
        intrinsics = create_camera_sensor(width, height, float(cfg.f_mm))[2].repeat(length, 1, 1)
    return {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx, weights_only=True)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose, weights_only=True),
        "K_fullimg": intrinsics,
        "cam_angvel": compute_cam_angvel(rotation_world_to_camera),
        "f_imgseq": torch.load(paths.vit_features, weights_only=True),
    }


def _verify_gvhmr_decode(clip: CanonicalClip, working_video: Path, reader_factory) -> None:
    """Ensure GVHMR's own video reader sees the canonical RGB frames."""

    import numpy as np

    reader = reader_factory(str(working_video))
    sentinel = object()
    try:
        for index, (actual, expected) in enumerate(itertools.zip_longest(reader, clip.rgb_frames, fillvalue=sentinel)):
            if actual is sentinel or expected is sentinel or not np.array_equal(actual, expected):
                raise VideoContractError(f"GVHMR decoded a different canonical frame at index {index}.")
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()


def run_gvhmr(
    *,
    clip: CanonicalClip,
    working_video: str | Path,
    output_dir: str | Path,
    gvhmr_root: str | Path,
    device: str,
    camera_motion: str = "simple_vo",
    focal_length_mm: float | None = None,
) -> MotionResult:
    """Recover human and source-camera motion from the canonical clip."""

    camera_motion = str(camera_motion).strip().lower().replace("-", "_")
    if camera_motion not in {"static", "simple_vo", "dpvo"}:
        raise ValueError(f"Unknown camera-motion solver: {camera_motion!r}.")
    if focal_length_mm is not None:
        focal_length_mm = float(focal_length_mm)
        if not 1 <= focal_length_mm <= 1000:
            raise ValueError("focal_length_mm must be between 1 and 1000 mm.")
    static_camera = camera_motion == "static"
    use_dpvo = camera_motion == "dpvo"
    root, revision = validate_gvhmr(gvhmr_root, require_dpvo=use_dpvo)
    working_video = Path(working_video).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with gvhmr_imports(root), _legacy_checkpoint_loading():
        install_pytorch3d_compat()
        import hydra
        import torch
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        from hmr4d.utils.net_utils import detach_to_cpu
        from hmr4d.utils.video_io_utils import get_video_reader
        from hydra import compose, initialize_config_module
        from omegaconf import open_dict

        _register_inference_store()
        overrides = [
            hydra_override("video_name", working_video.stem),
            f"static_cam={str(static_camera).lower()}",
            "verbose=false",
            f"use_dpvo={str(use_dpvo).lower()}",
            hydra_override("output_root", output_root),
        ]
        if focal_length_mm is not None:
            overrides.append(f"f_mm={focal_length_mm}")
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(
                config_name="demo",
                overrides=overrides,
            )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
        with open_dict(cfg):
            cfg.video_path = str(working_video)

        _verify_gvhmr_decode(clip, working_video, get_video_reader)
        _run_preprocess(cfg)
        data = _load_data(cfg)
        if int(data["length"]) != len(clip.frames):
            raise RuntimeError(f"GVHMR decoded {int(data['length'])} frames, expected {len(clip.frames)}.")
        observed_keypoints_2d = data["kp2d"].detach().cpu()
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().to(device)
        with torch.inference_mode():
            prediction = detach_to_cpu(model.predict(data, static_cam=static_camera))
        del model, data
        torch.cuda.empty_cache()

    result = MotionResult(
        gvhmr_revision=revision,
        fps=clip.fps,
        frame_timestamps_sec=tuple(float(frame.canonical_timestamp) for frame in clip.frames),
        source_frame_indices=tuple(frame.source_index for frame in clip.frames),
        source_pts=tuple(frame.source_pts for frame in clip.frames),
        source_size_bytes=clip.source_size_bytes,
        source_mtime_ns=clip.source_mtime_ns,
        image_height=clip.height,
        image_width=clip.width,
        smpl_params_global={name: prediction["smpl_params_global"][name] for name in SMPL_PARAMETER_NAMES},
        smpl_params_incam={name: prediction["smpl_params_incam"][name] for name in SMPL_PARAMETER_NAMES},
        K_fullimg=prediction["K_fullimg"],
        observed_keypoints_2d=observed_keypoints_2d,
        camera_motion=camera_motion,
        focal_length_mm=focal_length_mm,
    )
    result.validate(expected_frames=len(clip.frames))
    return result
