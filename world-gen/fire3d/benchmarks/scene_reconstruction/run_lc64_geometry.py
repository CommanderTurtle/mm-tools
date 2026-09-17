#!/usr/bin/env python3
"""Generate oracle-ID/oracle-pose scene geometry and optional PBR appearance.

This runner deliberately excludes detection, segmentation prediction, and pose prediction.
It uses GT instance masks to collect visible RGB-D points, computes
DINO/AnyUp features online, and predicts each visible object's geometry through:

    LC64 SS flow -> LC64 SS-VAE -> LC64 shape flow -> LC64 shape VAE/TRELLIS2

Predictions are exported in FF canonical object space. ``--predict-appearance`` adds PBR
flow on the frozen generated Shape-X2 support and jointly decodes UV-textured GLBs.
``evaluate_geometry.py`` still places geometry with the exact GT pose and scale.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "trellis2_x2"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from eval.reconstruction.shape_decode import decode_x2_object_meshes  # noqa: E402
from fire3d.runtime.model_loader import (  # noqa: E402
    DEFAULT_SHAPE_DECODER,
    call_model_inference,
    load_decoders,
    load_flow_models,
    seed_everything,
    stable_seed,
    write_point_ply,
)
from eval.reconstruction.lc64_benchmark_appearance import (  # noqa: E402
    appearance_features_from_shared_dino,
    concatenate_sparse_object_pairs,
    load_pbr_flow_model,
    predict_pbr_sparse,
)
from eval.reconstruction.lc64_shape_pbr_decode import (  # noqa: E402
    DecoderPaths,
    compose_textured_scene,
    decode_scene_to_raw_ovoxels,
    load_decoder_bundle,
    postprocess_and_bake_scene,
)
from eval.reconstruction.batched_ovoxel_postprocess import PostprocessConfig  # noqa: E402
from utils.perception_voxel_transfer import (  # noqa: E402
    DEFAULT_SCENE_SCALE,
    DEFAULT_VOXEL_RESOLUTION,
    load_perception_voxel_source,
    transfer_voxel_labels,
)
from utils.background_room_box import (  # noqa: E402
    RoomBox,
    filter_background_instance_near_room_box,
    fit_room_box_isotropic_canonical_transform,
)
from utils.exact_camera_rgb import resolve_frames_dir  # noqa: E402
from utils.discrete import continue_transform_batch, discrete_transform_batch  # noqa: E402
from utils.project import (  # noqa: E402
    downsample_all,
    project_depth_to_world_patch_geometry_with_instance_mask,
)
from utils.read_frames import (  # noqa: E402
    read_cameras,
    read_depths,
    read_masks_v2,
    read_rgbs,
)
from utils.transforms import (  # noqa: E402
    get_transform_matrix_batch,
    point_normalize,
    transform_6d_from_transform_batch,
)


SCHEMA = "ff_scene_lc64_geometry_predictions_v1"


# The shipped LC64 bundle. The DEFAULT_*_RUN / DEFAULT_*_VAE constants imported
# above still point at the original training output directories, which no longer
# exist on this machine -- everything was consolidated into this bundle. Repoint
# the defaults so a no-flag invocation runs the shipped protocol.
BUNDLE = REPO_ROOT / "checkpoints/Fire3D/reconstruction"
BUNDLE_SS_RUN = BUNDLE / "flows/ss"
BUNDLE_SHAPE_RUN = BUNDLE / "flows/shape"
BUNDLE_PBR_RUN = BUNDLE / "flows/pbr"
BUNDLE_SS_VAE = BUNDLE / "vae/ss"
BUNDLE_SHAPE_VAE = BUNDLE / "vae/shape"
BUNDLE_PBR_VAE = BUNDLE / "vae/pbr"


# Process-level model cache: multi-scene invocations reuse loaded checkpoints
# instead of paying ~60 s of model loads per scene. Keyed on every argument
# that changes what gets loaded. Enabled via --model-cache (on for the
# multi-scene batch mode); single-scene runs keep the load-per-run behavior.
_MODEL_CACHE: dict[tuple, Any] = {}


def _cached_load(enabled: bool, key: tuple, loader):
    if not enabled:
        return loader()
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader()
    return _MODEL_CACHE[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument(
        "--batch-scenes",
        type=Path,
        default=None,
        help=(
            "JSON list of {scene_id, manifest, perception_result_dir?} to run "
            "sequentially in ONE process with models loaded once "
            "(~60 s saved per scene after the first). Mutually exclusive with "
            "--manifest/--scene-id."
        ),
    )
    parser.add_argument(
        "--model-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse loaded checkpoints across scenes in this process. "
            "Defaults on for --batch-scenes, off for single-scene runs."
        ),
    )
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--perception-result-dir",
        type=Path,
        default=None,
        help=(
            "Optional eval_perception.py scene directory. When set, predicted "
            "instance labels and raw OBBs replace oracle IDs/poses."
        ),
    )
    parser.add_argument(
        "--perception-label-transfer",
        choices=("repeat", "voxel_transfer"),
        default="voxel_transfer",
        help=(
            "Map saved perception labels to the denser DINO grid by legacy "
            "image-grid repetition or by exact predicted perception voxels."
        ),
    )
    parser.add_argument(
        "--perception-voxel-resolution",
        type=int,
        default=DEFAULT_VOXEL_RESOLUTION,
    )
    parser.add_argument(
        "--perception-scene-scale",
        type=float,
        default=DEFAULT_SCENE_SCALE,
    )
    parser.add_argument(
        "--background-room-box-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit a robust z-up rectangular room to the largest predicted "
            "instance and retain only its points near the six box faces."
        ),
    )
    parser.add_argument(
        "--background-unit-box-prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the flow models' canonical [-0.5, 0.5]^3 conditioning "
            "prune to the background. Disabling this keeps the normal "
            "foreground batch untouched, then reruns only the background "
            "through SS, Shape, and PBR without either unit-box prune."
        ),
    )
    parser.add_argument(
        "--background-canonical-transform",
        choices=("perception_obb", "room_box_isotropic_enclose"),
        default="perception_obb",
        help=(
            "Canonical transform used only for the predicted room background. "
            "perception_obb preserves the detector pose. "
            "room_box_isotropic_enclose keeps the fitted room-box center/yaw "
            "and expands one scalar scale until every retained room-shell "
            "point fits inside the canonical unit cube."
        ),
    )
    parser.add_argument(
        "--background-canonical-margin",
        type=float,
        default=0.01,
        help=(
            "Canonical margin reserved on every side when enclosing retained "
            "background points with --background-canonical-transform "
            "room_box_isotropic_enclose."
        ),
    )
    parser.add_argument(
        "--gt-background-instance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In GT-perception mode, reconstruct the room as instance 0 from the "
            "transforms pickle's layout_<scene> record (or a synthesised "
            "axis-aligned room box when that record is absent). Without this "
            "the GT path silently omits the background, because "
            "visible_object_ids starts at 1."
        ),
    )
    parser.add_argument(
        "--prune-oversized-instances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop non-background detections whose OBB volume exceeds the "
            "volume-ratio times the room-box prior's volume. This rejects "
            "room-scale merged detections and is a no-op when the room prior "
            "is absent."
        ),
    )
    parser.add_argument(
        "--oversized-instance-volume-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum non-background OBB volume as a ratio of the fitted room "
            "volume; 1.0 rejects only detections larger than the room."
        ),
    )
    parser.add_argument("--background-room-box-distance", type=float, default=0.12)
    parser.add_argument(
        "--background-room-box-adaptive-max-distance", type=float, default=0.25
    )
    parser.add_argument(
        "--background-room-box-trim-quantile", type=float, default=0.01
    )
    parser.add_argument("--background-room-box-yaw-samples", type=int, default=180)
    parser.add_argument(
        "--background-room-box-min-points", type=int, default=512
    )
    parser.add_argument(
        "--camera-pose-override",
        type=Path,
        default=None,
        help=(
            "Optional aligned camera NPZ for Imaginarium. When omitted, a "
            "camera_pose_override.json saved in --perception-result-dir is "
            "used automatically. No override preserves the historical loader."
        ),
    )
    parser.add_argument("--ss-run-dir", type=Path, default=BUNDLE_SS_RUN)
    parser.add_argument("--ss-checkpoint", default="model.pt")
    parser.add_argument(
        "--ss-model-family",
        choices=("ss_x2_offline",),
        default="ss_x2_offline",
    )
    parser.add_argument("--shape-run-dir", type=Path, default=BUNDLE_SHAPE_RUN)
    parser.add_argument("--shape-checkpoint", default="model.pt")
    parser.add_argument(
        "--shape-model-family",
        choices=("shape_x2_offline",),
        default="shape_x2_offline",
    )
    parser.add_argument("--ss-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ss-vae-root", type=Path, default=BUNDLE_SS_VAE)
    parser.add_argument("--ss-vae-checkpoint", default="decoder.pt")
    parser.add_argument("--ss-vae-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-vae-root", type=Path, default=BUNDLE_SHAPE_VAE)
    parser.add_argument("--shape-vae-checkpoint", default="decoder.pt")
    parser.add_argument("--shape-vae-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-decoder-pretrained", default=DEFAULT_SHAPE_DECODER)
    parser.add_argument(
        "--pbr-decoder-pretrained",
        default=(
            REPO_ROOT
            / "checkpoints/Fire3D/external/trellis2/"
            "tex_dec_next_dc_f16c32_fp16"
        ),
    )
    parser.add_argument(
        "--predict-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict PBR-X2 and export canonical textured GLBs after geometry inference.",
    )
    parser.add_argument(
        "--geometry-stage-mesh-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Decode Shape-X2 latents to meshes inside the geometry stage. "
            "Defaults to off when --predict-appearance is on: the appearance "
            "stage decodes the same latents again and its postprocessed "
            "canonical_geometry.ply is hardlinked to objects/pred_*.ply, so "
            "the geometry-stage decode (own remesh + decimation + export) is "
            "pure duplicated work. Forced on when appearance is off. Note "
            "--resume-existing cannot detect completed objects while this is "
            "off, because the geometry stage writes no per-object mesh."
        ),
    )
    parser.add_argument(
        "--export-world-object-meshes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also export world-space <scene>/pred_*.ply meshes in perception "
            "mode. Off by default: no production consumer reads them (geometry "
            "eval and the shaper benchmarks resolve mesh_path from the summary, "
            "renders use the composed GLB), and writing them costs one full "
            "mesh export per object."
        ),
    )
    parser.add_argument("--pbr-run-dir", type=Path, default=BUNDLE_PBR_RUN)
    parser.add_argument("--pbr-checkpoint", default="model.pt")
    parser.add_argument(
        "--pbr-model-family",
        choices=("pbr_x2_offline",),
        default="pbr_x2_offline",
    )
    parser.add_argument("--pbr-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pbr-vae-root", type=Path, default=BUNDLE_PBR_VAE)
    parser.add_argument("--pbr-vae-checkpoint", default="decoder.pt")
    parser.add_argument("--pbr-vae-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--appearance-texture-size", type=int, default=512)
    parser.add_argument("--appearance-decimation-target", type=int, default=100000)
    parser.add_argument(
        "--appearance-background-decimation-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply the decimation target for the reconstructed room, "
            "identified by the room-box prior rather than by position 0. "
            "1.0 gives the room the same face budget as every other instance "
            "(user directive 2026-09-03); it was 0.5 while the room was being "
            "treated as a cheap backdrop."
        ),
    )
    parser.add_argument("--appearance-decode-object-chunk-size", type=int, default=16)
    parser.add_argument(
        "--appearance-persist-raw-decoded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--appearance-mesh-batch-size", type=int, default=16)
    parser.add_argument(
        "--appearance-uv-cpu-workers",
        type=int,
        default=None,
        help=(
            "Maximum xatlas CPU workers inside each CuMesh batch. None uses "
            "min(mesh batch size, host CPU count)."
        ),
    )
    parser.add_argument(
        "--appearance-detailed-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize and record decoder, CuMesh topology/UV, and texture "
            "bake substages. Intended for profiling because synchronization "
            "can reduce overlap."
        ),
    )
    parser.add_argument(
        "--appearance-mesh-backend",
        choices=("batch", "scalar", "projection", "hybrid"),
        default="batch",
        help=(
            "batch (default) runs the reference CuMesh chart-UV texturing "
            "with the batched remesh -- the quality-gated path. hybrid "
            "(projection atlas foreground) is faster but fails the visual "
            "gate with black speckle (2026-09-01): projection atlases are "
            "not overlap-free, so keep it opt-in until a visibility-aware "
            "fill exists."
        ),
    )
    parser.add_argument(
        "--appearance-hybrid-foreground-fill-mode",
        choices=("telea", "gpu_dilate", "nearest_push", "jfa"),
        default="gpu_dilate",
    )
    parser.add_argument(
        "--appearance-hybrid-foreground-dilation-pixels", type=int, default=4
    )
    parser.add_argument(
        "--appearance-hybrid-foreground-surface-mapping",
        choices=("source_bvh", "processed"),
        default="processed",
    )
    parser.add_argument("--appearance-hybrid-atlas-size", type=int, default=2048)
    parser.add_argument(
        "--pbr-condition-snap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Snap the PBR flow's condition points onto the decoded 512-res "
            "shape surface (nearest dual vertex, per object, in canonical "
            "space) before the flow runs. Restores the training-time contract "
            "that condition points lie on the shape; costs one extra "
            "shape-only VAE decode per scene."
        ),
    )
    parser.add_argument(
        "--pbr-condition-snap-max-distance",
        type=float,
        default=0.15,
        help=(
            "Canonical-space cap: points whose nearest shape vertex is farther "
            "stay unmoved (likely mis-segmented background, not misalignment)."
        ),
    )
    parser.add_argument(
        "--pbr-condition-snap-fraction",
        type=float,
        default=1.0,
        help="1.0 snaps fully onto the vertex; smaller moves that fraction.",
    )
    parser.add_argument(
        "--appearance-xatlas-block-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="xatlas PackOptions.block_align: 4x4-block chart alignment; "
        "reported large packing speedups for slightly looser atlases.",
    )
    parser.add_argument(
        "--appearance-hybrid-foreground-decimation-target", type=int, default=20000
    )
    parser.add_argument("--appearance-projection-num-views", type=int, default=128)
    parser.add_argument(
        "--appearance-projection-assignment-resolution", type=int, default=512
    )
    parser.add_argument("--appearance-projection-padding-pixels", type=int, default=4)
    parser.add_argument(
        "--appearance-projection-view-assignment-mode",
        choices=("visible_pixels", "coverage_first"),
        default="coverage_first",
    )
    parser.add_argument(
        "--appearance-projection-preferred-visibility-ratio",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--appearance-projection-raster-instance-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--appearance-projection-allow-scalar-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--appearance-topology-initial-target-multiplier", type=int, default=3
    )
    parser.add_argument(
        "--appearance-topology-simplify-threshold", type=float, default=1e-8
    )
    parser.add_argument(
        "--appearance-topology-chart-area-penalty", type=float, default=0.1
    )
    parser.add_argument(
        "--appearance-topology-remesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rebuild topology with CuMesh narrow-band dual contouring "
            "(o_voxel to_glb remesh=True branch) instead of the cleanup schedule."
        ),
    )
    parser.add_argument(
        "--appearance-topology-remesh-band",
        type=float,
        default=None,
        help=(
            "Narrow-band width in voxel units for the dual-contouring remesh. "
            "Default resolves per dataset: 2.0 for ScanNet++ and 1.0 "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--appearance-topology-remesh-project",
        type=float,
        default=0.0,
        help="Ratio for projecting remeshed vertices back onto the source surface.",
    )
    parser.add_argument(
        "--appearance-surface-mapping",
        choices=("source_bvh", "processed"),
        default="source_bvh",
    )
    parser.add_argument(
        "--appearance-sparse-query-mode",
        choices=("exact", "power2"),
        default="power2",
    )
    parser.add_argument(
        "--appearance-texture-fill-mode",
        choices=("telea", "gpu_dilate", "nearest_push", "jfa"),
        default="gpu_dilate",
    )
    parser.add_argument("--appearance-texture-dilation-pixels", type=int, default=32)
    parser.add_argument(
        "--appearance-texture-erode-iterations",
        type=int,
        default=2,
        help="Boundary-shell erosion for the nearest_push fill (Video2Game default 2).",
    )
    # 1024 conditioning points per object (512 FPS + 512 random under the
    # default hybrid sampler). Measured on computer_room_03: the dark
    # terrace-riser threads lose their riser alignment entirely (riser
    # median 1.99 -> 0.19 cells) and the desktop renders clean, at +10%
    # wall time and +14% voxels. NOTE: flows/{ss,shape,pbr}/config.yaml all
    # record max_cond_len 512, so this runs them at twice their trained
    # conditioning length -- a deliberate choice, not an oversight.
    parser.add_argument("--max-cond-len", type=int, default=1024)
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=REPO_ROOT / "third_party/dinov3",
    )
    parser.add_argument(
        "--dino-model-path",
        type=Path,
        default=(
            REPO_ROOT
            / "checkpoints/Fire3D/external/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
    )
    parser.add_argument(
        "--anyup-repo-dir",
        type=Path,
        default=REPO_ROOT / "third_party/anyup",
    )
    parser.add_argument("--dino-upsample", type=int, default=4)
    parser.add_argument("--anyup-frame-batch-size", type=int, default=16)
    parser.add_argument("--inference-num-steps", type=int, default=12)
    parser.add_argument("--sample-method", choices=("random", "fps", "robust_fps", "hybrid"), default="hybrid")
    parser.add_argument(
        "--pbr-sample-method",
        choices=("random", "fps", "robust_fps", "hybrid"),
        default=None,
        help=(
            "Conditioning-point sampler for the PBR flow only; SS and shape "
            "keep --sample-method. Default None inherits --sample-method, so "
            "the shipped protocol is unchanged unless this is set."
        ),
    )
    parser.add_argument(
        "--pbr-shape-source-run",
        type=Path,
        default=None,
        help=(
            "Diagnostic: read each object's Shape-X2 latents from this other "
            "run's reconstruction/<dataset>/<scene>/debug/pred_XXXX/shape_x2.npz "
            "instead of this run's own. Bisects the SS/shape stage from the PBR "
            "stage -- everything else (conditioning, seed, decode) stays local, "
            "so the only substituted variable is the geometry latents PBR is "
            "conditioned on via concat_cond."
        ),
    )
    parser.add_argument(
        "--pbr-guidance-strength",
        type=float,
        default=None,
        help=(
            "Classifier-free guidance strength for the PBR flow. None keeps "
            "FlowEulerCfgSampler's own default of 3.0, which is the value the "
            "2026-08-29 reference used. Raising it sharpens the conditional "
            "signal; the decoded albedo of current runs carries roughly half "
            "the chroma of that reference, so this is the direct control."
        ),
    )
    parser.add_argument(
        "--pbr-max-cond-len",
        type=int,
        default=None,
        help=(
            "Conditioning-point budget for the PBR flow only; SS and shape "
            "keep --max-cond-len. Default None inherits it. max_cond_len is a "
            "forward-time attribute with no weight dependence, so PBR can run "
            "at its trained 512 while geometry stays at 1024."
        ),
    )
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument(
        "--empty-occupancy-policy",
        choices=("skip", "argmax"),
        default="argmax",
        help="argmax preserves benchmark coverage if the SS decoder predicts no cell over threshold.",
    )
    parser.add_argument("--object-batch-size", type=int, default=32)
    parser.add_argument(
        "--ss-shape-object-batch-size",
        type=int,
        default=None,
        help=(
            "Independent SS/Shape flow and SS-VAE object batch size. None "
            "inherits --object-batch-size for backward compatibility."
        ),
    )
    parser.add_argument(
        "--pbr-object-batch-size",
        type=int,
        default=None,
        help=(
            "Independent PBR flow object batch size. None inherits "
            "--object-batch-size for backward compatibility."
        ),
    )
    parser.add_argument("--mesh-decode-batch-size", type=int, default=1)
    parser.add_argument("--shape-decode-resolution", type=int, default=512)
    # Unified with --appearance-decimation-target so both mesh stages simplify
    # to the same budget.
    parser.add_argument("--shape-mesh-decimation-target", type=int, default=100000)
    parser.add_argument("--shape-mesh-remesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-mesh-remesh-band", type=float, default=1.0)
    parser.add_argument("--shape-mesh-remesh-project", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--object-selection",
        choices=("all", "background"),
        default="all",
        help=(
            "Objects reconstructed by this invocation. background is a "
            "diagnostic/refresh path that still loads the full scene condition "
            "but runs flow, VAE, mesh postprocess, composition, and export only "
            "for the room-box background."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--input-audit-only",
        action="store_true",
        help="Validate RGB-D/mask projection and GT transforms without loading any neural model.",
    )
    args = parser.parse_args()
    if args.dino_upsample <= 0 or 16 % args.dino_upsample:
        parser.error("--dino-upsample must be a positive divisor of 16")
    if args.perception_voxel_resolution <= 0 or args.perception_scene_scale <= 0:
        parser.error("perception voxel resolution and scene scale must be positive")
    if args.background_room_box_distance < 0:
        parser.error("background room-box distance must be non-negative")
    if (
        args.background_room_box_adaptive_max_distance
        < args.background_room_box_distance
    ):
        parser.error(
            "background room-box adaptive max distance must be at least distance"
        )
    if not 0 <= args.background_room_box_trim_quantile < 0.25:
        parser.error("background room-box trim quantile must be in [0, 0.25)")
    if args.background_room_box_yaw_samples <= 0:
        parser.error("background room-box yaw samples must be positive")
    if args.background_room_box_min_points <= 0:
        parser.error("background room-box min points must be positive")
    if not 0 <= args.background_canonical_margin < 0.5:
        parser.error("background canonical margin must be in [0, 0.5)")
    if (
        args.background_canonical_transform != "perception_obb"
        and not args.background_room_box_prior
    ):
        parser.error(
            "a room-box background canonical transform requires "
            "--background-room-box-prior"
        )
    if args.object_batch_size <= 0:
        parser.error("--object-batch-size must be positive")
    if (
        args.ss_shape_object_batch_size is not None
        and args.ss_shape_object_batch_size <= 0
    ):
        parser.error("--ss-shape-object-batch-size must be positive")
    if args.pbr_object_batch_size is not None and args.pbr_object_batch_size <= 0:
        parser.error("--pbr-object-batch-size must be positive")
    if args.appearance_decode_object_chunk_size <= 0:
        parser.error("--appearance-decode-object-chunk-size must be positive")
    if (
        args.appearance_uv_cpu_workers is not None
        and args.appearance_uv_cpu_workers <= 0
    ):
        parser.error("--appearance-uv-cpu-workers must be positive")
    if (
        args.appearance_projection_num_views <= 0
        or args.appearance_projection_assignment_resolution <= 0
        or args.appearance_projection_padding_pixels < 0
    ):
        parser.error("projection counts must be positive and padding non-negative")
    if not 0 <= args.appearance_projection_preferred_visibility_ratio <= 1:
        parser.error("projection preferred visibility ratio must be in [0, 1]")
    if args.appearance_projection_raster_instance_batch_size <= 0:
        parser.error("projection raster instance batch size must be positive")
    if args.appearance_topology_initial_target_multiplier < 1:
        parser.error("appearance topology initial target multiplier must be positive")
    if args.appearance_topology_simplify_threshold <= 0:
        parser.error("appearance topology simplify threshold must be positive")
    if args.appearance_topology_chart_area_penalty < 0:
        parser.error("appearance topology chart area penalty must be non-negative")
    if args.appearance_texture_dilation_pixels < 0:
        parser.error("appearance texture dilation pixels must be non-negative")
    if args.batch_scenes is not None:
        if args.manifest is not None or args.scene_id is not None:
            parser.error("--batch-scenes is mutually exclusive with --manifest/--scene-id")
    elif args.manifest is None or args.scene_id is None:
        parser.error("either --batch-scenes or both --manifest and --scene-id are required")
    if args.model_cache is None:
        args.model_cache = args.batch_scenes is not None
    if args.geometry_stage_mesh_decode is None:
        # With appearance on, the appearance stage decodes the same latents
        # again with its own remesh + decimation, so the geometry-stage decode
        # is duplicated work. Without appearance it is the only mesh producer.
        args.geometry_stage_mesh_decode = not args.predict_appearance
    elif not args.geometry_stage_mesh_decode and not args.predict_appearance:
        parser.error(
            "--no-geometry-stage-mesh-decode requires --predict-appearance; "
            "nothing else produces the object meshes"
        )
    return args


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def timed_network_call(device: torch.device, function):
    """Return ``(result, seconds)`` for queued device work in one network call."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end) / 1000.0)
    started = time.perf_counter()
    result = function()
    return result, float(time.perf_counter() - started)


def synchronized_timestamp(device: torch.device) -> float:
    """Return a wall-clock timestamp after queued device work is complete."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _quat_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> list[float]:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _room_box_from_record(record: dict[str, Any]) -> RoomBox:
    return RoomBox(
        center=np.asarray(record["center"], dtype=np.float64),
        rotation=np.asarray(record["rotation_box_to_scene"], dtype=np.float64),
        half_extents=np.asarray(record["half_extents"], dtype=np.float64),
        yaw_degrees=float(record["yaw_degrees"]),
        trim_quantile=float(record["trim_quantile"]),
    )


def apply_background_canonical_transform(
    *,
    mode: str,
    margin: float,
    all_points: np.ndarray,
    local_ids: np.ndarray,
    scene_to_object: list[np.ndarray],
    normalized_obbs: list[dict[str, Any]],
    normalization: np.ndarray,
    background_prior: dict[str, Any],
) -> None:
    """Apply a fitted canonical transform to only the room background row."""
    if mode == "perception_obb":
        return
    if mode != "room_box_isotropic_enclose":
        raise ValueError(f"Unsupported background canonical transform: {mode}")

    background_id = int(background_prior["background_local_instance_id"])
    mask = np.asarray(local_ids) == background_id
    points = np.asarray(all_points, dtype=np.float64)[mask]
    if not len(points):
        raise ValueError("Cannot fit a canonical background transform without points")

    previous = np.asarray(scene_to_object[background_id], dtype=np.float64)
    previous_canonical = (
        np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
        )
        @ previous.T
    )[:, :3]
    fitted, audit = fit_room_box_isotropic_canonical_transform(
        points,
        _room_box_from_record(background_prior["box"]),
        margin=float(margin),
    )
    object_to_scene = np.linalg.inv(np.asarray(fitted, dtype=np.float64))
    object_to_world = (
        np.linalg.inv(np.asarray(normalization, dtype=np.float64))
        @ object_to_scene
    )
    scale = float(audit["isotropic_scale"])

    audit.update(
        {
            "applied": True,
            "background_local_instance_id": background_id,
            "previous_scene_to_object": previous.tolist(),
            "previous_canonical_bounds": [
                previous_canonical.min(axis=0).tolist(),
                previous_canonical.max(axis=0).tolist(),
            ],
            "previous_num_outside_unit_box": int(
                np.count_nonzero(
                    np.any(np.abs(previous_canonical) > 0.5, axis=1)
                )
            ),
            "object_to_world": object_to_world.tolist(),
        }
    )
    background_prior["canonical_transform"] = audit
    scene_to_object[background_id] = fitted

    # Keep normalized metadata consistent for downstream audits. The retained
    # raw detector OBB remains untouched as provenance; world placement reads
    # the explicit override recorded above.
    normalized_obbs[background_id] = {
        **normalized_obbs[background_id],
        "translate": list(audit["center"]),
        "rotation": _matrix_to_quat_wxyz(
            np.asarray(audit["rotation_box_to_scene"], dtype=np.float64)
        ),
        "scale": scale,
        "canonical_transform_source": mode,
    }


def _repeat_instance_grid(
    labels: np.ndarray,
    *,
    frames: int,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    expected = frames * source_height * source_width
    if labels.size != expected:
        raise ValueError(
            f"Perception label count {labels.size} != expected grid {expected} "
            f"({frames}x{source_height}x{source_width})"
        )
    if target_height % source_height or target_width % source_width:
        raise ValueError(
            f"Cannot integer-upsample perception grid {source_height}x{source_width} "
            f"to {target_height}x{target_width}"
        )
    labels = labels.reshape(frames, source_height, source_width)
    labels = np.repeat(labels, target_height // source_height, axis=1)
    labels = np.repeat(labels, target_width // source_width, axis=2)
    return labels[:, :target_height, :target_width].reshape(-1)


def apply_perception_conditioning(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    """Replace oracle instance IDs/poses with one saved perception prediction."""
    result_dir = args.perception_result_dir
    if result_dir is None:
        return data
    labels_path = result_dir / "point_instance_masks.pkl"
    raw_obbs_path = result_dir / "raw_oriented_bboxes.json"
    if not labels_path.is_file() or not raw_obbs_path.is_file():
        raise FileNotFoundError(
            f"Perception result needs {labels_path.name} and {raw_obbs_path.name}: {result_dir}"
        )
    with labels_path.open("rb") as handle:
        label_payload = pickle.load(handle)
    perception_labels = np.asarray(label_payload["pred"], dtype=np.int64).reshape(-1)
    raw_payload = json.loads(raw_obbs_path.read_text(encoding="utf-8"))
    raw_obbs = sorted(raw_payload["objects"], key=lambda row: int(row["index"]))

    frames = int(data["num_frames"])
    original_height = int(data["image_height"] * 2)
    original_width = int(data["image_width"] * 2)
    source_height = original_height // 16
    source_width = original_width // 16
    target_height = int(data["image_height"] // data["feature_stride"])
    target_width = int(data["image_width"] // data["feature_stride"])
    transfer_mode = str(args.perception_label_transfer)
    transfer_audit = None
    source_frame_translation_residual = None
    if transfer_mode == "repeat":
        # eval_perception uses original-image stride 16. The flow grid is
        # original-image stride 8 for up4, so the compatibility path repeats 2x2.
        flow_labels = _repeat_instance_grid(
            perception_labels,
            frames=frames,
            source_height=source_height,
            source_width=source_width,
            target_height=target_height,
            target_width=target_width,
        )
        perception_grid = {
            "label_transfer": transfer_mode,
            "source_stride": 16,
            "source_shape": [frames, source_height, source_width],
            "target_shape": [frames, target_height, target_width],
            "upsample": [
                target_height // source_height,
                target_width // source_width,
            ],
        }
    else:
        (
            source_model_points,
            source_labels,
            raw_to_model,
            source_frame_translation_residual,
        ) = load_perception_voxel_source(result_dir)
        if not np.array_equal(source_labels, perception_labels):
            raise ValueError("Perception PLY source labels disagree with label payload")
        normalization_translation = np.asarray(
            data["norm_transform"], dtype=np.float64
        )[:3, 3]
        dense_raw_points = (
            np.asarray(data["all_points"], dtype=np.float64)
            - normalization_translation
        )
        # raw -> perception model frame is rigid, not necessarily a plain
        # shift: single_image can optionally canonicalise room yaw, so this
        # carries a rotation whenever that loader option chooses a nonzero yaw.
        dense_model_points = (
            dense_raw_points @ np.asarray(raw_to_model, dtype=np.float64)[:3, :3].T
            + np.asarray(raw_to_model, dtype=np.float64)[:3, 3]
        )
        flow_labels, transfer_audit = transfer_voxel_labels(
            source_model_points,
            source_labels,
            dense_model_points,
            scene_scale=float(args.perception_scene_scale),
            resolution=int(args.perception_voxel_resolution),
        )
        perception_grid = {
            "label_transfer": transfer_mode,
            "source_stride": 16,
            "source_shape": [frames, source_height, source_width],
            "target_shape": [frames, target_height, target_width],
            "voxel_resolution": int(args.perception_voxel_resolution),
            "scene_scale": float(args.perception_scene_scale),
            "voxel_size": float(args.perception_scene_scale)
            / int(args.perception_voxel_resolution),
            "source_frame_translation_residual": (
                source_frame_translation_residual
            ),
            "transfer_audit": transfer_audit,
        }
    if flow_labels.size != data["all_points"].shape[0]:
        raise ValueError(
            f"Upsampled perception labels {flow_labels.size} != flow points "
            f"{data['all_points'].shape[0]}"
        )

    normalization = np.asarray(data["norm_transform"], dtype=np.float64)
    linear = normalization[:3, :3]
    scale_factor = float(np.mean(np.linalg.norm(linear, axis=0)))
    normalized_obbs = []
    retained_raw_obbs = []
    local_ids = np.full(flow_labels.shape, -1, dtype=np.int64)
    scene_to_object = []
    object_ids = []
    object_names = []
    point_counts = []
    for raw_obb in raw_obbs:
        prediction_index = int(raw_obb["index"])
        mask = flow_labels == prediction_index
        point_count = int(np.count_nonzero(mask))
        if point_count == 0:
            continue
        local_id = len(object_ids)
        local_ids[mask] = local_id
        raw_center = np.asarray(raw_obb["translate"], dtype=np.float64)
        center = (normalization @ np.r_[raw_center, 1.0])[:3]
        raw_rotation = _quat_wxyz_to_matrix(raw_obb.get("rotation", [1, 0, 0, 0]))
        normalized_rotation = linear @ raw_rotation
        u, _, vt = np.linalg.svd(normalized_rotation)
        normalized_rotation = u @ vt
        if np.linalg.det(normalized_rotation) < 0:
            u[:, -1] *= -1
            normalized_rotation = u @ vt
        raw_scale = np.asarray(raw_obb.get("scale", 1.0), dtype=np.float64).reshape(-1)
        scalar_scale = float(raw_scale[0]) * scale_factor
        object_to_scene = np.eye(4, dtype=np.float64)
        object_to_scene[:3, :3] = normalized_rotation * scalar_scale
        object_to_scene[:3, 3] = center
        scene_to_object.append(np.linalg.inv(object_to_scene).astype(np.float32))
        object_ids.append(prediction_index)
        object_names.append(f"pred_{prediction_index:04d}")
        point_counts.append(point_count)
        normalized_obbs.append(
            {
                **raw_obb,
                "translate": center.tolist(),
                "scale": scalar_scale,
            }
        )
        retained_raw_obbs.append(raw_obb)

    background_room_box_prior = None
    if getattr(args, "background_room_box_prior", False):
        local_ids, background_room_box_prior = (
            filter_background_instance_near_room_box(
                data["all_points"],
                local_ids,
                distance_threshold=float(args.background_room_box_distance),
                trim_quantile=float(args.background_room_box_trim_quantile),
                yaw_samples=int(args.background_room_box_yaw_samples),
                minimum_points=int(args.background_room_box_min_points),
                adaptive_max_distance=float(
                    args.background_room_box_adaptive_max_distance
                ),
            )
        )
        background_local_id = int(
            background_room_box_prior["background_local_instance_id"]
        )
        background_room_box_prior["background_prediction_index"] = int(
            object_ids[background_local_id]
        )
        point_counts = [
            int(np.count_nonzero(local_ids == local_id))
            for local_id in range(len(object_ids))
        ]

        if getattr(args, "prune_oversized_instances", True):
            from utils.oversized_instance_prune import prune_oversized_instances

            local_ids, oversized_audit = prune_oversized_instances(
                local_ids=local_ids,
                object_ids=object_ids,
                object_names=object_names,
                point_counts=point_counts,
                scene_to_object=scene_to_object,
                normalized_obbs=normalized_obbs,
                retained_raw_obbs=retained_raw_obbs,
                background_prior=background_room_box_prior,
                volume_ratio=float(args.oversized_instance_volume_ratio),
            )
            background_room_box_prior["oversized_instance_prune"] = oversized_audit
            if oversized_audit.get("num_pruned"):
                names = [row["object_name"] for row in oversized_audit["pruned"]]
                print(
                    f"[oversized-prune] dropped {names} "
                    f"(> {oversized_audit['volume_ratio_threshold']:.2f}x room "
                    f"volume {oversized_audit['room_volume']:.1f} m^3)",
                    flush=True,
                )

        apply_background_canonical_transform(
            mode=str(
                getattr(args, "background_canonical_transform", "perception_obb")
            ),
            margin=float(getattr(args, "background_canonical_margin", 0.01)),
            all_points=data["all_points"],
            local_ids=local_ids,
            scene_to_object=scene_to_object,
            normalized_obbs=normalized_obbs,
            normalization=normalization,
            background_prior=background_room_box_prior,
        )

    foreground = local_ids >= 0
    if not object_ids or not np.any(foreground):
        raise RuntimeError("Perception produced no reconstructable predicted instances")
    data.update(
        {
            "foreground_mask": foreground,
            "points": data["all_points"][foreground],
            "instance_ids": local_ids[foreground],
            "scene_to_object": np.stack(scene_to_object),
            "object_ids": object_ids,
            "object_names": object_names,
            "point_counts": point_counts,
            "perception_result_dir": str(result_dir),
            "perception_raw_obbs": retained_raw_obbs,
            "perception_normalized_obbs": normalized_obbs,
            "perception_grid": perception_grid,
            "background_room_box_prior": background_room_box_prior,
        }
    )
    return data


def load_manifest_scene(path: Path, scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ff_scene_geometry_manifest_v1":
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')}")
    matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest scene named {scene_id}, found {len(matches)}")
    return manifest, matches[0]


def resolve_camera_pose_override(args: argparse.Namespace) -> Path | None:
    if args.camera_pose_override is not None:
        return args.camera_pose_override.resolve()
    if args.perception_result_dir is None:
        return None
    metadata_path = args.perception_result_dir / "camera_pose_override.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = metadata.get("path")
    if not source_path:
        raise KeyError(f"Camera-pose metadata is missing path: {metadata_path}")
    return Path(source_path).expanduser().resolve()


def apply_camera_pose_override(
    *,
    override_path: Path,
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
    rgbs: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Select registered frames and replace GT cameras with aligned estimates."""
    if not override_path.is_file():
        raise FileNotFoundError(f"Missing camera-pose override: {override_path}")
    with np.load(override_path, allow_pickle=False) as override:
        required = {
            "image_names",
            "frame_indices",
            "gt_c2w",
            "aligned_predicted_c2w",
        }
        missing = sorted(required.difference(override.files))
        if missing:
            raise KeyError(
                f"Camera override {override_path} is missing keys: {missing}"
            )
        image_names = np.asarray(override["image_names"]).astype(str)
        frame_indices = np.asarray(override["frame_indices"], dtype=np.int64)
        expected_gt = np.asarray(override["gt_c2w"], dtype=np.float64)
        aligned_c2ws = np.asarray(
            override["aligned_predicted_c2w"], dtype=np.float64
        )

    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError(
            f"Camera override frame_indices must be non-empty: {override_path}"
        )
    if np.any(frame_indices < 0) or np.any(frame_indices >= len(c2ws)):
        raise IndexError(
            f"Camera override indices outside [0, {len(c2ws)}): {override_path}"
        )
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError(f"Duplicate camera override indices: {override_path}")
    if len(image_names) != len(frame_indices):
        raise ValueError(
            f"Camera override image-name count does not match indices: {override_path}"
        )
    if expected_gt.shape != (len(frame_indices), 4, 4):
        raise ValueError(
            f"Unexpected GT camera shape {expected_gt.shape}: {override_path}"
        )
    if aligned_c2ws.shape != expected_gt.shape:
        raise ValueError(
            f"Unexpected predicted camera shape {aligned_c2ws.shape}: "
            f"{override_path}"
        )

    source_gt = np.asarray(c2ws[frame_indices], dtype=np.float64)
    gt_residual = float(np.max(np.abs(source_gt - expected_gt[:, :3, :])))
    if gt_residual > 1e-5:
        raise ValueError(
            "Camera override GT poses do not match the selected source frames "
            f"(max residual {gt_residual:.3e}): {override_path}"
        )

    metadata = {
        "path": str(override_path),
        "image_names": image_names.tolist(),
        "frame_indices": frame_indices.tolist(),
        "num_source_frames": int(len(c2ws)),
        "num_selected_frames": int(len(frame_indices)),
        "gt_pose_max_residual": gt_residual,
    }
    return (
        np.asarray(intrinsics)[frame_indices],
        aligned_c2ws[:, :3, :],
        np.asarray(rgbs)[frame_indices],
        np.asarray(depths)[frame_indices],
        np.asarray(masks)[frame_indices],
        metadata,
    )


def load_single_image_scene_inputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    scene: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Load one preprocessed single-view scene in the ground-aligned frame.

    Reads the same `data/<scene>/aligned_pcd.ply` grid that
    `utils.data_single_image` feeds to perception, so both stages share one
    coordinate frame. The cloud is the organised half-resolution grid; invalid
    samples are stored as NaN and are filled from their nearest valid neighbour
    here, matching how the multi-view path fills invalid depth.

    The release has no per-frame GT instance masks and no per-object transform
    records, so oracle conditioning is impossible and
    ``--perception-result-dir`` is required. Every object field returned is
    empty; ``apply_perception_conditioning`` fills them from the saved
    perception prediction, as it does for the multi-view datasets.
    """
    import cv2
    import trimesh
    from scipy.interpolate import griddata
    from utils.data_single_image import single_image_root

    if args.perception_result_dir is None:
        raise ValueError(
            "single_image scenes have no GT instance masks or object transforms; "
            "--perception-result-dir is required for this dataset"
        )

    # Perception resolves this dataset through FF_SINGLE_IMAGE_ROOT. Use that
    # same loader root here so a protocol root override cannot split the two
    # stages across different copies of the dataset.
    dataset_root = Path(single_image_root())
    scene_dir = dataset_root / "data" / scene.get("scene_dir", scene["scene_id"])
    rgb_path = scene_dir / "rgb.jpeg"
    pcd_path = scene_dir / "aligned_pcd.ply"
    for path in (rgb_path, pcd_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; rebuild it with "
                "the published single-image inference archive"
            )

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    full_height, full_width = rgb.shape[:2]
    grid_height, grid_width = full_height // 2, full_width // 2

    cloud = trimesh.load(pcd_path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    if points.shape[0] != grid_height * grid_width:
        raise ValueError(
            f"{pcd_path} has {points.shape[0]} points, expected "
            f"{grid_height * grid_width} for rgb {full_height}x{full_width}"
        )
    points = points.reshape(grid_height, grid_width, 3)

    invalid = ~np.isfinite(points).all(axis=-1)
    if invalid.any():
        if invalid.all():
            raise ValueError(f"{pcd_path} contains no finite points")
        ys, xs = np.nonzero(~invalid)
        ty, tx = np.nonzero(invalid)
        points[ty, tx] = griddata(
            np.column_stack([ys, xs]),
            points[ys, xs],
            np.column_stack([ty, tx]),
            method="nearest",
        )

    rgb_grid = cv2.resize(rgb, (grid_width, grid_height), interpolation=cv2.INTER_AREA)

    # DINO tokenises on a 16-pixel patch grid, so the flow grid only matches the
    # sampled point grid when both sides are multiples of 16. 484x648 is not, so
    # centre-crop to the largest valid region rather than resampling.
    cropped_height = (grid_height // 16) * 16
    cropped_width = (grid_width // 16) * 16
    top = (grid_height - cropped_height) // 2
    left = (grid_width - cropped_width) // 2
    points = points[top : top + cropped_height, left : left + cropped_width]
    rgb_grid = rgb_grid[top : top + cropped_height, left : left + cropped_width]

    feature_stride = 16 // int(args.dino_upsample)
    sampled = points[::feature_stride, ::feature_stride].reshape(-1, 3)
    sampled, norm_transform = point_normalize(np.asarray(sampled, dtype=np.float32))
    sampled = np.asarray(sampled, dtype=np.float32)
    norm_transform = np.asarray(norm_transform, dtype=np.float64)

    expected_features = int(
        (cropped_height // feature_stride) * (cropped_width // feature_stride)
    )
    if expected_features != sampled.shape[0]:
        raise ValueError(
            f"Sampled point count {sampled.shape[0]} does not match expected DINO "
            f"grid {expected_features}"
        )

    return {
        "manifest": manifest,
        "scene": scene,
        "dataset_root": dataset_root,
        "rgbs": rgb_grid[None, ...].astype(np.float32),
        "all_points": sampled,
        "foreground_mask": np.zeros(sampled.shape[0], dtype=bool),
        "points": sampled[:0],
        "instance_ids": np.zeros((0,), dtype=np.int64),
        "scene_to_object": np.zeros((0, 4, 4), dtype=np.float32),
        "object_ids": [],
        "object_names": [],
        "point_counts": [],
        "feature_stride": feature_stride,
        "image_height": int(cropped_height),
        "image_width": int(cropped_width),
        "num_frames": 1,
        "norm_transform": norm_transform,
        "camera_pose_override": None,
    }


def load_scannetpp_scene_inputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    scene: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Load one ScanNet++ scene in its configured inference frame.

    Reuses `utils.data_scannetpp` for frame selection and for the wall estimate,
    yaw and rotation centre, so reconstruction lands in exactly the frame
    perception saw. With wall alignment disabled this shared transform is the
    identity. Frames are halved and centre-cropped to a multiple of 16 so the
    DINO patch grid matches the sampled point grid.

    The dense grid is not wall-pruned: voxel_transfer assigns labels from the
    pruned perception cloud, so cells beyond the walls simply receive no label.
    A protocol may still drop invalid or confidence-rejected depth samples. In
    that case a full-grid keep mask is retained and applied to DINO features
    after extraction so point/feature rows stay aligned.

    ScanNet++ has no GT instance masks or object transforms here, so
    --perception-result-dir is required and every object field is left empty
    for apply_perception_conditioning() to fill.
    """
    from utils.data_scannetpp import (
        depth_confidence_threshold,
        depth_invalid_policy,
        load_scannetpp_data,
        load_scene_frames,
        rotate_points,
        scene_wall_alignment,
    )

    if args.perception_result_dir is None:
        raise ValueError(
            "ScanNet++ scenes have no GT instance masks or object transforms; "
            "--perception-result-dir is required for this dataset"
        )

    scene_id = scene["scene_id"]
    entry = next(
        (d for d in load_scannetpp_data() if d["scene_id"] == scene_id), None
    )
    if entry is None:
        raise ValueError(f"ScanNet++ scene {scene_id} is not loadable")

    rgbs, depths, intrinsics, c2ws, _ = load_scene_frames(entry)
    _, rotation_center, rotation_angle, rotation_diag = scene_wall_alignment(
        rgbs, depths, intrinsics, c2ws
    )

    masks = np.zeros(depths.shape, dtype=np.int32)
    rgbs, depths, masks, intrinsics, height, width = downsample_all(
        rgbs, depths, masks, intrinsics, depths.shape[1], depths.shape[2], factor=2
    )

    cropped_height = (height // 16) * 16
    cropped_width = (width // 16) * 16
    if cropped_height <= 0 or cropped_width <= 0:
        raise ValueError(f"ScanNet++ frame {height}x{width} is too small to crop")
    if (cropped_height, cropped_width) != (height, width):
        top = (height - cropped_height) // 2
        left = (width - cropped_width) // 2
        rgbs = rgbs[:, top : top + cropped_height, left : left + cropped_width, :]
        depths = depths[:, top : top + cropped_height, left : left + cropped_width]
        masks = masks[:, top : top + cropped_height, left : left + cropped_width]
        intrinsics = intrinsics.copy()
        intrinsics[:, 0, 2] -= left
        intrinsics[:, 1, 2] -= top
        height, width = cropped_height, cropped_width

    feature_stride = 16 // int(args.dino_upsample)
    invalid_policy = depth_invalid_policy()
    points, _, _, _, grid_keep_mask = (
        project_depth_to_world_patch_geometry_with_instance_mask(
            depths,
            intrinsics,
            c2ws,
            masks,
            downsample=feature_stride,
            invalid_policy=invalid_policy,
            return_grid_keep_mask=True,
        )
    )
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    grid_keep_mask = np.asarray(grid_keep_mask, dtype=bool).reshape(-1)
    expected_features = int(
        len(rgbs) * (height // feature_stride) * (width // feature_stride)
    )
    if expected_features != grid_keep_mask.size:
        raise ValueError(
            f"Depth validity grid {grid_keep_mask.size} does not match expected "
            f"DINO grid {expected_features}"
        )
    points = points[grid_keep_mask]
    if points.shape[0] == 0:
        raise ValueError("No valid ScanNet++ depth samples remain after filtering")
    points = rotate_points(points, center=rotation_center, angle=rotation_angle)
    points, norm_transform = point_normalize(np.asarray(points, dtype=np.float32))
    points = np.asarray(points, dtype=np.float32)
    norm_transform = np.asarray(norm_transform, dtype=np.float64)

    if int(grid_keep_mask.sum()) != points.shape[0]:
        raise ValueError(
            f"Projected point count {points.shape[0]} does not match retained "
            f"depth grid {int(grid_keep_mask.sum())}"
        )

    return {
        "manifest": manifest,
        "scene": scene,
        "dataset_root": dataset_root,
        "rgbs": np.asarray(rgbs, dtype=np.float32),
        "all_points": points,
        "foreground_mask": np.zeros(points.shape[0], dtype=bool),
        "points": points[:0],
        "instance_ids": np.zeros((0,), dtype=np.int64),
        "scene_to_object": np.zeros((0, 4, 4), dtype=np.float32),
        "object_ids": [],
        "object_names": [],
        "point_counts": [],
        "feature_stride": feature_stride,
        "image_height": int(height),
        "image_width": int(width),
        "num_frames": int(len(rgbs)),
        "norm_transform": norm_transform,
        "camera_pose_override": None,
        "scannetpp_rotation": rotation_diag,
        "dino_source_grid_points": expected_features,
        "dino_grid_keep_mask": grid_keep_mask,
        "depth_filter": {
            "invalid_policy": invalid_policy,
            "confidence_threshold": depth_confidence_threshold(),
            "source_grid_points": expected_features,
            "retained_grid_points": int(grid_keep_mask.sum()),
        },
    }


def resolve_scene_rgb_frames_dir(
    manifest: dict[str, Any],
    scene: dict[str, Any],
    dataset_root: Path,
) -> Path:
    original = dataset_root / scene["frames_dir"]
    if manifest.get("dataset") not in {"ithor", "imaginarium"}:
        return original
    return Path(
        resolve_frames_dir(
            original,
            dataset_subdir=str(manifest["dataset_subdir"]),
            scene_id=str(scene["scene_id"]),
            video_id=int(scene.get("video_id", 0)),
        )
    )


def load_scene_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest, scene = load_manifest_scene(args.manifest, args.scene_id)
    dataset_root = args.fire3d_test_root / manifest["dataset_subdir"]
    if manifest.get("dataset") == "single_image":
        return load_single_image_scene_inputs(args, manifest, scene, dataset_root)
    if manifest.get("dataset") == "scannetpp":
        return load_scannetpp_scene_inputs(args, manifest, scene, dataset_root)

    intrinsics, c2ws, (height, width) = read_cameras(str(dataset_root / scene["camera_path"]))
    rgb_frames_dir = resolve_scene_rgb_frames_dir(manifest, scene, dataset_root)
    rgbs = read_rgbs(
        str(rgb_frames_dir),
        height,
        width,
        parallel=True,
        max_workers=args.read_workers,
    )
    depths = read_depths(
        str(dataset_root / scene["depths_dir"]),
        height,
        width,
        parallel=True,
        max_workers=args.read_workers,
    )
    masks, _ = read_masks_v2(
        str(dataset_root / scene["masks_dir"]),
        height,
        width,
        parallel=True,
        max_workers=args.read_workers,
    )
    camera_pose_override = None
    override_path = resolve_camera_pose_override(args)
    if override_path is not None:
        (
            intrinsics,
            c2ws,
            rgbs,
            depths,
            masks,
            camera_pose_override,
        ) = apply_camera_pose_override(
            override_path=override_path,
            intrinsics=intrinsics,
            c2ws=c2ws,
            rgbs=rgbs,
            depths=depths,
            masks=masks,
        )
    rgbs, depths, masks, intrinsics, height, width = downsample_all(
        rgbs, depths, masks, intrinsics, height, width, factor=2
    )

    feature_stride = 16 // int(args.dino_upsample)
    points, _, _, raw_instance_ids = project_depth_to_world_patch_geometry_with_instance_mask(
        depths,
        intrinsics,
        c2ws,
        masks,
        downsample=feature_stride,
    )
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    raw_instance_ids = np.asarray(raw_instance_ids, dtype=np.int64).reshape(-1)
    points, norm_transform = point_normalize(points)
    points = np.asarray(points, dtype=np.float32)
    norm_transform = np.asarray(norm_transform, dtype=np.float64)

    object_ids = [int(value) for value in scene["visible_object_ids"]]
    object_names = [f"object_{object_id:04d}" for object_id in object_ids]
    with (dataset_root / scene["transforms_path"]).open("rb") as handle:
        transform_records = pickle.load(handle)
    missing = [name for name in object_names if name not in transform_records]
    if missing:
        raise KeyError(f"Scene transforms are missing visible objects: {missing}")

    # GT mode used to drop the room entirely. Not because the pose is missing:
    # every scene's transforms pickle carries a `layout_<scene>` record with
    # scale/angles/trans, and instance id 0 is the largest label in the GT masks
    # (7.5 M pixels on bedroom_01). It was dropped only because
    # `visible_object_ids` starts at 1, so the room was never named. Prepend it
    # as instance 0 so GT conditioning reconstructs the same set as the
    # predicted path, where the room-box prior supplies the background.
    background_name = None
    background_source = None
    if getattr(args, "gt_background_instance", True) and 0 not in object_ids:
        layout_keys = [key for key in transform_records if key.startswith("layout")]
        if layout_keys:
            background_name = layout_keys[0]
            background_source = "gt_layout_record"
        else:
            # No layout record: synthesise the pose the user specified -- zero
            # rotation, centred in the room, scale = the largest xyz extent.
            raw_points = np.asarray(points, dtype=np.float64)
            lower = raw_points.min(axis=0)
            upper = raw_points.max(axis=0)
            background_name = f"layout_{scene['scene_id']}"
            transform_records[background_name] = {
                "scale": float(np.max(upper - lower)),
                "angles": [0.0, 0.0, 0.0],
                "trans": (0.5 * (lower + upper)).tolist(),
                "mesh_name": None,
            }
            background_source = "synthesised_axis_aligned_room_box"
        object_ids = [0] + object_ids
        object_names = [background_name] + object_names
        print(
            f"[gt] background instance 0 from {background_source} "
            f"({background_name})",
            flush=True,
        )

    scales = np.asarray([transform_records[name]["scale"] for name in object_names], dtype=np.float64)
    angles = np.asarray([transform_records[name]["angles"] for name in object_names], dtype=np.float64)
    translations = np.asarray([transform_records[name]["trans"] for name in object_names], dtype=np.float64)
    new_scales, new_angles, new_translations = transform_6d_from_transform_batch(
        scales,
        angles,
        translations,
        norm_transform,
    )
    d_scales, d_angles, d_translations = discrete_transform_batch(
        new_scales,
        new_angles,
        new_translations,
    )
    c_scales, c_angles, c_translations = continue_transform_batch(
        d_scales,
        d_angles,
        d_translations,
    )
    object_to_scene = get_transform_matrix_batch(c_scales, c_angles, c_translations)
    scene_to_object = np.linalg.inv(object_to_scene).astype(np.float32)

    max_raw_id = max(int(raw_instance_ids.max(initial=0)), max(object_ids, default=0))
    id_map = np.full((max_raw_id + 1,), -1, dtype=np.int64)
    for local_id, raw_id in enumerate(object_ids):
        id_map[raw_id] = local_id
    in_range = (raw_instance_ids >= 0) & (raw_instance_ids < id_map.shape[0])
    local_instance_ids = np.full_like(raw_instance_ids, -1)
    local_instance_ids[in_range] = id_map[raw_instance_ids[in_range]]
    foreground = local_instance_ids >= 0

    expected_features = int(len(rgbs) * (height // feature_stride) * (width // feature_stride))
    if expected_features != points.shape[0]:
        raise ValueError(
            f"Projected point count {points.shape[0]} does not match expected DINO grid "
            f"{expected_features} ({len(rgbs)}x{height // feature_stride}x{width // feature_stride})"
        )
    point_counts = [int(np.count_nonzero(local_instance_ids == index)) for index in range(len(object_ids))]
    if not np.any(foreground):
        raise RuntimeError("No GT-visible object has a point on the DINO feature grid")

    return {
        "manifest": manifest,
        "scene": scene,
        "dataset_root": dataset_root,
        "rgb_frames_dir": str(rgb_frames_dir),
        "rgbs": np.asarray(rgbs, dtype=np.float32),
        "all_points": points,
        "foreground_mask": foreground,
        "points": points[foreground],
        "instance_ids": local_instance_ids[foreground],
        "scene_to_object": scene_to_object,
        "object_ids": object_ids,
        "object_names": object_names,
        "point_counts": point_counts,
        "feature_stride": feature_stride,
        "image_height": int(height),
        "image_width": int(width),
        "num_frames": int(len(rgbs)),
        "norm_transform": norm_transform,
        "camera_pose_override": camera_pose_override,
    }


def filter_dino_features_to_projected_depth_grid(
    all_features: torch.Tensor, data: dict[str, Any]
) -> torch.Tensor:
    """Drop DINO rows whose matching depth sample was rejected."""

    keep = data.get("dino_grid_keep_mask")
    if keep is None:
        return all_features
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    if all_features.shape[0] != keep.size:
        raise ValueError(
            f"DINO feature count {all_features.shape[0]} != depth validity grid "
            f"{keep.size}"
        )
    keep_tensor = torch.from_numpy(keep).to(device=all_features.device)
    return all_features[keep_tensor]


def input_audit(data: dict[str, Any]) -> dict[str, Any]:
    counts = np.asarray(data["point_counts"], dtype=np.int64)
    source_grid_points = int(
        data.get("dino_source_grid_points", data["all_points"].shape[0])
    )
    return {
        "schema": SCHEMA,
        "dataset": data["manifest"]["dataset"],
        "scene_id": data["scene"]["scene_id"],
        "rgb_frames_dir": data.get("rgb_frames_dir"),
        "num_frames": data["num_frames"],
        "image_height": data["image_height"],
        "image_width": data["image_width"],
        "feature_stride": data["feature_stride"],
        "num_dino_grid_points": int(data["all_points"].shape[0]),
        "num_dino_source_grid_points": source_grid_points,
        "num_depth_rejected_grid_points": (
            source_grid_points - int(data["all_points"].shape[0])
        ),
        "num_visible_objects": len(data["object_ids"]),
        "num_foreground_grid_points": int(data["points"].shape[0]),
        "num_objects_with_zero_grid_points": int(np.count_nonzero(counts == 0)),
        "min_grid_points_per_object": int(counts.min()) if counts.size else 0,
        "median_grid_points_per_object": float(np.median(counts)) if counts.size else 0.0,
        "max_grid_points_per_object": int(counts.max()) if counts.size else 0,
        "objects": [
            {"object_id": object_id, "num_grid_points": count}
            for object_id, count in zip(data["object_ids"], data["point_counts"])
        ],
        "camera_pose_override": data.get("camera_pose_override"),
        "perception_grid": data.get("perception_grid"),
        "depth_filter": data.get("depth_filter"),
        "background_room_box_prior": data.get("background_room_box_prior"),
    }


def resolve_ss_shape_object_batch_size(args: argparse.Namespace) -> int:
    return int(args.ss_shape_object_batch_size or args.object_batch_size)


def resolve_pbr_object_batch_size(args: argparse.Namespace) -> int:
    return int(args.pbr_object_batch_size or args.object_batch_size)


def inference_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        "scope": (
            "perception_to_geometry_predicted_id_and_pose"
            if args.perception_result_dir is not None
            else "geometry_only_oracle_id_and_pose"
        ),
        "dino_upsample": args.dino_upsample,
        "max_cond_len": args.max_cond_len,
        "inference_num_steps": args.inference_num_steps,
        "sample_method": args.sample_method,
        "pbr_sample_method": args.pbr_sample_method or args.sample_method,
        "pbr_max_cond_len": resolve_pbr_max_cond_len(args),
        "pbr_guidance_strength": (
            3.0 if args.pbr_guidance_strength is None
            else float(args.pbr_guidance_strength)
        ),
        "pbr_condition_snap": bool(args.pbr_condition_snap),
        "pbr_condition_snap_max_distance": float(args.pbr_condition_snap_max_distance),
        "pbr_condition_snap_fraction": float(args.pbr_condition_snap_fraction),
        "occupancy_threshold": args.occupancy_threshold,
        "empty_occupancy_policy": args.empty_occupancy_policy,
        "object_batch_size": args.object_batch_size,
        "ss_shape_object_batch_size": resolve_ss_shape_object_batch_size(args),
        "pbr_object_batch_size": resolve_pbr_object_batch_size(args),
        "appearance_decode_object_chunk_size": (
            args.appearance_decode_object_chunk_size
        ),
        "appearance_mesh_batch_size": args.appearance_mesh_batch_size,
        "appearance_uv_cpu_workers": args.appearance_uv_cpu_workers,
        "appearance_detailed_profile": bool(args.appearance_detailed_profile),
        "normal_flow_weights": not args.ss_use_ema and not args.shape_use_ema,
        "ss_model_family": args.ss_model_family,
        "shape_model_family": args.shape_model_family,
        "perception_label_transfer": (
            args.perception_label_transfer
            if args.perception_result_dir is not None
            else None
        ),
        "perception_voxel_resolution": (
            args.perception_voxel_resolution
            if args.perception_result_dir is not None
            and args.perception_label_transfer == "voxel_transfer"
            else None
        ),
        "perception_scene_scale": (
            args.perception_scene_scale
            if args.perception_result_dir is not None
            and args.perception_label_transfer == "voxel_transfer"
            else None
        ),
        "background_room_box_prior": bool(args.background_room_box_prior),
        "background_unit_box_prune": bool(args.background_unit_box_prune),
        "background_canonical_transform": getattr(
            args, "background_canonical_transform", "perception_obb"
        ),
        "background_canonical_margin": float(
            getattr(args, "background_canonical_margin", 0.01)
        ),
        "object_selection": getattr(args, "object_selection", "all"),
        "background_room_box_distance": (
            args.background_room_box_distance
            if args.background_room_box_prior
            else None
        ),
        "background_room_box_adaptive_max_distance": (
            args.background_room_box_adaptive_max_distance
            if args.background_room_box_prior
            else None
        ),
        "background_room_box_trim_quantile": (
            args.background_room_box_trim_quantile
            if args.background_room_box_prior
            else None
        ),
        "background_room_box_yaw_samples": (
            args.background_room_box_yaw_samples
            if args.background_room_box_prior
            else None
        ),
        "background_room_box_min_points": (
            args.background_room_box_min_points
            if args.background_room_box_prior
            else None
        ),
    }
    if args.predict_appearance:
        settings.update(
            {
                "scope": (
                    "perception_to_geometry_and_appearance_predicted_id_and_pose"
                    if args.perception_result_dir is not None
                    else "geometry_and_appearance_oracle_id_and_pose"
                ),
                "predict_appearance": True,
                "pbr_normal_flow_weights": not args.pbr_use_ema,
                "pbr_model_family": args.pbr_model_family,
                "appearance_mesh_backend": args.appearance_mesh_backend,
                "appearance_projection_num_views": (
                    args.appearance_projection_num_views
                    if args.appearance_mesh_backend == "projection"
                    else None
                ),
                "appearance_projection_assignment_resolution": (
                    args.appearance_projection_assignment_resolution
                    if args.appearance_mesh_backend == "projection"
                    else None
                ),
                "appearance_projection_padding_pixels": (
                    args.appearance_projection_padding_pixels
                    if args.appearance_mesh_backend == "projection"
                    else None
                ),
                "appearance_projection_view_assignment_mode": (
                    args.appearance_projection_view_assignment_mode
                    if args.appearance_mesh_backend == "projection"
                    else None
                ),
                "appearance_projection_preferred_visibility_ratio": (
                    args.appearance_projection_preferred_visibility_ratio
                    if args.appearance_mesh_backend == "projection"
                    else None
                ),
                "appearance_topology_initial_target_multiplier": (
                    args.appearance_topology_initial_target_multiplier
                ),
                "appearance_topology_simplify_threshold": (
                    args.appearance_topology_simplify_threshold
                ),
                "appearance_topology_chart_area_penalty": (
                    args.appearance_topology_chart_area_penalty
                ),
                "appearance_surface_mapping": args.appearance_surface_mapping,
                "appearance_sparse_query_mode": args.appearance_sparse_query_mode,
                "appearance_texture_fill_mode": args.appearance_texture_fill_mode,
                "appearance_texture_erode_iterations": (
                    args.appearance_texture_erode_iterations
                ),
                "appearance_texture_dilation_pixels": (
                    args.appearance_texture_dilation_pixels
                ),
            }
        )
    return settings


def common_tensors(
    data: dict[str, Any],
    point_features: torch.Tensor,
    selected: list[int],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "points": torch.from_numpy(data["points"]).to(device=device, dtype=torch.float32).unsqueeze(0),
        "colors": point_features.unsqueeze(0),
        "instance_ids": torch.from_numpy(data["instance_ids"]).to(device=device, dtype=torch.long).unsqueeze(0),
        "object_transforms": torch.from_numpy(data["scene_to_object"]).to(device=device, dtype=torch.float32).unsqueeze(0),
        "selected_indices": torch.tensor(selected, device=device, dtype=torch.long).unsqueeze(0),
        "max_num_objects": len(data["object_ids"]),
    }


def background_local_instance_id(data: dict[str, Any]) -> int:
    """Resolve the background row used by flow conditioning."""

    prior = data.get("background_room_box_prior") or {}
    if prior.get("background_local_instance_id") is not None:
        local_id = int(prior["background_local_instance_id"])
        if not 0 <= local_id < len(data["object_ids"]):
            raise ValueError(
                f"Background local ID {local_id} is outside "
                f"[0, {len(data['object_ids'])})"
            )
        return local_id

    candidates = [
        local_id
        for local_id, object_id in enumerate(data["object_ids"])
        if int(object_id) == 0
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Background unit-box pruning was disabled, but the scene has no "
            "unique background row from either the room-box prior or object ID 0"
        )
    return candidates[0]


def selected_local_ids_for_run(
    data: dict[str, Any], object_selection: str
) -> list[int]:
    """Resolve the object rows this invocation is responsible for."""
    if object_selection == "all":
        return list(range(len(data["object_ids"])))
    if object_selection == "background":
        return [background_local_instance_id(data)]
    raise ValueError(f"Unsupported object selection: {object_selection}")


def flow_unit_box_prune_passes(
    data: dict[str, Any],
    selected: list[int],
    *,
    background_unit_box_prune: bool,
) -> list[tuple[list[int], bool]]:
    """Plan an unchanged normal pass plus an optional background replacement."""

    passes = [(list(selected), True)]
    if background_unit_box_prune:
        return passes
    background_id = background_local_instance_id(data)
    if background_id in selected:
        passes.append(([background_id], False))
    return passes


def canonical_points_for_object(data: dict[str, Any], local_id: int) -> np.ndarray:
    mask = data["instance_ids"] == local_id
    points = data["points"][mask]
    if points.shape[0] == 0:
        return points
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ data["scene_to_object"][local_id].T)[:, :3].astype(np.float32)


def object_output_dir(args: argparse.Namespace, object_name: str) -> Path:
    return args.output_root / args.scene_id / "debug" / object_name


def mesh_output_path(args: argparse.Namespace, object_name: str) -> Path:
    return args.output_root / args.scene_id / "objects" / f"{object_name}.ply"


def completed_local_ids(args: argparse.Namespace, data: dict[str, Any]) -> set[int]:
    if not args.resume_existing:
        return set()
    completed = set()
    for local_id, name in enumerate(data["object_names"]):
        if not mesh_output_path(args, name).is_file():
            continue
        if args.predict_appearance and not (
            object_output_dir(args, name) / "shape_x2.npz"
        ).is_file():
            continue
        completed.add(local_id)
    return completed


@torch.no_grad()
def predict_chunk(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    selected: list[int],
    point_features: torch.Tensor,
    ss_model: Any,
    shape_model: Any,
    ss_decoder: Any,
    x2_decoder: Any,
    shape_decoder: Any,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    device: torch.device,
    prune_out_of_range: bool = True,
) -> list[dict[str, Any]]:
    selected_count = len(selected)
    selected_names = [data["object_names"][index] for index in selected]
    model_conditioning_seed = stable_seed(
        args.seed,
        (
            f"{data['manifest']['dataset']}:{data['scene']['scene_id']}:"
            f"{','.join(selected_names)}:conditioning-model-order-v1"
        ),
    )
    common = common_tensors(data, point_features, selected, device)
    ss_placeholder = torch.zeros(
        (selected_count, 8, 2, 2, 2), device=device, dtype=torch.float32
    )
    empty_coords = torch.empty((0, 4), device=device, dtype=torch.int32)
    # FPS uses a random start point. Reset the same stable seed before both
    # stages so SS and shape consume an identical per-object token ordering.
    seed_everything(model_conditioning_seed)
    def infer_ss():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            return call_model_inference(
                ss_model,
                **common,
                object_feats=ss_placeholder,
                object_coords=empty_coords,
                return_denoised_latents=True,
                sample_method=args.sample_method,
                inference_num_steps=args.inference_num_steps,
                prune_out_of_range=prune_out_of_range,
            )

    ss_result, ss_flow_seconds = timed_network_call(device, infer_ss)
    ss_latents = ss_result["feat_preds"].float()
    def decode_ss():
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            return ss_decoder(ss_latents).float()

    occupancy_logits, ss_vae_seconds = timed_network_call(device, decode_ss)
    expected = (selected_count, 1, 8, 8, 8)
    if tuple(occupancy_logits.shape) != expected:
        raise ValueError(f"Expected SS decoder logits {expected}, got {tuple(occupancy_logits.shape)}")
    occupancy_probs = torch.sigmoid(occupancy_logits)[:, 0]

    sparse_parts = []
    occupied_xyz: list[torch.Tensor] = []
    for chunk_id in range(selected_count):
        occupied = occupancy_probs[chunk_id] >= float(args.occupancy_threshold)
        if not bool(occupied.any()) and args.empty_occupancy_policy == "argmax":
            occupied.reshape(-1)[int(torch.argmax(occupancy_probs[chunk_id]).item())] = True
        xyz = torch.nonzero(occupied, as_tuple=False).to(dtype=torch.int32)
        occupied_xyz.append(xyz)
        if xyz.shape[0]:
            sparse_parts.append(
                torch.cat(
                    [
                        torch.full((xyz.shape[0], 1), chunk_id, device=device, dtype=torch.int32),
                        xyz,
                    ],
                    dim=1,
                )
            )
    if any(xyz.shape[0] == 0 for xyz in occupied_xyz):
        empty = [selected[index] for index, xyz in enumerate(occupied_xyz) if xyz.shape[0] == 0]
        raise RuntimeError(f"SS decoder produced empty occupancy for local objects {empty}")
    sparse_coords = torch.cat(sparse_parts, dim=0)
    shape_placeholder = torch.zeros(
        (sparse_coords.shape[0], int(shape_model.shape_x2_channels)),
        device=device,
        dtype=torch.float32,
    )
    seed_everything(model_conditioning_seed)
    def infer_shape():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            return call_model_inference(
                shape_model,
                **common,
                object_feats=shape_placeholder,
                object_coords=sparse_coords,
                return_denoised_latents=True,
                sample_method=args.sample_method,
                inference_num_steps=args.inference_num_steps,
                prune_out_of_range=prune_out_of_range,
            )

    shape_result, shape_flow_seconds = timed_network_call(device, infer_shape)
    pred_norm = shape_result["feat_preds"].float()
    if getattr(shape_model, "normalize_shape_x2", True):
        pred_raw = pred_norm * shape_model.x2_std.float() + shape_model.x2_mean.float()
    else:
        pred_raw = pred_norm
    decoder_timing: dict[str, float] = {}
    if not args.geometry_stage_mesh_decode:
        # The appearance stage decodes these latents again and exports the
        # postprocessed mesh; skip the duplicate decode/remesh/decimation here.
        meshes: list[Any] = [None] * selected_count
    else:
        meshes = decode_x2_object_meshes(
            x2_decoder=x2_decoder,
            input_mean=input_mean,
            input_std=input_std,
            shape_decoder=shape_decoder,
            feats_raw=pred_raw,
            coords=sparse_coords,
            object_count=selected_count,
            mesh_decode_batch_size=args.mesh_decode_batch_size,
            shape_decode_resolution=args.shape_decode_resolution,
            decimation_target=args.shape_mesh_decimation_target,
            remesh=args.shape_mesh_remesh,
            remesh_band=args.shape_mesh_remesh_band,
            remesh_project=args.shape_mesh_remesh_project,
            timing_accumulator=decoder_timing,
        )
    if len(meshes) != selected_count:
        raise ValueError(f"Expected {selected_count} decoded meshes, got {len(meshes)}")

    outputs = []
    for chunk_id, local_id in enumerate(selected):
        sparse_mask = sparse_coords[:, 0] == int(chunk_id)
        local_coords = sparse_coords[sparse_mask].clone()
        local_coords[:, 0] = 0
        outputs.append(
            {
                "local_id": local_id,
                "ss_latent": ss_latents[chunk_id],
                "occupancy_logits": occupancy_logits[chunk_id, 0],
                "occupancy_prob": occupancy_probs[chunk_id],
                "occupied_xyz": occupied_xyz[chunk_id],
                "shape_x2_features": pred_raw[sparse_mask],
                "shape_x2_coords": local_coords,
                "mesh": meshes[chunk_id],
                "unit_box_prune": bool(prune_out_of_range),
                "model_conditioning_seed": model_conditioning_seed,
                "network_timing_seconds": {
                    "ss_flow_amortized": ss_flow_seconds / selected_count,
                    "ss_vae_decoder_amortized": ss_vae_seconds / selected_count,
                    "shape_flow_amortized": shape_flow_seconds / selected_count,
                    "shape_x2_decoder_amortized": decoder_timing.get(
                        "shape_x2_decoder", 0.0
                    )
                    / selected_count,
                    "trellis_shape_decoder_gpu_amortized": decoder_timing.get(
                        "trellis_shape_decoder_gpu", 0.0
                    )
                    / selected_count,
                    "strict_network_total_amortized": (
                        ss_flow_seconds
                        + ss_vae_seconds
                        + shape_flow_seconds
                        + decoder_timing.get("shape_x2_decoder", 0.0)
                        + decoder_timing.get("trellis_shape_decoder_gpu", 0.0)
                    )
                    / selected_count,
                    "batch_object_count": selected_count,
                },
            }
        )
    return outputs


def _perception_object_to_world(data: dict[str, Any], local_id: int) -> np.ndarray:
    """Rigid+scale transform from FF-canonical object space to scene world."""
    prior = data.get("background_room_box_prior") or {}
    fitted = prior.get("canonical_transform") or {}
    if (
        int(prior.get("background_local_instance_id", -1)) == int(local_id)
        and fitted.get("applied")
    ):
        return np.asarray(fitted["object_to_world"], dtype=np.float64)

    raw_obb = data["perception_raw_obbs"][local_id]
    rotation = _quat_wxyz_to_matrix(raw_obb.get("rotation", [1, 0, 0, 0]))
    scale = np.asarray(raw_obb.get("scale", 1.0), dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    object_to_world = np.eye(4, dtype=np.float64)
    object_to_world[:3, :3] = rotation @ np.diag(scale[:3])
    object_to_world[:3, 3] = np.asarray(raw_obb["translate"], dtype=np.float64)
    return object_to_world


def export_object(args: argparse.Namespace, data: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    local_id = int(result["local_id"])
    object_id = int(data["object_ids"][local_id])
    object_name = data["object_names"][local_id]
    debug_dir = object_output_dir(args, object_name)
    debug_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        debug_dir / "occupancy.npz",
        ss_latent=result["ss_latent"].detach().cpu().numpy(),
        logits=result["occupancy_logits"].detach().cpu().numpy(),
        probability=result["occupancy_prob"].detach().cpu().numpy(),
        occupied_xyz=result["occupied_xyz"].detach().cpu().numpy(),
    )
    raw_mask = data["instance_ids"] == local_id
    write_point_ply(debug_dir / "conditioning_points_scene.ply", data["points"][raw_mask])
    write_point_ply(
        debug_dir / "conditioning_points_canonical.ply",
        canonical_points_for_object(data, local_id),
    )

    mesh = result["mesh"]
    mesh_path = mesh_output_path(args, object_name)
    status = "decoded"
    error = None
    if not args.geometry_stage_mesh_decode:
        # No mesh was decoded here; the appearance stage backfills mesh_path
        # with its postprocessed canonical geometry. Empty/garbage latents
        # still fail there, at postprocess input validation.
        mesh_path = None
    elif mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        status = "mesh_decode_failed"
        error = "decoder returned an empty mesh"
    else:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(mesh_path)
    world_mesh_path = None
    if (
        status == "decoded"
        and mesh is not None
        and args.export_world_object_meshes
        and args.perception_result_dir is not None
    ):
        object_to_world = _perception_object_to_world(data, local_id)
        world_mesh = mesh.copy()
        world_mesh.apply_transform(object_to_world)
        world_mesh_path = args.output_root / args.scene_id / f"{object_name}.ply"
        world_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        world_mesh.export(world_mesh_path)
    shape_x2_path = None
    if args.predict_appearance:
        shape_x2_path = debug_dir / "shape_x2.npz"
        np.savez_compressed(
            shape_x2_path,
            feats=result["shape_x2_features"].detach().cpu().numpy(),
            coords=result["shape_x2_coords"].detach().cpu().numpy(),
        )
    record = {
        "dataset": data["manifest"]["dataset"],
        "scene_id": data["scene"]["scene_id"],
        "object_id": object_id,
        "object_name": object_name,
        "status": status,
        "error": error,
        "num_condition_points": int(data["point_counts"][local_id]),
        "num_occupied_voxels": int(result["occupied_xyz"].shape[0]),
        "unit_box_prune": bool(result.get("unit_box_prune", True)),
        "model_conditioning_seed": int(result["model_conditioning_seed"]),
        "prediction_space": "ff_canonical",
        "mesh_path": (
            str(mesh_path) if mesh_path is not None and mesh_path.is_file() else None
        ),
        "world_mesh_path": str(world_mesh_path) if world_mesh_path is not None else None,
        "debug_dir": str(debug_dir),
        "network_timing_seconds": result["network_timing_seconds"],
    }
    if args.predict_appearance:
        record.update(
            {
                "shape_x2_path": str(shape_x2_path),
                "pbr_x2_path": None,
                "textured_glb": None,
            }
        )
    atomic_json(debug_dir / "summary.json", record)
    return record


def _load_local_shape(
    record: dict[str, Any], batch_index: int, expected_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(record["shape_x2_path"])
    with np.load(path, allow_pickle=False) as archive:
        feats = torch.from_numpy(np.asarray(archive["feats"], dtype=np.float32))
        coords = torch.from_numpy(np.asarray(archive["coords"], dtype=np.int32))
    if feats.ndim != 2 or feats.shape[1] != int(expected_channels):
        raise ValueError(
            f"Expected shape features [N,{expected_channels}] in {path}, "
            f"got {tuple(feats.shape)}"
        )
    if coords.shape != (feats.shape[0], 4) or torch.unique(coords[:, 0]).tolist() != [0]:
        raise ValueError(f"Invalid local Shape-X2 coordinates in {path}: {tuple(coords.shape)}")
    coords = coords.clone()
    coords[:, 0] = int(batch_index)
    return feats, coords


def _raw_world_to_object_transforms(
    data: dict[str, Any], selected_local_ids: list[int]
) -> np.ndarray:
    """Build world-to-canonical transforms from the retained raw predicted OBBs."""

    if "perception_raw_obbs" not in data:
        raise ValueError("Raw world transforms are only available for perception-conditioned runs")
    transforms = [
        np.linalg.inv(_perception_object_to_world(data, local_id))
        for local_id in selected_local_ids
    ]
    return np.stack(transforms)


def resolve_pbr_max_cond_len(args: argparse.Namespace) -> int:
    """PBR conditioning budget, falling back to the shared --max-cond-len."""
    if args.pbr_max_cond_len is not None:
        return int(args.pbr_max_cond_len)
    return int(args.max_cond_len)


def resolve_remesh_band(args: argparse.Namespace, dataset: str) -> float:
    """Per-dataset default for the remesh narrow band; an explicit value wins."""

    if args.appearance_topology_remesh_band is not None:
        return float(args.appearance_topology_remesh_band)
    return 2.0 if dataset == "scannetpp" else 1.0


def run_appearance(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    records: list[dict[str, Any]],
    shared_dino_features: torch.Tensor,
    shared_dino_config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Predict PBR on frozen geometry support, then jointly decode and bake it."""
    appearance_root = args.output_root / args.scene_id / "appearance"
    appearance_started = synchronized_timestamp(device)
    phase_timing: dict[str, float] = {}

    phase_started = synchronized_timestamp(device)
    pbr_max_cond_len = resolve_pbr_max_cond_len(args)
    pbr_key = (
        "pbr_flow", str(args.pbr_run_dir), str(args.pbr_checkpoint),
        args.pbr_model_family, args.pbr_use_ema, pbr_max_cond_len,
        args.dino_upsample, args.anyup_frame_batch_size, str(device),
    )
    pbr_model, pbr_metadata = _cached_load(
        args.model_cache,
        pbr_key,
        lambda: load_pbr_flow_model(
            run_dir=args.pbr_run_dir,
            checkpoint=args.pbr_checkpoint,
            device=device,
            max_cond_len=pbr_max_cond_len,
            dino_upsample=args.dino_upsample,
            anyup_frame_batch_size=args.anyup_frame_batch_size,
            use_ema=args.pbr_use_ema,
            model_family=args.pbr_model_family,
            shared_dino_config=shared_dino_config,
        ),
    )
    phase_timing["pbr_model_load"] = synchronized_timestamp(device) - phase_started

    phase_started = synchronized_timestamp(device)
    rgb_tensor = torch.from_numpy(data["rgbs"]).to(device=device, dtype=torch.float32)

    def infer_appearance_features(
        rgb_input=rgb_tensor,
        shared_features=shared_dino_features,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return appearance_features_from_shared_dino(
                pbr_model,
                rgb_input,
                shared_features,
                grid_keep_mask=data.get("dino_grid_keep_mask"),
            )

    all_appearance, appearance_feature_device_seconds = timed_network_call(
        device, infer_appearance_features
    )
    del rgb_tensor, shared_dino_features
    foreground_indices = torch.from_numpy(np.flatnonzero(data["foreground_mask"])).to(
        device=device, dtype=torch.long
    )
    point_features = all_appearance[foreground_indices].float()
    del all_appearance, foreground_indices
    phase_timing["appearance_feature_adapter"] = (
        synchronized_timestamp(device) - phase_started
    )

    if getattr(args, "pbr_shape_source_run", None) is not None:
        # Diagnostic bisection: swap in another run's Shape-X2 latents. Rewrite
        # the record path itself rather than intercepting the PBR loader, or the
        # aggregation step would still read this run's own shape and reject the
        # pair on a token-count mismatch.
        source_run = Path(args.pbr_shape_source_run)
        for record in records:
            local = Path(record["shape_x2_path"])
            candidate = source_run / "debug" / local.parent.name / local.name
            if not candidate.is_file():
                raise FileNotFoundError(f"Shape source missing: {candidate}")
            record["shape_x2_path"] = str(candidate)
        print(
            f"[bisect] Shape-X2 latents substituted from {source_run}",
            flush=True,
        )
    by_object_id = {int(record["object_id"]): record for record in records}
    eligible_local_ids = [
        local_id
        for local_id, object_id in enumerate(data["object_ids"])
        if by_object_id.get(int(object_id), {}).get("status") == "decoded"
        and (
            not args.resume_existing
            or not by_object_id[int(object_id)].get("pbr_x2_path")
            or not Path(by_object_id[int(object_id)]["pbr_x2_path"]).is_file()
        )
    ]
    flow_data = data
    snap_audit = None
    if args.pbr_condition_snap and eligible_local_ids:
        phase_started = synchronized_timestamp(device)
        from eval.reconstruction.lc64_shape_pbr_decode import decode_shape_vertices
        from utils.pbr_condition_snap import snap_condition_points

        snap_decoder_paths = DecoderPaths(
            shape_x2_root=args.shape_vae_root,
            pbr_x2_root=args.pbr_vae_root,
            shape_x2_checkpoint=args.shape_vae_checkpoint,
            pbr_x2_checkpoint=args.pbr_vae_checkpoint,
            shape_x2_use_ema=args.shape_vae_use_ema,
            pbr_x2_use_ema=args.pbr_vae_use_ema,
            shape_decoder=args.shape_decoder_pretrained,
        )
        # same cache key as the joint decode below, so the bundle loads once
        snap_models = _cached_load(
            args.model_cache,
            ("decoder_bundle", str(snap_decoder_paths), str(device)),
            lambda: load_decoder_bundle(snap_decoder_paths, device),
        )
        targets_by_local_id: dict[int, np.ndarray] = {}
        chunk = max(int(args.appearance_decode_object_chunk_size), 1)
        for start in range(0, len(eligible_local_ids), chunk):
            selected = eligible_local_ids[start : start + chunk]
            shape_parts, coord_parts = [], []
            for batch_index, local_id in enumerate(selected):
                record = by_object_id[int(data["object_ids"][local_id])]
                shape, coords = _load_local_shape(
                    record, batch_index, int(pbr_model.shape_x2_channels)
                )
                shape_parts.append(shape)
                coord_parts.append(coords)
            vertices = decode_shape_vertices(
                models=snap_models,
                shape_feats=torch.cat(shape_parts).to(
                    device=device, dtype=torch.float32
                ),
                coords=torch.cat(coord_parts).to(device=device, dtype=torch.int32),
                num_objects=len(selected),
                resolution=int(args.shape_decode_resolution),
            )
            for local_id, verts in zip(selected, vertices):
                targets_by_local_id[local_id] = verts.numpy()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        snapped_points, snap_audit = snap_condition_points(
            points=data["points"],
            instance_ids=data["instance_ids"],
            scene_to_object=data["scene_to_object"],
            targets_by_local_id=targets_by_local_id,
            object_names=list(data["object_names"]),
            max_distance=float(args.pbr_condition_snap_max_distance),
            fraction=float(args.pbr_condition_snap_fraction),
            debug_dir=appearance_root / "condition_snap",
        )
        del targets_by_local_id
        flow_data = dict(data)
        flow_data["points"] = snapped_points
        print(
            f"[pbr-snap] {snap_audit['num_objects_snapped']} objects: "
            f"moved {100 * snap_audit['scene_moved_fraction']:.1f}% of condition "
            f"points (mean NN distance before "
            f"{snap_audit['scene_nn_distance_before_mean']:.4f})",
            flush=True,
        )
        phase_timing["pbr_condition_snap"] = (
            synchronized_timestamp(device) - phase_started
        )

    phase_started = synchronized_timestamp(device)
    pbr_flow_device_seconds = 0.0
    background_no_prune_id = (
        None
        if args.background_unit_box_prune
        else background_local_instance_id(data)
    )
    pbr_batch_size = resolve_pbr_object_batch_size(args)
    for start in range(0, len(eligible_local_ids), pbr_batch_size):
        selected = eligible_local_ids[start : start + pbr_batch_size]
        shape_parts, coord_parts = [], []
        selected_records = []
        for batch_index, local_id in enumerate(selected):
            record = by_object_id[int(data["object_ids"][local_id])]
            shape, coords = _load_local_shape(
                record, batch_index, int(pbr_model.shape_x2_channels)
            )
            shape_parts.append(shape)
            coord_parts.append(coords)
            selected_records.append(record)
        shape = torch.cat(shape_parts).to(device)
        coords = torch.cat(coord_parts).to(device)
        common = common_tensors(flow_data, point_features, selected, device)
        seed_everything(
            stable_seed(
                args.seed,
                f"{data['manifest']['dataset']}:{args.scene_id}:pbr:{start}",
            )
        )
        predicted, batch_pbr_flow_seconds = timed_network_call(
            device,
            lambda common_input=common, shape_input=shape, coords_input=coords: predict_pbr_sparse(
                model=pbr_model,
                common=common_input,
                shape_features=shape_input,
                coords=coords_input,
                selected_indices=torch.tensor(selected, dtype=torch.long).unsqueeze(0),
                sample_method=(args.pbr_sample_method or args.sample_method),
                inference_num_steps=args.inference_num_steps,
                guidance_strength=args.pbr_guidance_strength,
                device=device,
            ),
        )
        pbr_flow_device_seconds += batch_pbr_flow_seconds
        background_override = None
        if background_no_prune_id in selected:
            background_batch_index = selected.index(background_no_prune_id)
            background_mask = coords[:, 0] == int(background_batch_index)
            background_shape = shape[background_mask]
            background_coords = coords[background_mask].clone()
            background_coords[:, 0] = 0
            background_common = common_tensors(
                flow_data, point_features, [background_no_prune_id], device
            )
            seed_everything(
                stable_seed(
                    args.seed,
                    (
                        f"{data['manifest']['dataset']}:{args.scene_id}:"
                        "pbr:background-no-unit-box-prune-v1"
                    ),
                )
            )
            print(
                "[background-unit-box] rerunning PBR for "
                f"{data['object_names'][background_no_prune_id]} without "
                "canonical unit-box pruning",
                flush=True,
            )
            background_predicted, background_pbr_flow_seconds = timed_network_call(
                device,
                lambda: predict_pbr_sparse(
                    model=pbr_model,
                    common=background_common,
                    shape_features=background_shape,
                    coords=background_coords,
                    selected_indices=torch.tensor(
                        [background_no_prune_id], dtype=torch.long
                    ).unsqueeze(0),
                    sample_method=(args.pbr_sample_method or args.sample_method),
                    inference_num_steps=args.inference_num_steps,
                    guidance_strength=args.pbr_guidance_strength,
                    prune_out_of_range=False,
                    device=device,
                ),
            )
            pbr_flow_device_seconds += background_pbr_flow_seconds
            background_override = (background_predicted, background_coords)
            del background_common, background_shape

        for batch_index, (local_id, record) in enumerate(
            zip(selected, selected_records)
        ):
            if local_id == background_no_prune_id:
                assert background_override is not None
                object_predicted, local_coords = background_override
            else:
                mask = coords[:, 0] == int(batch_index)
                object_predicted = predicted[mask]
                local_coords = coords[mask].clone()
                local_coords[:, 0] = 0
            pbr_path = Path(record["debug_dir"]) / "pbr_x2.npz"
            np.savez_compressed(
                pbr_path,
                feats=object_predicted.detach().cpu().numpy(),
                coords=local_coords.detach().cpu().numpy(),
            )
            record["pbr_x2_path"] = str(pbr_path)
            record["pbr_unit_box_prune"] = bool(
                local_id != background_no_prune_id
            )
            atomic_json(Path(record["debug_dir"]) / "summary.json", record)
        del shape, coords, common, predicted, background_override
        if device.type == "cuda":
            torch.cuda.empty_cache()

    phase_timing["pbr_flow_and_sparse_export"] = (
        synchronized_timestamp(device) - phase_started
    )

    phase_started = synchronized_timestamp(device)
    del point_features
    if not args.model_cache:
        del pbr_model
    gc.collect()
    if device.type == "cuda":
        # With --model-cache the flows stay resident; release torch's reserved
        # pool so the CuMesh allocator (raw cudaMalloc) has headroom during the
        # remesh/simplify peak.
        torch.cuda.empty_cache()
    phase_timing["pbr_model_cleanup"] = (
        synchronized_timestamp(device) - phase_started
    )
    phase_started = synchronized_timestamp(device)

    paired = [
        record
        for record in records
        if record.get("status") == "decoded"
        and record.get("shape_x2_path")
        and record.get("pbr_x2_path")
    ]
    object_id_to_local = {
        int(object_id): local_id
        for local_id, object_id in enumerate(data["object_ids"])
    }
    paired_local_ids = [
        object_id_to_local[int(record["object_id"])] for record in paired
    ]
    background_local_id = (
        (data.get("background_room_box_prior") or {}).get(
            "background_local_instance_id"
        )
    )
    background_pair_position = (
        paired_local_ids.index(int(background_local_id))
        if background_local_id is not None
        and int(background_local_id) in paired_local_ids
        else None
    )
    sparse_summary = concatenate_sparse_object_pairs(
        paired,
        shape_output=appearance_root / "sparse/shape_x2.npz",
        pbr_output=appearance_root / "sparse/pbr_x2.npz",
    )
    phase_timing["sparse_pair_aggregation"] = (
        synchronized_timestamp(device) - phase_started
    )
    decoder_paths = DecoderPaths(
        shape_x2_root=args.shape_vae_root,
        pbr_x2_root=args.pbr_vae_root,
        shape_x2_checkpoint=args.shape_vae_checkpoint,
        pbr_x2_checkpoint=args.pbr_vae_checkpoint,
        shape_x2_use_ema=args.shape_vae_use_ema,
        pbr_x2_use_ema=args.pbr_vae_use_ema,
        shape_decoder=args.shape_decoder_pretrained,
        pbr_decoder=str(args.pbr_decoder_pretrained),
    )
    phase_started = synchronized_timestamp(device)
    decoder_models = _cached_load(
        args.model_cache,
        ("decoder_bundle", str(decoder_paths), str(device)),
        lambda: load_decoder_bundle(decoder_paths, device),
    )
    phase_timing["appearance_decoder_load"] = (
        synchronized_timestamp(device) - phase_started
    )
    phase_started = synchronized_timestamp(device)
    raw_summary, raw_objects = decode_scene_to_raw_ovoxels(
        shape_path=Path(sparse_summary["shape_x2"]),
        pbr_path=Path(sparse_summary["pbr_x2"]),
        output_dir=appearance_root / "raw_decoded",
        object_count=len(paired),
        decoder_paths=decoder_paths,
        device=device,
        resolution=args.shape_decode_resolution,
        chunk_size=args.appearance_decode_object_chunk_size,
        overwrite=not args.resume_existing,
        models=decoder_models,
        persist_raw=args.appearance_persist_raw_decoded,
        return_raw=True,
        detailed_profile=args.appearance_detailed_profile,
    )
    phase_timing["shape_pbr_decode_to_raw"] = (
        synchronized_timestamp(device) - phase_started
    )
    if not args.model_cache:
        del decoder_models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    phase_started = synchronized_timestamp(device)
    mesh_summary = postprocess_and_bake_scene(
        decode_summary=raw_summary,
        output_dir=appearance_root / "textured_objects",
        device=device,
        resolution=args.shape_decode_resolution,
        decimation_target=args.appearance_decimation_target,
        background_object_position=background_pair_position,
        background_decimation_multiplier=(
            args.appearance_background_decimation_multiplier
        ),
        texture_size=args.appearance_texture_size,
        mesh_batch_size=args.appearance_mesh_batch_size,
        backend=args.appearance_mesh_backend,
        hybrid_foreground_fill_mode=args.appearance_hybrid_foreground_fill_mode,
        hybrid_foreground_dilation_pixels=(
            args.appearance_hybrid_foreground_dilation_pixels
        ),
        hybrid_foreground_surface_mapping=(
            args.appearance_hybrid_foreground_surface_mapping
        ),
        hybrid_foreground_decimation_target=(
            args.appearance_hybrid_foreground_decimation_target
        ),
        hybrid_atlas_size=args.appearance_hybrid_atlas_size,
        projection_num_views=args.appearance_projection_num_views,
        projection_assignment_resolution=(
            args.appearance_projection_assignment_resolution
        ),
        projection_padding_pixels=args.appearance_projection_padding_pixels,
        projection_view_assignment_mode=(
            args.appearance_projection_view_assignment_mode
        ),
        projection_preferred_visibility_ratio=(
            args.appearance_projection_preferred_visibility_ratio
        ),
        projection_raster_instance_batch_size=(
            args.appearance_projection_raster_instance_batch_size
        ),
        projection_allow_scalar_fallback=(
            args.appearance_projection_allow_scalar_fallback
        ),
        topology_config=PostprocessConfig(
            initial_target_multiplier=(
                args.appearance_topology_initial_target_multiplier
            ),
            simplify_threshold=args.appearance_topology_simplify_threshold,
            chart_area_penalty_weight=(
                args.appearance_topology_chart_area_penalty
            ),
            remesh=args.appearance_topology_remesh,
            remesh_band=resolve_remesh_band(args, data["manifest"]["dataset"]),
            remesh_project=args.appearance_topology_remesh_project,
            remesh_resolution=args.appearance_resolution
            if hasattr(args, "appearance_resolution")
            else 512,
            xatlas_block_align=args.appearance_xatlas_block_align,
            uv_cpu_workers=args.appearance_uv_cpu_workers,
        ),
        surface_mapping=args.appearance_surface_mapping,
        sparse_query_mode=args.appearance_sparse_query_mode,
        texture_fill_mode=args.appearance_texture_fill_mode,
        texture_dilation_pixels=args.appearance_texture_dilation_pixels,
        texture_erode_iterations=args.appearance_texture_erode_iterations,
        overwrite=not args.resume_existing,
        raw_objects=raw_objects,
        detailed_profile=args.appearance_detailed_profile,
    )
    phase_timing["mesh_postprocess_texture_export"] = (
        synchronized_timestamp(device) - phase_started
    )
    phase_started = synchronized_timestamp(device)
    object_id_to_local_for_export = {
        int(object_id): local_id
        for local_id, object_id in enumerate(data["object_ids"])
    }
    for pair_record, mesh_record in zip(paired, mesh_summary["objects"]):
        pair_record["textured_glb"] = mesh_record["canonical_glb"]
        if not args.geometry_stage_mesh_decode:
            # The geometry stage skipped its duplicate decode, so publish the
            # appearance stage's postprocessed geometry under the canonical
            # objects/pred_*.ply contract (hardlink; copy across devices).
            target = mesh_output_path(args, pair_record["object_name"])
            target.parent.mkdir(parents=True, exist_ok=True)
            source = Path(mesh_record["canonical_ply"])
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            pair_record["mesh_path"] = str(target)
            if (
                args.export_world_object_meshes
                and args.perception_result_dir is not None
            ):
                # The shaper benchmark evaluators compare <scene>/pred_*.ply
                # against GT meshes with no pose application, so these must be
                # world-space. With the geometry-stage decode skipped, this is
                # the only place that can still produce them.
                import trimesh as _trimesh

                local_id = object_id_to_local_for_export[
                    int(pair_record["object_id"])
                ]
                world_mesh = _trimesh.load(source, force="mesh", process=False)
                world_mesh.apply_transform(
                    _perception_object_to_world(data, local_id)
                )
                world_path = (
                    args.output_root
                    / args.scene_id
                    / f"{pair_record['object_name']}.ply"
                )
                world_path.parent.mkdir(parents=True, exist_ok=True)
                world_mesh.export(world_path)
                pair_record["world_mesh_path"] = str(world_path)
        atomic_json(Path(pair_record["debug_dir"]) / "summary.json", pair_record)

    selected_object_ids = [int(record["object_id"]) for record in paired]
    selected_local_ids = paired_local_ids
    training_scene = compose_textured_scene(
        mesh_summary=mesh_summary,
        object_transforms=np.asarray(data["scene_to_object"]),
        selected_instance_ids=selected_local_ids,
        source_instance_ids=selected_object_ids,
        output_path=appearance_root / "predicted_textured_training_scene.glb",
        overwrite=not args.resume_existing,
    )
    world_scene = None
    if args.perception_result_dir is not None:
        world_to_object = _raw_world_to_object_transforms(data, selected_local_ids)
        # The compact transform table is already ordered exactly like the paired
        # textured objects, so compose against dense local positions [0, K).
        world_scene = compose_textured_scene(
            mesh_summary=mesh_summary,
            object_transforms=world_to_object,
            selected_instance_ids=list(range(len(paired))),
            source_instance_ids=selected_object_ids,
            output_path=appearance_root / "predicted_textured_world_scene.glb",
            overwrite=not args.resume_existing,
        )
    elif len(selected_local_ids) > 0 and data.get("norm_transform") is not None:
        # GT-annotation mode (no perception): scene_to_object maps the
        # normalized scene to canonical object space and norm_transform maps
        # dataset world to the normalized scene (translation only), so their
        # product is the world-to-object transform. This keeps the GT-mode
        # world GLB in the same dataset-world frame as perception-mode output,
        # so the unified renderer's dataset cameras apply unchanged.
        norm = np.asarray(data["norm_transform"], dtype=np.float64)
        scene_to_object = np.asarray(data["scene_to_object"], dtype=np.float64)
        world_to_object = np.stack(
            [scene_to_object[local_id] @ norm for local_id in selected_local_ids]
        )
        world_scene = compose_textured_scene(
            mesh_summary=mesh_summary,
            object_transforms=world_to_object,
            selected_instance_ids=list(range(len(paired))),
            source_instance_ids=selected_object_ids,
            output_path=appearance_root / "predicted_textured_world_scene.glb",
            overwrite=not args.resume_existing,
        )
    phase_timing["scene_composition_and_metadata"] = (
        synchronized_timestamp(device) - phase_started
    )
    phase_timing["appearance_total"] = (
        synchronized_timestamp(device) - appearance_started
    )
    settings = inference_settings(args)
    summary = {
        "schema": "ff_scene_lc64_appearance_predictions_v1",
        "dataset": data["manifest"]["dataset"],
        "scene_id": args.scene_id,
        "pbr_flow": pbr_metadata,
        "pbr_condition_snap": snap_audit,
        "remesh_band": resolve_remesh_band(args, data["manifest"]["dataset"]),
        "sparse": sparse_summary,
        "decoder": {
            "shape_vae_root": str(args.shape_vae_root),
            "shape_vae_checkpoint": args.shape_vae_checkpoint,
            "pbr_vae_root": str(args.pbr_vae_root),
            "pbr_vae_checkpoint": args.pbr_vae_checkpoint,
            "pbr_vae_use_ema": bool(args.pbr_vae_use_ema),
            "detailed_profile": raw_summary.get("detailed_profile"),
        },
        "raw_decode": str(appearance_root / "raw_decoded/decode_summary.json"),
        "mesh_summary": str(appearance_root / "textured_objects/mesh_summary.json"),
        "scene_world_to_training": np.asarray(data["norm_transform"]).tolist(),
        "composed_training_scene": training_scene,
        "composed_world_scene": world_scene,
        "objects": paired,
        "phase_timing_seconds": phase_timing,
        "network_timing": {
            "definition": (
                "CUDA-event device time for appearance feature adaptation and "
                "one-step PBR flow only; wall phases include tensor/file work."
            ),
            "appearance_feature_adapter_scene_seconds": (
                appearance_feature_device_seconds
            ),
            "pbr_flow_total_seconds": pbr_flow_device_seconds,
            "pbr_flow_amortized_seconds_per_object": (
                pbr_flow_device_seconds / max(len(paired), 1)
            ),
        },
    }
    atomic_json(appearance_root / "appearance_summary.json", summary)
    return records, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pipeline_started = time.perf_counter()
    phase_timing: dict[str, float] = {}
    phase_started = time.perf_counter()
    data = load_scene_inputs(args)
    data = apply_perception_conditioning(args, data)
    target_local_ids = selected_local_ids_for_run(
        data, getattr(args, "object_selection", "all")
    )
    target_local_id_set = set(target_local_ids)
    audit = input_audit(data)
    settings = inference_settings(args)
    scene_root = args.output_root / args.scene_id
    atomic_json(scene_root / "input_audit.json", audit)
    phase_timing["data_load_and_preprocess"] = time.perf_counter() - phase_started
    print(json.dumps(audit, indent=2), flush=True)
    if args.input_audit_only:
        return {"input_audit": audit, "status": "input_audit_only"}

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(stable_seed(args.seed, f"{data['manifest']['dataset']}:{args.scene_id}"))
    phase_started = synchronized_timestamp(device)
    flow_key = (
        "flow_models", str(args.ss_run_dir), str(args.ss_checkpoint),
        args.ss_model_family, args.ss_use_ema, str(args.shape_run_dir),
        str(args.shape_checkpoint), args.shape_model_family, args.shape_use_ema,
        args.max_cond_len, args.dino_upsample, args.anyup_frame_batch_size,
        str(device),
    )
    ss_model, shape_model, flow_metadata = _cached_load(
        args.model_cache, flow_key, lambda: load_flow_models(args, device)
    )
    shared_dino_config = dict(ss_model.config)
    decoder_key = (
        "decoders", str(args.ss_vae_root), args.ss_vae_checkpoint, args.ss_vae_use_ema,
        str(args.shape_vae_root), args.shape_vae_checkpoint, args.shape_vae_use_ema,
        args.shape_decoder_pretrained, str(device),
    )
    ss_decoder, x2_decoder, shape_decoder, stats, decoder_metadata = _cached_load(
        args.model_cache, decoder_key, lambda: load_decoders(args, device)
    )
    input_mean, input_std = stats
    phase_timing["geometry_model_load"] = (
        synchronized_timestamp(device) - phase_started
    )

    phase_started = synchronized_timestamp(device)
    rgb_tensor = torch.from_numpy(data["rgbs"]).to(device=device, dtype=torch.float32)
    def infer_shared_dino(model=ss_model, rgb_input=rgb_tensor):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            return model.get_dino_feats(rgb_input).float()

    all_features, shared_dino_seconds = timed_network_call(device, infer_shared_dino)
    del rgb_tensor
    all_features = filter_dino_features_to_projected_depth_grid(all_features, data)
    if all_features.shape[0] != data["all_points"].shape[0]:
        raise ValueError(
            f"DINO feature count {all_features.shape[0]} != projected point count "
            f"{data['all_points'].shape[0]}"
        )
    foreground_indices = torch.from_numpy(np.flatnonzero(data["foreground_mask"])).to(
        device=device, dtype=torch.long
    )
    point_features = all_features[foreground_indices]
    del foreground_indices
    phase_timing["shared_dino_feature_extract"] = (
        synchronized_timestamp(device) - phase_started
    )

    phase_started = synchronized_timestamp(device)
    old_records: dict[int, dict[str, Any]] = {}
    summary_path = scene_root / "inference_summary.json"
    if args.resume_existing and summary_path.is_file():
        try:
            old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            old_records = {int(record["object_id"]): record for record in old_summary.get("objects", [])}
        except (OSError, ValueError, KeyError, TypeError):
            old_records = {}
    completed = completed_local_ids(args, data) & target_local_id_set
    pending = [index for index in target_local_ids if index not in completed]
    records = []
    for index in sorted(completed):
        object_id = data["object_ids"][index]
        if object_id in old_records:
            records.append(old_records[object_id])
            continue
        debug_summary = object_output_dir(args, data["object_names"][index]) / "summary.json"
        if debug_summary.is_file():
            try:
                records.append(json.loads(debug_summary.read_text(encoding="utf-8")))
                continue
            except (OSError, ValueError, TypeError):
                pass
        records.append(
            {
                "dataset": data["manifest"]["dataset"],
                "scene_id": args.scene_id,
                "object_id": object_id,
                "object_name": data["object_names"][index],
                "status": "decoded",
                "error": None,
                "num_condition_points": data["point_counts"][index],
                "prediction_space": "ff_canonical",
                "mesh_path": str(mesh_output_path(args, data["object_names"][index])),
            }
        )

    ss_shape_batch_size = resolve_ss_shape_object_batch_size(args)
    for start in range(0, len(pending), ss_shape_batch_size):
        selected = pending[start : start + ss_shape_batch_size]
        names = [data["object_names"][index] for index in selected]
        print(f"[scene] predicting {names}", flush=True)
        try:
            chunk_results = None
            for pass_selected, prune_out_of_range in flow_unit_box_prune_passes(
                data,
                selected,
                background_unit_box_prune=args.background_unit_box_prune,
            ):
                if not prune_out_of_range:
                    print(
                        "[background-unit-box] rerunning SS/Shape for "
                        f"{data['object_names'][pass_selected[0]]} without "
                        "canonical unit-box pruning",
                        flush=True,
                    )
                pass_results = predict_chunk(
                    args=args,
                    data=data,
                    selected=pass_selected,
                    point_features=point_features,
                    ss_model=ss_model,
                    shape_model=shape_model,
                    ss_decoder=ss_decoder,
                    x2_decoder=x2_decoder,
                    shape_decoder=shape_decoder,
                    input_mean=input_mean,
                    input_std=input_std,
                    device=device,
                    prune_out_of_range=prune_out_of_range,
                )
                if chunk_results is None:
                    chunk_results = pass_results
                else:
                    replacements = {
                        int(result["local_id"]): result for result in pass_results
                    }
                    chunk_results = [
                        replacements.get(int(result["local_id"]), result)
                        for result in chunk_results
                    ]
            assert chunk_results is not None
            for result in chunk_results:
                record = export_object(args, data, result)
                records = [item for item in records if item["object_id"] != record["object_id"]]
                records.append(record)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            for local_id in selected:
                record = {
                    "dataset": data["manifest"]["dataset"],
                    "scene_id": args.scene_id,
                    "object_id": data["object_ids"][local_id],
                    "object_name": data["object_names"][local_id],
                    "status": "failed",
                    "error": error,
                    "num_condition_points": data["point_counts"][local_id],
                    "mesh_path": None,
                }
                records = [item for item in records if item["object_id"] != record["object_id"]]
                records.append(record)
            if not args.continue_on_error:
                raise
        records.sort(key=lambda item: int(item["object_id"]))
        partial = {
            "schema": SCHEMA,
            "dataset": data["manifest"]["dataset"],
            "scene_id": args.scene_id,
            "status": "running",
            "input_audit": audit,
            "flow": flow_metadata,
            "decoders": decoder_metadata,
            "objects": records,
        }
        atomic_json(summary_path, partial)

    records.sort(key=lambda item: int(item["object_id"]))
    if args.perception_result_dir is not None:
        obb_objects = []
        records_by_object_id = {
            int(record["object_id"]): record for record in records
        }
        for local_id, (raw_obb, object_name) in enumerate(
            zip(data["perception_raw_obbs"], data["object_names"])
        ):
            object_id = int(data["object_ids"][local_id])
            record = records_by_object_id.get(object_id)
            if record is None:
                continue
            prior = data.get("background_room_box_prior") or {}
            fitted = prior.get("canonical_transform") or {}
            if (
                int(prior.get("background_local_instance_id", -1)) == local_id
                and fitted.get("applied")
            ):
                object_to_world = _perception_object_to_world(data, local_id)
                linear = object_to_world[:3, :3]
                scale = np.linalg.norm(linear, axis=0)
                rotation = _matrix_to_quat_wxyz(linear / scale[None, :])
                translation = object_to_world[:3, 3]
            else:
                scale = np.asarray(
                    raw_obb.get("scale", 1.0), dtype=np.float64
                ).reshape(-1)
                if scale.size == 1:
                    scale = np.repeat(scale, 3)
                rotation = [
                    float(value)
                    for value in raw_obb.get("rotation", [1, 0, 0, 0])
                ]
                translation = np.asarray(raw_obb["translate"], dtype=np.float64)
            obb_objects.append(
                {
                    "index": object_id,
                    "name": object_name,
                    "translation": [
                        float(value) for value in translation
                    ],
                    "rotation": rotation,
                    "scale": [float(value) for value in scale[:3]],
                    "confidence": float(raw_obb.get("confidence", 1.0)),
                    "mesh_path": record.get("world_mesh_path"),
                }
            )
        atomic_json(
            scene_root / "object_obbs.json",
            {
                "schema": "ff_scene_perception_reconstruction_objects_v1",
                "scene_name": args.scene_id,
                "source_perception_result": str(args.perception_result_dir),
                "objects": obb_objects,
            },
        )
    del point_features, ss_model, shape_model, ss_decoder, x2_decoder, shape_decoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    phase_timing["geometry_prediction_decode_export"] = (
        synchronized_timestamp(device) - phase_started
    )

    appearance_summary = None
    phase_started = synchronized_timestamp(device)
    if args.predict_appearance:
        records, appearance_summary = run_appearance(
            args=args,
            data=data,
            records=records,
            shared_dino_features=all_features,
            shared_dino_config=shared_dino_config,
            device=device,
        )
    phase_timing["appearance_pipeline"] = (
        synchronized_timestamp(device) - phase_started
    )
    phase_started = synchronized_timestamp(device)
    del all_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    phase_timing["final_cleanup"] = synchronized_timestamp(device) - phase_started

    decoded = sum(record.get("status") == "decoded" for record in records)
    expected = len(target_local_ids)
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data["manifest"]["dataset"],
        "scene_id": args.scene_id,
        "status": "complete" if decoded == expected else "incomplete",
        "num_expected_objects": expected,
        "num_scene_objects": len(data["object_ids"]),
        "selected_local_ids": target_local_ids,
        "selected_object_names": [
            data["object_names"][local_id] for local_id in target_local_ids
        ],
        "num_decoded_objects": decoded,
        "elapsed_seconds": time.time() - started,
        "pipeline_wall_seconds": time.perf_counter() - pipeline_started,
        "phase_timing_seconds": phase_timing,
        "settings": settings,
        "input_audit": audit,
        "flow": flow_metadata,
        "decoders": decoder_metadata,
        "objects": records,
        "network_timing": {
            "definition": (
                "CUDA-event device time for neural forward/sampling and GPU decoder calls only; "
                "excludes data loading, tensor/file serialization, CPU mesh conversion, "
                "decimation, remeshing, metric calculation, and rendering"
            ),
            "shared_dino_scene_seconds": shared_dino_seconds,
            "shared_dino_amortized_seconds_per_object": shared_dino_seconds
            / max(expected, 1),
            "per_object": {
                record["object_name"]: record.get("network_timing_seconds")
                for record in records
                if record.get("network_timing_seconds") is not None
            },
        },
    }
    if args.predict_appearance:
        summary["appearance"] = appearance_summary
    atomic_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in ("status", "num_expected_objects", "num_decoded_objects", "elapsed_seconds")}, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.fire3d_test_root = args.fire3d_test_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.batch_scenes is None:
        args.manifest = args.manifest.resolve()
        result = run(args)
        if result.get("status") == "incomplete":
            raise SystemExit(2)
        return

    entries = json.loads(args.batch_scenes.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise SystemExit("--batch-scenes must be a non-empty JSON list")
    failures = []
    for n, entry in enumerate(entries, start=1):
        args.scene_id = entry["scene_id"]
        args.manifest = Path(entry["manifest"]).resolve()
        if entry.get("perception_result_dir"):
            args.perception_result_dir = Path(entry["perception_result_dir"])
        print(
            f"[batch] ({n}/{len(entries)}) {args.scene_id}"
            + (" (models cached)" if n > 1 else ""),
            flush=True,
        )
        try:
            result = run(args)
        except Exception as error:  # keep the batch going; report at the end
            traceback.print_exc()
            failures.append((args.scene_id, f"{type(error).__name__}: {error}"))
            result = None
        finally:
            # Resident models shrink the free pool; trim between scenes so
            # per-scene allocations (flows at batch 32, remesh grids) do not
            # accumulate fragmentation on top of the cached weights.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if result is not None and result.get("status") == "incomplete":
            failures.append((args.scene_id, "incomplete"))
    if failures:
        for scene_id, reason in failures:
            print(f"[batch] FAILED {scene_id}: {reason}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
