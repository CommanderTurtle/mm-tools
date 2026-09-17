"""LC64 PBR-X2 sparse flow conditioned on RGB and LC64 Shape-X2."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2

from modules import sparse as sp
from models.context_encoder import (
    filter_points_in_object_unit_box,
    fix_empty_context,
    get_context_encoder,
    prune_out_of_range_points,
)
from models.dino_anyup import (
    dino_feature_downsample,
    load_anyup_upsampler,
    move_module_to_device,
    run_dino_anyup_features,
    validate_dino_upsample,
)
from models.flow_model_sparse import ElasticSLatFlowModel
from models.flow_trainer_sparse import SparseFlowMatchingCFGTrainer
from utils.constants import POS_MAX
from utils.object_cond import sample_object_feats
from utils.voxelize import InstanceVoxelize


def make_transform():
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    def transform_fn(tensor):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif tensor.ndim != 4:
            raise ValueError(f"Expected [n,h,w] or [n,h,w,3], got {tensor.shape}")
        return normalize(tensor.permute(0, 3, 1, 2))

    return transform_fn


def _load_x2_stats(path, channels, label):
    if path is None:
        raise ValueError(f"Offline {label} training requires a latent stats path")
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {label} latent stats: {stats_path}")
    with stats_path.open("r") as f:
        stats = json.load(f)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(
            f"Expected {channels}-channel {label} stats, got mean={mean.numel()} std={std.numel()}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Non-finite {label} latent stats in {stats_path}")
    return mean.reshape(1, channels), torch.clamp(std.reshape(1, channels), min=1e-6)


class ObjectGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scene_scale = POS_MAX
        self.resolution = int(config.get("resolution", 1024))
        self.max_denoise_objects = int(config.get("max_denoise_objects", 64))
        self.voxelize = InstanceVoxelize(
            voxel_size=self.scene_scale / self.resolution,
            resolution=self.resolution,
            apply_standardization=False,
        )
        self.max_cond_len = int(config.get("max_cond_len", 512))

        self.context_encoder = get_context_encoder(config["context_encoder"])
        self.context_channels = int(config["context_encoder"]["channels"])
        appearance_cfg = config.get("appearance_condition", {})
        self.appearance_mode = str(appearance_cfg.get("mode", "dino_rgb")).lower()
        if self.appearance_mode not in {"dino", "dino_rgb", "dino_rgb_concat", "rgb"}:
            raise ValueError(f"Unknown appearance mode: {self.appearance_mode}")
        self.dino_channels = int(appearance_cfg.get("dino_channels", self.context_channels))
        self.rgb_channels = int(appearance_cfg.get("rgb_channels", self.context_channels))
        if self.appearance_mode == "dino_rgb_concat":
            expected_context_channels = self.dino_channels + self.rgb_channels
            if self.context_channels != expected_context_channels:
                raise ValueError(
                    "dino_rgb_concat requires context channels = DINO channels + RGB channels: "
                    f"{self.context_channels} != {self.dino_channels}+{self.rgb_channels}"
                )
        self.rgb_mlp = None
        if self.appearance_mode in {"dino_rgb", "dino_rgb_concat", "rgb"}:
            hidden = int(appearance_cfg.get("hidden_channels", 256))
            rgb_out_channels = (
                self.rgb_channels
                if self.appearance_mode == "dino_rgb_concat"
                else self.context_channels
            )
            self.rgb_mlp = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, rgb_out_channels),
            )
            if self.appearance_mode == "dino_rgb" and bool(
                appearance_cfg.get("zero_init_residual", True)
            ):
                nn.init.zeros_(self.rgb_mlp[-1].weight)
                nn.init.zeros_(self.rgb_mlp[-1].bias)

        dino_config = config.get("dino") or {}
        self.dino_upsample = validate_dino_upsample(
            int(dino_config.get("upsample", 1)),
            int(dino_config.get("downsample", 16)),
        )
        self.__dict__["_anyup_upsampler"] = None
        self.dino_model = None
        self.dino_transform = None
        if config.get("dino") is not None:
            self.dino_model, self.dino_transform = self.get_dino_model_and_transform()
            self.dino_model.eval()
            for parameter in self.dino_model.parameters():
                parameter.requires_grad = False

        flow_cfg = config.get("flow_model_pbr_x2") or config["flow_model_feat"]
        trainer_cfg = config.get("flow_trainer_pbr_x2") or config["flow_trainer_feat"]
        self.pbr_x2_channels = int(flow_cfg["out_channels"])
        self.shape_x2_channels = int(
            config.get("shape_x2", {}).get(
                "channels", int(flow_cfg["in_channels"]) - self.pbr_x2_channels
            )
        )
        if int(flow_cfg["in_channels"]) != self.shape_x2_channels + self.pbr_x2_channels:
            raise ValueError(
                "PBR flow in_channels must equal Shape-X2 channels + PBR-X2 channels: "
                f"{flow_cfg['in_channels']} != {self.shape_x2_channels}+{self.pbr_x2_channels}"
            )

        shape_cfg = config.get("shape_x2", {})
        pbr_cfg = config.get("pbr_x2", {})
        shape_mean, shape_std = _load_x2_stats(
            shape_cfg.get("x2_latent_stats_path"), self.shape_x2_channels, "Shape-X2"
        )
        pbr_mean, pbr_std = _load_x2_stats(
            pbr_cfg.get("x2_latent_stats_path"), self.pbr_x2_channels, "PBR-X2"
        )
        self.normalize_shape_x2 = bool(shape_cfg.get("normalize_condition", True))
        self.normalize_pbr_x2 = bool(pbr_cfg.get("normalize_target", True))
        self.register_buffer("shape_x2_mean", shape_mean, persistent=False)
        self.register_buffer("shape_x2_std", shape_std, persistent=False)
        self.register_buffer("pbr_x2_mean", pbr_mean, persistent=False)
        self.register_buffer("pbr_x2_std", pbr_std, persistent=False)

        self.flow_model_feat = ElasticSLatFlowModel(**flow_cfg)
        self.flow_trainer_feat = SparseFlowMatchingCFGTrainer(
            denoiser=self.flow_model_feat,
            trainer_name="pbr_x2",
            p_uncond=float(trainer_cfg["p_uncond"]),
        )
        self.profile_step_timing = os.environ.get("FF_PROFILE_STEP_TIMING", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_profile = {}

    @torch.no_grad()
    def get_dino_model_and_transform(self):
        dino = self.config["dino"]
        model = torch.hub.load(
            dino["repo_dir"],
            dino.get("model_name", "dinov3_vitl16"),
            source="local",
            weights=dino["model_path"],
        )
        return model, make_transform()

    def _get_anyup_upsampler(self, device):
        if self.dino_upsample <= 1:
            return None
        upsampler = self.__dict__.get("_anyup_upsampler")
        if upsampler is None:
            upsampler = load_anyup_upsampler(self.config.get("dino", {}), device=device)
            self.__dict__["_anyup_upsampler"] = upsampler
        return move_module_to_device(upsampler, device)

    @torch.no_grad()
    def get_dino_feats(self, rgbs):
        if self.dino_model is None:
            raise ValueError("model.dino is required when precomputed colors are not supplied")
        dino_downsample = int(self.config.get("dino", {}).get("downsample", 16))
        validate_dino_upsample(self.dino_upsample, dino_downsample)
        n, height, width, _ = rgbs.shape
        output_h, output_w = height // dino_downsample, width // dino_downsample
        height, width = output_h * dino_downsample, output_w * dino_downsample
        rgbs = rgbs[:, :height, :width]
        transformed = self.dino_transform(rgbs)
        if next(self.dino_model.parameters()).device != transformed.device:
            self.dino_model.to(transformed.device)
        if self.dino_upsample == 1:
            outputs = self.dino_model(transformed, is_training=True)
            return outputs["x_norm_patchtokens"].reshape(n * output_h * output_w, -1)

        upsampler = self._get_anyup_upsampler(transformed.device)
        image_downsample = dino_feature_downsample(self.dino_upsample, dino_downsample)
        anyup_h, anyup_w = height // image_downsample, width // image_downsample
        transformed = transformed[:, :, : anyup_h * image_downsample, : anyup_w * image_downsample]
        return run_dino_anyup_features(
            dino_model=self.dino_model,
            anyup_upsampler=upsampler,
            rgbs_tensor=transformed,
            dino_downsample=dino_downsample,
            dino_upsample=self.dino_upsample,
            anyup_q_chunk_size=self.config.get("dino", {}).get("anyup_q_chunk_size"),
            anyup_tile_grid=int(self.config.get("dino", {}).get("anyup_tile_grid", 0)),
            anyup_frame_batch_size=int(self.config.get("dino", {}).get("anyup_frame_batch_size", 1)),
        )

    def get_rgb_cell_feats(self, rgbs):
        """Average RGB over the same effective cells returned by DINO/AnyUp."""
        dino_downsample = int(self.config.get("dino", {}).get("downsample", 16))
        effective = dino_feature_downsample(self.dino_upsample, dino_downsample)
        n, height, width, _ = rgbs.shape
        # Match get_dino_feats: first trim to whole native DINO patches.
        height = (height // dino_downsample) * dino_downsample
        width = (width // dino_downsample) * dino_downsample
        if height == 0 or width == 0:
            raise ValueError(f"RGB image is smaller than DINO patch size: {tuple(rgbs.shape)}")
        rgb_nchw = rgbs[:, :height, :width].permute(0, 3, 1, 2).float()
        cells = F.avg_pool2d(rgb_nchw, kernel_size=effective, stride=effective)
        return cells.permute(0, 2, 3, 1).reshape(-1, 3)

    def _appearance_feats(self, rgbs):
        rgb_cells = None
        if self.appearance_mode in {"dino_rgb", "dino_rgb_concat", "rgb"}:
            rgb_cells = self.get_rgb_cell_feats(rgbs)
        if self.appearance_mode == "rgb":
            return self.rgb_mlp(rgb_cells * 2.0 - 1.0)
        dino_feats = self.get_dino_feats(rgbs)
        expected_dino_channels = (
            self.dino_channels
            if self.appearance_mode == "dino_rgb_concat"
            else self.context_channels
        )
        if dino_feats.shape[-1] != expected_dino_channels:
            raise ValueError(
                f"DINO channels {dino_feats.shape[-1]} != expected {expected_dino_channels}"
            )
        if self.appearance_mode == "dino":
            return dino_feats
        if rgb_cells.shape[0] != dino_feats.shape[0]:
            raise ValueError(
                f"RGB/DINO cell mismatch: rgb={rgb_cells.shape[0]} dino={dino_feats.shape[0]}"
            )
        rgb_feats = self.rgb_mlp(rgb_cells * 2.0 - 1.0)
        if self.appearance_mode == "dino_rgb_concat":
            appearance_feats = torch.cat((dino_feats, rgb_feats), dim=-1)
            if appearance_feats.shape[-1] != self.context_channels:
                raise RuntimeError(
                    f"Concatenated appearance channels {appearance_feats.shape[-1]} "
                    f"!= context channels {self.context_channels}"
                )
            return appearance_feats
        return dino_feats + rgb_feats

    def _normalize(self, feats, mean, std, channels, enabled, label):
        if feats.ndim != 2 or feats.shape[1] != channels:
            raise ValueError(f"Expected {label} [N,{channels}], got {tuple(feats.shape)}")
        if not enabled:
            return feats.float()
        return (feats.float() - mean.to(feats.device)) / std.to(feats.device)

    def _normalize_shape_x2_feats(self, feats):
        return self._normalize(
            feats,
            self.shape_x2_mean,
            self.shape_x2_std,
            self.shape_x2_channels,
            self.normalize_shape_x2,
            "Shape-X2",
        )

    def _normalize_pbr_x2_feats(self, feats):
        return self._normalize(
            feats,
            self.pbr_x2_mean,
            self.pbr_x2_std,
            self.pbr_x2_channels,
            self.normalize_pbr_x2,
            "PBR-X2",
        )

    def _denormalize_pbr_x2_feats(self, feats):
        if not self.normalize_pbr_x2:
            return feats.float()
        return feats.float() * self.pbr_x2_std.to(feats.device) + self.pbr_x2_mean.to(feats.device)

    def forward(
        self,
        points,
        instance_ids,
        object_transforms,
        object_shape_x2_feats,
        object_coords,
        selected_indices,
        max_num_objects,
        object_pbr_x2_feats=None,
        colors=None,
        rgbs=None,
        valid_point_masks=None,
        return_denoised_latents=False,
        sample_method="random",
        prune_out_of_range=True,
        inference_num_steps=50,
        guidance_strength=None,
        inference_only=False,
    ):
        if points.shape[0] != 1:
            raise ValueError("ObjectGen requires batch size 1 per GPU")
        voxel_coords = points[0]
        if colors is None:
            if rgbs is None:
                raise ValueError("Either colors or rgbs must be provided")
            voxel_feats = self._appearance_feats(rgbs[0])
            if valid_point_masks is not None:
                voxel_feats = voxel_feats[valid_point_masks[0]]
        else:
            if self.appearance_mode != "dino":
                raise ValueError("Precomputed colors support only appearance_condition.mode=dino")
            voxel_feats = colors[0]

        voxel_instance_ids = instance_ids[0]
        object_transforms = object_transforms[0]
        if prune_out_of_range:
            voxel_coords, voxel_feats, voxel_instance_ids = filter_points_in_object_unit_box(
                voxel_coords,
                voxel_feats,
                voxel_instance_ids,
                object_transforms,
                max_num_objects,
            )
        context, context_coords, context_mask = sample_object_feats(
            coords=voxel_coords,
            feats=voxel_feats,
            instance_ids=voxel_instance_ids,
            max_num_objects=max_num_objects,
            max_feats_len=self.max_cond_len,
            sample_method=sample_method,
        )
        context_coords_h = F.pad(context_coords, (0, 1), value=1.0)
        context_coords = (context_coords_h @ object_transforms.transpose(-2, -1))[..., :3]
        if prune_out_of_range:
            context_mask = prune_out_of_range_points(context_coords, context_mask)
        context = self.context_encoder(context, context_coords, context_mask)
        context, context_mask = fix_empty_context(context, context_mask)
        selected_indices = selected_indices[0]
        context = context[selected_indices]
        context_mask = context_mask[selected_indices]
        lengths = context_mask.sum(dim=1)
        cond = list(torch.split(context[context_mask], lengths.cpu().tolist(), dim=0))

        shape_sparse = sp.SparseTensor(
            feats=self._normalize_shape_x2_feats(object_shape_x2_feats),
            coords=object_coords,
        )
        if inference_only:
            if not return_denoised_latents:
                raise ValueError("inference_only requires return_denoised_latents=True")
            pbr_sparse = sp.SparseTensor(
                feats=torch.zeros(
                    (object_coords.shape[0], self.pbr_x2_channels),
                    device=object_shape_x2_feats.device,
                    dtype=object_shape_x2_feats.dtype,
                ),
                coords=object_coords,
            )
        else:
            if object_pbr_x2_feats is None:
                raise ValueError("Training/evaluation with targets requires object_pbr_x2_feats")
            pbr_sparse = sp.SparseTensor(
                feats=self._normalize_pbr_x2_feats(object_pbr_x2_feats),
                coords=object_coords,
            )

        results = {"flow_metrics_dict": {}, "flow_loss_dict": {}}
        if return_denoised_latents:
            pred_sparse = self.flow_trainer_feat.run_inference(
                x_0=pbr_sparse,
                cond=cond,
                concat_cond=shape_sparse,
                steps=inference_num_steps,
                guidance_strength=guidance_strength,
            )
            results["pbr_x2_pred_feats"] = pred_sparse.feats
            results["pbr_x2_pred_feats_raw"] = self._denormalize_pbr_x2_feats(pred_sparse.feats)
            # Compatibility with generic feature-flow evaluation helpers.
            results["feat_preds"] = pred_sparse.feats

        if not inference_only:
            loss, metrics = self.flow_trainer_feat.training_step(
                x_0=pbr_sparse,
                cond=cond,
                concat_cond=shape_sparse,
            )
            results["flow_loss_dict"] = {
                "loss_flow_pbr_x2": loss,
                "loss_flow_feat": loss,
            }
            results["flow_metrics_dict"].update(metrics)
            results["pbr_x2_target_feats"] = pbr_sparse.feats
            results["pbr_x2_target_feats_raw"] = object_pbr_x2_feats.float()
        results["pbr_x2_target_coords"] = object_coords
        results["shape_x2_condition_feats"] = shape_sparse.feats
        return results
