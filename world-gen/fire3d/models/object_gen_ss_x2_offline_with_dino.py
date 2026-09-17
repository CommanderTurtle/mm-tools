"""Dense LC64 SS-VAE X2 flow model with scene/object DINO conditioning."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2

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
from models.flow_model import FlowModel
from models.flow_trainer import FlowMatchingCFGTrainer
from utils.constants import POS_MAX
from utils.object_cond import sample_object_feats


def make_transform():
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    def transform_fn(tensor):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif tensor.ndim != 4:
            raise ValueError(
                f"Expected input [frames,height,width] or [frames,height,width,3], got {tensor.shape}"
            )
        return normalize(tensor.permute(0, 3, 1, 2))

    return transform_fn


def _load_ss_x2_stats(
    path,
    *,
    channels,
    resolution,
    expected_ss_step=None,
    expected_ss_artifact_id=None,
    expected_ss_encoder_sha256=None,
):
    if path is None:
        raise ValueError("Offline SS X2 training requires model.ss_x2.latent_stats_path")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing SS X2 latent statistics: {path}")
    stats = json.loads(path.read_text())
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(
            f"Expected {channels}-channel SS statistics, got mean={mean.numel()} std={std.numel()}"
        )
    if int(stats.get("latent_channels", channels)) != channels:
        raise ValueError(f"SS statistics latent_channels mismatch at {path}")
    if int(stats.get("latent_resolution", resolution)) != resolution:
        raise ValueError(f"SS statistics latent_resolution mismatch at {path}")
    ss_vae = stats.get("ss_vae") or {}
    if expected_ss_artifact_id is not None:
        actual_artifact = str(ss_vae.get("artifact_id", ""))
        if actual_artifact != str(expected_ss_artifact_id):
            raise ValueError(
                f"SS statistics artifact mismatch: {actual_artifact!r} != "
                f"{expected_ss_artifact_id!r}"
            )
    if expected_ss_step is not None and int(ss_vae.get("step", -1)) != int(expected_ss_step):
        raise ValueError(
            f"SS statistics checkpoint step mismatch: {ss_vae.get('step')} != {expected_ss_step}"
        )
    if expected_ss_encoder_sha256 is not None:
        actual = str(ss_vae.get("encoder_sha256", "")).lower()
        expected = str(expected_ss_encoder_sha256).lower()
        if actual != expected:
            raise ValueError(f"SS statistics encoder SHA-256 mismatch: {actual} != {expected}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Non-finite SS statistics at {path}")
    if torch.any(std <= 0):
        raise ValueError(f"SS statistics must have positive standard deviations at {path}")
    shape = (1, channels, 1, 1, 1)
    return mean.reshape(shape), torch.clamp(std.reshape(shape), min=1e-6)


class ObjectGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scene_scale = POS_MAX
        self.resolution = int(config.get("resolution", 1024))
        self.max_cond_len = int(config.get("max_cond_len", 512))
        self.context_encoder = get_context_encoder(config["context_encoder"])

        dino_config = config.get("dino") or {}
        self.dino_upsample = validate_dino_upsample(
            int(dino_config.get("upsample", 1)), int(dino_config.get("downsample", 16))
        )
        self.__dict__["_anyup_upsampler"] = None
        self.dino_model = None
        self.dino_transform = None
        if config.get("dino") is not None:
            self.dino_model, self.dino_transform = self.get_dino_model_and_transform()
            self.dino_model.eval()
            for parameter in self.dino_model.parameters():
                parameter.requires_grad = False

        flow_model_config = config["flow_model_ss_x2"]
        flow_trainer_config = config.get("flow_trainer_ss_x2", {})
        self.ss_channels = int(flow_model_config["in_channels"])
        self.ss_resolution = int(flow_model_config["resolution"])
        if self.ss_channels != 8 or self.ss_resolution != 2:
            raise ValueError(
                f"LC64 SS flow requires [8,2,2,2], got channels={self.ss_channels}, "
                f"resolution={self.ss_resolution}"
            )
        if int(flow_model_config["out_channels"]) != self.ss_channels:
            raise ValueError("SS flow input/output channel counts must match")

        ss_config = config.get("ss_x2", {})
        self.normalize_ss_x2 = bool(ss_config.get("normalize_target", True))
        mean, std = _load_ss_x2_stats(
            ss_config.get("latent_stats_path"),
            channels=self.ss_channels,
            resolution=self.ss_resolution,
            expected_ss_artifact_id=ss_config.get("artifact_id"),
            expected_ss_step=ss_config.get("step"),
            expected_ss_encoder_sha256=ss_config.get("encoder_sha256"),
        )
        self.register_buffer("ss_mean", mean, persistent=False)
        self.register_buffer("ss_std", std, persistent=False)

        self.flow_model_feat = FlowModel(**flow_model_config)
        self.flow_trainer_feat = FlowMatchingCFGTrainer(
            denoiser=self.flow_model_feat,
            trainer_name="ss_x2",
            t_schedule=flow_trainer_config.get(
                "t_schedule",
                {"name": "logitNormal", "args": {"mean": 0.0, "std": 1.0}},
            ),
            sigma_min=float(flow_trainer_config.get("sigma_min", 1e-5)),
            p_uncond=float(flow_trainer_config.get("p_uncond", 0.1)),
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
            raise ValueError("config.model.dino is required when colors are not provided")
        dino_downsample = int(self.config.get("dino", {}).get("downsample", 16))
        validate_dino_upsample(self.dino_upsample, dino_downsample)
        frames, height, width, _ = rgbs.shape
        output_height, output_width = height // dino_downsample, width // dino_downsample
        height, width = output_height * dino_downsample, output_width * dino_downsample
        rgbs = self.dino_transform(rgbs[:, :height, :width])
        dino_device = next(self.dino_model.parameters()).device
        if dino_device != rgbs.device:
            self.dino_model.to(rgbs.device)
        if self.dino_upsample == 1:
            outputs = self.dino_model(rgbs, is_training=True)
            return outputs["x_norm_patchtokens"].reshape(frames * output_height * output_width, -1)

        upsampler = self._get_anyup_upsampler(rgbs.device)
        image_downsample = dino_feature_downsample(self.dino_upsample, dino_downsample)
        anyup_height, anyup_width = height // image_downsample, width // image_downsample
        rgbs = rgbs[:, :, : anyup_height * image_downsample, : anyup_width * image_downsample]
        return run_dino_anyup_features(
            dino_model=self.dino_model,
            anyup_upsampler=upsampler,
            rgbs_tensor=rgbs,
            dino_downsample=dino_downsample,
            dino_upsample=self.dino_upsample,
            anyup_q_chunk_size=self.config.get("dino", {}).get("anyup_q_chunk_size"),
            anyup_tile_grid=int(self.config.get("dino", {}).get("anyup_tile_grid", 0)),
            anyup_frame_batch_size=int(
                self.config.get("dino", {}).get("anyup_frame_batch_size", 1)
            ),
        )

    def normalize_ss_latents(self, latents):
        expected = (self.ss_channels,) + (self.ss_resolution,) * 3
        if latents.ndim != 5 or tuple(latents.shape[1:]) != expected:
            raise ValueError(f"Expected SS targets [K,{','.join(map(str, expected))}], got {latents.shape}")
        if not torch.isfinite(latents).all():
            raise ValueError("SS targets contain non-finite values")
        if not self.normalize_ss_x2:
            return latents.float()
        return (latents.float() - self.ss_mean.float()) / self.ss_std.float()

    def denormalize_ss_latents(self, latents):
        if not self.normalize_ss_x2:
            return latents.float()
        return latents.float() * self.ss_std.float() + self.ss_mean.float()

    def forward(
        self,
        points,
        instance_ids,
        object_transforms,
        object_feats,
        object_coords,
        selected_indices,
        max_num_objects,
        colors=None,
        rgbs=None,
        valid_point_masks=None,
        return_denoised_latents=False,
        sample_method="random",
        prune_out_of_range=True,
        inference_num_steps=50,
        inference_only=False,
    ):
        if points.shape[0] != 1:
            raise ValueError("SS flow currently requires dataloader batch size 1")
        if object_coords.numel() != 0:
            raise ValueError("Dense SS flow expects an empty object_coords compatibility tensor")

        profile = self.profile_step_timing and torch.cuda.is_available()
        events = []

        def mark(name):
            if profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events.append((name, event))

        mark("start")
        voxel_coords = points[0]
        if colors is None:
            if rgbs is None:
                raise ValueError("Either colors or rgbs must be provided")
            voxel_feats = self.get_dino_feats(rgbs[0])
            if valid_point_masks is not None:
                voxel_feats = voxel_feats[valid_point_masks[0]]
        else:
            voxel_feats = colors[0]
        mark("dino_or_colors")

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
        mark("sample_object_feats")
        context_coords_h = F.pad(context_coords, (0, 1), value=1.0)
        context_coords = (context_coords_h @ object_transforms.transpose(-2, -1))[..., :3]
        if prune_out_of_range:
            context_mask = prune_out_of_range_points(context_coords, context_mask)
        context = self.context_encoder(context, context_coords, context_mask)
        context, context_mask = fix_empty_context(context, context_mask)
        mark("context_encoder")

        selected = selected_indices[0].long()
        context = context[selected]
        context_mask = context_mask[selected]
        if context.shape[0] != object_feats.shape[0]:
            raise ValueError(
                f"Condition/target count mismatch: conditions={context.shape[0]} "
                f"targets={object_feats.shape[0]}"
            )
        mark("condition_select")

        if inference_only:
            if not return_denoised_latents:
                raise ValueError("inference_only requires return_denoised_latents=True")
            predicted_normalized = self.flow_trainer_feat.run_inference(
                cond=context,
                cond_mask=context_mask,
                steps=inference_num_steps,
            )
            predicted = self.denormalize_ss_latents(predicted_normalized)
            return {
                "flow_metrics_dict": {},
                "flow_loss_dict": {},
                "feat_preds": predicted,
            }

        normalized_targets = self.normalize_ss_latents(object_feats)
        mark("ss_normalize")

        predicted = None
        if return_denoised_latents:
            predicted_normalized = self.flow_trainer_feat.run_inference(
                cond=context,
                cond_mask=context_mask,
                steps=inference_num_steps,
            )
            predicted = self.denormalize_ss_latents(predicted_normalized)
        loss, metrics = self.flow_trainer_feat.training_step(
            x_0=normalized_targets,
            cond=context,
            cond_mask=context_mask,
        )
        mark("flow_training")

        if profile:
            torch.cuda.synchronize()
            self._last_profile = {
                f"model/{events[index][0]}_to_{events[index + 1][0]}_ms": events[index][
                    1
                ].elapsed_time(events[index + 1][1])
                for index in range(len(events) - 1)
            }
            self._last_profile.update(
                {
                    "model/dino_or_colors_ms": self._last_profile.get(
                        "model/start_to_dino_or_colors_ms", 0.0
                    ),
                    "model/sample_object_feats_ms": self._last_profile.get(
                        "model/dino_or_colors_to_sample_object_feats_ms", 0.0
                    ),
                    "model/context_encoder_ms": self._last_profile.get(
                        "model/sample_object_feats_to_context_encoder_ms", 0.0
                    ),
                    "model/ss_x2_normalize_ms": self._last_profile.get(
                        "model/condition_select_to_ss_normalize_ms", 0.0
                    ),
                    "model/flow_training_ms": self._last_profile.get(
                        "model/ss_normalize_to_flow_training_ms", 0.0
                    ),
                }
            )

        result = {
            "flow_metrics_dict": metrics,
            "flow_loss_dict": {"loss_flow_feat": loss},
        }
        if return_denoised_latents:
            result["feat_preds"] = predicted
            result["shape_x2_target_feats"] = object_feats.float()
            result["ss_x2_target_latents"] = object_feats.float()
        return result
