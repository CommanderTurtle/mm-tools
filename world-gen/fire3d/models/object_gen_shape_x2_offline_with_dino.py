"""Offline Shape VAE X2 flow model with DINO/context conditioning.

This variant trains on precomputed raw LC512 Shape VAE X2 slats.  The model
only applies the 512-channel train-set normalization before flow matching.
"""

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
from models.flow_model_sparse import ElasticSLatFlowModel
from models.flow_trainer_sparse import SparseFlowMatchingCFGTrainer
from models.dino_anyup import (
    dino_feature_downsample,
    load_anyup_upsampler,
    move_module_to_device,
    run_dino_anyup_features,
    validate_dino_upsample,
)
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
            raise ValueError(f"Expected input shape [n, h, w] or [n, h, w, 3], got {tensor.shape}")
        tensor = tensor.permute(0, 3, 1, 2)
        return normalize(tensor)

    return transform_fn


def _load_shape_x2_stats(path, channels):
    if path is None:
        raise ValueError("Offline Shape X2 training requires model.shape_x2.x2_latent_stats_path")
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing Shape VAE X2 latent stats: {stats_path}")
    with stats_path.open("r") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(f"Expected {channels}-channel x2 stats, got mean={mean.numel()} std={std.numel()}")
    return mean.reshape(1, channels), torch.clamp(std.reshape(1, channels), min=1e-6)


class ObjectGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scene_scale = POS_MAX
        self.resolution = config.get("resolution", 1024)
        self.max_denoise_objects = config.get("max_denoise_objects", 64)
        self.voxelize = InstanceVoxelize(
            voxel_size=self.scene_scale / self.resolution,
            resolution=self.resolution,
            apply_standardization=False,
        )
        self.max_cond_len = config.get("max_cond_len", 64)

        self.context_encoder = get_context_encoder(config["context_encoder"])
        dino_config = config.get("dino") or {}
        self.dino_upsample = validate_dino_upsample(
            int(dino_config.get("upsample", 1)),
            int(dino_config.get("downsample", 16)),
        )
        self.__dict__["_anyup_upsampler"] = None
        self.dino_model = None
        self.dino_transform = None
        if config.get("dino", None) is not None:
            self.dino_model, self.dino_transform = self.get_dino_model_and_transform()
            self.dino_model.eval()
            for p in self.dino_model.parameters():
                p.requires_grad = False

        flow_model_cfg = config.get("flow_model_shape_x2") or config["flow_model_feat"]
        flow_trainer_cfg = config.get("flow_trainer_shape_x2") or config["flow_trainer_feat"]
        self.shape_x2_channels = int(flow_model_cfg.get("in_channels", 512))

        shape_x2_cfg = config.get("shape_x2", {})
        legacy_shape_vae_x2_cfg = config.get("shape_vae_x2", {})
        stats_path = shape_x2_cfg.get(
            "x2_latent_stats_path",
            legacy_shape_vae_x2_cfg.get("x2_latent_stats_path"),
        )
        self.normalize_shape_x2 = bool(shape_x2_cfg.get("normalize_target", True))
        x2_mean, x2_std = _load_shape_x2_stats(stats_path, self.shape_x2_channels)
        self.register_buffer("x2_mean", x2_mean, persistent=False)
        self.register_buffer("x2_std", x2_std, persistent=False)

        self.flow_model_feat = ElasticSLatFlowModel(**flow_model_cfg)
        self.flow_trainer_feat = SparseFlowMatchingCFGTrainer(
            denoiser=self.flow_model_feat,
            trainer_name="shape_x2",
            p_uncond=flow_trainer_cfg["p_uncond"],
        )
        self.profile_step_timing = os.environ.get("FF_PROFILE_STEP_TIMING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_profile = {}

    @torch.no_grad()
    def get_dino_model_and_transform(self):
        model_path = self.config["dino"]["model_path"]
        repo_dir = self.config["dino"]["repo_dir"]
        model_name = self.config["dino"].get("model_name", "dinov3_vitl16")
        model = torch.hub.load(repo_dir, model_name, source="local", weights=model_path)
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
        n, h, w, _ = rgbs.shape
        output_h, output_w = h // dino_downsample, w // dino_downsample
        h, w = output_h * dino_downsample, output_w * dino_downsample
        rgbs = rgbs[:, :h, :w]
        rgbs = self.dino_transform(rgbs)
        dino_device = next(self.dino_model.parameters()).device
        if dino_device != rgbs.device:
            self.dino_model.to(rgbs.device)
        if self.dino_upsample == 1:
            outputs = self.dino_model(rgbs, is_training=True)
            feats = outputs["x_norm_patchtokens"]
            return feats.reshape(n * output_h * output_w, -1)

        anyup_upsampler = self._get_anyup_upsampler(rgbs.device)
        image_downsample = dino_feature_downsample(self.dino_upsample, dino_downsample)
        anyup_output_h, anyup_output_w = h // image_downsample, w // image_downsample
        rgbs = rgbs[:, :, : anyup_output_h * image_downsample, : anyup_output_w * image_downsample]
        return run_dino_anyup_features(
            dino_model=self.dino_model,
            anyup_upsampler=anyup_upsampler,
            rgbs_tensor=rgbs,
            dino_downsample=dino_downsample,
            dino_upsample=self.dino_upsample,
            anyup_q_chunk_size=self.config.get("dino", {}).get("anyup_q_chunk_size"),
            anyup_tile_grid=int(self.config.get("dino", {}).get("anyup_tile_grid", 0)),
            anyup_frame_batch_size=int(self.config.get("dino", {}).get("anyup_frame_batch_size", 1)),
        )

    def _normalize_shape_x2_feats(self, object_feats):
        if object_feats.ndim != 2 or object_feats.shape[1] != self.shape_x2_channels:
            raise ValueError(f"Expected Shape VAE X2 feats [N,{self.shape_x2_channels}], got {tuple(object_feats.shape)}")
        if not self.normalize_shape_x2:
            return object_feats
        mean = self.x2_mean.to(device=object_feats.device, dtype=torch.float32)
        std = self.x2_std.to(device=object_feats.device, dtype=torch.float32)
        return (object_feats.float() - mean) / std

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
        """
        Args:
            object_feats: raw Shape VAE X2 LC512 feats (sum_i N_i, 512), float
            object_coords: batched sparse coords (sum_i N_i, 4), int32
        """

        profile = self.profile_step_timing and torch.cuda.is_available()
        profile_events = []

        def mark_profile(name):
            if profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                profile_events.append((name, event))

        mark_profile("start")

        batch_size = points.shape[0]
        assert batch_size == 1, "Batch size must be 1 for ObjectGen"

        voxel_coords = points[0]
        if colors is None:
            if rgbs is None:
                raise ValueError("Either colors or rgbs must be provided")
            dino_feats = self.get_dino_feats(rgbs[0])
            if valid_point_masks is not None:
                dino_feats = dino_feats[valid_point_masks[0]]
            voxel_feats = dino_feats
        else:
            voxel_feats = colors[0]
        mark_profile("dino_or_colors")
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
        mark_profile("sample_object_feats")

        context_coords_h = F.pad(context_coords, (0, 1), value=1.0)
        context_coords = (context_coords_h @ object_transforms.transpose(-2, -1))[..., :3]
        mark_profile("context_transform")

        if prune_out_of_range:
            context_mask = prune_out_of_range_points(context_coords, context_mask)
        context = self.context_encoder(context, context_coords, context_mask)
        context, context_mask = fix_empty_context(context, context_mask)
        mark_profile("context_encoder")

        selected_indices = selected_indices[0]
        context = context[selected_indices]
        context_mask = context_mask[selected_indices]

        lengths = context_mask.sum(dim=1)
        valid = context[context_mask]
        cond = list(torch.split(valid, lengths.cpu().tolist(), dim=0))
        mark_profile("cond_pack")

        if inference_only:
            if not return_denoised_latents:
                raise ValueError("inference_only requires return_denoised_latents=True")
            # Sparse flow inference replaces the feature values with Gaussian noise but
            # preserves these generated SS coordinates. Placeholder values therefore do
            # not carry target shape information.
            topology = sp.SparseTensor(feats=object_feats.float(), coords=object_coords)
            feats_preds = self.flow_trainer_feat.run_inference(
                x_0=topology,
                cond=cond,
                steps=inference_num_steps,
            ).feats
            return {
                "flow_metrics_dict": {},
                "flow_loss_dict": {},
                "feat_preds": feats_preds,
                "shape_x2_target_coords": topology.coords,
            }

        shape_x2_feats = self._normalize_shape_x2_feats(object_feats)
        shape_x2_sparse = sp.SparseTensor(
            feats=shape_x2_feats,
            coords=object_coords,
        )
        mark_profile("shape_x2_normalize")

        if return_denoised_latents:
            feats_preds = self.flow_trainer_feat.run_inference(
                x_0=shape_x2_sparse,
                cond=cond,
                steps=inference_num_steps,
            )
            feats_preds = feats_preds.feats

        loss_flow_feat, metrics_dict_feat = self.flow_trainer_feat.training_step(
            x_0=shape_x2_sparse,
            cond=cond,
        )
        mark_profile("flow_training")
        if profile:
            torch.cuda.synchronize()
            self._last_profile = {
                f"model/{profile_events[i][0]}_to_{profile_events[i + 1][0]}_ms": profile_events[i][1].elapsed_time(profile_events[i + 1][1])
                for i in range(len(profile_events) - 1)
            }
            self._last_profile.update({
                "model/dino_or_colors_ms": self._last_profile.get("model/start_to_dino_or_colors_ms", 0.0),
                "model/sample_object_feats_ms": self._last_profile.get("model/dino_or_colors_to_sample_object_feats_ms", 0.0),
                "model/context_transform_ms": self._last_profile.get("model/sample_object_feats_to_context_transform_ms", 0.0),
                "model/context_encoder_ms": self._last_profile.get("model/context_transform_to_context_encoder_ms", 0.0),
                "model/cond_pack_ms": self._last_profile.get("model/context_encoder_to_cond_pack_ms", 0.0),
                "model/shape_x2_encode_ms": self._last_profile.get("model/cond_pack_to_shape_x2_normalize_ms", 0.0),
                "model/flow_training_ms": self._last_profile.get("model/shape_x2_normalize_to_flow_training_ms", 0.0),
            })

        flow_metrics_dict = {}
        flow_metrics_dict.update(metrics_dict_feat)

        flow_loss_dict = {
            "loss_flow_feat": loss_flow_feat,
        }

        flow_results_dict = {
            "flow_metrics_dict": flow_metrics_dict,
            "flow_loss_dict": flow_loss_dict,
        }
        if return_denoised_latents:
            flow_results_dict["feat_preds"] = feats_preds
            flow_results_dict["shape_x2_target_feats"] = shape_x2_sparse.feats
            flow_results_dict["shape_x2_target_coords"] = shape_x2_sparse.coords

        return flow_results_dict
