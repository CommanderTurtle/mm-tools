# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VQ-VAE decoding shared by SOMA and G1, compatible with their checkpoints."""

from __future__ import annotations

from typing import List, Optional

import torch as t
from torch import nn

from .features import (
    convert_sparse_cond_to_dense_cond_if_needed,
    extract_feature_from_motion_rep,
)
from .layers import DoubleCondDecoder, Encoder
from .quantizer import QuantizeEMAResetMultiHead, resolve_multihead_vq_dimensions


class VQVAE(nn.Module):
    """VQ-VAE that turns motion tokens back into motion features.

    Only its decoder runs at export.
    """

    ALLOWED_FEATURE_MODE = [
        "invalid",
        "pose",
        "root",
        "root_without_hip_height",
        "root_without_hip_height_without_heading",
        "root_without_heading",
        "root_without_hip_height_without_heading_with_mask",
        "joint_positions_and_rotations",
        "joint_positions_and_rotations_and_foot_contact",
        "joint_positions_and_rotations_and_hip_height",
    ]

    def __init__(
        self,
        pose_root_mode: str,
        motion_rep: nn.Module,
        feature_mode: Optional[List[str]] = None,
        nb_code: Optional[int] = None,
        code_dim: Optional[int] = None,
        output_emb_width: Optional[int] = None,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        num_heads: int = 4,
        codes_per_head: Optional[int] = None,
        code_dim_per_head: Optional[int] = None,
        norm: Optional[str] = None,
        cond_fusion_last_layer: bool = False,
        decoder_type: type[nn.Module] = DoubleCondDecoder,
    ):
        super().__init__()
        if feature_mode is None or len(feature_mode) != 4:
            raise ValueError("feature_mode must contain exactly four feature names.")
        invalid_features = [
            feature
            for feature in feature_mode
            if feature not in self.ALLOWED_FEATURE_MODE
        ]
        if invalid_features:
            raise ValueError(f"Unsupported feature modes: {invalid_features}.")
        if pose_root_mode not in ("pose", "root"):
            raise ValueError("pose_root_mode must be 'pose' or 'root'.")
        if codes_per_head is None and nb_code is None:
            nb_code = 512
        if code_dim_per_head is None and code_dim is None:
            code_dim = 512
        quantizer_dimensions = resolve_multihead_vq_dimensions(
            num_heads=num_heads,
            codes_per_head=codes_per_head,
            code_dim_per_head=code_dim_per_head,
            nb_code=nb_code,
            code_dim=code_dim,
        )
        if output_emb_width is None:
            output_emb_width = quantizer_dimensions.latent_dim
        if output_emb_width != quantizer_dimensions.latent_dim:
            raise ValueError(
                f"output_emb_width={output_emb_width} must match the resolved latent dimension "
                f"{quantizer_dimensions.latent_dim}."
            )
        self.code_dim = quantizer_dimensions.latent_dim
        self.num_code = quantizer_dimensions.combined_code_space
        self._motion_rep = motion_rep
        self._pose_root_mode = pose_root_mode
        self._down_t = down_t
        (
            self.encoder_input_feature_mode,
            self.decoder_input_feature_mode,
            self.decoder_target_cond_feature_mode,
            self.decoder_external_cond_feature_mode,
        ) = feature_mode
        example = t.zeros([1, 1, len(motion_rep.indices["all"])])
        encoder_dim, decoder_dim, target_dim, external_dim = (
            int(extract_feature_from_motion_rep(example, motion_rep, feature).shape[-1])
            for feature in feature_mode
        )
        self.encoder = Encoder(
            encoder_dim,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = decoder_type(
            decoder_dim,
            output_emb_width,
            down_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
            target_cond_dim=target_dim,
            external_cond_dim=external_dim,
            cond_fusion_last_layer=cond_fusion_last_layer,
        )
        self.quantizer = QuantizeEMAResetMultiHead(
            self.num_code, self.code_dim, num_heads
        )

    def extract_feature(self, x: t.Tensor, feature: str = ""):
        """Select a feature with extract_feature_from_motion_rep().

        An empty feature selects pose_root_mode.
        """
        feature = self._pose_root_mode if feature == "" else feature
        return extract_feature_from_motion_rep(x, self._motion_rep, feature)

    def forward_decoder(
        self,
        x,
        target_cond: t.Tensor,
        has_target_cond: t.Tensor = None,
        external_cond: t.Tensor = None,
        use_overall_indices: bool = True,
        token_mask: t.Tensor = None,
    ):
        """Decode tokens into motion features, conditioned on the target poses."""
        num_expected_frames = x.shape[1] * 2**self._down_t
        x_d = self.quantizer.dequantize(x, use_overall_indices=use_overall_indices)
        x_quantized = x_d.permute(0, 2, 1).contiguous()
        if target_cond is not None:
            target_cond, has_target_cond = convert_sparse_cond_to_dense_cond_if_needed(
                target_cond, has_target_cond, num_expected_frames
            )
            target_cond = self.extract_feature(
                target_cond, self.decoder_target_cond_feature_mode
            )
        x_decoder = self.decoder(
            x_quantized,
            external_cond,
            target_cond,
            has_target_cond,
            token_mask=token_mask,
        )
        return {"recon_state": x_decoder.permute(0, 2, 1)}

    @property
    def pose_root_mode(self):
        """What the VQ-VAE models: "pose" or "root"."""
        return self._pose_root_mode

    @property
    def motion_rep(self):
        """Motion feature layout of the decoded features."""
        return self._motion_rep
