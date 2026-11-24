# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LicenseRef-SAM-License
# From SAM 3D Body (https://github.com/facebookresearch/sam-3d-body), modified for FAH; licence: third_party_licenses/SAM-License.txt

from .prompt_encoder import PromptEncoder
from .promptable_decoder import PromptableDecoder


def build_decoder(cfg, context_dim=None):
    if cfg.TYPE == "sam":
        return PromptableDecoder(
            dims=cfg.DIM,
            context_dims=context_dim,
            depth=cfg.DEPTH,
            num_heads=cfg.HEADS,
            head_dims=cfg.DIM_HEAD,
            mlp_dims=cfg.MLP_DIM,
            layer_scale_init_value=cfg.LAYER_SCALE_INIT,
            drop_rate=cfg.DROP_RATE,
            attn_drop_rate=cfg.ATTN_DROP_RATE,
            drop_path_rate=cfg.DROP_PATH_RATE,
            ffn_type=cfg.FFN_TYPE,
            enable_twoway=cfg.ENABLE_TWOWAY,
            repeat_pe=cfg.REPEAT_PE,
            frozen=cfg.get("FROZEN", False),
            do_interm_preds=cfg.get("DO_INTERM_PREDS", False),
            do_keypoint_tokens=cfg.get("DO_KEYPOINT_TOKENS", False),
            keypoint_token_update=cfg.get("KEYPOINT_TOKEN_UPDATE", None),
        )
    else:
        raise ValueError("Invalid decoder type: ", cfg.TYPE)
