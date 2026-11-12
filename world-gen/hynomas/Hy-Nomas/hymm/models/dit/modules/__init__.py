# Base config shared by all Nomas-G/2-16164 variants
_NOMAS_G_16164_BASE = {
    "in_channels": 32,
    "out_channels": 32,
    "mm_double_blocks_depth": 54,
    "mm_single_blocks_depth": 0,
    "rope_dim_list": [16, 56, 56],
    "hidden_size": 2048,
    "heads_num": 16,
    "mlp_width_ratio": 4,
    "patch_size": [1, 1, 1],
}

_NOMAS_G_32324_BASE = {
    **_NOMAS_G_16164_BASE,
}

MODEL_CONFIG = {
    "HY-Nomas-G/2-16164-MMDiT": _NOMAS_G_16164_BASE,
    "HY-Nomas-G/2-32324-MMDiT": _NOMAS_G_32324_BASE,
}


def load_model(model_type, args, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    factor_kwargs["attn_mode"] = args.dit.attn_mode
    factor_kwargs["seg_ar_attention_backend"] = getattr(
        args.dit, "seg_ar_attention_backend", "magi"
    )

    if "16164" in model_type or "32324" in model_type:
        factor_kwargs["is_reshape_temporal_channels"] = False
    else:
        factor_kwargs["is_reshape_temporal_channels"] = True
    if args.dit.attn_mode != "flash" and args.dit.attn_mode != "flash3":
        # block attn
        factor_kwargs["ba_attn_chunk_size"] = args.dit.ba_attn_chunk_size
        factor_kwargs["ba_attn_topk"] = args.dit.ba_attn_topk
        factor_kwargs["ba_attn_causal"] = args.dit.ba_attn_causal
        factor_kwargs["ba_attn_slice_method"] = args.dit.ba_attn_slice_method
        factor_kwargs["ba_attn_full_attn_block_num"] = (
            args.dit.ba_attn_full_attn_block_num
        )
        factor_kwargs["ba_attn_full_attn_every_n_blocks"] = (
            args.dit.ba_attn_full_attn_every_n_blocks
        )

        # ssta & sta (only needed for ssta-related modes)
        if args.dit.attn_mode in ("sta_flex", "ssta_flex", "ptm_sparse_attn"):
            factor_kwargs["win_size"] = [
                int(s) for s in str(args.dit.win_size).split("_")
            ]
            factor_kwargs["win_type"] = args.dit.win_type
            factor_kwargs["win_ratio"] = args.dit.win_ratio
            factor_kwargs["tile_size"] = [
                int(s) for s in str(args.dit.tile_size).split("_")
            ]
            factor_kwargs["ssta_topk"] = args.dit.ssta_topk
            factor_kwargs["ssta_threshold"] = args.dit.ssta_threshold
            factor_kwargs["ssta_enforce_share_topk_block"] = (
                args.dit.ssta_enforce_share_topk_block
            )
            factor_kwargs["ssta_lambda"] = getattr(args, "ssta_lambda", 0.9)
            factor_kwargs["ssta_sampling_type"] = getattr(
                args, "ssta_sampling_type", "similarity"
            )
            factor_kwargs["ssta_adaptive_pool"] = getattr(args, "ssta_adaptive_pool", 0)
            factor_kwargs["ptm_attn_sparse_type"] = args.dit.ptm_attn_sparse_type
            factor_kwargs["ptm_attn_pad_type"] = args.dit.ptm_attn_pad_type
            factor_kwargs["ptm_attn_use_text_mask"] = args.dit.ptm_attn_use_text_mask
            factor_kwargs["ptm_attn_mask_share_within_head"] = (
                args.dit.ptm_attn_mask_share_within_head
            )
            factor_kwargs["df_magi_topk"] = getattr(args, "df_magi_topk", 0)
            factor_kwargs["df_magi_threshold"] = getattr(args, "df_magi_threshold", 0.0)
            factor_kwargs["df_magi_share_within_head"] = getattr(
                args, "df_magi_share_within_head", 1
            )
            factor_kwargs["df_magi_causal"] = getattr(args, "df_magi_causal", 1)
            factor_kwargs["df_avg_text_tokens"] = getattr(args, "df_avg_text_tokens", 0)

    if args.dit.use_moe:
        from .moe.hunyuan_moe import MoeConfig

        moe_config = MoeConfig()
        moe_config.num_experts = args.dit.num_experts
        moe_config.use_mixed_mlp_moe = args.dit.use_mixed_mlp_moe
        moe_config.moe_topk = args.dit.moe_topk
        moe_config.capacity_factor = args.dit.capacity_factor
        moe_config.hidden_act = args.dit.act_type
        moe_config.intermediate_size = (
            MODEL_CONFIG[model_type]["hidden_size"]
            * MODEL_CONFIG[model_type]["mlp_width_ratio"]
        )
        moe_config.hidden_size = MODEL_CONFIG[model_type]["hidden_size"]
        moe_config.moe_drop_tokens = args.dit.moe_drop_tokens
        moe_config.moe_random_routing_dropped_token = (
            args.dit.moe_random_routing_dropped_token
        )
        factor_kwargs["moe_config"] = moe_config

    if "Nomas" in model_type:
        factor_kwargs["text_states_dim"] = args.text_encoder.text_states_dim

    if model_type not in MODEL_CONFIG:
        raise NotImplementedError(
            f"model_type={model_type} not in {list(MODEL_CONFIG.keys())}"
        )

    if "Nomas" not in model_type:
        raise NotImplementedError(f"model_type={model_type} is not a Nomas variant")

    factor_kwargs["unpad_zero_tokens"] = getattr(args.dit, "unpad_zero_tokens", True)
    # Plan A: fixed max segment count for the optional learnable per-segment
    # text PE inside SingleTokenRefinerPerBlock. 0 = disabled (default).
    factor_kwargs["text_seg_pos_emb_num"] = int(
        getattr(args, "text_seg_pos_emb_num", 0)
    )

    # ======================== Camera Control ========================
    if getattr(args.camera, "use_camera_control", False):
        camera_injection_mode = getattr(args.camera, "injection_mode", None)
        # Valid options: "action_attn" | "extrinsics_attn" | "residual_norm"
        assert camera_injection_mode is not None, (
            "camera.injection_mode must be set when camera.use_camera_control is True. "
            'Valid options: "action_attn", "extrinsics_attn", "residual_norm"'
        )
        factor_kwargs["camera_injection_mode"] = camera_injection_mode
        model_config = MODEL_CONFIG[model_type]

        if camera_injection_mode in ("action_attn", "extrinsics_attn"):
            keyboard_dim_in = 8 if camera_injection_mode == "action_attn" else 64
            action_config = {
                "keyboard_dim_in": keyboard_dim_in,
                "hidden_size": 128,
                "img_hidden_size": model_config["hidden_size"],
                "keyboard_hidden_dim": 1024,
                "heads_num": 16,
                "patch_size": model_config.get("patch_size", [1, 1, 1]),
                "qk_norm": True,
                "qk_norm_type": "rms",
                "qkv_bias": False,
                "rope_dim_list": [8, 28, 28],
                "rope_theta": 256,
                # Per-segment block-diagonal cross-attention via
                # MagiAttention's flex_flash_attn_func. False = legacy
                # "image sees all keyboard" flash_attn path. Requires
                # multi-prompt segment boundaries (prompt_video_boundaries)
                # to be supplied at runtime, which nomas_cpt.forward already
                # forwards to ActionModule when available.
                "per_segment_camera_attn": bool(
                    getattr(args.camera, "per_segment_camera_attn", False)
                ),
                # Per-chunk (fixed latent-frame chunk) camera cross-attention.
                # Overrides the segment boundaries used by the camera cross-attn
                # with [0, cs, 2*cs, ..., T_lat] chunks; gate / DF grouping stay
                # per-segment. See ActionModule for details.
                "per_chunk_camera_attn": bool(
                    getattr(args.camera, "per_chunk_camera_attn", False)
                ),
                "camera_chunk_size": int(getattr(args.camera, "camera_chunk_size", 5)),
            }
            action_config["seg_ar_attention_backend"] = factor_kwargs[
                "seg_ar_attention_backend"
            ]
            factor_kwargs["action_config"] = action_config

    # ======================== Unified Model ========================
    from ..nomas_cpt import CPTHYNomasDiffusionTransformer

    model = CPTHYNomasDiffusionTransformer(
        **MODEL_CONFIG[model_type],
        **factor_kwargs,
    )

    return model
