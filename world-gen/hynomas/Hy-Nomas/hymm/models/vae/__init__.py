"""Load the 32-channel DCAE-V used for training and standard inference."""

import torch



def load_vae(
    vae_type: str = "dcae-v-f32t4c32",
    vae_precision: str = None,
    vae_path: str = None,
    logger=None,
    device=None,
):
    """Return DCAE-V, its checkpoint path, and spatial/temporal compression ratios.

    The optional PixelUMM decoder is loaded separately by the inference pipeline.
    """
    if vae_type != "dcae-v-f32t4c32":
        raise ValueError(
            f"Unsupported VAE type: {vae_type!r}; expected 'dcae-v-f32t4c32'"
        )

    if not vae_path:
        raise ValueError("model.vae.checkpoint is required for DCAE-V")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if vae_precision is not None and vae_precision not in dtype_map:
        raise ValueError(f"Unsupported VAE precision: {vae_precision!r}")
    dtype = dtype_map.get(vae_precision, torch.bfloat16)

    if logger is not None:
        logger.info(f"Loading DCAE-V ({vae_type}) from: {vae_path}")

    from .autoencoder_dcae_v import AutoencoderDCAEV

    vae = AutoencoderDCAEV.from_checkpoint(
        checkpoint_path=vae_path,
        dtype=dtype,
        device=device if device is not None else "cpu",
    )
    vae.requires_grad_(False)
    vae.eval()

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    return (
        vae,
        vae_path,
        vae.spatial_compression_ratio,
        vae.time_compression_ratio,
    )
