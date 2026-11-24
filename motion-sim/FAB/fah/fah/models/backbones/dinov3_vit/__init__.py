# SPDX-License-Identifier: LicenseRef-DINOv3-License
# Minimal vendored DINOv3 ViT (facebookresearch/dinov3, DINOv3 License: LICENSE.md in this folder)
from .vision_transformer import DinoVisionTransformer

# hub/backbones.py constructor arguments for the architectures used by SAM 3D Body
ARCHS = {
    "dinov3_vith16plus": dict(
        img_size=224, patch_size=16, in_chans=3, pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32", embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6.0,
        qkv_bias=True, drop_path_rate=0.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
        ffn_layer="swiglu", ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
    ),
}


def build_dinov3(name, **overrides):
    kwargs = dict(ARCHS[name])
    kwargs.update(overrides)
    return DinoVisionTransformer(**kwargs)
