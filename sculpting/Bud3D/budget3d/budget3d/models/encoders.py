"""The two frozen image encoders, exactly as Idol runs them.

Idol's own ``ImageEncoder`` is reused unchanged -- the DINOv3 LayerNorm, the FLUX.2 2x2
patchify, the VAE's BatchNorm statistics and the five zero tokens in front of feature2 all
matter, and reimplementing them is how a port silently drifts from the released model.
Only *where the weights come from* changes:

  DINOv3 ViT-H+/16  facebook/dinov3-vith16plus-pretrain-lvd1689m is gated with manual
                    approval.  The identical artefact (same sha-tagged .pth Meta released)
                    is converted to HF format by scripts/convert_dinov3_to_hf.py, which
                    verifies the forward pass against the reference logits shipped in
                    transformers' own conversion script.
  FLUX.2 VAE        black-forest-labs/FLUX.2-dev is gated too, but FLUX.2-klein-4B carries
                    the same VAE in its `vae/` subfolder, ungated: its config declares
                    `_name_or_path: black-forest-labs/FLUX.2-dev`, 32 latent channels and
                    the same 2x2 patch size.
"""
import os

import torch

from ..upstream import REPO_ROOT  # (importing upstream also puts Idol on sys.path)
from idol.conditioning import ImageEncoder as IdolImageEncoder

DINOV3_DIR = os.environ.get("BUDGET3D_DINOV3", os.path.join(
    REPO_ROOT, "weights", "dinov3", "vith16plus_lvd1689m"))
FLUX2_VAE_REPO = os.environ.get("BUDGET3D_FLUX2_VAE", "black-forest-labs/FLUX.2-klein-4B")


class IdolEncoder(IdolImageEncoder):
    """Idol's encoder with the VAE's *deterministic posterior mean*.

    The released sampler draws a sample from the VAE posterior, which injects a different
    noise realisation into the conditioning on every call.  For distillation the
    conditioning should be a fixed function of the image -- the teacher's targets and the
    student's input have to describe the same conditioning -- so the mean is used, as the
    specification requires.  Everything else is Idol's own function, unchanged.
    """

    deterministic_vae = True

    def _flux2_tokens(self, images):
        if not self.deterministic_vae:
            return super()._flux2_tokens(images)
        vae = self.flux2_vae
        latents = vae.encode(images.to(torch.bfloat16) * 2 - 1).latent_dist.mode()   # <- mean, not sample
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1)
                         + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
        padding = torch.zeros(batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
        return torch.cat([padding, tokens], dim=1)


def build_encoder(device="cuda", dinov3_dtype="bf16", deterministic_vae=True):
    """The frozen (DINOv3, FLUX.2 VAE) pair Idol conditions on."""
    if not os.path.isdir(DINOV3_DIR):
        raise FileNotFoundError(
            f"converted DINOv3 not found at {DINOV3_DIR}; run\n"
            f"  python scripts/convert_dinov3_to_hf.py --model-name vith16plus_lvd1689m "
            f"--save-dir weights/dinov3")
    enc = IdolEncoder(DINOV3_DIR, FLUX2_VAE_REPO, dinov3_dtype=dinov3_dtype, device=device)
    enc.deterministic_vae = deterministic_vae
    return enc
