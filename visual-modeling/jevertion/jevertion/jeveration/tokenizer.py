"""From codes to pixels: the TA-TiTok codebook, the CLIP text tower the decoder reads, and the two 512-px decoders.

    plain decoder     codes + text -> image                    (text-to-image)
    source decoder    codes + text + source image -> image     (image-to-image)
"""
import json
import os
import sys

import torch
import torch.nn as nn

from . import config
from .backbone import resolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "tatitok"))
from modeling.modules.blocks import TATiTokDecoder   # noqa: E402  (vendored, see third_party/tatitok)


def _fetch(source, default, filename):
    """a file of a local folder, or of a Hub repository at its pinned revision"""
    from huggingface_hub import hf_hub_download
    repo, revision = resolve(source, default)
    return os.path.join(repo, filename) if os.path.isdir(repo) else hf_hub_download(repo, filename, revision=revision)


class SourceDecoder(TATiTokDecoder):
    """TA-TiTok's text-aware decoder with a gated branch that reads the source image"""

    def __init__(self, cfg, n_tasks=4, gate_every=4):
        super().__init__(cfg)
        assert self.patch_size == 16
        self.cond_gate_every = gate_every
        self.cond_in = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
                                     nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(), nn.Conv2d(256, self.width, 2, 2))
        self.cond_task = nn.Embedding(n_tasks + 1, self.width)
        gates = nn.ModuleList()
        for _ in range(len(range(gate_every - 1, self.num_layers, gate_every))):
            m = nn.Module()
            m.gx, m.gc, m.proj = nn.Linear(self.width, 1), nn.Linear(self.width, 1), nn.Linear(self.width, self.width)
            gates.append(m)
        self.cond_gates = gates

    def forward(self, z_quantized, text_guidance, cond, task):
        """z_quantized [B, C, 1, N] code vectors, text_guidance [B, 77, 768], cond [B, 3, 512, 512] the source image in [0, 1],
        task [B] its task id (> 0) -> image [B, 3, 512, 512]"""
        n, c, h, w = z_quantized.shape
        assert h == 1 and w == self.num_latent_tokens, f"{h}, {w}, {self.num_latent_tokens}"
        x = self.decoder_embed(z_quantized.reshape(n, c * h, w).permute(0, 2, 1))
        seq_len = x.shape[1]
        mask_tokens = self.mask_token.repeat(n, self.grid_size ** 2, 1).to(x.dtype)
        # contiguous: the standard memory layout, whatever layout the caller's tensor has
        ce = self.cond_in(cond.to(self.cond_task.weight.dtype).contiguous()).flatten(2).transpose(1, 2)   # [B, grid^2, W]
        ce = (ce + self.cond_task(task).unsqueeze(1)).transpose(0, 1)                                 # [grid^2, B, W]
        cls = self.class_embedding.unsqueeze(0).expand(n, -1, -1).to(mask_tokens.dtype)
        mask_tokens = torch.cat([cls, mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        tg = self.text_guidance_proj(text_guidance) + self.text_guidance_positional_embedding
        x = self.ln_pre(torch.cat([x, tg], dim=1)).permute(1, 0, 2)
        g2, k = self.grid_size ** 2, 0
        for i in range(self.num_layers):
            x_in = x
            x = self.transformer[i](x)
            if (i + 1) % self.cond_gate_every == 0:
                m = self.cond_gates[k]
                k += 1
                sx = torch.sigmoid(m.gx(x_in[1:1 + g2]))               # how much of the patch token to replace
                sc = torch.sigmoid(m.gc(ce))                            # how much of the source feature to let in
                hq = x[1:1 + g2] * (1 - sx) + sc * m.proj(ce)
                x = torch.cat([x[:1], hq.to(x.dtype), x[1 + g2:]], 0)
        x = self.ln_post(x.permute(1, 0, 2)[:, 1:1 + g2])
        x = x.permute(0, 2, 1).reshape(n, self.width, self.grid_size, self.grid_size)
        return self.conv_out(self.ffn(x.contiguous()))


def load_decoder_weights(decoder, path):
    """Load a decoder checkpoint (its "ema_plain" entry, or a bare state dict). A source-decoder checkpoint also serves the
    plain decoder: its `cond_*` entries are dropped."""
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    sd = sd.get("ema_plain", sd)
    if not isinstance(decoder, SourceDecoder):
        sd = {k: v for k, v in sd.items() if not k.startswith("cond_")}
    decoder.load_state_dict(sd, strict=True)
    return decoder.eval().requires_grad_(False)


class ImageCodec(nn.Module):
    """codebook + CLIP text tower + the 512-px decoders; frozen"""

    def __init__(self, tatitok=None, clip=None):
        """tatitok, clip: local folders of TA-TiTok VQ-128 and CLIP ViT-L/14-336 (None: the public repositories)"""
        super().__init__()
        import open_clip
        from omegaconf import OmegaConf
        from safetensors import safe_open
        cfg = OmegaConf.create(json.load(open(_fetch(tatitok, config.TATITOK, "config.json"))))
        assert cfg.model.vq_model.num_latent_tokens == config.NUM_CODES and cfg.model.vq_model.codebook_size == config.CODEBOOK_SIZE
        cfg.dataset.preprocessing.crop_size = config.IMAGE_SIZE                     # the decoders' output size
        self.cfg = cfg
        with safe_open(_fetch(tatitok, config.TATITOK, "model.safetensors"), "pt") as f:
            self.register_buffer("codebook", f.get_tensor("quantize.embedding.weight").float(), persistent=False)
        # The OpenAI CLIP weights were trained with QuickGELU; open_clip applies it only when asked.
        model, _, _ = open_clip.create_model_and_transforms(config.CLIP_ARCH, force_quick_gelu=True,
                                                            pretrained=_fetch(clip, config.CLIP_TEXT, "open_clip_model.safetensors"))
        del model.visual
        model.transformer.batch_first = False
        self.clip, self.clip_tokenizer = model, open_clip.get_tokenizer(config.CLIP_ARCH)
        self.decoder = self.source_decoder = None
        self.eval().requires_grad_(False)

    def load_decoder(self, path):
        self.decoder = load_decoder_weights(TATiTokDecoder(self.cfg), path).to(self.codebook.device)

    def load_source_decoder(self, path):
        self.source_decoder = load_decoder_weights(SourceDecoder(self.cfg), path).to(self.codebook.device)

    @torch.no_grad()
    def text_guidance(self, texts):
        """list of B strings -> [B, 77, 768]: the per-token output of the CLIP text tower"""
        clip = self.clip
        ids = self.clip_tokenizer(texts).to(self.codebook.device)
        dtype = clip.transformer.get_cast_dtype()
        x = clip.token_embedding(ids).to(dtype) + clip.positional_embedding.to(dtype)
        x = clip.transformer(x.permute(1, 0, 2), attn_mask=clip.attn_mask).permute(1, 0, 2)
        return clip.ln_final(x)

    @torch.no_grad()
    def decode(self, codes, texts, source=None, task=None):
        """codes [B, N] + the decoder's texts (+ source images [B, 3, 512, 512] in [0, 1] with their task ids [B])
        -> images [B, 3, 512, 512], float32, not clamped"""
        z = self.codebook[codes].float().permute(0, 2, 1).unsqueeze(2)              # [B, C, 1, N], the decoder's layout
        tg = self.text_guidance(texts).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            if source is None:
                return self.decoder(z, tg).float()
            return self.source_decoder(z, tg, source, task).float()
