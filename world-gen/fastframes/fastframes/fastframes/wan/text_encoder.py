# Adapted from ``WanTextEncoder`` of Self-Forcing (Apache-2.0).
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
"""umT5-XXL prompt encoder of Wan2.1 (512 tokens, padding zeroed)."""
from pathlib import Path

import torch
import torch.nn as nn

from .t5 import umt5_xxl_encoder
from .tokenizers import HuggingfaceTokenizer

__all__ = ["TextEncoder"]


class TextEncoder(nn.Module):

    def __init__(self, wan_dir, text_len=512):
        super().__init__()
        wan_dir = Path(wan_dir)
        self.text_encoder = umt5_xxl_encoder().eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=True))
        self.tokenizer = HuggingfaceTokenizer(
            name=str(wan_dir / "google" / "umt5-xxl"), seq_len=text_len, clean="whitespace",
            local_files_only=True)

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, prompts):
        """List of prompts -> ``{"prompt_embeds": [B, 512, 4096]}``."""
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        context = self.text_encoder(ids, mask)
        for u, n in zip(context, mask.gt(0).sum(dim=1).long()):
            u[n:] = 0.0
        return {"prompt_embeds": context}
