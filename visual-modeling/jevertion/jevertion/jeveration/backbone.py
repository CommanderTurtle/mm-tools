"""The frozen vision-language backbone (Qwen3.5-4B) with the shared LoRA adapters, and its single forward pass.

Sequences are padded on the left: most of the backbone's layers are recurrences over the positions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def resolve(source, default):
    """a local path or a Hub repository -> (location, revision); None: the public model at its pinned revision"""
    return default if source is None else (source, None)


def load_backbone(source=None, dtype=torch.bfloat16):
    """-> (language model, vision tower) of the backbone"""
    from transformers import AutoModelForImageTextToText
    repo, revision = resolve(source, config.BACKBONE)
    full = AutoModelForImageTextToText.from_pretrained(repo, revision=revision, dtype=dtype)
    lm, visual = full.model.language_model, full.model.visual
    del full
    return lm, visual


def load_text_tokenizer(source=None):
    from transformers import AutoTokenizer
    repo, revision = resolve(source, config.BACKBONE)
    return AutoTokenizer.from_pretrained(repo, revision=revision)


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x with W frozen"""

    def __init__(self, base, rank, alpha):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        d = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scale
        return y + d.to(y.dtype)


def inject_lora(lm, rank=config.LORA_RANK, targets=config.LORA_TARGETS):
    """Wrap every nn.Linear of the language model whose attribute name is in `targets`; alpha = rank. Returns how many."""
    todo = [(n, m) for n, m in lm.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for n, m in todo:
        parent = lm.get_submodule(n.rsplit(".", 1)[0]) if "." in n else lm
        setattr(parent, n.split(".")[-1], LoRALinear(m, rank, rank))
    return len(todo)


def backbone_forward(lm, x, mask, n_decision, rope_pos=None):
    """One pass of the language model over the embeddings x [B, L, H] with the padding mask [B, L] -> the final hidden states
    [B, L, H]. The last n_decision positions are the decision tokens; in the full-attention layers they attend to each other
    in both directions. rope_pos [3, B, L]: rotary positions (t, h, w), given when the prefix holds an image."""
    from transformers.masking_utils import create_causal_mask
    b, n = x.shape[:2]
    pos = torch.arange(n, device=x.device).view(1, 1, -1).expand(4, b, -1)      # text, temporal, height, width
    block = torch.full((b, n), -1, dtype=torch.long, device=x.device)
    block[:, n - n_decision:] = 0
    full = create_causal_mask(config=lm.config, inputs_embeds=x, attention_mask=mask, past_key_values=None, position_ids=pos[0],
                              block_sequence_ids=block)
    if isinstance(full, torch.Tensor) and full.dtype == torch.bool:
        # A padding position attends to nothing, and on MPS scaled_dot_product_attention turns such a row into NaN (the
        # CPU gives 0), which the linear-attention layers then carry into the real positions. Letting every position see
        # itself leaves the real positions unchanged: none of them attends to a padding position.
        full = full | torch.eye(n, dtype=torch.bool, device=full.device)
    linear = lm._update_linear_attn_mask(mask, None)
    pe = lm.rotary_emb(x, pos[1:] if rope_pos is None else rope_pos)
    h = x
    for i, layer in enumerate(lm.layers[:lm.config.num_hidden_layers]):
        h = layer(h, position_embeddings=pe, attention_mask=linear if lm.config.layer_types[i] == "linear_attention" else full,
                  position_ids=pos[0], past_key_values=None, use_cache=False)
    return lm.norm(h)
