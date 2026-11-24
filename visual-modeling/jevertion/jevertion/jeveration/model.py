"""Jeveration: a frozen vision-language model that answers with typed decisions in one forward pass.

    image generation   [prefix] [128 decision tokens, initialised from the random state]
                       -> backbone, one pass -> image head -> a code distribution at every position, all sampled in parallel
    dense prediction   [picture + instruction] [18 x 24 spatial queries]
                       -> backbone, one pass -> the task's head on every query -> maps (dense.py)

Parameter names follow the released checkpoint.
"""
import torch
import torch.nn as nn

from . import config
from .backbone import backbone_forward, inject_lora
from .dense import assemble
from .heads import CondEmbed, ImageHead, TimeMLP
from .random_state import sample_random_state


class Jeveration(nn.Module):
    def __init__(self, lm, visual):
        """lm, visual: the backbone's language model and vision tower (backbone.load_backbone)"""
        super().__init__()
        self.lm, self.visual = lm, visual
        self.lm.requires_grad_(False)
        self.visual.requires_grad_(False)
        self.n_lora = inject_lora(self.lm)
        self.hidden = hidden = lm.config.hidden_size
        self.n_tok = n = config.NUM_CODES
        d, k = config.CODE_DIM, config.CODEBOOK_SIZE
        # image decisions
        self.z_in = nn.Linear(d, hidden)
        self.z_pos = nn.Parameter(torch.zeros(n, hidden))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.t_mlp = TimeMLP(hidden)
        self.h_mlp = TimeMLP(hidden)
        self.cond = CondEmbed()
        self.head = ImageHead(hidden)
        self.code_head = nn.Linear(d, k)
        self.code_wide = nn.Linear(config.HEAD_WIDTH, k)
        # dense decisions
        gh, gw = (s // config.DENSE_CELL for s in config.DENSE_INPUT_HW)
        self.dense_grid, self.dense_n = (gh, gw), gh * gw
        win = 2 * config.DENSE_SUB
        self.dense_queries = nn.Parameter(torch.zeros(gh * gw, hidden))
        self.dense_q_vis = nn.Linear(hidden, hidden)
        self.dense_heads = nn.ModuleDict({t: nn.Sequential(nn.Linear(hidden, config.DENSE_HEAD_WIDTH), nn.SiLU(),
                                                           nn.Linear(config.DENSE_HEAD_WIDTH, win * win * config.DENSE_CHANNELS[t]))
                                          for t in config.DENSE_TASKS})
        # the tokenizer's codebook in the scale the generator works in (set_codebook)
        self.register_buffer("codebook_std", None, persistent=False)
        self.requires_grad_(False)

    # ---- weights -------------------------------------------------------------------------------------------------------------
    def checkpoint_parameters(self):
        """the parameters a Jeveration checkpoint holds: everything except the backbone's own (frozen, public) weights"""
        return {n: p for n, p in self.named_parameters()
                if not n.startswith("visual.") and (not n.startswith("lm.") or n.endswith((".lora_A", ".lora_B")))}

    def load_weights(self, path):
        """load a Jeveration checkpoint (the names must match exactly) -> the code statistics (lat_mean, lat_std)"""
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        sd, own = ck["ema"], self.checkpoint_parameters()
        missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
        assert not missing and not unexpected, f"checkpoint mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}"
        with torch.no_grad():
            for n, p in own.items():
                p.copy_(sd[n].to(p.device, p.dtype))
        return ck["lat_mean"], ck["lat_std"]

    def set_codebook(self, codebook, lat_mean, lat_std):
        """codebook [K, C] (the tokenizer's) -> the standardised codebook the decision tokens are initialised from"""
        dev = self.z_pos.device
        cb = codebook.detach().float().cpu().to(torch.float16).float().to(dev)      # half precision, as in training
        self.codebook_std = (cb - lat_mean.to(dev).float().view(1, -1)) / lat_std.to(dev).float().view(1, -1)

    # ---- image decisions -----------------------------------------------------------------------------------------------------
    def decide(self, input_ids, attention_mask, state, t, prefix=None):
        """ONE backbone pass: the decision tokens, initialised from the random state `state` [B, N, C] with masked share t [B],
        after a text-to-image prompt (input_ids, attention_mask) or an image prefix (prompts.build_prefix)
        -> (projection of u to the code space [B, N, C], features u [B, N, width])"""
        rope = None
        if prefix is not None:
            emb, attention_mask, rope = prefix
            b = emb.shape[0]
        else:
            emb = self.lm.embed_tokens(input_ids)
            b = input_ids.shape[0]
        state = state.float()
        tc = (self.t_mlp(t) + self.h_mlp(torch.zeros_like(t)))[:, None]
        c = self.cond(t)
        tail = (self.z_in(state) + self.z_pos + tc).to(emb.dtype)
        if rope is not None:           # the decision tokens' rotary positions continue after the prefix, text-like
            last = rope[:, :, -1].amax(0)
            cont = last[:, None] + 1 + torch.arange(tail.shape[1], device=rope.device)
            rope = torch.cat([rope, cont.unsqueeze(0).expand(3, -1, -1)], 2)
        mask = torch.cat([attention_mask, attention_mask.new_ones(b, tail.shape[1])], 1)
        h = backbone_forward(self.lm, torch.cat([emb, tail], 1), mask, tail.shape[1], rope_pos=rope)
        proj, u = self.head(h[:, -self.n_tok:].float(), state, c)
        return proj.float(), u

    def code_logits(self, input_ids, attention_mask, state, t, prefix=None, heads_in_autocast=False):
        """the code distribution's logits [B, N, K]: codebook branch + wide branch. heads_in_autocast: the two branches run in
        bfloat16 (text-to-image) or in float32 (image-to-image), as each task was evaluated."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.z_pos.is_cuda):
            proj, u = self.decide(input_ids, attention_mask, state, t, prefix)
            if heads_in_autocast:
                return self.code_head(proj.float()) + self.code_wide(u.float())
        return self.code_head(proj.float()) + self.code_wide(u.float())

    @torch.no_grad()
    def sample_codes(self, input_ids, attention_mask, generator=None, prefix=None, heads_in_autocast=False):
        """One joint decision: draw the random state, run one forward pass, sample every code -> codes [B, N] (int64)"""
        b = input_ids.shape[0] if prefix is None else prefix[0].shape[0]
        state, t = sample_random_state(self.mask_emb, self.codebook_std, b, generator)
        logits = self.code_logits(input_ids, attention_mask, state, t, prefix, heads_in_autocast)
        probs = torch.softmax(logits.float(), -1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator).view(b, self.n_tok)

    # ---- dense decisions -----------------------------------------------------------------------------------------------------
    def dense_states(self, prefix, vis, grid, img_start):
        """ONE backbone pass: the spatial queries after an image prefix (prompts.build_prefix) -> their final hidden states
        [B, gh*gw, hidden]. vis: the picture's visual tokens; img_start: the image's first position in the prefix."""
        emb, mask, rope = prefix
        b, (gh, gw) = emb.shape[0], grid
        assert (gh, gw) == self.dense_grid, f"a picture of {gh} x {gw} cells, the model has {self.dense_grid} spatial queries"
        q = self.dense_queries.unsqueeze(0) + self.dense_q_vis(vis.float())
        x = torch.cat([emb, q.to(emb.dtype)], 1)
        m = torch.cat([mask, mask.new_ones(b, self.dense_n)], 1)
        dev = rope.device
        row = torch.arange(gh, device=dev).repeat_interleave(gw)
        col = torch.arange(gw, device=dev).repeat(gh)
        t_ = (rope[:, :, -1].amax(0) + 1)[:, None].expand(-1, self.dense_n)
        cells = torch.stack([t_, (img_start + row).expand(b, -1), (img_start + col).expand(b, -1)])
        h = backbone_forward(self.lm, x, m, self.dense_n, rope_pos=torch.cat([rope, cells], 2))
        return h[:, -self.dense_n:].float()

    def dense_readout(self, task, hq):
        """the queries' states [n, gh*gw, hidden] -> the task's raw map [n, C, gh*8, gw*8] (logits / log-depth / normal)"""
        gh, gw = self.dense_grid
        return assemble(self.dense_heads[task](hq), gh, gw, config.DENSE_SUB, config.DENSE_CHANNELS[task])
