"""Budget3D on speridlabs/idol-turbo: Idol's denoiser + two phase-LoRA experts, guidance-free, few steps.

idol-turbo keeps Idol's weights and adds rank-32 LoRAs to every attention / MLP Linear of the noise refiner,
the context refiner and the 24 joint blocks: a *slow* expert used while t >= 0.5 and a *fast* expert below.
Its sampler runs ~5 Euler steps with no classifier-free guidance (guidance is folded into the LoRAs).

Two things differ from the stock (global-switch) implementation, because Budget3D trains on batches that mix
trajectories on both sides of the phase boundary, under gradient checkpointing:

  * the expert is chosen **per row** by a mask held in a shared ``PhaseState``.  The mask is set once before a
    forward and stays put until the next one, so a checkpointed block recomputed during backward sees the same
    experts it used in the forward pass.  In training both experts are evaluated and selected with ``where``
    (rank 32, ~8 % of a Linear's FLOPs), which also keeps every LoRA parameter in the graph for DDP;
    at inference, when the whole batch is in one phase, only that expert is evaluated.
  * the context refiner carries LoRAs too, so the refined image condition depends on the phase: callers that
    cache it (the rollout buffer, the sampler) keep one copy per phase.

The phase is derived from the model's own ``t`` argument (Idol's 0..1000 scale) by ``bind_phase``.  The 5-step
sampler is ``budget3d.sampling.turbo.sample_turbo``.
"""
import os

import torch
import torch.nn as nn

PHASE_THRESHOLD = 0.5
# How a PhaseLoRALinear adds its low-rank delta.  True (default): the experts are selected at rank size and the delta
# is accumulated onto the base output by one addmm (one pass over the output).  False: the delta of each expert at
# full output size, selected with `where`, scaled, added (five passes) -- the form idol-turbo's own code uses.  Both
# compute the same function up to float rounding (tests/test_lora_fused.py); the one-pass form costs several times
# less memory traffic on these wide, long activations and trained 8 % faster (7.09 against 6.55 trajectories/s on
# 4x RTX PRO 6000).  BUDGET3D_LORA_FUSED=0, or set_lora_fused(False), selects the classic form.
LORA_FUSED = os.environ.get("BUDGET3D_LORA_FUSED", "1") == "1"
LORA_RANK = 32
LORA_ALPHA = 128.0
LORA_FILE = "idol_turbo_lora.safetensors"
TURBO_REPO = "speridlabs/idol-turbo"


def set_lora_fused(enabled):
    """Switch every PhaseLoRALinear between the one-pass (True) and the classic (False) form; returns the old setting."""
    global LORA_FUSED
    was, LORA_FUSED = LORA_FUSED, bool(enabled)
    return was


class PhaseState:
    """Which expert each row of the next forward uses (slow = True)."""

    def __init__(self, allow_uniform_fastpath=True):
        self.slow = None            # bool tensor [B]
        self.uniform = None         # True / False when every row is in the same phase, else None
        self.allow_uniform_fastpath = allow_uniform_fastpath

    def set_from_t01(self, t01):
        self.slow = (t01.float() >= PHASE_THRESHOLD).reshape(-1)
        if self.allow_uniform_fastpath:
            n = int(self.slow.sum())
            self.uniform = True if n == self.slow.numel() else (False if n == 0 else None)
        else:
            self.uniform = None

    def set_uniform(self, slow, batch, device):
        self.slow = torch.full((batch,), bool(slow), device=device)
        self.uniform = bool(slow) if self.allow_uniform_fastpath else None


class PhaseLoRALinear(nn.Module):
    """base Linear + slow / fast low-rank experts, chosen per row by ``state``."""

    def __init__(self, base, rank, alpha, state):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.state = state
        i, o = base.in_features, base.out_features
        dev = base.weight.device
        self.slow_A = nn.Parameter(torch.zeros(rank, i, device=dev))
        self.slow_B = nn.Parameter(torch.zeros(o, rank, device=dev))
        self.fast_A = nn.Parameter(torch.zeros(rank, i, device=dev))
        self.fast_B = nn.Parameter(torch.zeros(o, rank, device=dev))

    def _delta(self, x, A, B):
        return (x @ A.t()) @ B.t()

    def _mask(self, x):
        m = self.state.slow
        if m.numel() != x.shape[0]:
            # CFG / packed passes run [rows; rows]: the same mask repeated
            assert x.shape[0] % m.numel() == 0, (x.shape, m.shape)
            m = m.repeat(x.shape[0] // m.numel())
        return m.reshape(-1, *([1] * (x.dim() - 1)))

    def _forward_fused(self, x, out):
        """out + scale * (selected expert): the rows pick their expert on the rank-sized hidden, the two B
        matrices sit side by side, and addmm accumulates the product onto `out`."""
        st = self.state
        xa = x if torch.is_autocast_enabled() else x.to(self.slow_A.dtype)
        if st.uniform is True:
            h, B = xa @ self.slow_A.t(), self.slow_B
        elif st.uniform is False:
            h, B = xa @ self.fast_A.t(), self.fast_B
        else:
            hs, hf = xa @ self.slow_A.t(), xa @ self.fast_A.t()
            m = self._mask(x).to(hs.dtype)
            h, B = torch.cat((hs * m, hf * (1 - m)), dim=-1), torch.cat((self.slow_B, self.fast_B), dim=1)
        flat = torch.addmm(out.reshape(-1, out.shape[-1]), h.reshape(-1, h.shape[-1]).to(out.dtype),
                           B.t().to(out.dtype), alpha=self.scale)
        return flat.view(out.shape)

    def forward(self, x):
        out = self.base(x)
        st = self.state
        if st.slow is None:
            raise RuntimeError("PhaseLoRALinear: phase not set (call through a bound forward or set_uniform)")
        if LORA_FUSED:
            return self._forward_fused(x, out)
        xa = x if torch.is_autocast_enabled() else x.to(self.slow_A.dtype)
        if st.uniform is True:
            d = self._delta(xa, self.slow_A, self.slow_B)
        elif st.uniform is False:
            d = self._delta(xa, self.fast_A, self.fast_B)
        else:
            m = self._mask(x)
            d = torch.where(m, self._delta(xa, self.slow_A, self.slow_B), self._delta(xa, self.fast_A, self.fast_B))
        return out + (d * self.scale).to(out.dtype)


def inject(model, state, rank=LORA_RANK, alpha=LORA_ALPHA):
    """Wrap the same Linears idol-turbo adapts: attn.qkv, attn.out, mlp.mlp[0], mlp.mlp[2] of every block."""
    for blocks in (model.noise_refiner, model.context_refiner, model.blocks):
        for blk in blocks:
            blk.attn.qkv = PhaseLoRALinear(blk.attn.qkv, rank, alpha, state)
            blk.attn.out = PhaseLoRALinear(blk.attn.out, rank, alpha, state)
            blk.mlp.mlp[0] = PhaseLoRALinear(blk.mlp.mlp[0], rank, alpha, state)
            blk.mlp.mlp[2] = PhaseLoRALinear(blk.mlp.mlp[2], rank, alpha, state)
    return model


def lora_path(repo=TURBO_REPO):
    """Idol-turbo's LoRA file: a local path given as ``repo``, else downloaded from the Hub."""
    if os.path.isfile(repo):
        return repo
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, LORA_FILE)


def load_lora(model, path=None):
    from safetensors.torch import load_file
    sd = load_file(path or lora_path())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected LoRA keys: {unexpected[:5]}"
    want = {n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1] in ("slow_A", "slow_B", "fast_A", "fast_B")}
    assert want == set(sd), f"LoRA keys do not cover the injected modules: {len(want)} vs {len(sd)}"
    return len(sd)


def bind_phase(model, state):
    """model(x, t, ...) sets the per-row phase from t before running (t on Idol's 0..1000 scale)."""
    plain = model.forward

    def forward(x, t, *args, **kwargs):
        state.set_from_t01(t.float() / 1000.0)
        return plain(x, t, *args, **kwargs)

    model.forward = forward
    model.phase_state = state
    return model


def make_turbo(model, lora=None, allow_uniform_fastpath=True):
    """In place: inject + load idol-turbo's LoRAs into a loaded Budget3DModel (or Idol) model, bind the phase."""
    state = PhaseState(allow_uniform_fastpath)
    inject(model, state)
    n = load_lora(model, lora)
    bind_phase(model, state)
    return model, state, n
