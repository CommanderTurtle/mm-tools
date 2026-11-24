# Copyright 2026 Adventists AI. Apache-2.0.
"""E2: vector frames -> CosyVoice3 speech-token space -> DiT flow matching -> HiFT (24 kHz). Inference only.

Weights: <model>/e2/model.safetensors (adapter.*, decoder.*, aff.*) and <model>/e2/hift.safetensors.
The flow decoder and vocoder classes come from CosyVoice3 (vendored in cosyvoice_lite, Apache-2.0)."""
import json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
from . import paths
from .cosyvoice_lite.dit import DiT
from .cosyvoice_lite.flow_matching import CausalConditionalCFM
from .cosyvoice_lite.generator import CausalHiFTGenerator
from .cosyvoice_lite.f0_predictor import CausalConvRNNF0Predictor
from .cosyvoice_lite.prelookahead import PreLookaheadLayer

UNIT = 8
SPU = 2560


class CausalConv1d(nn.Conv1d):
    def __init__(s, ci, co, k, d=1, groups=1):
        super().__init__(ci, co, k, dilation=d, groups=groups); s.lpad = (k - 1) * d
    def forward(s, x): return super().forward(F.pad(x, (s.lpad, 0)))


class Block(nn.Module):
    def __init__(s, d, h, cross=False, ff=4, p=0.0):
        super().__init__()
        s.ln1, s.att = nn.LayerNorm(d), nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.cross = cross
        if cross: s.lnc, s.catt = nn.LayerNorm(d), nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.ln2, s.ff = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(p), nn.Linear(ff * d, d))
    def forward(s, x, mask=None, mem=None):
        h = s.ln1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]
        if s.cross: h = s.lnc(x); x = x + s.catt(h, mem, mem, need_weights=False)[0]
        return x + s.ff(s.ln2(x))


def band_mask(T, la, win, dev):
    i = torch.arange(T, device=dev)[:, None]; j = torch.arange(T, device=dev)[None]
    return ~((j <= i + la) & (j > i - win))


class Adapter(nn.Module):
    def __init__(s, din=2560, d=512, la=1, uwin=20, fwin=80, La=4, Lf=2, sub=UNIT, dout=80, la_all=False, la_layers=1):
        super().__init__(); s.la, s.uwin, s.fwin, s.nsub, s.la_all = la, uwin, fwin, sub, la_all; s.la_layers = la_layers
        s.ain = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d)); s.aconv = CausalConv1d(d, d, 3)
        s.ablocks = nn.ModuleList([Block(d, 8, p=0.05) for _ in range(La)])
        s.sub = nn.Parameter(torch.randn(sub, d) * 0.02); s.fconv = CausalConv1d(d, d, 5)
        s.fblocks = nn.ModuleList([Block(d, 8, p=0.05) for _ in range(Lf)])
        s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, dout)


class S2(nn.Module):
    def __init__(s, emb, pla, a):
        super().__init__()
        s.ad = Adapter(din=a["din"], d=a["d"], la=a["la"], La=a["La"], Lf=a["Lf"], uwin=a.get("uwin", 20), la_all=bool(a.get("la_all")),
                       la_layers=a.get("la_layers", 1), sub=4, dout=6561)
        s.emb, s.pla = emb, pla
        s.res = nn.Linear(a["d"], 160)
        s.res_cap = float(a.get("res_cap", 0.0)); s.res_scale = 1.0; s.soft_g = float(a.get("soft_g", 0.0))

    def forward(s, v, p=None):
        x = s.ad.ain(v); x = x + s.ad.aconv(x.transpose(1, 2)).transpose(1, 2)
        m0, m = band_mask(x.shape[1], s.ad.la, s.ad.uwin, x.device), band_mask(x.shape[1], 0, s.ad.uwin, x.device)
        for i, b in enumerate(s.ad.ablocks): x = b(x, m0 if i < getattr(s.ad, 'la_layers', 1) else m)
        B, K, d = x.shape
        f = (x[:, :, None] + s.ad.sub[None, None]).reshape(B, K * s.ad.nsub, d); f = f + s.ad.fconv(f.transpose(1, 2)).transpose(1, 2)
        mf = band_mask(f.shape[1], 0, s.ad.fwin, f.device)
        for b in s.ad.fblocks: f = b(f, mf)
        h = s.ad.ln(f); lg = s.ad.out(h)
        e = s.emb(lg.argmax(-1))
        if s.soft_g > 0:
            p_ = F.softmax(lg.float(), -1); e = (1 - s.soft_g) * e + s.soft_g * (p_ @ s.emb.weight)
        mu_tok = s.pla(e).repeat_interleave(2, dim=1)
        res = s.res(h).reshape(B, K * s.ad.nsub * 2, 80)
        if s.res_cap > 0: res = s.res_cap * torch.tanh(res / s.res_cap)
        res = res * s.res_scale
        return lg.transpose(1, 2), (mu_tok + res).transpose(1, 2), res


class _Cfg(dict):
    __getattr__ = dict.get


def build_decoder():
    est = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
              static_chunk_size=25 * 2, num_decoding_left_chunks=-1)
    cfm = _Cfg(sigma_min=1e-06, solver="euler", t_scheduler="cosine", training_cfg_rate=0.2, inference_cfg_rate=0.7, reg_loss_type="l1")
    return CausalConditionalCFM(in_channels=240, cfm_params=cfm, n_spks=1, spk_emb_dim=80, estimator=est)


def build_hift():
    return CausalHiFTGenerator(in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
                               nsf_voiced_threshold=10, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                               istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 7, 11],
                               resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], source_resblock_kernel_sizes=[7, 7, 11],
                               source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], lrelu_slope=0.1, audio_limit=0.99,
                               conv_pre_look_right=4, f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512))


def load_all(dev):
    D = paths.e2_dir(); cfg = json.load(open(os.path.join(D, "config.json")))
    sd = load_file(os.path.join(D, "model.safetensors"))
    emb = nn.Embedding(6561, 80); pla = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
    m = S2(emb, pla, cfg)
    m.load_state_dict({k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")}, strict=True)
    dec = build_decoder(); dec.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}, strict=True)
    aff = nn.Linear(192, 80); aff.load_state_dict({k[len("aff."):]: v for k, v in sd.items() if k.startswith("aff.")}, strict=True)
    hift = build_hift(); hift.load_state_dict(load_file(os.path.join(D, "hift.safetensors")), strict=True)
    return m.to(dev).eval(), dec.to(dev).eval(), aff.to(dev).eval(), hift.to(dev).eval(), cfg
