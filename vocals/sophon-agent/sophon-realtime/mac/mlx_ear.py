"""The thinker's own ear on MLX: our audio tower + projector (our weights), ported from the server's inference code.
No transcription anywhere: the user's audio becomes vectors that go into the thinker in place of the <|audio|> tokens."""
import json, math, numpy as np, mlx.core as mx, mlx.nn as nn
from transformers import WhisperFeatureExtractor
HOP = 160
def _out_len(L, n_window=50):
    c = n_window * 2; r = L % c; f = (r - 1) // 2 + 1
    return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (L // c) * 13
def _post_cnn(L):
    for _ in range(3): L = (L - 1) // 2 + 1 if L > 0 else 0
    return L
class Layer(nn.Module):
    def __init__(s, d, h, ffn):
        super().__init__(); s.h = h
        s.self_attn = nn.Module(); s.self_attn.q_proj = nn.Linear(d, d); s.self_attn.k_proj = nn.Linear(d, d); s.self_attn.v_proj = nn.Linear(d, d); s.self_attn.out_proj = nn.Linear(d, d)
        s.self_attn_layer_norm = nn.LayerNorm(d); s.final_layer_norm = nn.LayerNorm(d); s.fc1 = nn.Linear(d, ffn); s.fc2 = nn.Linear(ffn, d)
    def __call__(s, x, blocks):
        L, d = x.shape; a = s.self_attn; y = s.self_attn_layer_norm(x)
        q, k, v = [p(y).reshape(L, s.h, -1).transpose(1, 0, 2) for p in (a.q_proj, a.k_proj, a.v_proj)]
        outs = []; o = 0
        for n in blocks:
            outs.append(mx.fast.scaled_dot_product_attention(q[None, :, o:o + n], k[None, :, o:o + n], v[None, :, o:o + n], scale=(d // s.h) ** -0.5)[0]); o += n
        att = mx.concatenate(outs, axis=1).transpose(1, 0, 2).reshape(L, d)
        x = x + a.out_proj(att)
        return x + s.fc2(nn.gelu(s.fc1(s.final_layer_norm(x))))
class Ear(nn.Module):
    def __init__(s, cfg):
        super().__init__(); a = cfg["audio_config"]; d = a["d_model"]; ch = a["downsample_hidden_size"]; s.a = a; s.stack = cfg["stack_factor"]
        s.full_length = bool(cfg.get("audio_full_length", False))  # 4B: one LM token per 2 encoder frames (no Ultravox-style rounding down)
        s.conv2d1 = nn.Conv2d(1, ch, 3, 2, 1); s.conv2d2 = nn.Conv2d(ch, ch, 3, 2, 1); s.conv2d3 = nn.Conv2d(ch, ch, 3, 2, 1)
        fb = (((a["num_mel_bins"] + 1) // 2 + 1) // 2 + 1) // 2; s.conv_out = nn.Linear(ch * fb, d, bias=False)
        s.layers = [Layer(d, a["encoder_attention_heads"], a["encoder_ffn_dim"]) for _ in range(a["encoder_layers"])]
        s.ln_post = nn.LayerNorm(d)
        n, c = a["max_position_embeddings"], d; inc = math.log(10000) / (c // 2 - 1); inv = np.exp(-inc * np.arange(c // 2)); t = np.arange(n)[:, None] * inv[None]
        s._pos = mx.array(np.concatenate([np.sin(t), np.cos(t)], 1).astype(np.float32))
        din = d * s.stack; hid = cfg["hidden_size"]; dout = cfg["text_config"]["hidden_size"]
        s.ln_pre = nn.RMSNorm(din, eps=1e-6); s.linear_1 = nn.Linear(din, hid, bias=False); s.ln_mid = nn.RMSNorm(hid // 2, eps=1e-6); s.linear_2 = nn.Linear(hid // 2, dout, bias=False)
    def encode(s, feats, L):
        """feats: (128, T) log-mel numpy for one clip, L valid frames -> (n_out, 896)"""
        nw = s.a["n_window"]; cl = nw * 2; T = feats.shape[-1]
        if T % cl: feats = np.pad(feats, ((0, 0), (0, cl - T % cl))); T = feats.shape[-1]
        nc = T // cl; mask = (np.arange(T) < L).reshape(nc, cl).sum(-1)
        x = mx.array(feats.reshape(128, nc, cl).transpose(1, 0, 2)[..., None])  # (nc, mel, time, 1) NHWC
        x = nn.gelu(s.conv2d1(x)); x = nn.gelu(s.conv2d2(x)); x = nn.gelu(s.conv2d3(x))  # (nc, F, t, C)
        nc_, Fb, ts, C = x.shape; x = s.conv_out(x.transpose(0, 2, 3, 1).reshape(nc_, ts, C * Fb)) + s._pos[:ts].astype(x.dtype)
        keep = np.concatenate([np.arange(ts) + i * ts for i in range(nc) if False] + [np.arange(_post_cnn(int(m))) + i * ts for i, m in enumerate(mask)])
        h = x.reshape(-1, x.shape[-1])[mx.array(keep)]
        n_out = _out_len(L, nw); win = 13 * (s.a["n_window_infer"] // cl); blocks = [win] * (n_out // win) + ([n_out % win] if n_out % win else [])
        for l in s.layers: h = l(h, blocks)
        return s.ln_post(h)
    def project(s, h, ntok):
        T, C = h.shape; Tp = (T + s.stack - 1) // s.stack * s.stack
        if Tp > T: h = mx.concatenate([h, mx.zeros((Tp - T, C), h.dtype)])
        x = s.linear_1(s.ln_pre(h.reshape(Tp // s.stack, C * s.stack))); dm = x.shape[-1] // 2
        x = x[:, :dm] * nn.silu(x[:, dm:]); return s.linear_2(s.ln_mid(x))[:ntok]
def load_ear(thinker_dir, dtype=mx.float32):
    import os
    if os.path.exists(f"{thinker_dir}/ear.safetensors"):          # converted layout (convert.py): audio tower + projector in one file
        cfg = json.load(open(f"{thinker_dir}/config.json")); ear = Ear(cfg); W = {}
        for k, v in mx.load(f"{thinker_dir}/ear.safetensors").items():
            if k.startswith("audio_tower."):
                k = k[len("audio_tower."):]
                if k.startswith("conv2d") and k.endswith("weight"): v = v.transpose(0, 2, 3, 1)
            elif k.startswith("multi_modal_projector."): k = k[len("multi_modal_projector."):]
            W[k] = v.astype(dtype)
        ear.load_weights(list(W.items()), strict=True); mx.eval(ear.parameters())
        return ear, WhisperFeatureExtractor.from_pretrained(thinker_dir)
    cfg = json.load(open(f"{thinker_dir}/audio/config.json")); ear = Ear(cfg); W = {}
    for k, v in mx.load(f"{thinker_dir}/audio/model.safetensors").items():
        if not k.startswith("audio_tower."): continue
        k = k[len("audio_tower."):]
        if k.startswith("conv2d") and k.endswith("weight"): v = v.transpose(0, 2, 3, 1)
        W[k] = v.astype(dtype)
    import glob
    for f in sorted(glob.glob(f"{thinker_dir}/model-w4-*.safetensors")):
        for k, v in mx.load(f).items():
            if k.startswith("multi_modal_projector."): W[k[len("multi_modal_projector."):]] = v.astype(dtype)
    ear.load_weights(list(W.items()), strict=True); mx.eval(ear.parameters())
    fe = WhisperFeatureExtractor.from_pretrained(f"{thinker_dir}/audio"); return ear, fe
def embed_audio(ear, fe, a16k):
    step = HOP * 8 * ear.stack; n = max(len(a16k), 2 * HOP); n = -(-n // step) * step; a = np.pad(a16k.astype(np.float32), (0, n - len(a16k)))
    x = fe([a], sampling_rate=16000, padding="longest", pad_to_multiple_of=HOP, truncation=False, return_attention_mask=True, return_tensors="np")
    feats, L = x["input_features"][0], int(x["attention_mask"][0].sum())
    out = []
    for off in range(0, L, 3000):
        Lc = min(L - off, 3000); f = feats[:, off:off + 3000]
        if off > 0 and f.shape[-1] < 3000: f = np.pad(f, ((0, 0), (0, 3000 - f.shape[-1])))
        ntok = math.ceil(_out_len(Lc) / ear.stack) if ear.full_length else math.ceil(Lc / (8 * ear.stack))
        out.append(ear.project(ear.encode(f, Lc), int(ntok)))
    return mx.concatenate(out), feats
