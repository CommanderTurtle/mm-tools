"""Learned camera-motion head (docs/world.md): the relative metric
camera pose between two full frames, for world-frame hands without SLAM, or as DPVO's scale / failure prior.
Model: frozen DINOv3 ViT-S/16 tokens of frames a and b (--size 192 x 256) + a Fourier ray embedding of every token (from K:
intrinsics-aware across Aria / ARKit / EgoStandard) + a learned hand embedding on tokens inside the hand boxes + a
frame-slot embedding -> transformer over [camera token, tokens of a, tokens of b] -> rotation R_ab (6D), metric translation
t_ab (camera b's centre in camera a, m), log-variances of both (T_ab = inv(T_world_a) T_world_b).
Data: train/build_cam_frames.py caches (HOT3D train, EgoDex); windows of --window frames at a random frame stride
(--strides), a random 4:3 crop (K follows), window-consistent photometric jitter, hand-box jitter / dropout, random time
reversal. Held-out split by sequence (--val_frac); --exclude lists sequences never used (e.g. HOT3D evaluation sequences).
Loss per window: every consecutive pair + --long_pairs random pairs (lag up to window - 1): Laplace NLL of the rotation
error (deg) and the metric translation error (cm) with the predicted log-variances; + --w_chain x the error of the chained
consecutive predictions 0 -> k vs GT for every k (deg + cm), which keeps the chained trajectory unbiased.
    torchrun --nproc_per_node "${TRAIN_PROCESSES:?set TRAIN_PROCESSES}" -m train.camera_motion --batch_size "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --data $DATA_ROOT/camframes --out $CKPT_ROOT/camera_motion/<run>
"""
import argparse
import glob
import json
import math
import os
import time
import zlib

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------- geometry
def crop_params(hw, size, scale=1.0, rng=None):
    """A crop of the model's aspect (size = (H, W)) inside an image of shape hw: the largest such crop x scale, centred
    (rng None) or at a random position -> (x0, y0, crop w, crop h)."""
    h, w = hw
    a = size[0] / size[1]
    cw, ch = (w, w * a) if h / w > a else (h / a, h)
    cw, ch = cw * scale, ch * scale
    x0 = (w - cw) / 2 if rng is None else rng.uniform(0, w - cw)
    y0 = (h - ch) / 2 if rng is None else rng.uniform(0, h - ch)
    return x0, y0, cw, ch


def crop_image(img, crop, size):
    """img (h, w, 3) -> (size H, size W, 3) of the crop; exact sub-pixel crop via an affine warp."""
    x0, y0, cw, ch = crop
    sx, sy = size[1] / cw, size[0] / ch
    M = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], np.float32)
    interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
    if sx < 0.5:  # area-like downsampling before the warp
        img = cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
        M = np.array([[1, 0, -x0 * sx], [0, 1, -y0 * sy]], np.float32)
        interp = cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (size[1], size[0]), flags=interp, borderMode=cv2.BORDER_REFLECT)


def crop_K(K, crop, size):
    x0, y0, cw, ch = crop
    sx, sy = size[1] / cw, size[0] / ch
    K = np.asarray(K, np.float64).copy()
    K[0, 0], K[0, 2] = K[0, 0] * sx, (K[0, 2] - x0) * sx
    K[1, 1], K[1, 2] = K[1, 1] * sy, (K[1, 2] - y0) * sy
    return K


def box_tokens(boxes, crop, size, patch=16, pad=0.15):
    """Hand boxes (k, 5: x0 y0 x1 y1 valid, image px) -> (H / patch, W / patch) bool, tokens touching a padded box."""
    x0c, y0c, cw, ch = crop
    sx, sy = size[1] / cw, size[0] / ch
    gh, gw = size[0] // patch, size[1] // patch
    m = np.zeros((gh, gw), bool)
    for x0, y0, x1, y1, ok in boxes:
        if ok <= 0:
            continue
        d = pad * max(x1 - x0, y1 - y0)
        u0, u1 = ((x0 - d - x0c) * sx) / patch, ((x1 + d - x0c) * sx) / patch
        v0, v1 = ((y0 - d - y0c) * sy) / patch, ((y1 + d - y0c) * sy) / patch
        c0, c1 = max(int(np.floor(u0)), 0), min(int(np.ceil(u1)), gw)
        r0, r1 = max(int(np.floor(v0)), 0), min(int(np.ceil(v1)), gh)
        if c1 > c0 and r1 > r0:
            m[r0:r1, c0:c1] = True
    return m


def rot6d_to_R(x):
    a, b = x[..., :3], x[..., 3:]
    c0 = F.normalize(a, dim=-1)
    c1 = F.normalize(b - (c0 * b).sum(-1, keepdim=True) * c0, dim=-1)
    return torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], -1)


def rot_err_deg(R1, R2):
    """Geodesic angle (deg) via the chordal distance: |R1 - R2|_F = 2 sqrt(2) sin(theta / 2) (stable gradients)."""
    c = ((R1 - R2).square().sum((-1, -2)) + 1e-10).sqrt() / (2 * math.sqrt(2))
    return torch.rad2deg(2 * torch.asin(c.clamp(max=1.0)))


def ray_features(K, size, patch=16, bands=8):
    """K (B, 3, 3) for the model input -> (B, N, 2 + 4 bands) Fourier features of each token centre's ray (x / z, y / z)."""
    gh, gw = size[0] // patch, size[1] // patch
    v, u = torch.meshgrid(torch.arange(gh, device=K.device) * patch + patch / 2,
                          torch.arange(gw, device=K.device) * patch + patch / 2, indexing="ij")
    x = (u[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    y = (v[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    r = torch.stack([x, y], -1).reshape(len(K), -1, 2).float()
    e = r[..., None] * (2.0 ** torch.arange(bands, device=K.device) * math.pi)
    return torch.cat([r, e.sin().flatten(-2), e.cos().flatten(-2)], -1)


# ---------------------------------------------------------------------------- model
# graph_safe_raft adapts torchvision's RAFT (torchvision/models/optical_flow/raft.py; BSD 3-Clause License, Copyright (c)
# Soumith Chintala 2016; THIRD_PARTY_NOTICES.md)
def graph_safe_raft(raft):
    """torchvision's RAFT builds coordinate grids / correlation offsets on the CPU every call; create them on the GPU instead
    (same values) so a causal step can be captured in a CUDA graph."""
    import types

    import torchvision.models.optical_flow.raft as tr

    if not getattr(tr, "_graph_safe", False):
        mk = tr.make_coords_grid
        tr.make_coords_grid = lambda b, h, w, device="cuda": mk(b, h, w, device="cuda")
        tr._graph_safe = True
    cb = raft.corr_block
    n = 2 * cb.radius + 1
    d = torch.linspace(-cb.radius, cb.radius, n)
    cb.delta = torch.stack(torch.meshgrid(d, d, indexing="ij"), -1).view(1, n, n, 2).cuda()

    def index_pyramid(self, centroids_coords):
        b, _, h, w = centroids_coords.shape
        c = centroids_coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        out = []
        for vol in self.corr_pyramid:
            out.append(tr.grid_sample(vol, c + self.delta, align_corners=True, mode="bilinear").view(b, h, w, -1))
            c = c / 2
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()

    cb.index_pyramid = types.MethodType(index_pyramid, cb)


class GraphedStep:
    """CameraMotion.step as one CUDA graph (fixed shapes; inputs copied into static buffers, outputs are static buffers:
    consume / clone them before the next call). Standalone so the DPVO env can use it without the fah package."""

    def __init__(self, model):
        self.model, self.graph = model, None

    @torch.no_grad()
    def __call__(self, b):
        if self.graph is None:
            self.static = {k: v.clone() for k, v in b.items()}
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    self.model.step(self.static)
            torch.cuda.current_stream().wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.model.step(self.static)
        for k, v in b.items():
            self.static[k].copy_(v)
        self.graph.replay()
        return self.out


class CameraMotion(nn.Module):
    def __init__(self, backbone="vit_small_patch16_dinov3.lvd1689m", size=(192, 256), dim=256, layers=6, heads=8,
                 bands=8, pretrained=True, corr=True, raft=False):
        super().__init__()
        import timm

        self.size, self.patch, self.bands, self.corr, self.use_raft = tuple(size), 16, bands, corr, raft
        self.vit = timm.create_model(backbone, pretrained=pretrained, img_size=self.size, num_classes=0, dynamic_img_size=False)
        self.vit.requires_grad_(False)
        self.num_prefix = self.vit.num_prefix_tokens
        vd = self.vit.embed_dim
        self.proj = nn.Sequential(nn.LayerNorm(vd), nn.Linear(vd, dim))
        self.ray = nn.Linear(2 + 4 * bands, dim)
        self.slot = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.hand = nn.Parameter(torch.randn(dim) * 0.02)
        self.cam = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        if corr:  # explicit correspondences: soft-argmax flow of every token through the frozen-feature correlation
            self.log_tau = nn.Parameter(torch.tensor(math.log(0.05)))
            self.flow = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        if raft:  # pixel-precise correspondences: frozen RAFT-small optical flow a -> b, pooled per token
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None)
            self.raft.requires_grad_(False)
            self.raft_flow = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 11))
        nn.init.zeros_(self.out[-1].weight)
        with torch.no_grad():  # identity rotation, zero translation, unit variances at the start
            self.out[-1].bias.copy_(torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), False)

    @torch.no_grad()
    def encode(self, img):
        """img (B, 3, H, W) RGB uint8 / [0, 255] at self.size -> backbone tokens (B, N, vit dim); cache per frame."""
        x = (img.float() / 255 - self.mean) / self.std
        return self.vit.forward_features(x)[:, self.num_prefix:]

    def rays(self, K):
        """K (B, 3, 3) -> (B, N, dim + 2): the ray embedding and, last two channels, the raw token rays (x / z, y / z)."""
        r = ray_features(K, self.size, self.patch, self.bands)
        return torch.cat([self.ray(r), r[..., :2].to(self.ray.weight.dtype)], -1)

    def flows(self, fa, fb, r):
        """Soft-argmax correspondence of every token of a in b (and of b in a) on the cosine similarity of the frozen
        tokens: (displacement in ray units x 10, peak probability, normalised entropy) -> (B, N, 4) each."""
        A, Bf = F.normalize(fa.float(), dim=-1), F.normalize(fb.float(), dim=-1)
        c = A @ Bf.transpose(1, 2) / self.log_tau.exp()
        out = []
        for P in (c.softmax(-1), c.transpose(1, 2).softmax(-1)):
            d = (P @ r - r) * 10
            ent = -(P * (P + 1e-9).log()).sum(-1, keepdim=True) / math.log(P.shape[-1])
            out.append(torch.cat([d, P.amax(-1, keepdim=True), ent], -1))
        return out

    @torch.no_grad()
    def raft_tokens(self, ia, ib, K):
        """RAFT-small flow a -> b (6 updates) at the input size, per token: mean displacement in ray units (x 10) and its
        spread -> (B, N, 4)."""
        self.raft.eval()
        fl = self.raft(ia.float() / 127.5 - 1, ib.float() / 127.5 - 1, num_flow_updates=6)[-1].float()
        fl = fl / torch.stack([K[:, 0, 0], K[:, 1, 1]], 1)[:, :, None, None]
        m = F.avg_pool2d(fl, self.patch)
        sd = (F.avg_pool2d(fl.square(), self.patch) - m.square()).clamp(min=0).sqrt()
        return torch.cat([m * 10, sd * 10], 1).flatten(2).transpose(1, 2)

    def pair(self, fa, fb, ray, ha, hb, ia=None, ib=None, K=None):
        """Tokens of frames a, b (B, N, vit dim), rays() of their K (B, N, dim + 2), hand-token masks (B, N) ->
        R_ab (B, 3, 3), t_ab (B, 3) m, log-variance of the rotation error (deg) and of the translation error (cm).
        With raft: also the images ia, ib (B, 3, H, W) uint8 RGB and K (B, 3, 3)."""
        emb, r = ray[..., :-2], ray[..., -2:].float()
        a = self.proj(fa) + emb + self.slot[0] + ha[..., None].to(emb.dtype) * self.hand
        b = self.proj(fb) + emb + self.slot[1] + hb[..., None].to(emb.dtype) * self.hand
        if self.corr:
            qa, qb = self.flows(fa, fb, r)
            a, b = a + self.flow(qa.to(a.dtype)), b + self.flow(qb.to(b.dtype))
        if self.use_raft:
            q = self.raft_flow(self.raft_tokens(ia, ib, K).to(a.dtype))
            a, b = a + q, b + q
        x = torch.cat([self.cam.expand(len(a), -1, -1).to(a.dtype), a, b], 1)
        o = self.out(self.trunk(x)[:, 0]).float()
        return rot6d_to_R(o[:, :6]), o[:, 6:9] * 0.1, o[:, 9].clamp(-3, 5), o[:, 10].clamp(-3, 5)

    def step(self, b):
        """One causal step (CUDA-graph friendly, fah.cuda_graph.GraphedForward): b = dict(tok_a, img_a, img_b, ha, hb,
        ray, K) -> dict(R, t, lr, lt, tok_b) -- frame b's tokens are reused as tok_a of the next step."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok_b = self.encode(b["img_b"])
            R, t, lr, lt = self.pair(b["tok_a"], tok_b, b["ray"], b["ha"], b["hb"], b["img_a"], b["img_b"], b["K"])
        return dict(R=R, t=t, lr=lr, lt=lt, tok_b=tok_b)

    def forward(self, b, long_pairs=4, w_chain=0.2, rng=None, w_long=1.0):  # training / validation loss of a batch of windows
        return window_loss(self, b, long_pairs, w_chain, rng if rng is not None else np.random.default_rng(0), w_long)


# ---------------------------------------------------------------------------- data
class CamWindows(Dataset):
    def __init__(self, root, sources, split="train", exclude=(), val_frac=0.03, window=17, strides=(1, 2, 3),
                 size=(192, 256), aug=True, length=10 ** 7, crop_min=0.75, fast=0.0):
        self.window, self.strides, self.size, self.aug, self.length = window, strides, tuple(size), aug, length
        self.crop_min = crop_min  # random crops of crop_min..1 of the largest 4:3 crop (focal length / FOV variety)
        self.fast = fast  # share of windows taken as the fastest of 8 candidates (camera displacement)
        self.clips, self.by_src = [], {}
        from train.release_guards import egodex_camera_holdout

        for src in sources:
            for f in sorted(glob.glob(os.path.join(root, src, "*", "meta.npz"))):
                # Also protect existing caches built before the frame-builder holdout guard.
                if src == "egodex" and egodex_camera_holdout(os.path.basename(os.path.dirname(f))):
                    continue
                m = np.load(f)
                seq = str(m["sequence"])
                if seq in exclude:
                    continue
                if ((zlib.crc32(seq.encode()) % 1000) < val_frac * 1000) != (split == "val"):
                    continue
                n = len(m["offsets"]) - 1
                if n < window:
                    continue
                self.by_src.setdefault(src, []).append(len(self.clips))
                self.clips.append(dict(dir=os.path.dirname(f), n=n, offsets=m["offsets"], T=m["T_world_cam"], K=m["K"],
                                       boxes=m["boxes"], hw=tuple(int(x) for x in m["size"])))
        w = np.array([sources[s] for s in self.by_src], np.float64)
        self.src_names, self.src_p = list(self.by_src), w / w.sum()

    def __len__(self):
        return self.length

    def frame(self, c, i):
        with open(os.path.join(c["dir"], "frames.bin"), "rb") as fb:
            fb.seek(int(c["offsets"][i]))
            buf = fb.read(int(c["offsets"][i + 1] - c["offsets"][i]))
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    def __getitem__(self, k):
        rng = np.random if self.aug else np.random.RandomState(k)
        src = self.src_names[rng.choice(len(self.src_names), p=self.src_p)]
        c = self.clips[self.by_src[src][rng.randint(len(self.by_src[src]))]]
        st = [s for s in self.strides if (self.window - 1) * s < c["n"]]
        s = st[rng.randint(len(st))]
        t0 = rng.randint(c["n"] - (self.window - 1) * s)
        if self.aug and self.fast > 0 and rng.rand() < self.fast:  # oversample fast camera motion (walking-like steps)
            cand = [rng.randint(c["n"] - (self.window - 1) * s) for _ in range(8)]
            p = c["T"][:, :3, 3]
            t0 = max(cand, key=lambda a: np.linalg.norm(p[a + (self.window - 1) * s] - p[a]))
        idx = t0 + s * np.arange(self.window)
        if self.aug and rng.rand() < 0.5:
            idx = idx[::-1]
        crop = crop_params(c["hw"], self.size, rng.uniform(self.crop_min, 1.0) if self.aug else 1.0, rng if self.aug else None)
        if self.aug:
            ga, gb, gs = rng.uniform(0.7, 1.3), rng.uniform(-25, 25), rng.uniform(0.6, 1.4)
            wb = rng.uniform(0.9, 1.1, 3)
        imgs, hands = [], []
        for i in idx:
            x = crop_image(self.frame(c, i), crop, self.size)
            if self.aug:
                x = x.astype(np.float32) * ga + gb + rng.uniform(-5, 5)
                g = x.mean(2, keepdims=True)
                x = np.clip((g + (x - g) * gs) * wb, 0, 255).astype(np.uint8)
                if rng.rand() < 0.1:
                    x = cv2.GaussianBlur(x, (5, 5), 0)
                if rng.rand() < 0.1:
                    x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 80))])[1], cv2.IMREAD_COLOR)
            imgs.append(x[:, :, ::-1].transpose(2, 0, 1))
            b = c["boxes"][i].copy()
            if self.aug:
                wh = np.maximum(b[:, 2:4] - b[:, 0:2], 1)
                b[:, :4] += rng.uniform(-0.1, 0.1, (len(b), 4)) * np.tile(wh, (1, 2))
                b[rng.rand(len(b)) < 0.1, 4] = 0
            hands.append(box_tokens(b, crop, self.size))
        T = c["T"][idx]
        T = np.linalg.inv(T[0]) @ T  # relative to the first frame of the window
        return dict(img=torch.from_numpy(np.ascontiguousarray(np.stack(imgs))), K=torch.from_numpy(crop_K(c["K"], crop, self.size)).float(),
                    T=torch.from_numpy(T).float(), hand=torch.from_numpy(np.stack(hands)))


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)  # many loader workers on a shared node


# ---------------------------------------------------------------------------- training
def window_loss(model, b, long_pairs, w_chain, rng, w_long=1.0):
    B, W = b["img"].shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = model.encode(b["img"].flatten(0, 1)).unflatten(0, (B, W))  # (B, W, N, D)
        ray = model.rays(b["K"])
    hand = b["hand"].flatten(2)  # (B, W, N)
    I = list(range(W - 1)) + [0] * long_pairs
    J = list(range(1, W)) + [0] * long_pairs
    for p in range(long_pairs):
        lag = int(rng.integers(2, W))
        i = int(rng.integers(0, W - lag))
        I[W - 1 + p], J[W - 1 + p] = i, i + lag
    P = len(I)
    im = (b["img"][:, I].flatten(0, 1), b["img"][:, J].flatten(0, 1), b["K"].repeat_interleave(P, 0)) if model.use_raft else ()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        R, t, lr, lt = model.pair(feats[:, I].flatten(0, 1), feats[:, J].flatten(0, 1), ray.repeat_interleave(P, 0),
                                  hand[:, I].flatten(0, 1), hand[:, J].flatten(0, 1), *im)
    R, t, lr, lt = R.view(B, P, 3, 3), t.view(B, P, 3), lr.view(B, P), lt.view(B, P)
    T = b["T"]
    G = torch.linalg.inv(T[:, I]) @ T[:, J]
    er = rot_err_deg(R, G[..., :3, :3])
    et = (t - G[..., :3, 3]).abs().sum(-1) * 100
    wp = torch.ones(P, device=R.device)
    wp[W - 1:] = w_long  # --w_long: weight of the long pairs (direct multi-second displacement)
    nll = ((er * torch.exp(-lr) + lr + et * torch.exp(-lt) + 3 * lt) * wp).sum(-1).mean() / wp.sum()
    # chained consecutive predictions 0 -> k vs GT
    M = torch.eye(4, device=R.device).repeat(B, W - 1, 1, 1)
    M[..., :3, :3], M[..., :3, 3] = R[:, :W - 1], t[:, :W - 1]
    acc, ch_r, ch_t = torch.eye(4, device=R.device).repeat(B, 1, 1), [], []
    for k in range(W - 1):
        acc = acc @ M[:, k]
        ch_r.append(rot_err_deg(acc[:, :3, :3], T[:, k + 1, :3, :3]))
        ch_t.append((acc[:, :3, 3] - T[:, k + 1, :3, 3]).abs().sum(-1) * 100)
    ch_r, ch_t = torch.stack(ch_r, 1), torch.stack(ch_t, 1)
    loss = nll + w_chain * (ch_r + ch_t).mean()
    parts = dict(nll=nll.item(), rot_deg=er[:, :W - 1].mean().item(), trans_cm=et[:, :W - 1].mean().item(),
                 rot_long_deg=er[:, W - 1:].mean().item() if long_pairs else 0.0,
                 trans_long_cm=et[:, W - 1:].mean().item() if long_pairs else 0.0,
                 chain_end_deg=ch_r[:, -1].mean().item(), chain_end_cm=ch_t[:, -1].mean().item(),
                 gt_step_deg=rot_err_deg(G[:, :W - 1, :3, :3], torch.eye(3, device=R.device).expand(B, W - 1, 3, 3)).mean().item())
    return loss, parts


@torch.no_grad()
def evaluate(model, loader, dev, n_batches, long_pairs):
    model.eval()
    rng = np.random.default_rng(0)
    acc = {}
    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
        _, parts = model(b, long_pairs, 0.0, rng)
        for kk, v in parts.items():
            acc.setdefault(kk, []).append(v)
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(args):
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    size = tuple(int(x) for x in args.size.split("x"))
    sources = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.sources.split(",")}
    exclude = set(json.load(open(args.exclude))) if args.exclude else set()
    model = CameraMotion(args.backbone, size, args.dim, args.layers, corr=not args.no_corr, raft=args.raft,
                         pretrained=not args.init).to(dev)
    if args.init:  # warm start (e.g. from an earlier head; also supplies the frozen backbone / RAFT weights)
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    step = 0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):  # resume (preemption-safe)
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index]) if ddp else model
    strides = tuple(int(x) for x in args.strides.split(","))
    ds = CamWindows(args.data, sources, "train", exclude, args.val_frac, args.window, strides, size, crop_min=args.crop_min,
                    fast=args.fast)
    vs = CamWindows(args.data, sources, "val", exclude, args.val_frac, args.window, strides, size, aug=False, length=10 ** 6)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "data_inventory.json"), "w") as f:
            json.dump({split: [os.path.relpath(c["dir"], args.data) for c in dataset.clips]
                       for split, dataset in (("train", ds), ("val", vs))}, f, indent=2)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True,
                        worker_init_fn=seed_worker, persistent_workers=True)
    val = DataLoader(vs, batch_size=args.batch_size, num_workers=2)
    run = None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        print(f"train {len(ds.clips)} clips {dict((s, len(v)) for s, v in ds.by_src.items())}, val {len(vs.clips)} clips, "
              f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params, from step {step}", flush=True)
        if args.wandb:
            import wandb

            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "FAH"), entity=os.environ.get("WANDB_ENTITY"),
                             name=args.run_name, config=vars(args), dir=args.out, resume="allow",
                             id=(args.run_name or os.path.basename(args.out.rstrip("/"))).replace("/", "_"))
    t0 = time.time()
    it = iter(loader)
    while step < args.steps:
        sc = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = args.lr * sc
        b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
        loss, parts = net(b, args.long_pairs, args.w_chain, rng, args.w_long)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if rank == 0 and step % 50 == 0:
            log = dict(loss=loss.item(), **parts, lr=opt.param_groups[0]["lr"], win_per_s=50 * args.batch_size * world / (time.time() - t0))
            t0 = time.time()
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in log.items()), flush=True)
            if run:
                run.log({f"train/{k}": v for k, v in log.items()}, step=step)
        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            m = evaluate(model, val, dev, args.eval_batches, args.long_pairs)
            print(f"[eval step {step}] " + " ".join(f"{k}={v:.4g}" for k, v in m.items()), flush=True)
            if run:
                run.log({f"val/{k}": v for k, v in m.items()}, step=step)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, args=vars(args), metrics=m), last + ".tmp")
            os.replace(last + ".tmp", last)
        if ddp and step % args.eval_every == 0:
            dist.barrier()
    if rank == 0:
        torch.save(dict(model=model.state_dict(), step=step, args=vars(args)), os.path.join(args.out, "model.pt"))
    if run:
        run.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--sources", default="hot3d:1.0", help="cache sources with sampling weights, e.g. hot3d:0.5,egodex:0.5")
    p.add_argument("--exclude", default="", help="json list of sequence ids never used")
    p.add_argument("--val_frac", type=float, default=0.03)
    p.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--size", default="192x256")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--no_corr", action="store_true", help="no correlation-flow token features")
    p.add_argument("--raft", action="store_true", help="RAFT-small optical-flow token features (frozen)")
    p.add_argument("--window", type=int, default=17)
    p.add_argument("--strides", default="1,2,3")
    p.add_argument("--long_pairs", type=int, default=4)
    p.add_argument("--fast", type=float, default=0.0, help="share of windows oversampled for fast camera motion")
    p.add_argument("--crop_min", type=float, default=0.75, help="smallest random crop (fraction of the largest 4:3 crop)")
    p.add_argument("--w_chain", type=float, default=0.2)
    p.add_argument("--w_long", type=float, default=1.0, help="weight of the long pairs in the NLL (direct displacement)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, required=True, help="training samples per process; choose for your setup")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (opt-in; WANDB_PROJECT / WANDB_ENTITY from the environment)")
    p.add_argument("--init", default="", help="initial weights (a model.pt of the same architecture)")
    main(p.parse_args())
