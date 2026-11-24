"""Inference precision helpers for FAB-L / FAB-S under BF16 autocast."""

import torch
import torch.nn as nn

CAST = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)


def bf16_weights(model):
    """Store the Linear / Conv weights in BF16 once. Under BF16 autocast these layers cast their FP32 weights on every
    call (for a ViT-L backbone ~0.6 GB of casts per frame); pre-cast weights give the same values without the copies.
    TorchScript modules (the MHR rig, FP32) are left alone. Returns the number of converted layers."""
    skip = {id(m) for s in model.modules() if isinstance(s, torch.jit.ScriptModule) for m in s.modules()}
    n = 0
    for m in model.modules():
        if isinstance(m, CAST) and id(m) not in skip:
            m.to(torch.bfloat16)
            n += 1
    return n


def compile_body(model, mode="max-autotune-no-cudagraphs"):
    """torch.compile (inductor) of the backbone and the decoder; capturable in the caller's CUDA graph. Fused kernels
    round BF16 differently, so outputs are not bit-identical: validate accuracy (bench/eval_hmr.py --compile)."""
    model.body.backbone.forward = torch.compile(model.body.backbone.forward, mode=mode, dynamic=False)
    model.body.decoder.forward = torch.compile(model.body.decoder.forward, mode=mode, dynamic=False)


class GraphBody:
    """The FAB-L / FAB-S body path under BF16 autocast, replayed from one CUDA graph per person count (inputs copied into
    static buffers; [0, 1] crops). Returns the graph's output buffers: valid until the next call."""

    def __init__(self, model):
        self.model, self.graphs = model, {}
        model.body.inputs_in_01 = True

    def _run(self, img, c, s, o, K, A):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(img, c, s, o, K, affine=A)

    def __call__(self, img, c, s, o, K, A):
        x = (img, c, s, o, K, A)
        if len(img) not in self.graphs:
            static = [t.clone() for t in x]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._run(*static)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run(*static)
            self.graphs[len(img)] = (g, static, out)
        g, static, out = self.graphs[len(img)]
        for dst, src in zip(static, x):
            dst.copy_(src, non_blocking=True)
        g.replay()
        return out


def fast_inference(model):
    """bf16_weights + the eager rig's pose correctives as CSR (call after enable_fast, with the model on its device)."""
    n = bf16_weights(model)
    hp = model.body.head_pose
    for m in (hp.kp_mhr, hp.full_mhr):
        if m is not None:
            m.sparse_correctives()
    return n
