"""Loading Idol's released weights and trained Budget3D weights, and the gate that the student starts as the teacher."""
import json
import os

import torch
from safetensors.torch import load_file
from tensordict import TensorDict

from ..octree import dense_layout
from ..upstream import LatentSeqMMFlowModel
from .denoiser import Budget3DModel

# the configuration of the added modules that the released checkpoints were trained with
ADAPTER_DEFAULTS = dict(mix_bott=256, ipe_bottleneck=256, refine_width=128, refine_depth=2, refine_heads=4,
                        refine_zt_skip=True, use_hf_skip=True, hf_rho=0.25, use_ipe=True, use_scale_emb=True,
                        multiscale_cascade=True, ipe_scale=1.0, reconstruct_point_features=True)


def idol_weights_dir(repo="speridlabs/idol"):
    """The local snapshot of the released weights (downloads it on first use)."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["config.json", "*.safetensors"])


def load_config(weights_dir):
    with open(os.path.join(weights_dir, "config.json")) as f:
        return json.load(f)


def load_teacher(weights_dir, device="cpu", dtype=torch.float32):
    """The released denoiser, frozen, in fp32 masters (Idol's own fp16 cast undone)."""
    cfg = load_config(weights_dir)
    model = LatentSeqMMFlowModel(**cfg["denoiser"]["args"]).float()
    model.dtype = torch.float32
    model.load_state_dict(load_file(os.path.join(weights_dir, "denoiser.safetensors")), strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)


def load_student(weights_dir, device="cpu", **adapter_kwargs):
    """The released denoiser plus the zero-initialised Budget3D adapters."""
    cfg = load_config(weights_dir)
    args = {**ADAPTER_DEFAULTS, **adapter_kwargs, **cfg["denoiser"]["args"]}
    model = Budget3DModel(**args)
    missing, unexpected = model.load_state_dict(load_file(os.path.join(weights_dir, "denoiser.safetensors")),
                                                strict=False)
    assert not unexpected, f"unexpected keys in the released checkpoint: {unexpected[:6]}"
    bad = [k for k in missing if not k.startswith("bp_") and k not in
           ("perm_to_octree", "inv_perm", "cube_to_cells", "cell_ctr", "cube_ctr")]
    assert not bad, f"released weights missing for backbone tensors: {bad[:6]}"
    model.assert_zero_init()
    return model.to(device).bind_geometry()


def read_checkpoint(path, ema=True):
    """Trained weights -> (state dict, step).  Accepts a training checkpoint (``checkpoint-*.pth``: the EMA weights, or
    the raw ones with ``ema=False``) and the released ``.safetensors`` file (the EMA weights, step in its metadata)."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        return load_file(path), int(meta["step"]) if str(meta.get("step", "")).isdigit() else None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (ck["ema"] if (ema and "ema" in ck) else ck["model"]), ck.get("step")


def load_trained(student, path, ema=True):
    """Load trained weights into a student built by load_student (after make_turbo for the turbo variant); every
    tensor of the file must land.  Returns the training step."""
    state, step = read_checkpoint(path, ema)
    missing, unexpected = student.load_state_dict(state, strict=False)
    assert not unexpected, f"checkpoint keys the model does not have: {unexpected[:6]}"
    assert len(set(state) - set(missing)) == len(state), "some checkpoint tensors were not loaded"
    return step


def load_budget3d(ckpt=None, turbo=False, lora=None, device="cpu", ema=True, weights_dir=None):
    """A ready-to-sample model -> (model, training step).

    ``ckpt``: a released ``.safetensors`` file or a training checkpoint; None gives the untrained start (the released
    Idol, or Idol-turbo with ``turbo``, plus zero adapters).  ``turbo``: inject Idol-turbo's phase-LoRA experts first
    (``lora``: their file, downloaded from speridlabs/idol-turbo if omitted), then load the trained weights on top.
    """
    model = load_student(weights_dir or idol_weights_dir(), device=device)
    if turbo:
        from .turbo import make_turbo
        make_turbo(model, lora)
    step = load_trained(model, ckpt, ema) if ckpt else 0
    return model.bind_geometry().eval(), step


def load_decoders(weights_dir, device="cpu"):
    """Idol's frozen octree-occupancy and Gaussian decoders (unchanged) -> (dec, dec_gs, config)."""
    from idol.models import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
    cfg = load_config(weights_dir)

    def build(cls, key, name):
        m = cls(**cfg[key]["args"])
        m.load_state_dict(load_file(os.path.join(weights_dir, f"{name}.safetensors")), strict=True)
        return m.eval().requires_grad_(False).to(device)
    return build(OctreeProbabilityFixedlenDecoder, "decoder", "decoder"), build(GaussianFixedlenDecoder, "decoder_gs", "decoder_gs"), cfg


@torch.no_grad()
def decode_gaussians(latent, dec, dec_gs, cfg, num_gaussians=262144):
    """A [1, 8192, 16] latent -> Idol's 3D Gaussians (anchor points sampled from the octree occupancy)."""
    from idol.models import OctreeProbabilityFixedlenDecoder
    per = dec_gs.rep_config["num_gaussians"]
    anchors = OctreeProbabilityFixedlenDecoder.sample(dec, latent.float(), num_points=max(1, num_gaussians // per),
                                                      level=cfg["max_voxel_level"])
    return dec_gs.to_representation(anchors, dec_gs(anchors, latent.float()))[0]


@torch.no_grad()
def check_dense_parity(student, teacher, batch=1, seed=0, device="cpu", tol=0.0):
    """At the dense layout the student must compute exactly what the released model computes.

    Both run in the same precision, so the only admissible difference is zero (``tol=0``);
    a non-zero tolerance is for comparing under bf16 autocast, where it is not bit-exact.
    """
    # the released model has no autocast, so compare in the same precision: a tolerance
    # here would hide exactly the drift this gate exists to catch
    was = student.bf16_blocks
    student.bf16_blocks = False
    for m in student.bp_refiners.values():
        m.bf16_blocks = False
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, c = teacher.q_token_length, teacher.in_channels
    x = TensorDict({"latent": torch.randn(batch, n, c, generator=g).to(device),
                    "camera": torch.randn(batch, 1, teacher.cam_channels, generator=g).to(device)},
                   batch_size=batch)
    cond = TensorDict({"feature1": torch.randn(batch, 4101, 1280, generator=g).to(device),
                       "feature2": torch.randn(batch, 4101, 128, generator=g).to(device)}, batch_size=batch)
    t = torch.full((batch,), 731.0, device=device)
    ref = teacher(x, t, cond)
    got = student(x, t, cond, layout=dense_layout(batch, device))
    report = {}
    for key in ("latent", "camera"):
        d = (got[key].float() - ref[key].float()).abs().max().item()
        report[key] = d
    student.bf16_blocks = was
    for m in student.bp_refiners.values():
        m.bf16_blocks = was
    for key, d in report.items():
        assert d <= tol, f"dense parity failed on {key}: max |diff| = {d:.3e} (tol {tol:g})"
    return report
