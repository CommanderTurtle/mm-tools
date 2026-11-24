"""The turbo parity gate: at the dense layout, with the Budget3D adapters still zero, the student must compute
exactly what idol-turbo computes -- checked against the *official* turbo.py (global slow/fast switch) on Idol's
own model class, for each phase and for a batch that mixes both phases row by row."""
import importlib.util

import torch
from tensordict import TensorDict

from ..octree import dense_layout
from .load import load_teacher
from .turbo import LORA_ALPHA, LORA_RANK, TURBO_REPO, set_lora_fused


def official_turbo_module():
    from huggingface_hub import hf_hub_download
    spec = importlib.util.spec_from_file_location("idol_turbo_official", hf_hub_download(TURBO_REPO, "turbo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def check_turbo_parity(student, weights, lora, device="cuda", batch=2, seed=0, tol=1e-5, fused=False):
    """Returns {case: max |difference|} and asserts each is within ``tol``.  By default the student's LoRA runs in the
    classic form for the comparison, which is written as the official code writes it and gives 0.0; ``fused=True``
    checks the one-pass form instead (differs by float rounding, use a tolerance around 5e-4)."""
    from safetensors.torch import load_file
    was_fused = set_lora_fused(fused)
    off = official_turbo_module()
    ref = load_teacher(weights, device=device)
    off._inject(ref, LORA_RANK, LORA_ALPHA)
    ref.to(device)
    missing, unexpected = ref.load_state_dict(load_file(lora), strict=False)
    assert not unexpected, unexpected[:4]

    was = student.bf16_blocks
    mods = [m for m in student.modules() if hasattr(m, "bf16_blocks")]
    for m in mods:
        m.bf16_blocks = False
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, c = ref.q_token_length, ref.in_channels
    x = TensorDict({"latent": torch.randn(batch, n, c, generator=g).to(device),
                    "camera": torch.randn(batch, 1, ref.cam_channels, generator=g).to(device)}, batch_size=batch)
    cond = TensorDict({"feature1": torch.randn(batch, 4101, 1280, generator=g).to(device),
                       "feature2": torch.randn(batch, 4101, 128, generator=g).to(device)}, batch_size=batch)
    report = {}
    official = {}
    for phase, tval in (("slow", 800.0), ("fast", 200.0)):
        t = torch.full((batch,), tval, device=device)
        off._set_active(ref, phase)
        official[phase] = ref(x, t, cond)
        got = student(x, t, cond, layout=dense_layout(batch, device))
        for k in ("latent", "camera"):
            report[f"{phase}/{k}"] = (got[k].float() - official[phase][k].float()).abs().max().item()
    # mixed batch: row 0 slow, row 1 fast -- must equal the per-phase official rows
    t = torch.tensor([800.0, 200.0][:batch], device=device)
    got = student(x, t, cond, layout=dense_layout(batch, device))
    for k in ("latent", "camera"):
        exp = torch.cat([official["slow"][k][:1], official["fast"][k][1:2]])
        report[f"mixed/{k}"] = (got[k].float() - exp.float()).abs().max().item()
    for m in mods:
        m.bf16_blocks = was
    set_lora_fused(was_fused)
    del ref
    for key, d in report.items():
        assert d <= tol, f"turbo parity failed on {key}: max |diff| = {d:.3e} (tol {tol:g})"
    return report
