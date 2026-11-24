"""Sampling at a token budget (the CLI is scripts/sample.py).

    sampler     Budget3D-Idol: 50 Euler steps with classifier-free guidance; Idol's framing of an input image
    turbo       Budget3D-Idol-Turbo: ~5 guidance-free Euler steps
    fast        optimised inference: bf16 blocks, fused attention, torch.compile
"""
