"""The stance pin's penetration lookup (bench/stance_pin.Surface.req / req_cols) as one fused Triton kernel, forward and backward.

Surface.req evaluates, for every (frame, collision point) element, 25 footprint offsets on two terrain grids (bilinear) and keeps
the highest sphere-centre requirement; in PyTorch that is ~60 elementwise kernels over (elements x 25) tensors plus their
backward. This kernel computes the same values with the same IEEE float32 operations in the same order (explicit round-to-nearest
add / sub / mul / div, no FMA contraction), and the backward reproduces autograd's gradient for that expression exactly: the
max passes the gradient to the first maximal offset only, torch.maximum / torch.minimum ties split it in half, clamp_min passes
it where x >= 0, and each bilinear weight's gradient is accumulated in the order the autograd engine adds it. The result is
bitwise the PyTorch expression (checked against it in tests/test_req_kernel.py and on the chain's outputs).
"""
try:
    import triton  # noqa: F401

    AVAILABLE = True
except ImportError:  # no Triton build for this platform: Surface.req / Terrain.forward keep their PyTorch expressions
    AVAILABLE = False


def req_fused(xy, slot_xy, const, g0, g1):
    """bench.stance_pin.Surface.req_cols(xy, slot_xy, const) for xy / slot_xy (..., 2) and req_const's tensors of that shape."""
    from fab.retarget._req_triton import req_fused as f

    return f(xy, slot_xy, const, g0, g1)


def terrain_fused(H, x0, y0, res, xy):
    """bench.videomimic.scene_retarget.Terrain.forward (no levels) for xy (..., 2): the same values and gradient."""
    from fab.retarget._req_triton import terrain_fused as f

    return f(H, x0, y0, res, xy)
