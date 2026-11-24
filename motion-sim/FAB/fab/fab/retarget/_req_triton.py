"""The Triton kernels of fab.retarget.req_kernel (imported only where Triton is available)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as ld

K_OFF = 25  # footprint offsets per element (Surface.req_const)
KP = 32  # padded


@triton.jit
def _bilinear(H, NY, NX, c, px, py, x0, y0, res, nx, ny):
    """One grid at (px, py): the clamped bilinear height and what the backward needs (all (BM, KP) tensors)."""
    u0 = ld.div_rn(ld.sub_rn(px, x0), res)
    v0 = ld.div_rn(ld.sub_rn(py, y0), res)
    limu = ld.sub_rn(nx, 1.001)
    limv = ld.sub_rn(ny, 1.001)
    cu = tl.where(u0 < 0.0, 0.0, u0)  # clamp_min(0)
    cv = tl.where(v0 < 0.0, 0.0, v0)
    u = tl.where(cu > limu, limu, cu)  # torch.minimum (no NaNs here)
    v = tl.where(cv > limv, limv, cv)
    fi = tl.floor(v)
    fj = tl.floor(u)
    i = fi.to(tl.int64)
    j = fj.to(tl.int64)
    a = ld.sub_rn(v, fi)
    b = ld.sub_rn(u, fj)
    base = (c * NY + i) * NX + j
    h00 = tl.load(H + base)
    h01 = tl.load(H + base + 1)
    h10 = tl.load(H + base + NX)
    h11 = tl.load(H + base + NX + 1)
    oma = ld.sub_rn(1.0, a)
    omb = ld.sub_rn(1.0, b)
    q00 = ld.mul_rn(ld.mul_rn(oma, omb), h00)
    q01 = ld.mul_rn(ld.mul_rn(oma, b), h01)
    q10 = ld.mul_rn(ld.mul_rn(a, omb), h10)
    q11 = ld.mul_rn(ld.mul_rn(a, b), h11)
    h = ld.add_rn(ld.add_rn(ld.add_rn(q00, q01), q10), q11)
    return h, u0, v0, cu, cv, limu, limv, a, b, oma, omb, h00, h01, h10, h11


@triton.jit
def _bilinear_bwd(g, u0, v0, cu, cv, limu, limv, a, b, oma, omb, h00, h01, h10, h11, res):
    """d h / d (px, py) times g, as autograd accumulates it for the expression in _bilinear / Surface._grid."""
    gw00 = ld.mul_rn(g, h00)
    gw01 = ld.mul_rn(g, h01)
    gw10 = ld.mul_rn(g, h10)
    gw11 = ld.mul_rn(g, h11)
    # w00 = (1 - a) * (1 - b), w01 = (1 - a) * b, w10 = a * (1 - b), w11 = a * b (MulBackward0: grad * other)
    gA1 = ld.mul_rn(gw00, omb)
    gB1 = ld.mul_rn(gw00, oma)
    gA2 = ld.mul_rn(gw01, b)
    gb01 = ld.mul_rn(gw01, oma)
    ga10 = ld.mul_rn(gw10, omb)
    gB2 = ld.mul_rn(gw10, a)
    ga11 = ld.mul_rn(gw11, b)
    gb11 = ld.mul_rn(gw11, a)
    # the engine runs the nodes latest-first: a gets w11, w10, (1 - a)#2, (1 - a)#1; b gets w11, (1 - b)#2, w01, (1 - b)#1
    ga = ld.add_rn(ld.add_rn(ld.add_rn(ga11, ga10), -gA2), -gA1)
    gb = ld.add_rn(ld.add_rn(ld.add_rn(gb11, -gB2), gb01), -gB1)
    # v = minimum(clamp_min(v0, 0), lim): minimum ties halve, clamp_min passes where v0 >= 0; v0 = (p - y0) / res
    gcv = tl.where(cv == limv, ld.mul_rn(ga, 0.5), tl.where(cv > limv, 0.0, ga))
    gv0 = tl.where(v0 >= 0.0, gcv, 0.0)
    gcu = tl.where(cu == limu, ld.mul_rn(gb, 0.5), tl.where(cu > limu, 0.0, gb))
    gu0 = tl.where(u0 >= 0.0, gcu, 0.0)
    return ld.div_rn(gu0, res), ld.div_rn(gv0, res)


@triton.jit
def _req_fwd(xa, xb, da, db, rs, cid, out, kst,
             H0, NY0, NX0, X00, Y00, R0, NXF0, NYF0,
             H1, NY1, NX1, X01, Y01, R1, NXF1, NYF1,
             M, BM: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BM + tl.arange(0, BM)
    k = tl.arange(0, 32)
    mm = m < M
    km = k < 25
    msk = mm[:, None] & km[None, :]
    m2 = m[:, None]
    k2 = k[None, :]
    c = tl.load(cid + m, mask=mm, other=0)[:, None]
    ax = tl.load(xa + m * 2, mask=mm, other=0.0)[:, None]
    ay = tl.load(xa + m * 2 + 1, mask=mm, other=0.0)[:, None]
    bx = tl.load(xb + m * 2, mask=mm, other=0.0)[:, None]
    by = tl.load(xb + m * 2 + 1, mask=mm, other=0.0)[:, None]
    dax = tl.load(da + (m2 * 25 + k2) * 2, mask=msk, other=0.0)
    day = tl.load(da + (m2 * 25 + k2) * 2 + 1, mask=msk, other=0.0)
    dbx = tl.load(db + (m2 * 25 + k2) * 2, mask=msk, other=0.0)
    dby = tl.load(db + (m2 * 25 + k2) * 2 + 1, mask=msk, other=0.0)
    r_s = tl.load(rs + m2 * 25 + k2, mask=msk, other=0.0)
    h0, _u0, _v0, _cu, _cv, _lu, _lv, _a, _b, _oa, _ob, _h00, _h01, _h10, _h11 = _bilinear(
        H0, NY0, NX0, c, ld.add_rn(ax, dax), ld.add_rn(ay, day), tl.load(X00 + c), tl.load(Y00 + c), tl.load(R0 + c),
        tl.load(NXF0 + c), tl.load(NYF0 + c))
    nx1 = tl.load(NXF1 + c)
    ny1 = tl.load(NYF1 + c)
    h1, u01, v01, _cu1, _cv1, _lu1, _lv1, _a1, _b1, _oa1, _ob1, _g00, _g01, _g10, _g11 = _bilinear(
        H1, NY1, NX1, c, ld.add_rn(bx, dbx), ld.add_rn(by, dby), tl.load(X01 + c), tl.load(Y01 + c), tl.load(R1 + c), nx1, ny1)
    ok = (u01 >= 0.0) & (u01 <= ld.sub_rn(nx1, 1.0)) & (v01 >= 0.0) & (v01 <= ld.sub_rn(ny1, 1.0))
    h1 = tl.where(ok, h1, -1000.0)
    val = ld.add_rn(tl.where(h0 < h1, h1, h0), r_s)
    val = tl.where(msk, val, float("-inf"))
    best = tl.max(val, 1)
    kk = tl.min(tl.where(val == best[:, None], k2, 32), 1)  # the first maximal offset (torch.max(dim))
    tl.store(out + m, best, mask=mm)
    tl.store(kst + m, kk.to(tl.int32), mask=mm)


@triton.jit
def _req_bwd(xa, xb, da, db, cid, kst, gout, ga_, gb_,
             H0, NY0, NX0, X00, Y00, R0, NXF0, NYF0,
             H1, NY1, NX1, X01, Y01, R1, NXF1, NYF1,
             M, BM: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BM + tl.arange(0, BM)
    mm = m < M
    k = tl.load(kst + m, mask=mm, other=0).to(tl.int64)
    c = tl.load(cid + m, mask=mm, other=0)
    g = tl.load(gout + m, mask=mm, other=0.0)
    ax = tl.load(xa + m * 2, mask=mm, other=0.0)
    ay = tl.load(xa + m * 2 + 1, mask=mm, other=0.0)
    bx = tl.load(xb + m * 2, mask=mm, other=0.0)
    by = tl.load(xb + m * 2 + 1, mask=mm, other=0.0)
    dax = tl.load(da + (m * 25 + k) * 2, mask=mm, other=0.0)
    day = tl.load(da + (m * 25 + k) * 2 + 1, mask=mm, other=0.0)
    dbx = tl.load(db + (m * 25 + k) * 2, mask=mm, other=0.0)
    dby = tl.load(db + (m * 25 + k) * 2 + 1, mask=mm, other=0.0)
    r0 = tl.load(R0 + c)
    h0, u0a, v0a, cua, cva, lua, lva, aa, ba, omaa, omba, w00, w01, w10, w11 = _bilinear(
        H0, NY0, NX0, c, ld.add_rn(ax, dax), ld.add_rn(ay, day), tl.load(X00 + c), tl.load(Y00 + c), r0, tl.load(NXF0 + c), tl.load(NYF0 + c))
    r1 = tl.load(R1 + c)
    nx1 = tl.load(NXF1 + c)
    ny1 = tl.load(NYF1 + c)
    h1, u0b, v0b, cub, cvb, lub, lvb, ab, bb, omab, ombb, b00, b01, b10, b11 = _bilinear(
        H1, NY1, NX1, c, ld.add_rn(bx, dbx), ld.add_rn(by, dby), tl.load(X01 + c), tl.load(Y01 + c), r1, nx1, ny1)
    ok = (u0b >= 0.0) & (u0b <= ld.sub_rn(nx1, 1.0)) & (v0b >= 0.0) & (v0b <= ld.sub_rn(ny1, 1.0))
    h1 = tl.where(ok, h1, -1000.0)
    # torch.maximum(h0, h1): ties halve the gradient
    gh = ld.mul_rn(g, 0.5)
    g0 = tl.where(h0 == h1, gh, tl.where(h0 < h1, 0.0, g))
    g1 = tl.where(h0 == h1, gh, tl.where(h0 > h1, 0.0, g))
    g1 = tl.where(ok, g1, 0.0)
    gax, gay = _bilinear_bwd(g0, u0a, v0a, cua, cva, lua, lva, aa, ba, omaa, omba, w00, w01, w10, w11, r0)
    gbx, gby = _bilinear_bwd(g1, u0b, v0b, cub, cvb, lub, lvb, ab, bb, omab, ombb, b00, b01, b10, b11, r1)
    tl.store(ga_ + m * 2, gax, mask=mm)
    tl.store(ga_ + m * 2 + 1, gay, mask=mm)
    tl.store(gb_ + m * 2, gbx, mask=mm)
    tl.store(gb_ + m * 2 + 1, gby, mask=mm)


BM_F, BM_B = 64, 256


def _gargs(g):
    H = g["H"]
    assert H.dtype == torch.float32 and H.is_contiguous()
    return (H, H.shape[1], H.shape[2], g["x0"].contiguous(), g["y0"].contiguous(), g["res"].contiguous(), g["nx"].contiguous(),
            g["ny"].contiguous())


class _Req(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xa, xb, da, db, rs, cid, g0, g1):
        M = xa.shape[0]
        out = torch.empty(M, device=xa.device, dtype=torch.float32)
        kst = torch.empty(M, device=xa.device, dtype=torch.int32)
        a0, a1 = _gargs(g0), _gargs(g1)
        _req_fwd[(triton.cdiv(M, BM_F),)](xa, xb, da, db, rs, cid, out, kst, *a0, *a1, M, BM=BM_F, enable_fp_fusion=False)
        ctx.save_for_backward(xa, xb, da, db, cid, kst)
        ctx.g = (a0, a1)
        return out

    @staticmethod
    def backward(ctx, gout):
        xa, xb, da, db, cid, kst = ctx.saved_tensors
        a0, a1 = ctx.g
        M = xa.shape[0]
        ga = torch.empty_like(xa)
        gb = torch.empty_like(xb)
        _req_bwd[(triton.cdiv(M, BM_B),)](xa, xb, da, db, cid, kst, gout.contiguous(), ga, gb, *a0, *a1, M, BM=BM_B, enable_fp_fusion=False)
        return ga, gb, None, None, None, None, None, None


def req_fused(xy, slot_xy, const, g0, g1):
    """Surface.req_cols(xy, slot_xy, const) for xy / slot_xy (..., 2) and req_const's tensors of the same leading shape."""
    _, cid, d, ds, rs = const
    lead = xy.shape[:-1]
    M = int(torch.Size(lead).numel())
    xa = xy.reshape(M, 2)
    xb = slot_xy.reshape(M, 2)
    xa = xa if xa.is_contiguous() else xa.contiguous()
    xb = xb if xb.is_contiguous() else xb.contiguous()
    out = _Req.apply(xa, xb, d.reshape(M, K_OFF, 2).contiguous(), ds.reshape(M, K_OFF, 2).contiguous(), rs.reshape(M, K_OFF).contiguous(),
                     cid.reshape(M).contiguous(), g0, g1)
    return out.reshape(lead)


# --- the solver's terrain lookup (bench/videomimic/scene_retarget.Terrain.forward, single-level grid) -------------------------
# Terrain.forward takes python-float x0 / y0 / res: PyTorch subtracts them as float32 scalars and divides by multiplying with the
# reciprocal of the double res rounded to float32 (div_true_kernel_cuda's CPU-scalar path, checked); its clamp bounds are float32(nx - 1.001)
# computed in double. The kernel does the same, and the backward is autograd's (clamp passes where lo <= x <= hi, the division's
# backward is the same reciprocal multiply, nan_to_num passes finite inputs).

FLT_MAX = 3.4028234663852886e38


@triton.jit
def _terr_fwd_bwd(xy, H, out, g, gxy, x0, y0, inv, hiu, hiv, NX, M, BWD: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BM + tl.arange(0, BM)
    mm = m < M
    x = tl.load(xy + m * 2, mask=mm, other=0.0)
    y = tl.load(xy + m * 2 + 1, mask=mm, other=0.0)
    fx = (x == x) & (x != float("inf")) & (x != float("-inf"))
    fy = (y == y) & (y != float("inf")) & (y != float("-inf"))
    x = tl.where(x != x, 0.0, tl.where(x == float("inf"), 3.4028234663852886e38, tl.where(x == float("-inf"), -3.4028234663852886e38, x)))
    y = tl.where(y != y, 0.0, tl.where(y == float("inf"), 3.4028234663852886e38, tl.where(y == float("-inf"), -3.4028234663852886e38, y)))
    u0 = ld.mul_rn(ld.sub_rn(x, x0), inv)
    v0 = ld.mul_rn(ld.sub_rn(y, y0), inv)
    u = tl.where(u0 < 0.0, 0.0, u0)
    u = tl.where(u > hiu, hiu, u)
    v = tl.where(v0 < 0.0, 0.0, v0)
    v = tl.where(v > hiv, hiv, v)
    fi = tl.floor(v)
    fj = tl.floor(u)
    base = fi.to(tl.int64) * NX + fj.to(tl.int64)
    a = ld.sub_rn(v, fi)
    b = ld.sub_rn(u, fj)
    h00 = tl.load(H + base, mask=mm, other=0.0)
    h01 = tl.load(H + base + 1, mask=mm, other=0.0)
    h10 = tl.load(H + base + NX, mask=mm, other=0.0)
    h11 = tl.load(H + base + NX + 1, mask=mm, other=0.0)
    oma = ld.sub_rn(1.0, a)
    omb = ld.sub_rn(1.0, b)
    if BWD:
        gg = tl.load(g + m, mask=mm, other=0.0)
        gw00 = ld.mul_rn(gg, h00)
        gw01 = ld.mul_rn(gg, h01)
        gw10 = ld.mul_rn(gg, h10)
        gw11 = ld.mul_rn(gg, h11)
        ga = ld.add_rn(ld.add_rn(ld.add_rn(ld.mul_rn(gw11, b), ld.mul_rn(gw10, omb)), -ld.mul_rn(gw01, b)), -ld.mul_rn(gw00, omb))
        gb = ld.add_rn(ld.add_rn(ld.add_rn(ld.mul_rn(gw11, a), -ld.mul_rn(gw10, a)), ld.mul_rn(gw01, oma)), -ld.mul_rn(gw00, oma))
        gu = tl.where((u0 >= 0.0) & (u0 <= hiu), gb, 0.0)
        gv = tl.where((v0 >= 0.0) & (v0 <= hiv), ga, 0.0)
        gx = ld.mul_rn(ld.mul_rn(gu, inv), tl.where(fx, 1.0, 0.0))
        gy = ld.mul_rn(ld.mul_rn(gv, inv), tl.where(fy, 1.0, 0.0))
        tl.store(gxy + m * 2, gx, mask=mm)
        tl.store(gxy + m * 2 + 1, gy, mask=mm)
    else:
        q00 = ld.mul_rn(ld.mul_rn(oma, omb), h00)
        q01 = ld.mul_rn(ld.mul_rn(oma, b), h01)
        q10 = ld.mul_rn(ld.mul_rn(a, omb), h10)
        q11 = ld.mul_rn(ld.mul_rn(a, b), h11)
        tl.store(out + m, ld.add_rn(ld.add_rn(ld.add_rn(q00, q01), q10), q11), mask=mm)


BM_T = 256


def _terr_scalars(x0, y0, res, ny, nx):
    import numpy as np

    f = np.float32
    return float(f(x0)), float(f(y0)), float(f(1.0 / res)), float(f(nx - 1.001)), float(f(ny - 1.001))  # 1 / res in double, as PyTorch


class _Terr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, H, sc):
        M = xy.shape[0]
        out = torch.empty(M, device=xy.device, dtype=torch.float32)
        _terr_fwd_bwd[(triton.cdiv(M, BM_T),)](xy, H, out, out, xy, *sc, H.shape[1], M, BWD=False, BM=BM_T, enable_fp_fusion=False)
        ctx.save_for_backward(xy, H)
        ctx.sc = sc
        return out

    @staticmethod
    def backward(ctx, g):
        xy, H = ctx.saved_tensors
        M = xy.shape[0]
        gxy = torch.empty_like(xy)
        _terr_fwd_bwd[(triton.cdiv(M, BM_T),)](xy, H, g, g.contiguous(), gxy, *ctx.sc, H.shape[1], M, BWD=True, BM=BM_T,
                                               enable_fp_fusion=False)
        return gxy, None, None


def terrain_fused(H, x0, y0, res, xy):
    """scene_retarget.Terrain.forward (no levels) for xy (..., 2): the same values and gradient."""
    lead = xy.shape[:-1]
    M = int(torch.Size(lead).numel())
    x2 = xy.reshape(M, 2)
    x2 = x2 if x2.is_contiguous() else x2.contiguous()
    assert H.dtype == torch.float32 and H.is_contiguous() and xy.dtype == torch.float32
    return _Terr.apply(x2, H, _terr_scalars(x0, y0, res, H.shape[0], H.shape[1])).reshape(lead)
