"""Mask-guided fingertip extension: the network's fingertips follow the MANO tip-vertex convention of its training labels,
which sits short of the visible end of the finger (most visible on close-ups). Given the hand's instance mask, each tip is
moved outward along its distal phalanx (DIP -> tip) to where the mask ends, at most `max_ext` x the phalanx length; a tip
outside its mask, a finger whose ray leaves the mask immediately, or one whose ray is still inside the mask at the cap
(curled / occluded fingertip: the ray crosses the palm or another finger) is left unchanged.
"""
import numpy as np

TIPS = (4, 8, 12, 16, 20)
DIPS = (3, 7, 11, 15, 19)


def tip_rays(kp2d, max_ext=0.6, half_width=1, min_len=4.0, mask_px=0.0, min_ratio=6.0):
    """The mask pixels extend_tips reads: per finger with a long enough distal phalanx, its tip pixel then the ray steps x
    widths -> (rays, xs, ys); rays = [(finger, unit dir, phalanx length, steps, widths)], xs / ys int pixels in that order."""
    kp = np.asarray(kp2d, np.float64)
    rays, xs, ys = [], [], []
    for f, (d, t) in enumerate(zip(DIPS, TIPS)):
        v = kp[t] - kp[d]
        L = float(np.linalg.norm(v))
        if L < max(min_len, min_ratio * mask_px):
            continue
        u, n = v / L, np.array([-v[1], v[0]]) / L
        s = np.arange(1.0, max_ext * L + 1e-6, 1.0)  # all ray steps at once, +-half_width px across the ray
        w = np.arange(-half_width, half_width + 1, dtype=np.float64)
        q = kp[t][None, None] + s[:, None, None] * u + w[None, :, None] * n  # (steps, widths, 2)
        xs += [np.array([int(round(kp[t, 0]))]), np.round(q[..., 0]).astype(int).ravel()]
        ys += [np.array([int(round(kp[t, 1]))]), np.round(q[..., 1]).astype(int).ravel()]
        rays.append((f, u, L, s, len(w)))
    if not rays:
        return rays, np.zeros(0, int), np.zeros(0, int)
    return rays, np.concatenate(xs), np.concatenate(ys)


def tips_from_hits(kp2d, rays, hits, gap=2, mask_px=0.0):
    """extend_tips from the mask values at tip_rays' pixels (hits, bool, same order) -> (21, 2) kp, (5,) extension."""
    kp = np.asarray(kp2d, np.float64).copy()
    ext = np.zeros(5)
    o = 0
    for f, u, L, s, nw in rays:
        h = hits[o:o + 1 + len(s) * nw]
        o += len(h)
        if not h[0]:  # tip outside its mask
            continue
        hit = h[1:].reshape(len(s), nw).any(1)  # (steps,)
        # the ray exits the mask once gap + 1 consecutive steps miss; the extension is the last hit before that
        miss = ~hit
        run = np.convolve(miss.astype(int), np.ones(gap + 1, int), mode="valid") == gap + 1  # window of gap + 1 misses
        if not run.any():  # still inside the mask at the cap: the ray runs over the palm / another finger
            continue
        first = int(np.argmax(run))  # index of the first missing step of that run
        hits_before = np.nonzero(hit[:first])[0]
        last = float(s[hits_before[-1]]) if len(hits_before) else 0.0
        last = max(last - 0.5 * mask_px, 0.0)
        kp[TIPS[f]] = kp[TIPS[f]] + last * u
        ext[f] = last / L
    return kp, ext


def extend_tips(kp2d, mask, max_ext=0.6, gap=2, half_width=1, min_len=4.0, mask_px=0.0, min_ratio=6.0):
    """kp2d (21, 2) image px, mask (H, W) bool -> (21, 2) copy with the tips moved, and the per-finger extension (5,)
    in units of the distal phalanx length. mask_px = the mask's native resolution in image px (a YOLO-seg prototype
    cell): fingers whose distal phalanx is shorter than min_ratio cells are left alone (a coarse mask overshoots small
    fingertips) and half a cell is taken off the extension (the upsampled mask boundary lies about that far out)."""
    H, W = mask.shape
    rays, xs, ys = tip_rays(kp2d, max_ext, half_width, min_len, mask_px, min_ratio)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    hits = np.zeros(len(xs), bool)
    hits[ok] = mask[ys[ok], xs[ok]]
    return tips_from_hits(kp2d, rays, hits, gap, mask_px)
