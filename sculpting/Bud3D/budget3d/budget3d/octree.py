"""Budget3D layouts: the Sobol-to-octree bijection, 3D salience, and the token layouts.

Idol's 8192 latent tokens are a stratified 3D Sobol sequence whose occupancy is exactly
16x16x16x2: every 16^3 voxel holds two antipodal points, one per x-half ([0, 1/32) vs
[1/32, 2/32)).  That makes the sequence a relabelling of a three-level octree and gives
three isotropic cubic scales:

    s = 1   one Sobol point                            8192 tokens  (100 %)
    s = 2   one 16^3 cell, side 1/16, two points       4096 tokens  ( 50 %)
    s = 4   one  8^3 cube, side 1/8, sixteen points     512 tokens  (6.25 %)

A layout is a top-down split of that octree: ``split_4`` says which of the 512 cubes open
into their 8 cells, ``split_2`` which of those cells open into their 2 points.  The token
count is exact,

    N = 512 + 7 * split_4.sum() + split_2.sum()      in {512} u [519, 8192]

so a budget is realised *by rank* rather than by bisecting a threshold: taking the k
highest-scoring cubes is the same family of layouts as thresholding the score at its k-th
largest value, and it lands on the requested N exactly.  Because (k_4, k_2) are shared
across a micro-batch while the *locations* stay per-example, every sequence in a batch has
the same length N and the same per-scale token counts: no padding, no varlen kernels, and
the training layout is built by the very same code path as the inference layout.
"""
import torch

N_SOBOL = 8192
N_CELL = 4096          # 16^3 cells, each holding 2 Sobol points
N_CUBE = 512           # 8^3 cubes, each holding 8 cells / 16 Sobol points
SCALES = (1, 2, 4)
SCALE_IDX = {1: 0, 2: 1, 4: 2}     # row of bp_scale_emb used by each scale

# base merge thresholds (tau_4, tau_2), used only by the threshold-offset layout mode
THRESHOLDS = (2.1, 1.5)
SCORE_SCALE = 0.08


# ---------------------------------------------------------------------------------------
# the bijection
# ---------------------------------------------------------------------------------------
def octree_permutation(pos):
    """pos [8192, 3] in [0, 1]^3 -> (perm_to_octree, inv_perm), both [8192] long.

    ``pos[perm_to_octree].view(16, 16, 16, 2, 3)`` is the Sobol set laid out on the octree
    grid; ``inv_perm`` maps octree order back to Idol's own token order.  Raises if the
    16^3 x 2 occupancy does not hold exactly, because every count below depends on it.
    """
    pos = pos.detach().float()
    ix = (pos[:, 0] * 16).long().clamp(0, 15)
    iy = (pos[:, 1] * 16).long().clamp(0, 15)
    iz = (pos[:, 2] * 16).long().clamp(0, 15)
    hx = (pos[:, 0] * 32).long() % 2
    oct_key = ((ix * 16 + iy) * 16 + iz) * 2 + hx
    if torch.unique(oct_key).numel() != N_SOBOL:
        raise ValueError("pos_pe is not an exact 16^3 x 2 stratified Sobol set; "
                         "the Budget3D scale hierarchy does not apply to this checkpoint")
    perm_to_octree = torch.argsort(oct_key, stable=True)
    return perm_to_octree, torch.argsort(perm_to_octree)


def cell_centers(device=None, dtype=torch.float32):
    """[4096, 3]: centre of each 16^3 cell in [0, 1]^3, in octree (cell) order."""
    g = (torch.arange(16, device=device, dtype=dtype) + 0.5) / 16
    c = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1)
    return c.reshape(N_CELL, 3)


def cube_centers(device=None, dtype=torch.float32):
    """[512, 3]: centre of each 8^3 cube in [0, 1]^3, in octree (cube) order."""
    g = (torch.arange(8, device=device, dtype=dtype) + 0.5) / 8
    c = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1)
    return c.reshape(N_CUBE, 3)


def cells_to_cubes(x):
    """[B, 16, 16, 16, ...] -> [B, 8, 8, 8, 8, ...]: group each 2x2x2 block of cells.

    The sub-cell axis that appears is ordered (a, b, c) over the x, y, z halves, which is
    the ordering ``cube_child_offsets`` describes and the refiner's slots assume.
    """
    B = x.shape[0]
    tail = x.shape[4:]
    x = x.reshape(B, 8, 2, 8, 2, 8, 2, *tail)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + len(tail)))
    return x.reshape(B, 8, 8, 8, 8, *tail)


def cube_child_offsets(device=None, dtype=torch.float32):
    """[8, 3]: centre of each child cell relative to its cube centre, the cube being [-1, 1]^3."""
    o = torch.tensor([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=device, dtype=dtype)
    return o - 0.5


def cell_child_offsets(device=None, dtype=torch.float32):
    """[2, 3]: the two Sobol slots' offsets inside a cell, the cell being [-1, 1]^3.

    Only the x-half is known analytically (slot 0 in [0, 1/2), slot 1 in [1/2, 1) of the
    cell's x extent); y and z carry no slot structure, so they stay at the centre.
    """
    return torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], device=device, dtype=dtype)


# ---------------------------------------------------------------------------------------
# salience
# ---------------------------------------------------------------------------------------
def detail_scores(z_oct):
    """z_oct [B, 16, 16, 16, 2, C] -> (score_4 [B, 512], score_2 [B, 4096]).

    The spec's rule: the spread of the latent over the points a token would merge,
    ``log2(1 + sqrt(mean_c Var_pts(z)) / 0.08)``, which is monotone in that spread, so
    ranking by it ranks by detail.
    """
    B = z_oct.shape[0]
    cell = z_oct.reshape(B, N_CELL, 2, -1)
    s2 = cell.var(dim=2, unbiased=False).mean(-1)
    cube = cells_to_cubes(z_oct).reshape(B, N_CUBE, 16, -1)
    s4 = cube.var(dim=2, unbiased=False).mean(-1)
    to_bits = lambda v: torch.log2(1 + torch.sqrt(v + 1e-8) / SCORE_SCALE)
    return to_bits(s4), to_bits(s2)


def random_scores(B, device, blob_p=0.70, generator=None):
    """Content-independent scores: smooth 3D Gaussian blobs (``blob_p``) or uniform noise.

    This is the spatial debiasing of the recipe -- it keeps the coarse and fine adapters
    calibrated everywhere in [0, 1]^3 instead of only where real objects put their detail.
    """
    rnd = lambda *s: torch.rand(*s, device=device, generator=generator)
    field = torch.empty(B, 16, 16, 16, device=device)
    centres = cell_centers(device).view(16, 16, 16, 3)
    for i in range(B):
        if torch.rand(1, generator=generator).item() < blob_p:
            n = int(torch.randint(1, 5, (1,), generator=generator).item())
            f = torch.zeros(16, 16, 16, device=device)
            for _ in range(n):
                mu = rnd(3)
                sigma = 0.08 + 0.22 * rnd(1).item()
                d2 = ((centres - mu) ** 2).sum(-1)
                f = f + torch.exp(-0.5 * d2 / sigma ** 2)
            field[i] = f / f.amax().clamp_min(1e-6)
        else:
            field[i] = rnd(16, 16, 16)
    s2 = field.reshape(B, N_CELL)
    s4 = cells_to_cubes(field.unsqueeze(-1)).reshape(B, N_CUBE, 8).mean(-1)
    return s4, s2


# ---------------------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------------------
def k4_bounds(n_tokens):
    """The range of cube splits that can realise ``n_tokens`` with a valid tree.

    k_2 = N - 512 - 7 k_4 must satisfy 0 <= k_2 <= 8 k_4, i.e.
    ceil((N - 512) / 15) <= k_4 <= min(512, floor((N - 512) / 7)).
    """
    n = int(n_tokens)
    if n == N_CUBE:
        return 0, 0
    lo = -(-(n - N_CUBE) // 15)
    hi = min(N_CUBE, (n - N_CUBE) // 7)
    if hi < lo:
        raise ValueError(f"{n} tokens is not realisable; use 512 or 519..8192")
    return lo, hi


def split_counts(n_tokens, k4=None):
    """(k_4, k_2) for a token count.  ``k_4`` is clamped into the valid range; without it
    the coarse-first choice ``min(512, (N - 512) // 7)`` is used, which is the layout a
    pure budget with no salience would produce."""
    n = int(n_tokens)
    lo, hi = k4_bounds(n)
    k4 = hi if k4 is None else int(min(max(int(k4), lo), hi))
    k2 = n - N_CUBE - 7 * k4
    assert 0 <= k2 <= 8 * k4 or n == N_CUBE, (n, k4, k2)
    return k4, k2


def balance_splits(score4, score2, n_tokens, thresholds=THRESHOLDS, iters=24):
    """Choose how a budget is spent between the two split levels, from the content.

    A shared offset delta shifts both merge thresholds; raising it merges more and so
    lowers the token count monotonically.  Bisecting delta until the threshold rule lands
    on ``n_tokens`` says how many cubes *the content* wants open; that count (shared across
    the micro-batch, so every sequence keeps the same length and the same per-scale counts)
    is then realised exactly by rank.  Without this the budget would be spent coarse-first
    and a 50 % layout would degenerate to the uniform s=2 grid, ignoring salience entirely.
    """
    n = snap_realizable_tokens(n_tokens)
    lo_k4, hi_k4 = k4_bounds(n)
    if lo_k4 == hi_k4:
        return split_counts(n, lo_k4)
    tau4, tau2 = thresholds
    b = score4.shape[0]
    span = 16.0
    lo, hi = -span, span

    def counts(delta):
        split4 = score4 > (tau4 + delta)
        elig = split4.view(b, 8, 8, 8)[:, :, None, :, None, :, None].expand(b, 8, 2, 8, 2, 8, 2).reshape(b, N_CELL)
        split2 = elig & (score2 > (tau2 + delta))
        return split4.sum(1).float().mean(), N_CUBE + 7 * split4.sum(1).float().mean() + split2.sum(1).float().mean()

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if counts(mid)[1] > n:
            lo = mid
        else:
            hi = mid
    k4 = int(round(float(counts(hi)[0])))
    return split_counts(n, k4)


def snap_realizable_tokens(n_tokens):
    """Round a requested token count to the nearest realisable one (no padding ever)."""
    n = int(round(float(n_tokens)))
    n = max(N_CUBE, min(N_SOBOL, n))
    if N_CUBE < n < N_CUBE + 7:
        n = N_CUBE if n - N_CUBE < 4 else N_CUBE + 7
    return n


def token_counts(k4, k2):
    """(n4, n2, n1): how many tokens each scale contributes, given the split counts."""
    return N_CUBE - k4, 8 * k4 - k2, 2 * k2


class Layout:
    """One micro-batch's token layout: shared (k_4, k_2), per-example locations.

    idx4 [B, n4]  cube indices carrying an s=4 token
    idx2 [B, n2]  cell indices carrying an s=2 token
    idx1 [B, k2]  cell indices that opened into their 2 Sobol points (s=1)
    The token sequence is [s=4 tokens | s=2 tokens | s=1 tokens], length N.
    """

    def __init__(self, idx4, idx2, idx1, k4, k2):
        self.idx4, self.idx2, self.idx1 = idx4, idx2, idx1
        self.k4, self.k2 = k4, k2
        self.n4, self.n2, self.n1 = token_counts(k4, k2)
        self.n_tokens = self.n4 + self.n2 + self.n1
        self.batch = max(idx4.shape[0], idx2.shape[0], idx1.shape[0])
        # every one of the 4096 cells is covered exactly once -- by an s=1 pair, an s=2
        # token, or one of the 8 cells of an s=4 cube.  k2 + (8 k4 - k2) + 8 (512 - k4) =
        # 4096 identically, so a layout can never leave a Sobol point unwritten.
        assert self.k2 + self.n2 + 8 * self.n4 == N_CELL, (k4, k2, self.n4, self.n2, self.n1)
        assert idx4.shape[1] == self.n4 and idx2.shape[1] == self.n2 and idx1.shape[1] == self.k2

    @property
    def is_dense(self):
        return self.k4 == N_CUBE and self.k2 == N_CELL

    def to(self, device):
        return Layout(self.idx4.to(device), self.idx2.to(device), self.idx1.to(device), self.k4, self.k2)

    def scale_of_token(self, device=None):
        """[N] long: the bp_scale_emb row of every position in the sequence."""
        return torch.cat([
            torch.full((self.n4,), SCALE_IDX[4], device=device, dtype=torch.long),
            torch.full((self.n2,), SCALE_IDX[2], device=device, dtype=torch.long),
            torch.full((self.n1,), SCALE_IDX[1], device=device, dtype=torch.long)])

    def __repr__(self):
        return (f"Layout(N={self.n_tokens}, k4={self.k4}, k2={self.k2}, "
                f"n4={self.n4}, n2={self.n2}, n1={self.n1})")


def _rows_of(mask, n_per_row):
    """mask [B, M] bool with exactly n_per_row True per row -> [B, n_per_row] column indices."""
    B = mask.shape[0]
    if n_per_row == 0:
        return mask.new_zeros((B, 0), dtype=torch.long)
    idx = mask.nonzero(as_tuple=True)[1]
    return idx.view(B, n_per_row)


def build_layout(score4, score2, n_tokens, thresholds=THRESHOLDS):
    """Build the exact-budget layout from per-example salience.

    score4 [B, 512], score2 [B, 4096] -> Layout with exactly ``n_tokens`` tokens per example.
    The k_4 highest-scoring cubes open; among the cells they expose, the k_2
    highest-scoring open into Sobol points.  Ties are broken by index, deterministically.
    """
    k4, k2 = balance_splits(score4, score2, n_tokens, thresholds)
    B, device = score4.shape[0], score4.device
    split4 = torch.zeros(B, N_CUBE, dtype=torch.bool, device=device)
    if k4:
        split4.scatter_(1, score4.topk(k4, dim=1).indices, True)
    # a cell may only open inside an open cube
    elig = split4.view(B, 8, 8, 8)[:, :, None, :, None, :, None].expand(B, 8, 2, 8, 2, 8, 2).reshape(B, N_CELL)
    split2 = torch.zeros(B, N_CELL, dtype=torch.bool, device=device)
    if k2:
        masked = score2.masked_fill(~elig, float("-inf"))
        split2.scatter_(1, masked.topk(k2, dim=1).indices, True)
    n4, n2, _ = token_counts(k4, k2)
    return Layout(_rows_of(~split4, n4), _rows_of(elig & ~split2, n2), _rows_of(split2, k2), k4, k2)


def dense_layout(batch, device):
    """The 8192-token layout: every Sobol point is its own token (the pretrained behaviour)."""
    k4, k2 = N_CUBE, N_CELL
    cells = torch.arange(N_CELL, device=device).expand(batch, N_CELL)
    empty = torch.zeros(batch, 0, dtype=torch.long, device=device)
    return Layout(empty, empty, cells.contiguous(), k4, k2)


def uniform_layout(batch, scale, device):
    """The uniform layout at one scale: all 4096 cells (s=2) or all 512 cubes (s=4)."""
    if scale == 2:   # every cube open, no cell open -> all 4096 cells
        cells = torch.arange(N_CELL, device=device).expand(batch, N_CELL).contiguous()
        empty = torch.zeros(batch, 0, dtype=torch.long, device=device)
        return Layout(empty, cells, empty, N_CUBE, 0)
    if scale == 4:   # nothing open -> all 512 cubes
        cubes = torch.arange(N_CUBE, device=device).expand(batch, N_CUBE).contiguous()
        empty = torch.zeros(batch, 0, dtype=torch.long, device=device)
        return Layout(cubes, empty, empty, 0, 0)
    raise ValueError(scale)


def threshold_layout_tokens(score4, score2, delta, thresholds=THRESHOLDS):
    """Token count that the merge thresholds shifted by ``delta`` would produce, averaged
    over the batch and snapped to a realisable value.

    This is the spec's threshold-offset mode: it is used to *pick N*, after which the
    layout itself is realised by rank so that the batch stays rectangular.
    """
    tau4, tau2 = thresholds
    split4 = score4 > (tau4 + delta)
    B = score4.shape[0]
    elig = split4.view(B, 8, 8, 8)[:, :, None, :, None, :, None].expand(B, 8, 2, 8, 2, 8, 2).reshape(B, N_CELL)
    split2 = elig & (score2 > (tau2 + delta))
    n = N_CUBE + 7 * split4.sum(1).float().mean() + split2.sum(1).float().mean()
    return snap_realizable_tokens(n.item())


def point_scales(layout, inv_perm):
    """[B, 8192] in Idol's token order: the scale (1, 2 or 4) of the token owning each point.

    The coarse-boosted velocity loss reads this -- a point predicted through a merged token
    is weighted up, because that is the prediction the refiner has to get right.
    """
    b, dev = layout.batch, layout.idx1.device
    s = torch.zeros(b, N_CELL, 2, dtype=torch.long, device=dev)
    if layout.k2:
        s.scatter_(1, layout.idx1[:, :, None].expand(-1, -1, 2).contiguous(),
                   torch.full((b, layout.idx1.shape[1], 2), 1, dtype=torch.long, device=dev))
    if layout.n2:
        s.scatter_(1, layout.idx2[:, :, None].expand(-1, -1, 2).contiguous(),
                   torch.full((b, layout.idx2.shape[1], 2), 2, dtype=torch.long, device=dev))
    if layout.n4:
        j = torch.arange(N_CUBE, device=dev)
        jx, jy, jz = j // 64, (j // 8) % 8, j % 8
        off = torch.tensor([(a, bb, c) for a in (0, 1) for bb in (0, 1) for c in (0, 1)], device=dev)
        table = (((2 * jx[:, None] + off[:, 0]) * 16 + (2 * jy[:, None] + off[:, 1])) * 16
                 + (2 * jz[:, None] + off[:, 2]))
        cells = table[layout.idx4].reshape(b, -1)
        s.scatter_(1, cells[:, :, None].expand(-1, -1, 2).contiguous(),
                   torch.full((b, cells.shape[1], 2), 4, dtype=torch.long, device=dev))
    return s.reshape(b, N_SOBOL)[:, inv_perm]
