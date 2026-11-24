"""Budget3D on Idol: the released denoiser run on an adaptive number of latent tokens.

The released model always reads the same 8192 Sobol tokens.  Here a layout chooses, per
example, which regions of the object keep one token per Sobol point and which are merged
into 16^3 cells or 8^3 cubes.  The trunk runs on the shorter sequence; a token-local
refiner expands every merged token's single velocity back into one velocity per Sobol
point, so the model's output is always the full [B, 8192, 16] latent Idol's decoders expect.

Every added module is zero-initialised and the dense layout takes a fast path that calls
exactly the released computation, so at step 0 the student *is* the teacher, bit for bit
(``check_dense_parity``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from torch.utils.checkpoint import checkpoint

from .adapters import MipNeRFIPE3D, RMSNorm, RefineStage, ScaleMixer3D, _zero, wiener_hf_gain
from ..octree import (N_CELL, N_CUBE, N_SOBOL, Layout, cell_centers, cell_child_offsets, cells_to_cubes,
                      cube_centers, cube_child_offsets, dense_layout, octree_permutation)
from ..upstream import LatentSeqMMFlowModel

BP_PREFIX = "bp_"


def bf16_matmuls(enabled=True):
    """bf16 for the transformer blocks only.

    The precision rule of the recipe: only the Linear matmuls inside Idol's blocks,
    noise_refiner and context_refiner, and inside the Budget3D refine blocks, run in
    bfloat16.  The residual stream, the norms, pos_pe, t_embedder, adaLN_modulation, the
    shift tables and the output heads stay fp32 -- 26 residual blocks of bf16 accumulation
    is enough to drift, and out_layer feeding a fp32 scatter must not change dtype.
    """
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=enabled and torch.cuda.is_available())


MAX_REFINE_ROWS = 32768      # keep the refiner's batch dimension under CUDA's 65535 grid limit


def _chunked(stage, limit, lat, cond, rel_pos, z_stage, carry=None):
    """Run a refine stage over its rows in slices, concatenating the results.

    One call per row is one attention sequence, so the batch dimension here is
    B * (tokens of that scale) * children -- large enough to exceed CUDA's grid limit on
    big batches. Slicing is mathematically identical: the rows are independent.
    """
    rows = lat.shape[0]
    if rows <= limit:
        return stage(lat, cond, rel_pos, z_stage, carry)
    outs, hs = [], []
    for i in range(0, rows, limit):
        j = min(i + limit, rows)
        o, h = stage(lat[i:j], cond[i:j], rel_pos, z_stage[i:j],
                     None if carry is None else carry[i:j])
        outs.append(o)
        hs.append(h)
    return torch.cat(outs), torch.cat(hs)


def _gather(x, idx):
    """x [B, M, C], idx [B, n] -> [B, n, C]."""
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class Budget3DModel(LatentSeqMMFlowModel):
    """Idol's LatentSeqMMFlowModel plus the Budget3D adapters (~8 M parameters)."""

    def __init__(self, mix_bott=256, ipe_bottleneck=256, refine_width=128, refine_depth=2, refine_heads=4,
                 refine_zt_skip=True, use_hf_skip=True, hf_rho=0.25, use_ipe=True, use_scale_emb=True,
                 multiscale_cascade=True, ipe_scale=1.0, reconstruct_point_features=True, **idol_args):
        super().__init__(**idol_args)
        self.float()                      # Idol's __init__ halves the block Linears; keep fp32 masters
        self.dtype = torch.float32
        d = idol_args["model_channels"]
        c = idol_args["in_channels"]
        self.use_hf_skip, self.hf_rho = use_hf_skip, hf_rho
        self.multiscale_cascade = multiscale_cascade
        self.grad_checkpointing = False
        self.bf16_blocks = True       # set False to run everything in fp32 (the parity gate)
        self.reconstruct_point_features = reconstruct_point_features

        self.bp_mixers = nn.ModuleDict({"2": ScaleMixer3D(d, 2, mix_bott), "4": ScaleMixer3D(d, 8, mix_bott)})
        self.bp_ipe = MipNeRFIPE3D(d, ipe_bottleneck, ipe_scale=ipe_scale) if use_ipe else None
        self.bp_scale_emb = nn.Parameter(torch.zeros(4, d)) if use_scale_emb else None
        self.bp_refiners = nn.ModuleDict({
            "2": RefineStage(2, refine_width, refine_depth, refine_heads, d, c, refine_zt_skip),
            "4": RefineStage(8, refine_width, refine_depth, refine_heads, d, c, refine_zt_skip)})
        # training-only head: predicts the teacher's per-point trunk feature from the
        # refiner's hidden state (named to match the reference inventory: bp_feat_proj.out)
        self.bp_feat_proj = nn.Sequential()
        self.bp_feat_proj.add_module("norm", RMSNorm(refine_width))
        self.bp_feat_proj.add_module("out", _zero(nn.Linear(refine_width, d)))

        # geometry, filled in by `bind_geometry` once pos_pe holds the released Sobol set
        for name, size in (("perm_to_octree", N_SOBOL), ("inv_perm", N_SOBOL)):
            self.register_buffer(name, torch.zeros(size, dtype=torch.long), persistent=False)
        self.register_buffer("cube_to_cells", torch.zeros(N_CUBE, 8, dtype=torch.long), persistent=False)
        self.register_buffer("cell_ctr", torch.zeros(N_CELL, 3), persistent=False)
        self.register_buffer("cube_ctr", torch.zeros(N_CUBE, 3), persistent=False)

    # -----------------------------------------------------------------------------------
    def bind_geometry(self):
        """Derive the Sobol/octree tables from the loaded ``pos_pe``. Call after loading weights."""
        perm, inv = octree_permutation(self.pos_pe[0])
        dev = self.pos_pe.device
        self.perm_to_octree, self.inv_perm = perm.to(dev), inv.to(dev)
        j = torch.arange(N_CUBE, device=dev)
        jx, jy, jz = j // 64, (j // 8) % 8, j % 8
        off = torch.tensor([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=dev)
        cells = (((2 * jx[:, None] + off[:, 0]) * 16 + (2 * jy[:, None] + off[:, 1])) * 16
                 + (2 * jz[:, None] + off[:, 2]))
        self.cube_to_cells = cells
        self.cell_ctr, self.cube_ctr = cell_centers(dev), cube_centers(dev)
        return self

    @property
    def bp_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith(BP_PREFIX)]

    @property
    def backbone_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith(BP_PREFIX)]

    def assert_zero_init(self):
        """The contract: every added output projection is zero, so the dense forward is the teacher's."""
        must_be_zero = [("bp_mixers.2.up", self.bp_mixers["2"].up), ("bp_mixers.4.up", self.bp_mixers["4"].up),
                        ("bp_feat_proj.out", self.bp_feat_proj.out)]
        if self.bp_ipe is not None:
            must_be_zero.append(("bp_ipe.mlp.2", self.bp_ipe.mlp[2]))
        for key, stage in self.bp_refiners.items():
            must_be_zero += [(f"bp_refiners.{key}.out", stage.out),
                             (f"bp_refiners.{key}.embed_zt", stage.embed_zt)]
            for i, blk in enumerate(stage.blocks):
                must_be_zero += [(f"bp_refiners.{key}.blocks.{i}.ada", blk.ada),
                                 (f"bp_refiners.{key}.blocks.{i}.proj", blk.proj),
                                 (f"bp_refiners.{key}.blocks.{i}.mlp.2", blk.mlp[2])]
        for name, mod in must_be_zero:
            for what, tensor in (("weight", mod.weight), ("bias", mod.bias)):
                assert tensor is None or float(tensor.detach().abs().max()) == 0, f"{name}.{what} is not zero"
        if self.bp_scale_emb is not None:
            assert float(self.bp_scale_emb.detach().abs().max()) == 0, "bp_scale_emb is not zero"

    # -----------------------------------------------------------------------------------
    # the pieces, split so that sampling can cache what does not change
    # -----------------------------------------------------------------------------------
    def embed_context(self, cond):
        """Image tokens -> refined condition stream [B, 4101, D]. No timestep dependence."""
        h = self.cond_embedder(cond["feature1"]) + self.cond_embedder2(cond["feature2"])
        h = h.type(self.dtype)
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h = self._run(repo, block, h)
        return h.float()

    def embed_tokens(self, z, layout):
        """z [B, 8192, 16] -> token features [B, N, D] for this layout (condition-invariant)."""
        h = self.input_layer(z).type(self.dtype)
        h = h + self.pos_embedder(self.pos_pe).type(self.dtype)
        if layout.is_dense:
            return h
        b, d = h.shape[0], h.shape[-1]
        h_oct = h[:, self.perm_to_octree]
        h2 = self.bp_mixers["2"](h_oct.reshape(b, 16, 16, 16, 2, d)) if self.multiscale_cascade \
            else h_oct.reshape(b, 16, 16, 16, 2, d).mean(-2)
        h4 = self.bp_mixers["4"](cells_to_cubes(h2)) if self.multiscale_cascade \
            else cells_to_cubes(h2).mean(-2)
        tok4 = _gather(h4.reshape(b, N_CUBE, d), layout.idx4)
        tok2 = _gather(h2.reshape(b, N_CELL, d), layout.idx2)
        tok1 = _gather(h_oct, self._points_of(layout.idx1))
        if self.bp_ipe is not None:
            tok4 = tok4 + self.bp_ipe(_gather(self.cube_ctr.expand(b, -1, -1), layout.idx4), 4).type(tok4.dtype)
            tok2 = tok2 + self.bp_ipe(_gather(self.cell_ctr.expand(b, -1, -1), layout.idx2), 2).type(tok2.dtype)
        if self.bp_scale_emb is not None:
            e = self.bp_scale_emb.type(h.dtype)
            tok4, tok2, tok1 = tok4 + e[2], tok2 + e[1], tok1 + e[0]
        return torch.cat([tok4, tok2, tok1], dim=1)

    @staticmethod
    def _points_of(cell_idx):
        """cell indices [B, k] -> their 2 Sobol point indices [B, 2k], slot-major within a cell."""
        return (2 * cell_idx.unsqueeze(-1) + torch.arange(2, device=cell_idx.device)).flatten(1)

    def _run(self, repo, block, h, t_mod=None):
        """One block, with its learned rotary positions computed in fp32.

        The precision rule puts only the Linears inside blocks/noise_refiner/context_refiner
        in bf16.  The RePo layers are separate modules and stay fp32 deliberately: their
        output is a set of rotation angles scaled by frequencies up to 16, and computing
        cos/sin from a bf16 angle loses enough precision to move the model's output by
        several percent.
        """
        def go(y):
            rot = repo(y)                                     # fp32, outside the autocast
            with bf16_matmuls(self.bf16_blocks):
                return block(y, mod=t_mod, rotary_emb=rot)
        if self.grad_checkpointing and self.training:
            return checkpoint(go, h, use_reentrant=False)
        return go(h)

    def refine_latent_and_camera(self, h_x, cam, t_mod):
        """noise_refiner and cam_refiner: neither reads the condition, so under CFG both run
        once on B and only their outputs are duplicated to 2B."""
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = self._run(repo, block, h_x, t_mod)
        return h_x.float(), self.cam_refiner(cam).type(self.dtype)

    def joint(self, h_x, h_cond, h_cam, t_mod):
        """The 24 joint MMDiT blocks over [latent | image | camera]."""
        n, n_cam = h_x.shape[1], h_cam.shape[1]
        h = torch.cat([h_x, h_cond, h_cam], dim=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = self._run(repo, block, h, t_mod)
        h = h.float()
        return h[:, :n], h[:, -n_cam:]

    def trunk(self, h_x, h_cond, cam, t_mod):
        """Run the refiners and the 24 joint blocks; return the modulated token and camera streams."""
        h_x, h_cam = self.refine_latent_and_camera(h_x, cam, t_mod)
        return self.joint(h_x, h_cond, h_cam, t_mod)

    def head(self, h_x, h_cam, t_emb, z_dtype, cam_dtype):
        """Idol's final norm plus shift-table modulation."""
        h_x = F.layer_norm(h_x.type(z_dtype), h_x.shape[-1:])
        h_cam = F.layer_norm(h_cam.type(cam_dtype), h_cam.shape[-1:])
        shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        return h_x * (1 + scale) + shift, h_cam * (1 + scale) + shift

    # -----------------------------------------------------------------------------------
    def forward(self, x, t, cond=None, h_cond=None, layout=None, return_features=False):
        """x: TensorDict(latent [B, 8192, 16], camera [B, 1, 5]); t [B] in [0, 1000].

        ``layout`` defaults to the dense 8192-token layout, which reproduces the released
        model exactly.  ``h_cond`` short-circuits the condition refiner for sampling.
        With ``return_features`` the per-Sobol-point trunk features are returned too, which
        is what the feature-distillation term reads.
        """
        z, cam = x["latent"], x["camera"]
        b = z.shape[0]
        if layout is None:
            layout = dense_layout(b, z.device)
        if h_cond is None:
            h_cond = self.embed_context(cond)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb).type(self.dtype)

        h_x = self.embed_tokens(z, layout)
        h_x, h_cam = self.trunk(h_x, h_cond, cam, t_mod)
        with bf16_matmuls(False):
            h_tok, h_cam = self.head(h_x, h_cam, t_emb, z.dtype, cam.dtype)
            v_tok = self.out_layer(h_tok)

        if layout.is_dense:
            with bf16_matmuls(False):
                cam_v = self.cam_out_layer(h_cam)
            out = TensorDict({"latent": v_tok, "camera": cam_v}, batch_size=x.batch_size)
            if return_features:
                out["feature"] = h_tok
            return out

        with bf16_matmuls(False):
            v_pts, f_pts = self._expand(v_tok, h_tok, z, t, layout, return_features)
            cam_v = self.cam_out_layer(h_cam)
        latent = v_pts.reshape(b, N_SOBOL, -1)[:, self.inv_perm]
        out = TensorDict({"latent": latent, "camera": cam_v}, batch_size=x.batch_size)
        if return_features:
            out["feature"] = f_pts.reshape(b, N_SOBOL, -1)[:, self.inv_perm]
        return out

    @torch.no_grad()
    def forward_dual(self, z, cam, t, h_pos, h_neg, return_features=False):
        """Both CFG branches of the *dense* model in one pass -- the frozen teacher's targets.

        z, cam and t are identical in the two branches, and neither ``embed_tokens`` nor
        ``noise_refiner``/``cam_refiner`` reads the condition, so the whole prefix runs once
        on B and only its output is duplicated to 2B for the 24 joint blocks.  That is the
        same arithmetic as two separate passes, with the condition-invariant part not done
        twice.
        """
        b = z.shape[0]
        layout = dense_layout(b, z.device)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb).type(self.dtype)

        h_x = self.embed_tokens(z, layout)
        h_x, h_cam = self.refine_latent_and_camera(h_x, cam, t_mod)
        h_cond = torch.cat([h_pos.expand(b, -1, -1), h_neg.expand(b, -1, -1)])
        h_x, h_cam = self.joint(torch.cat([h_x, h_x]), h_cond, torch.cat([h_cam, h_cam]),
                                torch.cat([t_mod, t_mod]))
        with bf16_matmuls(False):
            h_tok, h_cam = self.head(h_x, h_cam, torch.cat([t_emb, t_emb]), z.dtype, cam.dtype)
            v = self.out_layer(h_tok)
            v_cam = self.cam_out_layer(h_cam)
        out = {"v_pos": v[:b], "v_neg": v[b:], "cam_pos": v_cam[:b], "cam_neg": v_cam[b:]}
        if return_features:
            out["h_pos"], out["h_neg"] = h_tok[:b], h_tok[b:]
        return out

    def compile_blocks(self):
        """torch.compile the transformer blocks and the RePo layers, as Idol's own optimised
        sampling path does.

        Only safe for the frozen teacher: it always runs at the dense layout, so every shape
        is fixed, whereas the student's sequence length changes every step and would
        recompile endlessly.  The RePo layers must be switched to real (cos, sin) rotations
        first -- Inductor cannot generate code for complex operators and silently produces
        wrong results if left on the complex path.  The two forms are the same rotation.
        """
        import types

        from ..upstream import idol_layers
        compiled = {}
        kinds = (idol_layers.UnifiedTransformerBlock, idol_layers.RePo3DRotaryEmbedding)
        for mod in self.modules():
            if isinstance(mod, idol_layers.RePo3DRotaryEmbedding):
                mod.real_valued = True
            if isinstance(mod, kinds):
                key = type(mod)
                if key not in compiled:
                    compiled[key] = torch.compile(key.forward, dynamic=False)
                mod.forward = types.MethodType(compiled[key], mod)
        return self

    def use_real_rope(self):
        """Switch the RePo layers to real (cos, sin) rotations without compiling, so an
        uncompiled model matches a compiled one."""
        from ..upstream import idol_layers
        for mod in self.modules():
            if isinstance(mod, idol_layers.RePo3DRotaryEmbedding):
                mod.real_valued = True
        return self

    @torch.no_grad()
    def forward_cfg(self, z, cam, t, h_pos, h_neg, layout, cfg=3.0, return_hat=True):
        """One guided step with the condition-invariant prefix computed once.

        ``input_layer``, the ScaleMixer cascade, the IPE, ``noise_refiner`` and
        ``cam_refiner`` never read the condition, and z, cam and t are the same in both CFG
        branches -- so they run once on B and only their outputs are duplicated to 2B for
        the 24 joint blocks.  ``h_pos`` / ``h_neg`` are cached per image, not per step.
        """
        b = z.shape[0]
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb).type(self.dtype)
        h_x = self.embed_tokens(z, layout)
        h_x, h_cam = self.refine_latent_and_camera(h_x, cam, t_mod)

        h_cond = torch.cat([h_pos.expand(b, -1, -1), h_neg.expand(b, -1, -1)])
        h_x, h_cam = self.joint(torch.cat([h_x, h_x]), h_cond, torch.cat([h_cam, h_cam]),
                                torch.cat([t_mod, t_mod]))
        with bf16_matmuls(False):
            h_tok, h_cam = self.head(h_x, h_cam, torch.cat([t_emb, t_emb]), z.dtype, cam.dtype)
            v_tok = self.out_layer(h_tok)
            v_cam = self.cam_out_layer(h_cam)
        v_cam = (1 + cfg) * v_cam[:b] - cfg * v_cam[b:]

        if layout.is_dense:
            v = (1 + cfg) * v_tok[:b] - cfg * v_tok[b:]
        else:
            two = Layout(layout.idx4.repeat(2, 1), layout.idx2.repeat(2, 1), layout.idx1.repeat(2, 1),
                         layout.k4, layout.k2)
            with bf16_matmuls(False):
                v_pts, _ = self._expand(v_tok, h_tok, torch.cat([z, z]), torch.cat([t, t]), two, False)
            v = v_pts.reshape(2 * b, N_SOBOL, -1)[:, self.inv_perm]
            v = (1 + cfg) * v[:b] - cfg * v[b:]
        return v, v_cam

    # -----------------------------------------------------------------------------------
    def _expand(self, v_tok, h_tok, z, t, layout, return_features):
        """Merged-token velocities -> one velocity per Sobol point, in octree order.

        Returns (v [B, 4096, 2, 16], features [B, 4096, 2, D] or None): the full Sobol set
        indexed by (cell, slot).
        """
        b, d = h_tok.shape[0], h_tok.shape[-1]
        n4, n2 = layout.n4, layout.n2
        dev = z.device
        z_oct = z[:, self.perm_to_octree].reshape(b, N_CELL, 2, -1)
        c = z_oct.shape[-1]
        v_pts = z.new_zeros(b, N_CELL, 2, c)
        f_pts = h_tok.new_zeros(b, N_CELL, 2, d) if return_features else None
        # the gain reads the diffusion time in [0, 1]; Idol's t argument is 1000 * t
        gain = wiener_hf_gain(t.float() / 1000.0, self.hf_rho).to(dev) if self.use_hf_skip else None

        def scatter(cells, v_cell, f_cell):
            idx = cells.unsqueeze(-1).unsqueeze(-1)
            v_pts.scatter_(1, idx.expand(-1, -1, 2, c), v_cell)
            if f_cell is not None:
                f_pts.scatter_(1, idx.expand(-1, -1, 2, d), f_cell)

        def per_row(g, n):
            """[B] -> [B * n, 1, 1], the gain of each parent token's example."""
            return g.reshape(b, 1).expand(b, n).reshape(b * n, 1, 1)

        # --- s = 1: the released prediction, one token per Sobol point
        if layout.k2:
            v1 = v_tok[:, n4 + n2:].reshape(b, layout.k2, 2, c)
            f1 = h_tok[:, n4 + n2:].reshape(b, layout.k2, 2, d) if return_features else None
            scatter(layout.idx1, v1, f1)

        # --- s = 2: one 16^3 cell -> its 2 Sobol points
        if n2:
            z_cell = _take(z_oct, layout.idx2)                                   # [B, n2, 2, C]
            m = b * n2
            v, h = _chunked(self.bp_refiners["2"], MAX_REFINE_ROWS,
                            v_tok[:, n4:n4 + n2].reshape(m, c), h_tok[:, n4:n4 + n2].reshape(m, d),
                            cell_child_offsets(dev), z_cell.reshape(m, 2, c))
            zs = z_cell.reshape(m, 2, c)
            if gain is not None:
                v = v + per_row(gain, n2) * (zs - zs.mean(1, keepdim=True))
            f = None
            if return_features:
                f = (h_tok[:, n4:n4 + n2].reshape(m, 1, d)
                     + self.bp_feat_proj(h).type(h_tok.dtype)).reshape(b, n2, 2, d)
            scatter(layout.idx2, v.reshape(b, n2, 2, c), f)

        # --- s = 4: one 8^3 cube -> its 8 cells -> their 16 Sobol points
        if n4:
            cells = self.cube_to_cells[layout.idx4]                              # [B, n4, 8]
            z_cell = _take(z_oct, cells.flatten(1)).reshape(b, n4, 8, 2, c)
            par_v, par_h = v_tok[:, :n4], h_tok[:, :n4]
            m = b * n4
            cell_v, cell_h = _chunked(self.bp_refiners["4"], MAX_REFINE_ROWS,
                                      par_v.reshape(m, c), par_h.reshape(m, d),
                                      cube_child_offsets(dev), z_cell.mean(-2).reshape(m, 8, c))
            m8 = m * 8
            cond8 = par_h.unsqueeze(2).expand(-1, -1, 8, -1).reshape(m8, d)
            zs = z_cell.reshape(m8, 2, c)
            # the refiner sees one sequence per child cell, so its batch dimension is
            # B * n4 * 8 -- which passes CUDA's 65535 grid limit at B >= 16 and fails with
            # "invalid configuration argument". Chunk it so any batch size works.
            v, h = _chunked(self.bp_refiners["2"], MAX_REFINE_ROWS,
                            cell_v.reshape(m8, c), cond8, cell_child_offsets(dev), zs,
                            carry=cell_h.reshape(m8, -1))
            if gain is not None:
                # the merged region is the whole 8^3 cube: subtract the mean over its 16 points
                cube_mean = z_cell.reshape(b, n4, 16, c).mean(2)                 # [B, n4, C]
                cube_mean = cube_mean.unsqueeze(2).expand(b, n4, 8, c).reshape(m8, 1, c)
                v = v + per_row(gain, n4 * 8) * (zs - cube_mean)
            f = None
            if return_features:
                f = (cond8.reshape(m8, 1, d)
                     + self.bp_feat_proj(h).type(h_tok.dtype)).reshape(b, n4 * 8, 2, d)
            scatter(cells.reshape(b, n4 * 8), v.reshape(b, n4 * 8, 2, c), f)
        return v_pts, f_pts


def _take(x, idx):
    """x [B, M, A, C], idx [B, k] -> [B, k, A, C]."""
    a, c = x.shape[2], x.shape[3]
    return x.gather(1, idx[:, :, None, None].expand(-1, -1, a, c))
