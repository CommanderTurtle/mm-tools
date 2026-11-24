"""MHR-Hand fit: the right hand of SAM 3D Body's Momentum Human Rig (MHR, hand mode) fitted to FAH's dense MANO-topology
surface and 21 joints -- the MHR counterpart of the MANO mesh stage (fah/fast_mano.py MeshStage).

Parameters per hand, as SAM 3D Body's hand decoder sets them: the 27 right-hand joint angles (MHR model parameters), the
hand scale components (scale comps 8 = wrist scale and 18-27 = bone lengths; the others only move the body), the identity
components that move the hand (shape comps 40-44; the other 40 move hand vertices < 0.1 mm per unit) and a camera-frame
rotation / translation of the wrist-centric hand. The rig is mapped to MANO topology by a sparse MHR -> MANO-778 vertex
regressor (assets/mhr_to_mano778.npz) and to the 21 joints by the MHR -> MANO21 kinematic regressor; residuals vs the
predicted surface / joints (Huber, as MeshFitter), MHR's own joint limits and unit-Gaussian priors on the scale / shape components. Closed-form
init (per-part Procrustes of the MANO-topology points -> joint euler angles down the chain, palm size -> wrist scale),
optionally a learned one-pass initialiser (MHRInitNet, train/mhr_fitter.py), then Levenberg-Marquardt steps (rigid
Jacobian in closed form; the rig's by batched forward differences or in closed form, MHRHand.verts_jac); the whole stage
is one CUDA graph per batch size.

    fitter = MHRHandFitter.from_model(model, iters=2, init_net="mhr_fitter.pt")  # model: load_hand_model(...) (its MHR head)
    hand = fit_mhr_hand(fitter, points3d, joints3d, is_right, K=K, joints2d=joints2d)  # one hand (778, 3), (21, 3) or a batch
    hand["verts"] (2445, 3) MHR hand mesh + hand["faces"], hand["verts_mano"] (778, 3), hand["joints"] (21, 3),
    hand["joints_rig"] (21, 3), hand["params"] (hand_pose, hand_pca, scale, shape, global_rot, cam_t, model_params, mirrored)

Lower level: asset = build_asset(model.head_pose_hand) (CPU tensors, torch.save-able), MHRHandFitter(asset, ...)(tV, tJ)
on right-hand-frame batches. Left hands follow SAM 3D Body's convention: the right MHR hand fitted to the mirrored hand
(mirror the outputs back; fit_mhr_hand does). The MHR model comes from Meta's MHR release (Apache-2.0), the hand
parameterisation from SAM 3D Body (facebook/sam-3d-body-dinov3, SAM License).
"""
import os

import numpy as np
import torch

from .fast_mano import kabsch_horn, rodrigues
from .models.heads.fk_eager import compose, euler_xyz_to_quat, quat_mul, quat_rotate
from .models.heads.mhr_hand_fast import MHR_RIGHT_WRIST_JOINT

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
VREG_TEACHER = os.path.join(ASSETS, "mhr_to_mano778_regressor.npz")  # fitted on SAM 3D Body teacher meshes (bench MANO-topology export)
VREG = os.path.join(ASSETS, "mhr_to_mano778.npz")  # refit from MHR fits to GT MANO meshes of training splits (scripts/fit_mhr_mano_regressor.py)
JREG = os.path.join(ASSETS, "mhr_to_mano21_kin_regressor.npz")
SHAPE_COMPS = (40, 41, 42, 43, 44)
SCALE_COMPS = (8,) + tuple(range(18, 28))
FLIP = np.diag([1.0, -1.0, -1.0])  # MHR world -> SAM 3D Body camera axes (pred_vertices[..., 1:3] *= -1)
SIX_ID = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
SUB = np.arange(0, 778, 4)  # MANO-topology points the learned initialiser reads (as train/mesh_fitter.py)


def build_asset(head, vreg=VREG, jreg=JREG, shape_comps=SHAPE_COMPS, scale_comps=SCALE_COMPS, free_scales=False):
    """Everything the fit needs from SAM 3D Body's MHR head (hand mode) as CPU tensors: hand-mode model parameters at
    zero global rotation (right wrist at the origin), the linear maps hand pose / scale components -> joint parameters,
    the skeleton, skinning / pose correctives / identity blend shapes of two vertex sets (f: the regressors' vertices,
    o: the whole hand mesh), the regressors, parameter limits, the closed-form init's part structure. free_scales: the
    hand's 26 scale model parameters (wrist scale, bone lengths / offsets) fitted directly instead of SAM 3D Body's
    scale components."""
    import roma

    from .estimator import load_joint_regressor
    from .models.heads.mhr_hand_fast import hand_vertex_mask

    mhr = head.mhr
    ct = mhr.character_torch
    sk, lbs = ct.skeleton, ct.linear_blend_skinning
    c = lambda x: x.detach().float().cpu().clone()
    vm = hand_vertex_mask(mhr)
    nall = len(vm)
    Wv, Wj = load_joint_regressor(vreg, nall), load_joint_regressor(jreg, nall)
    o_idx = np.nonzero(vm)[0]
    f_idx = np.nonzero(Wv.any(0) | Wj.any(0))[0]
    assert vm[f_idx].all(), "regressor uses non-hand vertices"
    P = 136 + head.scale_mean.shape[0]
    pt = c(ct.parameter_transform.parameter_transform)
    assert not pt[:, P:].any()  # identity parameters do not move the skeleton
    pt = pt[:, :P]
    nj = pt.shape[0] // 7
    ridx = head.hand_joint_idxs_right.cpu().long()
    keep = torch.ones(P)
    keep[head.nonhand_param_idxs.cpu().long()] = 0
    L2W, rw, root = c(head.local_to_world_wrist), c(head.right_wrist_coords), c(head.root_coords)
    g0 = roma.rotmat_to_euler("xyz", L2W)  # MHRHead.mhr_forward hand mode at global_rot = 0, global_trans = 0
    mp0 = torch.zeros(P)
    mp0[:3] = -(roma.euler_to_rotmat("xyz", g0) @ (rw - root) + root) * 10
    mp0[3:6] = g0
    mp0[136:] = c(head.scale_mean)
    mp0 = mp0 * keep
    A_pose = torch.zeros(27, P)
    A_pose[torch.arange(27), ridx] = 1
    A_scale = torch.zeros(len(scale_comps), P)
    A_scale[:, 136:] = c(head.scale_comps)[list(scale_comps)]
    A_scale = A_scale * keep
    if free_scales:
        free = [int(i) for i in torch.nonzero(keep[136:]).flatten() + 136]
        A_scale = torch.zeros(len(free), P)
        A_scale[torch.arange(len(free)), torch.tensor(free)] = 1
        scale_comps = [8] + [i - 136 for i in free[1:]]  # row 0 = the wrist scale (model parameter 144 = scale 8)
    a = dict(P=P, nj=nj, n_shape=head.num_shape_comps, shape_comps=list(shape_comps), scale_comps=list(scale_comps), free_scales=free_scales,
             mp0=mp0, A_pose=A_pose, A_scale=A_scale, jp0=mp0 @ pt.T, J_pose=A_pose @ pt.T, J_scale=A_scale @ pt.T,
             offsets=c(sk.joint_translation_offsets), prerot=c(sk.joint_prerotations),
             levels=[(lv[0].clone(), lv[1].clone()) for lv in torch.split(sk.pmi.cpu(), list(sk._pmi_buffer_sizes), 1)],
             parents=sk.joint_parents.cpu().long().clone(), ibp=c(lbs.inverse_bind_pose),
             L2W=L2W, right_wrist=rw, root=root, scale_mean=c(head.scale_mean), scale_comps_all=c(head.scale_comps),
             hand_idx=ridx, keep=keep, hand_pose_mean=c(head.hand_pose_mean), o_idx=torch.as_tensor(o_idx))
    # joints with rotational degrees of freedom; dof i -> (joint, euler axis)
    nzr = a["J_pose"].abs().argmax(1)
    a["dof_joint"], a["dof_axis"] = nzr // 7, nzr % 7 - 3
    assert ((nzr % 7 >= 3) & (nzr % 7 < 6)).all()
    rot = sorted(set(a["dof_joint"].tolist()))
    a["rot_joints"] = torch.tensor(rot)
    # pose correctives (sparse -> ReLU -> dense): only the hand's posed joints have non-zero features in hand mode
    seq = mhr.pose_correctives_model.pose_dirs_predictor
    sp = getattr(seq, "0")
    W0 = torch.sparse_coo_tensor(sp.sparse_indices.cpu(), sp.sparse_weight.detach().cpu(), list(sp.sparse_shape)).to_dense()
    W2 = getattr(seq, "2").weight.detach().cpu()
    fcols = torch.tensor([(j - 2) * 6 + k for j in rot for k in range(6)])
    remap_full = -torch.ones(nall, dtype=torch.long)
    skin_v, skin_j, skin_w = lbs.vert_indices_flattened.long().cpu(), lbs.skin_indices_flattened.long().cpu(), lbs.skin_weights_flattened.cpu()
    for p, idx in (("f", f_idx), ("o", o_idx)):
        it = torch.as_tensor(idx)
        remap = remap_full.clone()
        remap[it] = torch.arange(len(idx))
        v = remap[skin_v]
        k = v >= 0
        rows = (it[:, None] * 3 + torch.arange(3)[None]).flatten()
        w2 = W2[rows]
        w0 = W0[:, fcols]
        hid = torch.nonzero((w2.abs().sum(0) > 0) & (w0.abs().sum(1) > 0)).flatten()
        a.update({p + "base": c(ct.blend_shape.base_shape)[it], p + "shape": c(ct.blend_shape.shape_vectors)[list(shape_comps)][:, it],
                  p + "sv": v[k], p + "sj": skin_j[k], p + "sw": skin_w[k].float().clone(), p + "w0": w0[hid].clone(), p + "w2": w2[:, hid].clone()})
    a["Wv"] = torch.as_tensor(Wv[:, f_idx])
    a["Wj"] = torch.as_tensor(Wj[:, f_idx])
    a["Wv_o"] = torch.as_tensor(Wv[:, o_idx])
    a["Wj_o"] = torch.as_tensor(Wj[:, o_idx])
    from .metadata import MHR_HAND21_TO_OPENPOSE

    km = head.keypoint_mapping[21:42].detach().cpu()  # MHR70 right-hand keypoints: one skeleton joint each (MHRHead.enable_fast_hand)
    assert (km[:, :nall] == 0).all() and ((km > 0).sum(1) == 1).all()
    a["kp_joints"] = (km.argmax(1) - nall)[MHR_HAND21_TO_OPENPOSE].clone()
    faces = head.faces.cpu().numpy()
    remap = -np.ones(nall, np.int64)
    remap[o_idx] = np.arange(len(o_idx))
    a["faces"] = torch.as_tensor(remap[faces[vm[faces].all(1)]])
    # parameter limits on the fitted model parameters (hand angles, hand scales)
    pl = ct.parameter_limits
    li = pl.minmax_parameter_index.cpu().long()
    used = (a["A_pose"].abs().sum(0) + a["A_scale"].abs().sum(0)) > 0
    m = used[li]
    a.update(lim_idx=li[m], lim_lo=c(pl.minmax_min)[m], lim_hi=c(pl.minmax_max)[m], lim_w=c(pl.minmax_weight)[m])
    # closed-form init: parts = the wrist (root) + the posed joints; every other joint belongs to its nearest posed ancestor
    par = a["parents"].tolist()
    part_of = [0] * nj
    for j in range(nj):  # parents precede children in MHR's joint order
        p = par[j]
        part_of[j] = 1 + rot.index(j) if j in rot else (part_of[p] if p >= 0 else 0)
    a["wrist"] = MHR_RIGHT_WRIST_JOINT
    a["part_of_joint"] = torch.tensor(part_of)
    a["part_parent"] = torch.tensor([0] + [part_of[par[j]] for j in rot])
    a["size_coef"] = float(a["A_scale"][0, 136 + 8]) if scale_comps[0] == 8 else 0.0
    return a


def euler_xyz(R):
    """Euler angles (a, b, c) of R = Rz(c) Ry(b) Rx(a) (..., 3, 3) (the convention of fk_eager.euler_xyz_to_quat)."""
    a = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    b = torch.asin((-R[..., 2, 0]).clamp(-1 + 1e-7, 1 - 1e-7))
    c = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([a, b, c], -1)


def quat_to_rotmat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def six_d(e):
    """First two columns of R = Rz Ry Rx for euler angles e (..., 3) (mhr_utils.batch6DFromXYZ, out of place)."""
    cx, cy, cz = torch.cos(e).unbind(-1)
    sx, sy, sz = torch.sin(e).unbind(-1)
    return torch.stack([cy * cz, cy * sz, -sy, -cx * sz + sx * sy * cz, cx * cz + sx * sy * sz, sx * cy], -1)


def euler_xyz_np(R):
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)), np.arctan2(R[1, 0], R[0, 0])])


def rotmat_xyz_np(e):
    """R = Rz(e2) Ry(e1) Rx(e0) (roma.euler_to_rotmat("xyz"))."""
    (cx, cy, cz), (sx, sy, sz) = np.cos(e), np.sin(e)
    return np.array([[cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz],
                     [cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz],
                     [-sy, sx * cy, cx * cy]])


def sparse_rows(W):
    """Dense (M, V) regressor -> padded (M, K) vertex indices and weights (K = most non-zeros in a row)."""
    W = torch.as_tensor(W, dtype=torch.float32)
    K = max(int((W != 0).sum(1).max()), 1)
    w, i = W.abs().topk(K, 1)
    return i, W.gather(1, i) * (w > 0)


def segments(sv, n):
    """Skin entries per vertex as a padded (n, Kmax) index table into the entry list (K = the index of an appended zero
    row): vertex sums as a gather + a fixed-order sum, run-to-run identical on CUDA (index_add's float atomics are not)."""
    K = len(sv)
    order = torch.argsort(sv * K + torch.arange(K))
    cnt = torch.bincount(sv, minlength=n)
    start = torch.cumsum(cnt, 0) - cnt
    seg = torch.full((n, max(int(cnt.max()), 1)), K, dtype=torch.long)
    seg[sv[order], torch.arange(K) - start[sv[order]]] = order
    return seg


class MHRHand(torch.nn.Module):
    """The hand-mode MHR rig as a function of x = [hand pose (27) | scale comps | shape comps]: FK of the whole skeleton
    (fk_eager's math, all parameter sets in one batch), pose correctives of the posed hand joints, LBS of a vertex set.
    Vertices in the wrist-centric MHR frame (global rotation 0), metres."""

    def __init__(self, asset, device="cuda"):
        super().__init__()
        a = asset
        self.npose, self.ns, self.nb = len(a["A_pose"]), len(a["scale_comps"]), len(a["shape_comps"])
        self.nlev = len(a["levels"])
        for k in ("jp0", "J_pose", "J_scale", "offsets", "prerot", "ibp", "rot_joints", "Wv", "Wj",
                  "mp0", "A_pose", "A_scale", "lim_idx", "lim_lo", "lim_hi", "lim_w"):
            self.register_buffer(k, torch.as_tensor(a[k]).clone())
        for k in ("Wv", "Wj", "Wv_o", "Wj_o"):  # the regressors as gathers (<= 12 vertices per point)
            i, w = sparse_rows(a[k])
            self.register_buffer(k + "_i", i)
            self.register_buffer(k + "_w", w)
        for p in ("f", "o"):
            for k in ("base", "shape", "sv", "sj", "sw", "w0", "w2"):
                self.register_buffer(p + k, torch.as_tensor(a[p + k]).clone())
            self.register_buffer(p + "seg", segments(torch.as_tensor(a[p + "sv"]).long(), len(a[p + "base"])))
        for k, (src, tgt) in enumerate(a["levels"]):
            self.register_buffer(f"src{k}", src.long().clone())
            self.register_buffer(f"tgt{k}", tgt.long().clone())
        self.register_buffer("six_id", torch.tensor(SIX_ID))
        if "kp_joints" in a:  # the rig's own 21 hand keypoints (skeleton joints), OpenPose order
            self.register_buffer("kp", torch.as_tensor(a["kp_joints"]).long().clone())
        else:
            self.kp = None
        self._jac_setup(a)
        self.to(device)

    def _jac_setup(self, a):
        """Structure of the closed-form Jacobian (verts_jac): the joints the fit moves, their descendants, the skin
        weight below each of them per fit-set vertex, the joint-parameter entries of the scale components."""
        par = [int(p) for p in a["parents"]]
        rot = [int(j) for j in a["rot_joints"]]
        Jp, Js = torch.as_tensor(a["J_pose"]), torch.as_tensor(a["J_scale"])
        qp = Jp.abs().argmax(1)  # pose dof -> joint-parameter entry
        qs = [int(q) for q in torch.nonzero(Js.abs().sum(0) > 0).flatten()]
        assert all(q % 7 in (0, 1, 2, 6) for q in qs)  # scale components move joint translations / scales only
        jj = sorted(set(rot) | {q // 7 for q in qs})
        desc = torch.tensor([[float(self._below(k, j, par)) for k in range(len(par))] for j in jj])  # (Jn, J)
        sv, sj, sw = (torch.as_tensor(a["f" + k]) for k in ("sv", "sj", "sw"))
        mem = desc[:, sj.long()].T.contiguous()  # (K, Jn) skin entry moves with joint jj[c]
        self.register_buffer("jac_mem", mem)
        self.register_buffer("jac_Wd", torch.zeros(len(a["fbase"]), len(jj)).index_add(0, sv.long(), sw.float()[:, None] * mem))
        jn = {j: c for c, j in enumerate(jj)}
        self.register_buffer("jac_jj", torch.tensor(jj))
        self.register_buffer("jac_rot_par", torch.tensor([par[j] for j in rot]))
        self.register_buffer("jac_dof_rot", torch.tensor([rot.index(int(q) // 7) for q in qp]))  # dof -> posed joint
        self.register_buffer("jac_dof_jn", torch.tensor([jn[int(q) // 7] for q in qp]))
        self.register_buffer("jac_dof_ax", qp % 7 - 3)
        self.register_buffer("jac_dof_coef", Jp.gather(1, qp[:, None])[:, 0])
        qt, qsc = [q for q in qs if q % 7 < 3], [q for q in qs if q % 7 == 6]
        self.register_buffer("jac_qt_jn", torch.tensor([jn[q // 7] for q in qt], dtype=torch.long))
        self.register_buffer("jac_qt_par", torch.tensor([par[q // 7] for q in qt], dtype=torch.long))
        self.register_buffer("jac_qt_ax", torch.tensor([q % 7 for q in qt], dtype=torch.long))
        self.register_buffer("jac_qs_jn", torch.tensor([jn[q // 7] for q in qsc], dtype=torch.long))
        self.register_buffer("jac_Jst", Js[:, qt].clone())
        self.register_buffer("jac_Jss", Js[:, qsc].clone())
        self.register_buffer("jac_prerot", quat_to_rotmat(torch.as_tensor(a["prerot"])[rot]))  # (16, 3, 3)

    @staticmethod
    def _below(k, j, par):
        while k >= 0 and k != j:
            k = par[k]
        return k == j

    def bufs(self):
        return dict(self.named_buffers())

    def skeleton(self, x, b):
        """x (N, >= 27 + ns) -> joint parameters (N, J, 7), global skeleton states (N, J, 8) [t (cm), q xyzw, s]."""
        N = x.shape[0]
        jp = (b["jp0"] + x[:, :self.npose] @ b["J_pose"] + x[:, self.npose:self.npose + self.ns] @ b["J_scale"]).reshape(N, -1, 7)
        t = jp[..., :3] + b["offsets"]
        q = quat_mul(b["prerot"].expand(N, -1, 4), euler_xyz_to_quat(jp[..., 3:6]))
        s = torch.exp(jp[..., 6:7] * 0.69314718246459961)
        g = torch.cat([t, q, s], -1)
        for k in range(self.nlev):
            src, tgt = b[f"src{k}"], b[f"tgt{k}"]
            g = g.index_copy(1, src, compose(g.index_select(1, tgt), g.index_select(1, src)))
        return jp, g

    def verts(self, x, b, p="f"):
        """x (N, n) -> vertices of set p (N, n_p, 3) and the global skeleton states."""
        N = x.shape[0]
        jp, g = self.skeleton(x, b)
        feat = (six_d(jp.index_select(1, b["rot_joints"])[..., 3:6]) - b["six_id"]).flatten(1)
        rest = b[p + "base"] + torch.einsum("cvd,nc->nvd", b[p + "shape"], x[:, self.npose + self.ns:self.npose + self.ns + self.nb])
        rest = rest + (torch.relu(feat @ b[p + "w0"].T) @ b[p + "w2"].T).reshape(N, -1, 3)
        T = compose(g, b["ibp"].expand(N, -1, -1)).index_select(1, b[p + "sj"])  # (N, K, 8)
        pk = rest.index_select(1, b[p + "sv"])
        moved = (T[..., :3] + T[..., 7:] * quat_rotate(T[..., 3:7], pk)) * b[p + "sw"][:, None]
        return self.seg_sum(moved, b[p + "seg"]) / 100, g

    @staticmethod
    def seg_sum(y, seg):
        """Per-vertex sums of skin-entry values y (N, K, ...) -> (N, n, ...), deterministic (segments)."""
        y = torch.cat([y, torch.zeros_like(y[:, :1])], 1)
        return y.index_select(1, seg.flatten()).reshape(y.shape[0], *seg.shape, *y.shape[2:]).sum(2)

    def regress(self, v, b, k):
        """v (N, n, ..., 3) -> regressor k's points (N, M, ..., 3)."""
        i, w = b[k + "_i"], b[k + "_w"]
        return (v.index_select(1, i.flatten()).reshape(v.shape[0], *i.shape, *v.shape[2:]) * w.reshape(*w.shape, *[1] * (v.dim() - 2))).sum(2)

    def verts_jac(self, x, b):
        """Fit-set vertices (N, n, 3) and their Jacobian (N, n, npose + ns + nb, 3) in closed form: an euler angle of
        joint j turns everything skinned below j about j's axis (w x (y - T_j)) plus its pose-corrective change; a
        bone translation moves it by the parent's frame axis; the wrist scale scales it about the wrist; shape and
        correctives through each skin entry's linear part. Wrist-centric MHR frame, metres per parameter unit."""
        N = x.shape[0]
        nv = b["fbase"].shape[0]
        jp, g = self.skeleton(x, b)
        e = jp.index_select(1, b["rot_joints"])[..., 3:6]  # (N, 16, 3)
        h = (six_d(e) - b["six_id"]).flatten(1) @ b["fw0"].T  # (N, H)
        rest = b["fbase"] + torch.einsum("cvd,nc->nvd", b["fshape"], x[:, self.npose + self.ns:self.npose + self.ns + self.nb])
        rest = rest + (torch.relu(h) @ b["fw2"].T).reshape(N, nv, 3)
        T = compose(g, b["ibp"].expand(N, -1, -1)).index_select(1, b["fsj"])  # (N, K, 8)
        sv, sw = b["fsv"], b["fsw"]
        lin = lambda y: T[:, :, None, 7:] * quat_rotate(T[:, :, None, 3:7].expand(*y.shape[:-1], 4), y)  # skin entries' linear parts
        MK = T[..., :3] + T[..., 7:] * quat_rotate(T[..., 3:7], rest.index_select(1, sv))  # (N, K, 3) cm
        v = self.seg_sum(MK * sw[:, None], b["fseg"]) / 100
        scat = lambda y: self.seg_sum(y * sw.reshape(-1, *[1] * (y.dim() - 2)), b["fseg"])
        # offsets of every vertex's skinned parts below each moved joint from that joint, skin-weighted (cm)
        Tj = g.index_select(1, b["jac_jj"])[..., :3]
        Y = scat(MK[:, :, None, :] * b["jac_mem"][None, :, :, None]) - b["jac_Wd"][None, :, :, None] * Tj[:, None]  # (N, nv, Jn, 3)
        Rg = quat_to_rotmat(g[..., 3:7])
        # euler axes in the world: R_parent prerot [Rz Ry e_x, Rz e_y, e_z]
        cb, sb, cc, sc = torch.cos(e[..., 1]), torch.sin(e[..., 1]), torch.cos(e[..., 2]), torch.sin(e[..., 2])
        z = torch.zeros_like(cb)
        L = torch.stack([torch.stack([cc * cb, sc * cb, -sb], -1), torch.stack([-sc, cc, z], -1), torch.stack([z, z, z + 1], -1)], -2)
        W = (Rg.index_select(1, b["jac_rot_par"]) @ b["jac_prerot"]) @ L.mT  # (N, 16, 3, 3): columns = axes
        om = W.index_select(1, b["jac_dof_rot"]).gather(3, b["jac_dof_ax"][None, :, None, None].expand(N, -1, 3, 1))[..., 0]  # (N, 27, 3)
        Jr = torch.cross(om[:, None].expand(N, nv, -1, 3), Y.index_select(2, b["jac_dof_jn"]), dim=-1)
        # pose correctives: d rest / d euler = W2 diag(relu') W0 d six_d / d euler
        cx, sx = torch.cos(e[..., 0]), torch.sin(e[..., 0])
        cy, sy = cb, sb
        dfa = torch.stack([z, z, z, sx * sc + cx * sy * cc, -sx * cc + cx * sy * sc, cx * cy], -1)
        dfb = torch.stack([-sy * cc, -sy * sc, -cy, sx * cy * cc, sx * cy * sc, -sx * sy], -1)
        dfc = torch.stack([-cy * sc, cy * cc, z, -cx * cc - sx * sy * sc, -cx * sc + sx * sy * cc, z], -1)
        df = torch.stack([dfa, dfb, dfc], 2)  # (N, 16, 3, 6)
        df = df.index_select(1, b["jac_dof_rot"]).gather(2, b["jac_dof_ax"][None, :, None, None].expand(N, -1, 1, 6))[:, :, 0]  # (N, 27, 6)
        H = b["fw0"].shape[0]
        W0 = b["fw0"].reshape(H, -1, 6).index_select(1, b["jac_dof_rot"])  # (H, 27, 6)
        dh = torch.einsum("hdk,ndk->nhd", W0, df) * (h > 0).float()[:, :, None]
        dr = (b["fw2"] @ dh).reshape(N, nv, 3, -1).transpose(2, 3)  # (N, nv, 27, 3) cm per rad
        Jr = (Jr + scat(lin(dr.index_select(1, sv)))) * b["jac_dof_coef"][None, None, :, None]
        # scale components: bone translations (parent frame axes, parent scale), the wrist scale (about the wrist)
        Rp = Rg.index_select(1, b["jac_qt_par"])  # (N, nqt, 3, 3)
        ax = Rp.gather(3, b["jac_qt_ax"][None, :, None, None].expand(N, -1, 3, 1))[..., 0] * g.index_select(1, b["jac_qt_par"])[..., 7:]
        dqt = b["jac_Wd"].index_select(1, b["jac_qt_jn"])[None, :, :, None] * ax[:, None]  # (N, nv, nqt, 3)
        dqs = 0.69314718246459961 * Y.index_select(2, b["jac_qs_jn"])
        Js = torch.einsum("cq,nvqd->nvcd", b["jac_Jst"], dqt) + torch.einsum("cq,nvqd->nvcd", b["jac_Jss"], dqs)
        Jb = scat(lin(b["fshape"].index_select(1, sv).permute(1, 0, 2)[None].expand(N, -1, -1, -1)))  # (N, nv, nb, 3)
        return v, torch.cat([Jr, Js, Jb], 2) / 100

    def model_params(self, x, b):
        return b["mp0"] + x[:, :self.npose] @ b["A_pose"] + x[:, self.npose:self.npose + self.ns] @ b["A_scale"]


class MHRInitNet(torch.nn.Module):
    """Learned one-pass initialiser (train/mhr_fitter.py): from the canonical prediction (every 4th surface point and the
    joints, wrist-centred, in the closed-form root frame) and the closed-form parameters, a residual on the hand pose /
    scale / shape, a small rotation and a translation (zero-initialised: starts as the closed-form init)."""

    def __init__(self, d_in, d_out, hidden=1024):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_out))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return self.net(f)


class MHRHandFitter:
    """Batched MHR-hand fit (see the module docstring). x = [pose 27 | scale ns | shape nb | rotation 3 | translation 3]:
    rotation = an axis-angle increment on the closed-form root rotation R0 (camera <- wrist-centric MHR frame),
    translation = the wrist-centric origin in camera coordinates relative to the target wrist."""

    def __init__(self, asset, device="cuda", iters=4, wv=0.3, wj=1.0, w_scale=1e-9, w_shape=1e-9, w_lim=0.0, limits=True,
                 lam0=1e-2, adaptive=False, huber=0.01, jac="analytic", fd="forward", fd_eps=1e-3, power=4, rows=None, init_net=None, graph=True,
                 compile=True, chunk=2):
        """iters: Levenberg-Marquardt steps; wv / wj: surface / joint weights (as MeshFitter); w_scale / w_shape: unit-
        Gaussian priors on the scale / shape components; w_lim: soft MHR limits on the hand scale parameters (SAM 3D
        Body does not apply them; the wrist scale's [-0.2, 0.2] is tight around its mean -0.18); limits: joint angles kept
        inside MHR's limits; lam0: (initial) damping; adaptive: Levenberg-Marquardt with accept / reject, damping x 1/3 or
        x 4 and an active set at the joint limits (default False: damped Gauss-Newton, fixed damping lam0, every step
        taken, angles clamped -- a continuous function of the inputs; accept / reject flips on near-ties, so 1e-6 input
        changes moved hands by up to ~20 mm); jac: articulation / scale / shape Jacobian, "analytic" (MHRHand.verts_jac,
        default), "fd" (forward differences in one batched rig pass; fp32 rounding in them makes the step noisy: 0.14 mm
        median vertex change under 1e-6 input changes on HOT3D, analytic 0.001 mm), "ad" (forward-mode autograd, checks),
        "auto" (fd on CUDA, else analytic); rows: the MANO vertices
        with surface residuals (e.g. the points a dense-subset model predicts; default all 778); power: the closed-form init
        weighs every MANO-topology point's share of a part (skinning weights through the regressor) to this power;
        init_net: learned initialiser after the closed-form one (MHRInitNet or its checkpoint, train/mhr_fitter.py)."""
        self.dev = torch.device(device)
        jac = ("fd" if self.dev.type == "cuda" else "analytic") if jac == "auto" else jac
        self.rig = MHRHand(asset, self.dev)
        self.a = asset
        self.iters, self.wv, self.wj, self.lam0, self.huber, self.jac, self.fd, self.chunk = iters, wv, wj, lam0, huber, jac, fd, chunk
        self.adaptive = adaptive
        npo, ns, nb = self.rig.npose, self.rig.ns, self.rig.nb
        self.rows = None if rows is None else torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self.dev)
        self.nm_all, self.nk = len(asset["Wv"]), len(asset["Wj"])
        self.npose, self.nm = npo, self.nm_all if rows is None else len(rows)
        self.n = n = npo + ns + nb + 6
        self.sl_rot, self.sl_t = slice(n - 6, n - 3), slice(n - 3, n)
        self.graph = graph and self.dev.type == "cuda"
        self.B = self.rig.bufs()
        for k in ("_i", "_w"):  # regressor rows with surface residuals
            self.B["Wvs" + k] = self.B["Wv" + k] if rows is None else self.B["Wv" + k].index_select(0, self.rows)
        sel = torch.as_tensor(asset["lim_idx"]) >= 136  # scale parameters: soft limits; joint angles: clamped
        self.B["slim_idx"], self.B["slim_lo"], self.B["slim_hi"] = (torch.as_tensor(asset[k])[sel].to(self.dev) for k in ("lim_idx", "lim_lo", "lim_hi"))
        self.n_data = 3 * (self.nm + self.nk)
        self.B["rw3"] = torch.cat([torch.full((self.nm,), (wv / self.nm) ** 0.5), torch.full((self.nk,), (wj / self.nk) ** 0.5)]).to(self.dev)
        self.B["rscale"] = self.B["rw3"][:, None].expand(-1, 3).reshape(-1).contiguous()
        self.B["prior_w"] = torch.cat([w_lim ** 0.5 * torch.as_tensor(asset["lim_w"])[sel],
                                       torch.full((ns,), w_scale ** 0.5), torch.full((nb,), w_shape ** 0.5)]).to(self.dev)
        self.B["eye_n"] = torch.eye(n, device=self.dev)
        self.B["eye3"] = torch.eye(3, device=self.dev)
        # central-difference steps: radians for the angles, component units for scale / shape, metres for the translation
        self.B["eps"] = torch.tensor([fd_eps] * npo + [10 * fd_eps] * (ns + nb) + [fd_eps] * 3 + [0.1 * fd_eps] * 3, device=self.dev)
        self.limits = limits
        self._init_parts(asset, power)
        dofs = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]  # mhr_utils.compact_model_params_to_cont_hand
        m3c = np.concatenate([np.full(2 * d, d == 3) for d in dofs])
        m3p = np.concatenate([np.full(d, d == 3) for d in dofs])
        T = lambda m: torch.as_tensor(np.nonzero(m)[0], device=self.dev)
        self.B.update(c3=T(m3c), c1=T(~m3c), p3=T(m3p), p1=T(~m3p), hmean=torch.as_tensor(asset.get("hand_pose_mean", torch.zeros(54)), device=self.dev).float(),
                      sc_idx=torch.as_tensor(asset["scale_comps"] if not asset.get("free_scales") else [], dtype=torch.long, device=self.dev),
                      sh_idx=torch.as_tensor(asset["shape_comps"], dtype=torch.long, device=self.dev), n_shape=torch.zeros(asset["n_shape"], device=self.dev))
        self.B["sub"] = torch.as_tensor(SUB, device=self.dev)
        if self.rig.kp is not None:
            self.B["kp_joints"] = self.rig.kp
        self.init_net = None
        if init_net is not None:
            if not isinstance(init_net, torch.nn.Module):
                ck = torch.load(os.path.expanduser(init_net), map_location="cpu", weights_only=False)
                net = MHRInitNet(ck["d_in"], ck["d_out"], ck.get("hidden", 1024))
                net.load_state_dict(ck["model"])
                init_net = net
            self.init_net = init_net.to(self.dev).eval()
        cc = (lambda f: torch.compile(f, fullgraph=True, dynamic=False)) if compile and self.dev.type == "cuda" else (lambda f: f)
        self.step, self.init_c, self.final_c, self.refine_c = cc(self._lm_step), cc(self._init), cc(self._final), cc(self._refine)
        self._pp = None
        self.graphs = {}

    def _init_parts(self, a, power):
        """Reference (all parameters 0) MANO-topology points, their soft part weights, the parts' reference rotations,
        the posed joints by kinematic level."""
        with torch.no_grad():
            x0 = torch.zeros(1, self.n, device=self.dev)
            v, g = self.rig.verts(x0, self.B, "f")
            P0 = (self.B["Wv"] @ v[0]).double()  # (nm, 3) MANO-topology points
            # part weights of every MANO vertex through the regressor and the skinning weights
            S = torch.zeros(len(a["fbase"]), a["nj"], dtype=torch.float64)  # on the CPU: index_put_ accumulates with atomics on CUDA
            S.index_put_((torch.as_tensor(a["fsv"]).long(), torch.as_tensor(a["fsj"]).long()), torch.as_tensor(a["fsw"]).double(), accumulate=True)
            S = S.to(self.dev)
            npart = len(a["rot_joints"]) + 1
            oh = ((self.B["Wv"].double() @ S) @ torch.nn.functional.one_hot(torch.as_tensor(a["part_of_joint"], device=self.dev), npart).double()) ** power
        cnt = oh.sum(0)
        # a part without points of its own (MHR skins no MANO-topology point mainly to the thumb's CMC joint) takes its
        # first child's rotation; the child keeps what the parent's euler axes cannot represent
        ppar = [int(p) for p in a["part_parent"][1:]]
        src = list(range(len(ppar)))
        for k in range(len(ppar)):
            if cnt[k + 1] < 1:
                src[k] = next((c for c in range(len(ppar)) if ppar[c] == k + 1), k)
        self.B["part_src"] = torch.tensor(src, device=self.dev)
        cnt = cnt.clamp(min=1e-6)
        self.B["oh"], self.B["ohm"] = oh.float(), (oh / cnt).float()
        self.B["P0"] = P0.float()
        self.B["mu0"] = (oh.T @ P0 / cnt[:, None]).float()  # (parts, 3) weighted part centroids
        self.B["m0"] = cnt.float()
        self.B["palm0"] = self.B["mu0"][0]
        self.B["palm_rms0"] = ((((P0 - self.B["palm0"].double()) ** 2).sum(1) * oh[:, 0]).sum() / cnt[0]).sqrt().float()
        G = quat_to_rotmat(g[0, :, 3:7])  # (J, 3, 3) reference global rotations
        self.B["G0"] = G.index_select(0, self.B["rot_joints"])  # (16, 3, 3)
        self.B["wrist0"] = g[0, a["wrist"], :3] / 100  # the wrist scale's centre
        rot = a["rot_joints"].tolist()
        depth = []
        for p in ppar:
            depth.append(0 if p == 0 else depth[p - 1] + 1)
        self.levels = [[k for k in range(len(rot)) if depth[k] == d] for d in range(max(depth) + 1)]
        self.ppar = ppar
        for d, ks in enumerate(self.levels):
            self.B[f"lv{d}"] = torch.tensor(ks, device=self.dev)
        dof_part = torch.tensor([rot.index(int(j)) for j in a["dof_joint"]], device=self.dev)
        self.B["dof_part"], self.B["dof_axis"] = dof_part, torch.as_tensor(a["dof_axis"], device=self.dev)
        mask = torch.zeros(len(rot), 3, device=self.dev)
        mask[dof_part, self.B["dof_axis"]] = 1
        self.B["dof_mask"] = mask  # (16, 3) euler axes each posed joint has
        lo, hi = torch.full((self.npose,), -1e3), torch.full((self.npose,), 1e3)
        hp = a["A_pose"].argmax(1).tolist()
        for i, li in enumerate(a["lim_idx"].tolist()):
            if li in hp:
                lo[hp.index(li)], hi[hp.index(li)] = a["lim_lo"][i], a["lim_hi"][i]
        if not self.limits:
            lo, hi = lo * 0 - 1e3, hi * 0 + 1e3
        self.B["pose_lo"], self.B["pose_hi"] = lo.to(self.dev), hi.to(self.dev)
        self.size_dof = self.npose if a["scale_comps"][0] == 8 and a["size_coef"] else None
        self.size_coef = a["size_coef"]

    def _init(self, tV):
        """Closed-form init from the target surface tV (B, 778, 3, centred on the wrist): per-part Kabsch rotations
        (reference -> target), then joint by joint down the chain the local rotation in the joint's reference frame,
        reduced to the joint's euler axes (the parent's reduced rotation is what the child is relative to). -> x0, R0."""
        b = self.B
        B = len(tV)
        mu = torch.einsum("vk,bvc->bkc", b["ohm"], tV)
        C = torch.einsum("vk,vi,bvj->bkij", b["oh"], b["P0"], tV) - b["m0"][:, None, None] * b["mu0"][:, :, None] * mu[:, :, None, :]
        Q = kabsch_horn(C)  # (B, parts, 3, 3): reference part -> target (weighted)
        R0 = Q[:, 0]
        D = (R0[:, None].mT @ Q[:, 1:]).index_select(1, b["part_src"])  # part rotation changes in the wrist-centric frame
        Dr = [None] * len(self.ppar)  # reduced (representable) rotation changes
        eul = torch.zeros(B, len(self.ppar), 3, device=self.dev)
        for d, ks in enumerate(self.levels):
            li = b[f"lv{d}"]
            Dp = torch.stack([b["eye3"].expand(B, 3, 3) if self.ppar[k] == 0 else Dr[self.ppar[k] - 1] for k in ks], 1)
            G0 = b["G0"].index_select(0, li)
            E = G0.mT @ Dp.mT @ D.index_select(1, li) @ G0  # local rotation in the joint's reference frame
            e = euler_xyz(E) * b["dof_mask"].index_select(0, li)
            Er = quat_to_rotmat(euler_xyz_to_quat(e))
            Dk = Dp @ G0 @ Er @ G0.mT
            for c, k in enumerate(ks):
                Dr[k] = Dk[:, c]
            eul = eul.index_copy(1, li, e)
        pose = eul.index_select(1, b["dof_part"]).gather(2, b["dof_axis"].expand(B, -1)[..., None])[..., 0]
        x0 = torch.zeros(B, self.n, device=self.dev)
        x0[:, :self.npose] = torch.minimum(torch.maximum(pose, b["pose_lo"]), b["pose_hi"])
        s = torch.ones(B, device=self.dev)
        if self.size_dof is not None:  # palm size ratio -> wrist scale (scales the hand about the wrist joint)
            rms = (((tV - mu[:, :1]) ** 2).sum(-1) * b["oh"][:, 0]).sum(1) / b["m0"][0]
            s = (rms.sqrt() / b["palm_rms0"]).clamp(0.7, 1.4)
            x0[:, self.size_dof] = torch.log2(s) / self.size_coef
        x0[:, self.sl_t] = mu[:, 0] - (R0 @ (b["wrist0"] + s[:, None] * (b["palm0"] - b["wrist0"]))[..., None])[..., 0]
        return x0, R0

    @classmethod
    def from_model(cls, model, vreg=VREG, jreg=JREG, **kw):
        """Fitter for a loaded model (its head_pose_hand: SAM 3D Body's MHR head with the TorchScript rig) or the head."""
        head = getattr(model, "head_pose_hand", model)
        return cls(build_asset(head, vreg=vreg, jreg=jreg), **kw)

    def features(self, tV, tJ, x0, R0):
        """Learned initialiser input: wrist-centred targets (B, 778, 3), (B, 21, 3), closed-form x0 / R0 -> (B, d_in)."""
        q = torch.cat([tV.index_select(1, self.B["sub"]), tJ], 1) @ R0  # rows @ R0 = R0^T p: the closed-form root frame
        return torch.cat([q.flatten(1) * 10.0, x0[:, :self.n - 6]], 1)

    def apply_init(self, o, x0, R0):
        """Learned residual o (B, n) on the closed-form init -> x (rotation folded into R0), R0."""
        nf, b = self.n - 6, self.B
        pose = torch.minimum(torch.maximum(x0[:, :self.npose] + o[:, :self.npose], b["pose_lo"]), b["pose_hi"])
        x = torch.cat([pose, x0[:, self.npose:nf] + o[:, self.npose:nf], torch.zeros_like(x0[:, self.sl_rot]), x0[:, self.sl_t] + 0.01 * o[:, self.sl_t]], 1)
        return x, R0 @ rodrigues(0.1 * o[:, self.sl_rot], b["eye3"])

    def _refine(self, tV, tJ, x0, R0):
        return self.apply_init(self.init_net(self.features(tV, tJ, x0, R0)), x0, R0)

    def _residuals(self, X, tV, tJ, R0, b, points=False, v=None):
        """X (N, n) parameter sets, their targets (N, nm, 3), (N, nk, 3) and root rotations (N, 3, 3) -> (N, rows)
        [, the rotated model points (N, nm + nk, 3) before the translation]. v: the fit-set vertices if known."""
        N = X.shape[0]
        v = self.rig.verts(X, b, "f")[0] if v is None else v
        R = R0 @ rodrigues(X[:, self.sl_rot], b["eye3"])
        Q = torch.cat([self.rig.regress(v, b, "Wvs"), self.rig.regress(v, b, "Wj")], 1)
        Q = (Q[:, :, None, :] * R[:, None]).sum(-1)  # rows @ R^T, elementwise
        d = Q + X[:, None, self.sl_t] - torch.cat([tV, tJ], 1)
        lim = self.rig.model_params(X, b).index_select(1, b["slim_idx"])
        viol = torch.relu(lim - b["slim_hi"]) - torch.relu(b["slim_lo"] - lim)
        prior = torch.cat([viol, X[:, self.npose:self.npose + self.rig.ns + self.rig.nb]], 1) * b["prior_w"]
        r = torch.cat([(d * b["rw3"][:, None]).reshape(N, -1), prior], 1)
        return (r, Q) if points else r

    def _cost(self, r, b):
        """Huber cost of the data rows (delta in metres per coordinate) + squared prior rows, per hand."""
        nd = self.n_data
        u = r[:, :nd].abs() / b["rscale"]
        h = torch.where(u <= self.huber, 0.5 * u * u, self.huber * (u - 0.5 * self.huber)) * b["rscale"] ** 2
        return h.sum(1) + 0.5 * r[:, nd:].square().sum(1)

    def _lm_step(self, x, lam, tV, tJ, R0, b):
        """Levenberg-Marquardt step with Huber (IRLS) weights on the data rows, joint angles clamped to MHR's limits.
        Jacobian: the rotation (re-centred on R0 every step) and translation columns in closed form, the articulation /
        scale / shape columns by forward (or central) differences (all perturbed parameter sets of all hands in one rig
        batch) or in closed form (jac "analytic"). The step is kept where it lowers the cost (damping / 3, rotation
        folded into R0), else rejected (x 4)."""
        B, n = x.shape
        nf, nd, nr = n - 6, self.n_data, self.n_data // 3
        if self.jac == "ad":  # forward-mode Jacobian (exact)
            r = self._residuals(x, tV, tJ, R0, b)
            one = lambda xx, v, j, rr: self._residuals(xx[None], v[None], j[None], rr[None], b)[0]
            Jt = torch.func.vmap(torch.func.jacfwd(one))(x, tV, tJ, R0).mT  # (B, n, rows)
        elif self.jac == "analytic":  # closed-form rig Jacobian (MHRHand.verts_jac): one rig evaluation per step
            v, Jv = self.rig.verts_jac(x, b)
            r, Q = self._residuals(x, tV, tJ, R0, b, points=True, v=v)
            JQ = torch.cat([self.rig.regress(Jv, b, "Wvs"), self.rig.regress(Jv, b, "Wj")], 1)  # (B, nr, nf, 3)
            JQ = torch.einsum("bij,brcj->bcri", R0, JQ) * b["rw3"][None, None, :, None]
            lim = self.rig.model_params(x, b).index_select(1, b["slim_idx"])
            act = ((lim > b["slim_hi"]) | (lim < b["slim_lo"])).float()  # active soft limits
            nl, ns, nb = len(b["slim_idx"]), self.rig.ns, self.rig.nb
            Jl = torch.zeros(B, nf, nl, device=x.device)
            Jl[:, self.npose:self.npose + ns] = b["A_scale"].index_select(1, b["slim_idx"])[None] * act[:, None]
            Jp = torch.zeros(B, nf, ns + nb, device=x.device)
            Jp[:, self.npose:] = torch.eye(ns + nb, device=x.device)
            Jf = torch.cat([JQ.reshape(B, nf, nd), torch.cat([Jl, Jp], 2) * b["prior_w"]], 2)
        else:
            E = b["eye_n"][:nf] * b["eps"][:nf, None]
            k = 1 + (2 if self.fd == "central" else 1) * nf
            X = torch.cat([x[:, None], x[:, None] + E[None]] + ([x[:, None] - E[None]] if self.fd == "central" else []), 1).reshape(-1, n)
            rep = lambda t: t[:, None].expand(B, k, *t.shape[1:]).reshape(-1, *t.shape[1:])
            ra, Q = self._residuals(X, rep(tV), rep(tJ), rep(R0), b, points=True)
            ra, Q = ra.reshape(B, k, -1), Q.reshape(B, k, nr, 3)[:, 0]
            r = ra[:, 0]
            lo = ra[:, 1 + nf:] if self.fd == "central" else r[:, None]
            Jf = (ra[:, 1:1 + nf] - lo) / ((2 if self.fd == "central" else 1) * b["eps"][:nf])[None, :, None]  # (B, nf, rows)
        if self.jac != "ad":
            # d(R0 rod(w) p + t) / dw_k at w = 0: R0[:, k] x (R0 p); d / dt_k: e_k (data rows only)
            Jw = torch.cross(R0.mT[:, :, None, :].expand(B, 3, nr, 3), Q[:, None].expand(B, 3, nr, 3), dim=-1)
            Jtr = b["eye3"][None, :, None, :].expand(B, 3, nr, 3)
            rig = torch.cat([Jw, Jtr], 1) * b["rw3"][None, None, :, None]  # (B, 6, nr, 3)
            Jt = torch.cat([Jf, torch.cat([rig.reshape(B, 6, nd), torch.zeros(B, 6, r.shape[1] - nd, device=x.device)], 2)], 1)
        w = torch.clamp(self.huber / (r[:, :nd].abs() / b["rscale"]).clamp(min=1e-9), max=1.0)
        w = torch.cat([w, torch.ones_like(r[:, nd:])], 1)
        Jtw = Jt * w[:, None]
        # normal equations per hand as plain matmuls (cuBLAS's batched GEMM is ~5x slower at this shape)
        H = torch.stack([Jt[i] @ Jtw[i].T for i in range(B)])
        g = (Jtw * r[:, None]).sum(-1)[..., None]
        if self.adaptive:  # joint angles at a limit that the step would push further out stay fixed (projected Newton active set)
            xp, gp = x[:, :self.npose], g[:, :self.npose, 0]
            act = ((xp <= b["pose_lo"] + 1e-6) & (gp > 0)) | ((xp >= b["pose_hi"] - 1e-6) & (gp < 0))
            m = torch.cat([1.0 - act.float(), torch.ones_like(x[:, self.npose:])], 1)
            H = H * m[:, :, None] * m[:, None, :] + torch.diag_embed(1 - m)
            g = g * m[..., None]
        H = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1)) + 1e-9 * b["eye_n"]
        L = torch.linalg.cholesky_ex(H)[0]  # H is SPD; _ex: no host sync; triangular solves (cholesky_solve is not capturable)
        xc = x - torch.linalg.solve_triangular(L.mT, torch.linalg.solve_triangular(L, g, upper=False), upper=True)[..., 0]
        xc = torch.cat([torch.minimum(torch.maximum(xc[:, :self.npose], b["pose_lo"]), b["pose_hi"]), xc[:, self.npose:]], 1)
        ok = self._cost(self._residuals(xc, tV, tJ, R0, b), b) < self._cost(r, b) if self.adaptive else None
        Rc = R0 @ rodrigues(xc[:, self.sl_rot], b["eye3"])
        xc = torch.cat([xc[:, :self.sl_rot.start], torch.zeros_like(xc[:, self.sl_rot]), xc[:, self.sl_t]], 1)
        if not self.adaptive:  # fixed damping, every step taken: the fit is a continuous function of its inputs
            return xc, lam, Rc
        return (torch.where(ok[:, None], xc, x), torch.where(ok, lam / 3, lam * 4).clamp(1e-7, 1e4),
                torch.where(ok[:, None, None], Rc, R0))

    def _final(self, x, R0, c):
        """Fitted hand mesh, its MANO-topology map and joints (camera frame), camera rotation, and the parameters that
        do not depend on the camera placement: [hand_pca (54) | scale (28) | shape (45) | model_params (204, global 0)]."""
        b = self.B
        R = R0 @ rodrigues(x[:, self.sl_rot], b["eye3"])
        v, g = self.rig.verts(x, b, "o")
        v = (v[:, :, None, :] * R[:, None]).sum(-1) + x[:, None, self.sl_t]
        kp = g.index_select(1, b["kp_joints"])[..., :3] / 100 if "kp_joints" in b else torch.zeros(len(x), 21, 3, device=x.device)
        kp = (kp[:, :, None, :] * R[:, None]).sum(-1) + x[:, None, self.sl_t]
        B, npo, ns, nb = len(x), self.npose, self.rig.ns, self.rig.nb
        pose = x[:, :npo]
        cont = torch.zeros(B, 54, device=x.device)
        if npo == 27:  # SAM 3D Body's hand code: 6D per 3-dof joint, sin / cos per 1-dof angle (compact_model_params_to_cont_hand)
            cont = cont.index_copy(1, b["c3"], six_d(pose.index_select(1, b["p3"]).reshape(B, -1, 3)).reshape(B, -1))
            q = pose.index_select(1, b["p1"])
            cont = cont.index_copy(1, b["c1"], torch.stack([q.sin(), q.cos()], -1).reshape(B, -1))
            cont = cont - b["hmean"]
        loc = torch.cat([cont, torch.zeros(B, 28, device=x.device).index_copy(1, b["sc_idx"], x[:, npo:npo + b["sc_idx"].shape[0]]),
                         torch.zeros(B, b["n_shape"].shape[0], device=x.device).index_copy(1, b["sh_idx"], x[:, npo + ns:npo + ns + nb]),
                         self.rig.model_params(x, b)], 1)
        return v + c, self.rig.regress(v, b, "Wv_o") + c, self.rig.regress(v, b, "Wj_o") + c, R, loc, kp + c

    def _run(self, s):
        b = self.B
        c = s["tJ"][:, :1]
        tV, tJ = s["tV"] - c, s["tJ"] - c
        with torch.no_grad():
            x, R0 = self.init_c(tV)
            if self.init_net is not None:
                x, R0 = self.refine_c(tV, tJ, x, R0)
            lam = torch.full((len(x),), self.lam0, device=self.dev)
            tVf = tV if self.rows is None else tV.index_select(1, self.rows)
            for _ in range(self.iters):
                x, lam, R0 = self.step(x, lam, tVf, tJ, R0, b)
            v, vm, j, R, loc, kp = self.final_c(x, R0, c)
            s["out_kp"].copy_(kp)
            s["out_p"].copy_(loc)
            s["out_v"].copy_(v)
            s["out_vm"].copy_(vm)
            s["out_j"].copy_(j)
            s["out_x"].copy_(x)
            s["out_R"].copy_(R)
            s["out_t"].copy_(x[:, self.sl_t] + c[:, 0])

    def __call__(self, tV, tJ):
        """tV (B, 778, 3), tJ (B, 21, 3) numpy, right-hand frame, camera coordinates -> dict of numpy arrays: verts
        (B, 2445, 3), verts_mano (B, 778, 3), joints (B, 21, 3), x (B, n), R (B, 3, 3), t (B, 3). Chunks of <= chunk
        hands (CUDA graphs: 2; the batched linear solve takes a non-capturable path for larger batches)."""
        if len(tV) > self.chunk:
            outs = [self(tV[i:i + self.chunk], tJ[i:i + self.chunk]) for i in range(0, len(tV), self.chunk)]
            return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        B = len(tV)
        if B not in self.graphs:
            Z = lambda *sh: torch.zeros(*sh, device=self.dev)
            s = dict(tV=Z(B, self.nm_all, 3), tJ=Z(B, self.nk, 3), out_v=Z(B, len(self.B["obase"]), 3), out_vm=Z(B, self.nm_all, 3), out_j=Z(B, self.nk, 3),
                     out_x=Z(B, self.n), out_R=Z(B, 3, 3), out_t=Z(B, 3), out_p=Z(B, 54 + 28 + len(self.B["n_shape"]) + self.a["P"]), out_kp=Z(B, 21, 3))
            s["tV"].copy_(torch.as_tensor(np.asarray(tV), dtype=torch.float32))
            s["tJ"].copy_(torch.as_tensor(np.asarray(tJ), dtype=torch.float32))
            g = None
            if self.graph:
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    self._run(s)  # compile + allocations outside the capture
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._run(s)
            self.graphs[B] = (s, g)
        s, g = self.graphs[B]
        inp = torch.as_tensor(np.concatenate([np.asarray(tV, np.float32).reshape(B, -1), np.asarray(tJ, np.float32).reshape(B, -1)], 1))
        inp = inp.to(self.dev, non_blocking=True)
        s["tV"].copy_(inp[:, :self.nm_all * 3].reshape(B, self.nm_all, 3))
        s["tJ"].copy_(inp[:, self.nm_all * 3:].reshape(B, self.nk, 3))
        if g is not None:
            g.replay()
        else:
            self._run(s)
        keys = ("out_v", "out_vm", "out_j", "out_x", "out_R", "out_t", "out_p", "out_kp")
        flat = torch.cat([s[k].reshape(B, -1) for k in keys], 1).cpu().numpy()  # one device-to-host copy
        out, o = {}, 0
        for k in keys:
            m = s[k][0].numel()
            out[k[4:]] = flat[:, o:o + m].reshape(B, *s[k].shape[1:]).copy()
            o += m
        out["verts"], out["verts_mano"], out["joints"], out["joints_rig"] = out.pop("v"), out.pop("vm"), out.pop("j"), out.pop("kp")
        return out

    def params(self, out, k=None, R=None, t=None):
        """SAM 3D Body / MHR parameters of fitted hand k (all hands if None); R, t: camera-frame rotation / translation of
        the wrist-centric hand (default the fit's; e.g. after a 2D placement). hand_pose: the 27 right-hand joint angles
        (MHR model parameters, SAM 3D Body's joint order); hand_pca: the hand decoder's 54 hand components (continuous
        6D / sin-cos code minus the mean, components = identity); scale (28) / shape (45): SAM 3D Body's components;
        global_rot: its wrist-centric euler angles (xyz); cam_t: its camera translation; model_params (204): the MHR rig
        input, mhr(shape, model_params, 0) / 100 with y, z negated + cam_t = the fitted mesh. The placement-independent
        part comes from the device (_final), the global part is set here."""
        if k is None:
            return [self.params(out, i) for i in range(len(out["x"]))]
        a = self.a
        if self._pp is None:
            npf = lambda k: np.asarray(a[k], np.float64)
            self._pp = dict(L2W=npf("L2W"), rw=npf("right_wrist"), root=npf("root"), keep=npf("keep"), comps=npf("scale_comps_all"),
                            A=npf("A_scale")[:, 136:], ns=len(a["scale_comps_all"]))
        pp = self._pp
        x = np.asarray(out["x"][k], np.float64)
        R = np.asarray(out["R"][k] if R is None else R, np.float64)
        t = np.asarray(out["t"][k] if t is None else t, np.float64)
        nsh = a["n_shape"]
        p = out["p"][k]
        hand_pca, scale, shape, mp = p[:54], p[54:54 + pp["ns"]], p[54 + pp["ns"]:54 + pp["ns"] + nsh], p[54 + pp["ns"] + nsh:].astype(np.float64)
        if a.get("free_scales"):  # SAM 3D Body components = least-squares fit of the free hand scales (model_params exact)
            d = pp["A"].T @ x[27:27 + self.rig.ns]
            kept = pp["keep"][136:] > 0
            scale = np.linalg.lstsq(pp["comps"][:, kept].T, d[kept], rcond=None)[0]
        Eg = FLIP @ R
        gb = euler_xyz_np(Eg @ pp["L2W"])  # MHRHead.mhr_forward, hand mode
        mp[:3] = -(rotmat_xyz_np(gb) @ (pp["rw"] - pp["root"]) + pp["root"]) * 10 * pp["keep"][:3]
        mp[3:6] = gb * pp["keep"][3:6]
        f = lambda v: np.asarray(v, np.float32)
        return dict(hand_pose=f(x[:27]), hand_pca=f(hand_pca), scale=f(scale), shape=f(shape), global_rot=f(euler_xyz_np(Eg)),
                    global_rotmat=f(R), cam_t=f(t), model_params=f(mp))


def place_2d(joints, uv, K):
    """Rigid placement of a fitted hand so that its joints (21, 3) project onto the 2D joints uv (21, 2) under K (PnP
    from the 3D placement; the mesh stage's mesh_align_2d) -> (R, t, c): x -> (x - c) R^T + t, or None."""
    import cv2

    c = np.asarray(joints, np.float64)[0].copy()
    ok, rv, tv = cv2.solvePnP(np.asarray(joints, np.float64) - c, np.ascontiguousarray(np.asarray(uv, np.float64)), np.asarray(K, np.float64), None,
                              rvec=np.zeros(3), tvec=c.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tv).all() or tv[2, 0] <= 0.05:
        return None
    return cv2.Rodrigues(rv)[0], tv.reshape(3), c


def fit_mhr_hand(fitter, points3d, joints3d, is_right=True, K=None, joints2d=None):
    """Fit the MHR hand (SAM 3D Body's rig) to MANO-topology surface points + 21 joints of one hand or a batch.

    fitter: MHRHandFitter (MHRHandFitter.from_model(model, iters=2, init_net="mhr_fitter.pt")); points3d (778, 3) or
    (B, 778, 3) and joints3d (21, 3) / (B, 21, 3): camera frame, metres, OpenPose joint order (FAH's verts3d_dense /
    joints3d); is_right: bool or (B,); K (3, 3) with joints2d (21, 2) / (B, 21, 2): optional, places the fitted hand
    rigidly onto the 2D joints (PnP). Returns a dict per hand (a list for a batch):
    verts (2445, 3) the MHR hand mesh (camera frame) with faces (F, 3) (winding of the hand's side), verts_mano (778, 3)
    its MANO-topology map, joints (21, 3) MANO-convention kinematic joints (MHR -> MANO21 regressor), joints_rig (21, 3)
    the rig's own hand keypoints (skeleton joints, SAM 3D Body's keypoint mapping), params (MHRHandFitter.params: SAM 3D
    Body / MHR parameters; mirrored = a left hand, fitted as SAM 3D Body does: the right rig on the mirrored hand)."""
    P, J = np.asarray(points3d, np.float64), np.asarray(joints3d, np.float64)
    single = P.ndim == 2
    P, J = (P[None], J[None]) if single else (P, J)
    side = np.broadcast_to(np.asarray(is_right, bool), (len(P),))
    M = np.stack([np.array([1.0 if r else -1.0, 1.0, 1.0]) for r in side])
    o = fitter(P * M[:, None], J * M[:, None])
    uv = None if joints2d is None else np.asarray(joints2d, np.float64).reshape(len(P), -1, 2)
    faces = np.asarray(fitter.a["faces"])
    res = []
    for k, m in enumerate(M):
        v, vm, j, jr = (o[n][k] * m for n in ("verts", "verts_mano", "joints", "joints_rig"))
        R, t = o["R"][k], o["t"][k]  # right-hand frame
        pl = place_2d(j, uv[k], K) if K is not None and uv is not None else None
        if pl is not None:
            Rr, tv, c = pl
            v, vm, j, jr = ((y - c) @ Rr.T + tv for y in (v, vm, j, jr))
            Rm = np.diag(m) @ Rr @ np.diag(m)  # the same rigid motion in the right-hand frame
            R, t = Rm @ R, Rm @ (t - c * m) + tv * m
        f32 = lambda y: np.asarray(y, np.float32)
        res.append(dict(verts=f32(v), faces=faces if side[k] else faces[:, ::-1].copy(), verts_mano=f32(vm), joints=f32(j), joints_rig=f32(jr),
                        params=dict(fitter.params(o, k, R=R, t=t), mirrored=not side[k])))
    return res[0] if single else res

