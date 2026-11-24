"""FAB inference API: one call from an RGB image (or a stream of video frames) to every FAB output per person.

    from fab.predictor import FABPredictor
    fab = FABPredictor("fab-l", fast=True)               # released models: "fab-l" (default), "fab-s", "fab-h"; or a path
    people = fab.predict(image_rgb)                      # single image (K optional; boxes optional, detector otherwise)
    for people in fab.predict_video(frames, K): ...      # video mode: detector every track_every frames, boxes tracked
    for t, people in fab.predict_video_smoothed(frames, K): ...  # + a causal 3-frame-lag smoother per track (G1 / SMPL)

Video contacts (the causal contact model in the world frame) need world cameras, so they run in world mode
(``fab-world``), not here.

Options (the defaults are the released settings):
    fast=True                  compiled body path, one CUDA graph per person count (the first call per count compiles,
                               about 1-2 min); uses TensorRT engines built on this GPU when present (``fab-build-trt``)
    smpl_fitter="auto"         the released SMPL fitter for the model: the points fitter for FAB-L / FAB-S, the mesh
                               fitter for FAB-H; "" for no SMPL fit
    profile="auto"             FAB-H's SMPL fit: "mesh" (its default) fits the SMPL to its MHR mesh, "points" to its dense
                               points (the FAB-L / FAB-S path); FAB-L / FAB-S always fit to their points
    dense_outputs="auto"       FAB-H's mesh profile: the dense head is not run; True runs it and returns its outputs
                               (dense_surface, keypoints2d, ...), which the fit does not read
    dense_fit=True             opt-in refined fit (fab/gn_fit.DenseFit): two Gauss-Newton steps on the dense-refined mesh
                               and the dense 2D keypoints (needs the 23-joint regressor, see docs/usage.md)
    foot_refine=1              the fitted feet re-solved on the dense foot points (closed form, one pass); 0 = off
    dense_25d_mix="auto"       test-time mix of the lifted (2.5D) and the regressed 3D points; a float sets it

Each person is a dict (numpy, camera frame, metres unless noted):
  box (4) xyxy px used for the crop        cam_t (3)
  mesh_mhr (18439, 3)                      mesh_smpl (6890, 3) SMPL topology; mesh_smpl_dense: refined by the dense head
  keypoints3d (70, 3) MHR keypoints        keypoints2d_projected (70, 2) px, projection of keypoints3d
  keypoints2d (70, 2) px, the dense head's own 2D (prefer it for 2D tasks)
  dense_surface_cam (468, 3) surface points in the camera frame (anchored at the MHR mesh's hips, as the foot step);
  dense_surface (468, 3) the head's raw points (camera-frame orientation / scale, up to a common offset: the head is hip-aligned); dense_surface2d (468, 2) px
  joints_h36m14 (14, 3) the dense head's evaluation skeleton (root-relative)
  g1_q (29,) Unitree G1 joint angles (URDF order, rad, clamped to limits); g1_root (3, 3) robot root rotation
  mesh_smpl_points (6890, 3) optional parameter-free mesh from the dense surface points (needs the points-to-mesh map,
    see docs/usage.md)
  parts (H, W) uint8 body-part map in the image (0 background, fab/body_parts.PART_NAMES); parts=False skips the branch
  parts_roi + roi (4,) the part map on the crop's image region x0, y0, x1, y1 only (full_maps=False: only these;
    fab/seg_maps.full_map rebuilds the full map)
  parts_crop (256, 256) uint8 + crop_affine (2, 3) image -> crop px (crop_maps=True: only these, no warp;
    fab/seg_maps.paste_roi makes the image map on demand)
  contact (16,) image contact probabilities: 8 static + 8 any, each l_heel, l_toe, r_heel, r_toe, l_hand, r_hand,
    l_knee, r_knee
  smpl_pose (72,) / smpl_betas (10,) / mesh_smpl_fit (6890, 3): neutral SMPL fitted to the dense surface by the
    one-pass learned fitter. FAB-H's default (profile "mesh") fits the same way to the surface points of its own
    SMPL-topology mesh
Without intrinsics K, the focal length is the image diagonal (SAM 3D Body's default).
"""

import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from fab import paths

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # ultralytics (detector engine, detector_backend="ultralytics") never pip-installs

BODY_KP = list(range(21)) + [41, 62]  # body, feet and wrists (crop boxes from keypoints)
H36M_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]  # the 14 evaluation joints of the H36M regressor (as bench.eval_hmr)
# the released person detector; a TensorRT engine for it is used when one was built on this GPU (fab-build-trt).
# detector="" means no detector.
DEPLOYED_DETECTOR = str(paths.shared("detector.pt"))
# smpl_fitter="deployed" / "auto": the released fitters by head type ("absolute": points-first dense heads, FAB-L / FAB-S;
# "residual": FAB-H's mesh fit)
DEPLOYED_FITTERS = {"absolute": str(paths.shared("smpl_fitter.pt")),
                    "residual": str(paths.shared("smpl_fitter_mesh.pt"))}


DEPLOYED_BODY = str(paths.model_file("fab-l"))


def detector_engine_unusable(engine):
    """None when the TensorRT detector engine can run here: tensorrt imports at the version fab-build-trt recorded for it
    (<engine>.json); otherwise the reason. Checked before ultralytics loads the engine (ultralytics would otherwise try to
    install another tensorrt, which cannot read the engine)."""
    rec = Path(str(engine) + ".json")
    if not rec.exists():
        return f"no build record {rec.name} (rebuild it with fab-build-trt)"
    want = json.loads(rec.read_text()).get("tensorrt")
    try:
        import tensorrt as trt
    except Exception as e:  # noqa: BLE001
        return f"tensorrt does not import: {type(e).__name__}"
    if trt.__version__ != want:
        return f"tensorrt {trt.__version__} here, the engine was built with {want}"
    return None


def loud_warning(msg):
    """A speed path switching itself off (never silent): a stderr line and a RuntimeWarning, at the switch-off."""
    import sys

    print(f"WARNING {msg}", file=sys.stderr, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
# FAB-L / FAB-S (points-first dense heads), the default profile "simple": every person -> the one-pass SMPL fit to the
# dense points -> 2 Gauss-Newton steps -> the foot refine onto the points' own feet, with the 2D foot term folded in (one
# solve); one 2.5D setting. FABPredictor(**SIMPLE_PROFILE)
SIMPLE_PROFILE = dict(smpl_fitter="deployed", foot_target="dense", foot_2d=1.0, foot_2d_mode="fold",
                      fit_gn_iters=2, fit_gn_robust=50.0, fit_gn_anchor=1e-3,
                      foot_2d_iters=1, dense_25d_mix=1.0, fit_graph=True, gpu_crop=True, fit_orient_mesh=1.0)
# fit_orient_mesh=1.0: the fit's whole-body rotation from the MHR mesh head (J24 Kabsch about the pelvis, after the
# Gauss-Newton steps); pose / shape / placement / the foot refine from the points.
# FAB-H (a checkpoint without a dense head): its paper fit, the default profile "fabh": the one-pass mesh fitter
# (smpl_fitter_mesh.pt) on the MHR mesh's 512 symmetric surface points, no foot refine
FABH_PROFILE = dict(smpl_fitter="deployed", foot_refine=0)
# FAB-H with its dense-point head (the released fab-h weights): profile "mesh", its default, is the same mesh fit as above (the
# MHR mesh's 512 symmetric surface points, no foot refine, feet onto the mesh, no 2.5D re-mix); the dense head is not run
# (dense_outputs="auto" / False), or with dense_outputs=True it runs and its outputs are returned, but the fit does not read
# them. profile="points" fits the SMPL to the dense points instead, the same path as FAB-L / FAB-S ("simple", SIMPLE_PROFILE).
FABH_MESH_PROFILE = dict(FABH_PROFILE, dense_25d_mix=None)
HEAD_ONLY = ("vis_only", "contact_only", "parts_only", "mask_only", "g1_only")  # train_fab frozen-body head runs


def body_args(state, path=""):
    """The body's training args, for every body-path default (foot target, ...): a bundle's stored "body_args"
    (scripts/build_bundle.py); else, for a frozen-body head run (train_fab --vis_only / --contact_only / --parts_only /
    --mask_only / --g1_only), the body args of its --init checkpoint (the body it was trained on); else the checkpoint's
    own args. A head checkpoint is a different run whose args lack the body's losses (e.g. no foot terms -> foot_target
    "auto" would pick "mesh" and move mesh_smpl_fit 5-20 mm). -> (args dict, source)"""
    ta = dict(state.get("args", {}) or {})
    if state.get("body_args"):
        return dict(state["body_args"]), "body_args"
    if any(ta.get(k) for k in HEAD_ONLY) or (ta.get("merged_heads") and not ta.get("head_args")):
        init = ta.get("init") or ""
        if init and Path(init).exists():
            st = torch.load(init, map_location="cpu", weights_only=False, mmap=True)
            return body_args(st, init)[0], f"init {init}"
        warnings.warn(f"{path}: a head checkpoint (frozen-body run) without body_args and no readable --init ({init!r}): "
                      "the body-path defaults (foot target) fall back to the head run's own args; build the bundle with "
                      "scripts/build_bundle.py")
    return ta, "own"


def default_K(H, W):
    """The intrinsics FABPredictor.predict uses when none are given: focal length = the image diagonal (px), principal
    point at the image centre."""
    f = float(np.hypot(H, W))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)


def d2h(ts):
    """{name: device tensor or None} -> {name: float32 numpy array}: one flat device -> host copy (one sync)."""
    names = [k for k, t in ts.items() if t is not None]
    if not names:
        return {}
    flat = torch.cat([ts[k].float().reshape(-1) for k in names]).cpu().numpy()
    out, o = {}, 0
    for k in names:
        m = ts[k].numel()
        out[k] = flat[o : o + m].reshape(tuple(ts[k].shape))
        o += m
    return out


def closed_form_translation(X, uv, K):
    """Translation t (3,) placing root-relative 3D points X (N, 3) onto pixels uv (N, 2) under intrinsics K:
    (u - cx) (Z + tz) = fx (X + tx), likewise for v, solved by linear least squares."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    n = len(X)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, -du, du * X[:, 2] - fx * X[:, 0]
    A[n:, 1], A[n:, 2], b[n:] = fy, -dv, dv * X[:, 2] - fy * X[:, 1]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def place_points_t(X, uv_crop, affine, K):
    """closed_form_translation on the GPU, batched and graph-capturable: root-relative points X (B, N, 3) onto their
    crop-px 2D uv_crop (B, N, 2) (affine (B, 2, 3) image -> crop px, K (B, 3, 3)) -> t (B, 3); float64 normal equations
    (the least-squares solution of the same system), solved by the adjugate."""
    X, uv, A, K = X.double(), uv_crop.double(), affine.double(), K.double()
    M, b = A[:, :, :2], A[:, :, 2]
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    Mi = torch.stack([M[:, 1, 1], -M[:, 0, 1], -M[:, 1, 0], M[:, 0, 0]], -1).view(-1, 2, 2) / det[:, None, None]
    uvi = torch.einsum("bij,bnj->bni", Mi, uv - b[:, None])  # crop px -> image px
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    du, dv = uvi[..., 0] - cx, uvi[..., 1] - cy
    bu, bv = du * X[..., 2] - fx * X[..., 0], dv * X[..., 2] - fy * X[..., 1]  # rows (fx, 0, -du) / (0, fy, -dv)
    n = X.shape[1]
    z = torch.zeros_like(det)
    r0 = torch.stack([n * fx[:, 0] ** 2, z, -fx[:, 0] * du.sum(1)], -1)
    r1 = torch.stack([z, n * fy[:, 0] ** 2, -fy[:, 0] * dv.sum(1)], -1)
    r2 = torch.stack([-fx[:, 0] * du.sum(1), -fy[:, 0] * dv.sum(1), (du ** 2).sum(1) + (dv ** 2).sum(1)], -1)
    rhs = torch.stack([fx[:, 0] * bu.sum(1), fy[:, 0] * bv.sum(1), -(du * bu).sum(1) - (dv * bv).sum(1)], -1)
    c0, c1, c2 = torch.cross(r1, r2, dim=-1), torch.cross(r2, r0, dim=-1), torch.cross(r0, r1, dim=-1)
    inv = torch.stack([c0, c1, c2], -1) / (r0 * c0).sum(-1)[:, None, None]
    return (inv @ rhs[..., None])[..., 0]


class GraphFitChain:
    """The absolute (points-first) head's fit stage in one CUDA graph per person count,
    straight from the forward's GPU outputs: the foot target (the MHR mesh, the dense-refined mesh or their mean,
    interpolated on the GPU), the one-pass fit, the Gauss-Newton steps, the folded foot_2d target, the foot refine,
    the pose as axis-angle and the points' placement -> one packed (n, D) tensor for the forward's single device -> host
    copy: [pose (72), betas, vertices (6890 x 3, before the placement), placement t (3)]."""

    def __init__(self, fab):
        self.fab, self.graphs = fab, {}
        dev = fab.device
        sidx, nn_, wgt = fab.interp
        self.sidx = torch.as_tensor(sidx, device=dev)
        self.nn = torch.as_tensor(nn_, device=dev)
        self.wgt = torch.as_tensor(wgt, dtype=torch.float32, device=dev)
        self.j14 = torch.as_tensor(fab.j14, dtype=torch.float32, device=dev)
        self.fit_pos = None if fab.fit_pos is None else torch.as_tensor(fab.fit_pos, device=dev)
        self.fold = fab.f2d_sets is not None and fab.foot_2d_mode == "fold"
        if self.fold and getattr(fab, "_fold_graph", None) is None:
            from fab.foot2d import GraphFoldTarget

            fab._fold_graph = GraphFoldTarget(fab.body.J_regressor.float(), fab.f2d_sets, fab.foot_2d_max,
                                              fab.f2d_graph.args[2], fab.model.size, fab.foot_2d)
        # fit_compile: the same steps as one torch.compile'd (inductor) function inside the same CUDA graph
        # (~2,460 small kernels fused; outputs at rounding level); eager _run when off or when compilation fails
        self.run, self.paths, self.compiled = self._run, {}, False  # paths: graph key -> "compiled" | "eager"
        if getattr(fab, "fit_compile", False) and str(dev).startswith("cuda"):
            self.run, self.compiled = torch.compile(self._run, dynamic=False), True

    def _run(self, xyz, smpl, cam_t, uv01, cam_int, affine, vis_logits=None, temporal=None):
        from fab.smpl_fit import rotmat_to_aa

        fab, n = self.fab, len(xyz)
        j14, surf = xyz[:, 70:84], xyz[:, 84:]
        ftm = smpl
        if fab.foot_target in ("mix", "dense"):  # with the dense points' own feet (the dense-refined mesh)
            hip = lambda J: J[:, 2:4].mean(1, keepdim=True)  # noqa: E731
            dd = surf - (hip(j14) - hip(torch.einsum("jv,bvc->bjc", self.j14, smpl))) - smpl[:, self.sidx]
            dm = smpl + (dd[:, self.nn] * self.wgt[None, :, :, None]).sum(2)
            ftm = dm if fab.foot_target == "dense" else 0.5 * (smpl + dm)
        pts = surf if self.fit_pos is None else surf[:, self.fit_pos]
        R0, b0, v0 = fab.fitter._place(pts, *fab.fitter.fit(pts))
        t = None
        if fab.fit_gn_ray != 1.0:  # anisotropic: the points' placement first, for their camera rays
            t = place_points_t(xyz[:, 70:], uv01[:, 70:] * fab.model.size, affine, cam_int).float()
        if fab.fit_gn_iters > 0:  # the always-on Gauss-Newton to the points (FABPredictor._gn_refine)
            ray = None
            if t is not None:
                Xc = torch.cat([surf, j14], 1) + t[:, None]
                ray = Xc / Xc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            R0, b0, v0 = fab._gn_refine(R0, b0, v0, surf, j14, temporal, ray=ray)
        if fab.fit_orient_mesh > 0:  # the whole-body rotation part of the way to the MHR mesh's
            R0, v0 = fab._orient_to_mesh(R0, v0, smpl)
        if self.fold:
            ftm = fab._fold_graph._run(ftm, cam_t, uv01[:, :70], cam_int, affine,
                                       vis_logits if fab.foot_2d_vis else None)
        if t is not None:  # the foot refine's target: its along-ray mismatch to the fit's foot x sqrt(fit_gn_ray)
            ftm = fab._ray_foot_target(ftm, v0, t)
        R, v = fab.foot.refine._closed(R0, b0, v0, ftm)
        aa = rotmat_to_aa(R.reshape(-1, 3, 3)).reshape(n, -1)
        if t is None:
            t = place_points_t(xyz[:, 70:], uv01[:, 70:] * fab.model.size, affine, cam_int).float()
        packed = torch.cat([aa, b0, v.reshape(n, -1), t], 1)
        return {"packed": packed, "b0": b0}


    @torch.no_grad()
    def __call__(self, xyz, smpl, cam_t, uv01, cam_int, affine, vis_logits=None, temporal=None):
        """temporal: None or (Rp (n, 24, 3, 3), bp (n, 10), w (n,), w_b (n,)): the previous frame's fit of each person
        and its prior weights (0 = none; FABPredictor.predict(prev=...))."""
        x = {k: t.float() for k, t in (("xyz", xyz), ("smpl", smpl), ("cam_t", cam_t), ("uv01", uv01),
                                       ("cam_int", cam_int), ("affine", affine), ("vis_logits", vis_logits))
             if t is not None}
        tx = None if temporal is None else tuple(t.float() for t in temporal)
        key = (len(xyz), vis_logits is not None, temporal is not None)
        if key not in self.graphs:
            static = {k: t.clone() for k, t in x.items()}
            st_t = None if tx is None else tuple(t.clone() for t in tx)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                if self.compiled:  # the eager steps once first: their lazy members (FastGN, its tensors) exist
                    self._run(**static, temporal=st_t)  # before tracing (dynamo guards on tensors made inside the frame)
                for _ in range(3):
                    try:
                        self.run(**static, temporal=st_t)
                    except Exception as e:  # noqa: BLE001  (fit_compile: inductor unavailable / failed -> eager steps)
                        if not self.compiled:
                            raise
                        import warnings

                        warnings.warn(f"FABPredictor: fit_compile FAILED, the fit chain runs eager ({type(e).__name__}: "
                                      f"{str(e)[:300]})", RuntimeWarning, stacklevel=2)
                        self.run, self.compiled = self._run, False
                        self.run(**static, temporal=st_t)
                self.paths[key] = "compiled" if self.compiled else "eager"  # which path each graph captured
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.run(**static, temporal=st_t)
            self.graphs[key] = (g, static, st_t, out)
        g, static, st_t, out = self.graphs[key]
        for k, t in x.items():
            static[k].copy_(t)
        if tx is not None:
            for dst, src in zip(st_t, tx):
                dst.copy_(src)
        g.replay()
        return out


class GraphEngine:
    """A fixed one-person TensorRT body engine (fab/tensorrt.TensorRTBody) replayed from one CUDA graph per
    person count (the engine run once per person inside the graph, its outputs copied into per-count buffers); returns
    the graph's output dict (valid until the next call), as precision.GraphBody. batched {N: a fixed N-person engine of
    the same recipe}: the people run in the largest batches available (e.g. 5 = 4 + 1), else one at a time."""

    def __init__(self, engine, batched=None):
        self.engine, self.graphs, self.batched = engine, {}, dict(batched or {})

    def _run(self, img, c, s, o, K, A):
        outs, i, n = [], 0, len(img)
        while i < n:
            k = max([b for b in self.batched if b <= n - i], default=1)
            eng = self.batched.get(k, self.engine)
            r = eng(img[i : i + k], c[i : i + k], s[i : i + k], o[i : i + k], K[i : i + k], affine=A[i : i + k])
            outs.append({kk: v.clone() for kk, v in r.items()})
            i += k
        return {k: torch.cat([x[k] for x in outs]) for k in outs[0]}

    @torch.no_grad()
    def __call__(self, img, c, s, o, K, A):
        x = tuple(t.float().contiguous() for t in (img, c, s, o, K, A))
        if len(img) not in self.graphs:
            static = [t.clone() for t in x]
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self._run(*static)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run(*static)
            self.graphs[len(img)] = (g, static, out)
        g, static, out = self.graphs[len(img)]
        for dst, src in zip(static, x):
            dst.copy_(src, non_blocking=True)
        g.replay()
        return out


class GraphBodyBox:
    """precision.GraphBody + the prompt box (B, 4) as a 7th static input, for a "+extent" person-mask head."""

    def __init__(self, model):
        self.model, self.graphs = model, {}
        model.body.inputs_in_01 = True

    def _run(self, img, c, s, o, K, A, box):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(img, c, s, o, K, affine=A, mask_box=box)

    def __call__(self, *x):
        if len(x[0]) not in self.graphs:
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
            self.graphs[len(x[0])] = (g, static, out)
        g, static, out = self.graphs[len(x[0])]
        for dst, src in zip(static, x):
            dst.copy_(src, non_blocking=True)
        g.replay()
        return out


class FABPredictor:
    def __init__(self, fab_ckpt="fab-l", teacher_dir="", detector="deployed", bm_dir="",
                 det_conf=0.25, track_every=10, track_scale=1.1, device="cuda", smpl_fitter="auto", fast=False,
                 dense_fit=False, reg23="",
                 foot_refine="auto", parts=True, parts_head="", mask=True, full_maps=True,
                 gpu_paste=True, crop_maps="auto", dense_25d_mix="auto",
                 foot_refine_gate=0.0, foot_refine_mode="kabsch", foot_target="auto", foot_2d="auto", foot_2d_max=60.0,
                 foot_2d_vis=True, place_incrop=False, foot_2d_iters=3, foot_2d_min=0.0, foot_2d_mode="auto",
                 gpu_crop=True, fit_graph=True, body_engine="auto",
                 fit_gn_iters="auto", fit_gn_robust=50.0, fit_gn_anchor=1e-3, fit_temporal=0.0, fit_temporal_betas=0.0,
                 fit_temporal_root=0.1, foot_sole_w=0.0, fit_gn_ray="auto", det_gpu=True,
                 body_batch="auto", fit_gn_root=0.0, fit_gn_core=1.0, fit_orient_mesh="auto", fit_compile=False,
                 detector_backend="fab", profile="auto", dense_outputs="auto"):
        from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from fab.decoder_model import build_fab_decoder
        from torchvision.transforms import ToTensor

        fab_ckpt = str(paths.model_file("fab-l" if fab_ckpt == "deployed" else fab_ckpt))
        paths.require(fab_ckpt, "FAB weights", "Download them with `hf download` (README.md) or set $FAB_WEIGHTS.")
        teacher_dir = Path(teacher_dir) if teacher_dir else paths.sam3d_dir()
        paths.require(teacher_dir / "model_config.yaml", "SAM 3D Body config", "Run `python -c \"import fab; fab.download_sam3d()\"`.")
        state = torch.load(fab_ckpt, map_location="cpu", weights_only=False)
        ta = state.get("args", {})
        # a bundle's own fit settings: "smpl_fitter" replaces the released fitter for smpl_fitter "deployed" / "auto",
        # "fit_gn_ray" the "auto" default (1.0); none in the released bundles
        self.bundle_defaults = dict(state.get("predictor_defaults") or {})
        # + <bundle dir>/config.json (or predictor_defaults.json): a bundle's output defaults kept beside its weights, e.g.
        # {"crop_maps": true}: the part head's 256 crop map
        side = Path(fab_ckpt).resolve().parent / "config.json"
        side = side if side.exists() else side.with_name("predictor_defaults.json")
        if side.exists():
            import json

            self.bundle_defaults.update(json.loads(side.read_text()))
        if self.bundle_defaults:
            print(f"FABPredictor: bundle predictor_defaults {self.bundle_defaults}", flush=True)
        # body-path defaults (foot target, ...) from the BODY's args, not a head run's (body_args)
        self.body_args, self.body_args_src = body_args(state, fab_ckpt)
        model, tb = build_fab_decoder(teacher_dir, ta.get("backbone", "dinov3_vitb16"),
                                      ta.get("size", 512), device=device, pretrained=False,
                                      dec_layers=ta.get("dec_layers", 0))
        w = dict(state["model"])
        w.update(state.get("ema") or {})
        if ta.get("arch") == "fabt":
            model.body.backbone, model.body.backbone_dtype = tb, torch.bfloat16
        else:
            del tb
        if "dense_head.surface_idx" in w:
            from fab.dense_head import dense_hr, dense_mask_depth, dense_mask_query, dense_part_depth, dense_xyz_layers
            from fab.dense_head import dense_head_kwargs

            model.add_dense_head(w["dense_head.surface_idx"].tolist(),
                                 residual=bool(int(w.get("dense_head.residual", torch.tensor(0)))), parts="dense_head.part_emb" in w,
                                 xyz_layers=dense_xyz_layers(w), part_depth=dense_part_depth(w),
                                 hr=dense_hr(w), mask_depth=dense_mask_depth(w), mask_query=dense_mask_query(w),
                                 **dense_head_kwargs(w)).to(device)
        if any(k.startswith("g1_head.") for k in w):
            model.add_g1_head(joints_input=w["g1_head.0.weight"].shape[0] > model.body.head_pose.proj.embed_dims).to(device)
        if any(k.startswith("contact_head.") for k in w) and model.dense_head is not None:  # explicit contacts
            from fab.contact_head import contact_head_args

            model.add_contact_head(**contact_head_args(w)).to(device)
        if any(k.startswith("vis_head.") for k in w) and model.dense_head is not None:  # per-keypoint visibility
            from fab.vis_head import vis_head_args

            model.add_vis_head(**vis_head_args(w)).to(device)
        model.load_state_dict(w, strict=False)
        # profile (the defaults of every "auto" body-path argument below): "simple" (SIMPLE_PROFILE) for absolute
        # (points-first) dense heads, FAB-L / FAB-S; for FAB-H with its dense head "mesh" (FABH_MESH_PROFILE, the default) or
        # "points" (= "simple"); "fabh" (FABH_PROFILE) for a FAB-H checkpoint without a dense head; "residual" for a residual
        # dense head (a training option, no released model: no SMPL fit unless smpl_fitter is given).
        # Explicit arguments always win.
        absolute0 = model.dense_head is not None and not int(model.dense_head.residual)
        if profile == "auto" and ta.get("arch") == "fabt" and model.dense_head is not None:  # FAB-H + dense head: the mesh fit
            profile = "mesh"
        self.profile_requested = profile
        if profile == "points":  # the points fit = the simple profile (the same path as FAB-L / FAB-S)
            profile = "simple"
        if profile == "auto":
            profile = "simple" if absolute0 else ("fabh" if model.dense_head is None else "residual")
        self.profile = profile
        pdef = {"simple": SIMPLE_PROFILE, "fabh": FABH_PROFILE, "mesh": FABH_MESH_PROFILE}.get(profile, {})
        # dense_outputs (profile "mesh"): "auto" / False = the dense head not run (the head-less FAB-H forward, outputs and
        # speed); True = it runs and its outputs are returned (dense_surface, keypoints2d, ...), unused by the fit
        if profile == "mesh" and dense_outputs in ("auto", False) and model.dense_head is not None:
            model.dense_head = None
        if smpl_fitter == "auto":  # the profile's fitter; an explicit "" always means no fit
            smpl_fitter = pdef.get("smpl_fitter", "")
        if profile == "simple":
            foot_target = pdef["foot_target"] if foot_target == "auto" else foot_target
        foot_refine = int(pdef.get("foot_refine", 1) if foot_refine == "auto" else foot_refine)
        fit_gn_iters = int(pdef.get("fit_gn_iters", 0) if fit_gn_iters == "auto" else fit_gn_iters)
        if dense_25d_mix == "auto":
            dense_25d_mix = pdef.get("dense_25d_mix", 0.5)
        if model.dense_head is not None and getattr(model.dense_head, "p25d_mix", 0.0) > 0 and dense_25d_mix is not None:
            # 2.5D points (--dense_25d): the test-time mix of the lifted and the regressed 3D (the profile's)
            model.dense_head.p25d_mix = float(dense_25d_mix)
        if parts_head and model.dense_head is not None:  # separately trained part / mask heads (other weights unchanged)
            pw = torch.load(parts_head, map_location="cpu", weights_only=False)
            pw = {**pw["model"], **(pw.get("ema") or {})}
            got_parts = model.dense_head.load_part_head({k: v for k, v in pw.items() if k.startswith("dense_head.part_")})
            got_mask = model.dense_head.load_mask_head({k: v for k, v in pw.items() if k.startswith("dense_head.mask_")})
            if not (got_parts or got_mask):
                raise ValueError(f"{parts_head}: no part or mask head")
            model.dense_head.to(device)
        if not parts and model.dense_head is not None:
            model.dense_head.part_emb = None  # skip the part branch (no other output reads it)
        if not mask and model.dense_head is not None:
            model.dense_head.mask_dconv = None  # skip the person-mask branch
        self.model = model.eval()
        self.full_maps = full_maps  # False: parts / mask only as <name>_roi + roi (x0, y0, x1, y1), no (H, W) arrays
        # absolute / 2.5D points placed by their own 2D: leave out the points within 2 % of the crop edge (truncated
        # points' heatmaps pin to it
        # occlusion image-short legs 27 -> 23 %, EMDB-2 world WA -0.1, never worse)
        self.place_incrop = place_incrop
        # fit_graph: an absolute head's fit stage (foot target, one-pass fit, Gauss-Newton, fold, foot refine, axis-angle,
        # placement) in one CUDA graph before the forward's single device -> host copy
        # (GraphFitChain); False: the per-module path
        self.fit_graph = fit_graph
        self.fit_compile = bool(fit_compile)  # GraphFitChain's steps through torch.compile
        # fit_gn_iters > 0 (always on when set): that many
        # Gauss-Newton iterations from the one-pass fit to the dense surface points + the 14 joints, every person,
        # Geman-McClure IRLS weights (fit_gn_robust mm) and a prior toward the one-pass fit's joint rotations / betas
        # (fit_gn_anchor), before the foot refine (absolute heads)
        self.fit_gn_iters, self.fit_gn_robust, self.fit_gn_anchor = int(fit_gn_iters), float(fit_gn_robust), \
            float(fit_gn_anchor)
        # fit_gn_ray < 1 (anisotropic): the along-ray (depth) part of every point residual weighted
        # fit_gn_ray in the Gauss-Newton, and of the foot refine's target-vs-fit mismatch (the image-plane part kept);
        # GraphFitChain only
        if fit_gn_ray == "auto":
            fit_gn_ray = self.bundle_defaults.get("fit_gn_ray", pdef.get("fit_gn_ray", 1.0))
        self.fit_gn_ray = float(fit_gn_ray)
        # opt-in, points only: fit_gn_root damps the Gauss-Newton's global
        # rotation steps (large: the one-pass fit's orientation kept); fit_gn_core multiplies the weight of the rigid core
        # (the pelvis / spine surface points and the J14 hips, shoulders, neck) in its point residual
        self.fit_gn_root, self.fit_gn_core = float(fit_gn_root), float(fit_gn_core)
        # fit_orient_mesh > 0: after the Gauss-Newton, the fit's whole-body rotation (J24 Kabsch
        # about its pelvis) turned this fraction of the way to the MHR mesh's; the articulation, shape and placement kept.
        # "auto": the profile's (simple 1.0; fabh / residual 0); 0 = the points' own orientation
        self.fit_orient_mesh = float(pdef.get("fit_orient_mesh", 0.0) if fit_orient_mesh == "auto" else fit_orient_mesh)
        # video mode (predict(prev=...) / predict_video): the Gauss-Newton's prior toward the previous frame's fit of the
        # same person, fit_temporal on the joint rotations (fit_temporal_root x it on the global rotation) and
        # fit_temporal_betas on the shape (world W: per-frame pose / scale consistency)
        self.fit_temporal, self.fit_temporal_betas = float(fit_temporal), float(fit_temporal_betas)
        self.fit_temporal_root = float(fit_temporal_root)
        self.gpu_paste = gpu_paste and str(device).startswith("cuda")  # crop maps warped on the GPU (False: cv2)
        if crop_maps == "auto":  # the bundle's own default (deploy3: True); a separate parts_head keeps the image maps
            crop_maps = bool(self.bundle_defaults.get("crop_maps", False)) and not parts_head
        self.crop_maps = crop_maps  # True: parts_crop / mask_crop (256 x 256) + crop_affine only, no image maps
        self.model.map_outputs = True  # part / mask crop maps inside the (graphed) forward
        dh = self.model.dense_head  # a "+extent" person-mask head reads the prompt box (crop units)
        self.mask_extent = dh is not None and dh.mask_dconv is not None and (getattr(dh, "mask_extent", False)
                                                                             or getattr(dh, "mask_corners", False))
        self.model.body.head_pose.enable_fast()
        self.body_graph, self.backend, why = None, "torch", ""
        if str(device).startswith("cuda"):  # BF16 weights + CSR pose correctives (same outputs, ~20 % faster)
            from fab.precision import GraphBody, compile_body, fast_inference

            fast_inference(self.model)
            self.device = device
            eng, why = self._engine(body_engine, fab_ckpt, fast)
            if eng is not None:  # the body forward as the verified TensorRT engine of this checkpoint
                self.body_graph, self.backend = GraphEngine(eng, self._batch_engines(body_batch, body_engine, fab_ckpt)), \
                    str(Path(body_engine if body_engine != "auto" else self._engine_path(fab_ckpt)))
                if self.body_graph.batched:
                    why = f"+ batch engines {sorted(self.body_graph.batched)}"
            elif fast:  # the deployed path: compiled backbone / decoder, one CUDA graph per person count
                compile_body(self.model)
                self.body_graph = (GraphBodyBox if self.mask_extent else GraphBody)(self.model)
        print(f"FABPredictor: body backend {self.backend}" + (f" ({why})" if why else ""), flush=True)
        self.transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                                  VisionTransformWrapper(ToTensor())])
        # gpu_crop: the same crop geometry (the transforms without the image), the bilinear warp + [0, 1] scaling on
        # the GPU instead of cv2.warpAffine / ToTensor on the CPU (about 4 ms -> < 1 ms per frame)
        self.geo = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False)])
        self.gpu_crop = gpu_crop and str(device).startswith("cuda")
        self.device, self.det_conf = device, det_conf
        # det_gpu (default on CUDA): the person detector without ultralytics' host pre-processing (detect(): the same boxes
        # as the host path, bit for bit on 3DPW val + COCO val; detection about 3x faster), the
        # image uploaded once for the detector and the GPU crop
        self.det_gpu = bool(det_gpu) and str(device).startswith("cuda")
        self.track_every, self.track_scale = track_every, track_scale
        self.detector = None
        if detector == "deployed":  # the released detector; on the fast path its TensorRT engine when one was built here
            detector = DEPLOYED_DETECTOR if Path(DEPLOYED_DETECTOR).exists() else ""
            if not detector:
                warnings.warn(f"no person detector at {DEPLOYED_DETECTOR}: pass boxes to predict() or download the weights")
            elif fast and str(device).startswith("cuda"):
                engine = paths.engines_dir(device) / "detector.engine"
                if engine.exists() and importlib.util.find_spec("ultralytics") is not None:  # the engine extra
                    why = detector_engine_unusable(engine)
                    if why:
                        warnings.warn(f"FABPredictor: detector engine {engine} not used ({why}); the PyTorch detector "
                                      f"(fab.yolo, the same boxes)")
                    else:
                        detector = str(engine)
        # detector_backend: "fab" (default): the detector network, its letterbox and its NMS in fab.yolo, no ultralytics
        # at inference (the ultralytics path's boxes, bit for bit: scripts/check_detector.py); "ultralytics": the
        # ultralytics package's predictor (the reference). A TensorRT detector engine always runs through ultralytics.
        if detector_backend not in ("fab", "ultralytics"):
            raise ValueError(f"detector_backend {detector_backend!r}: 'fab' or 'ultralytics'")
        self.detector_backend = detector_backend
        if detector:
            self._init_detector(detector)
        self.to_smpl, self.j14, self.interp = None, None, None
        bm = Path(bm_dir) if bm_dir else paths.body_models_dir()
        if bm.exists():
            from bench.body_models import SMPL
            from bench.mhr_smpl import MHRToSMPL

            self.to_smpl = MHRToSMPL(bm / "mhr").to(device)
            self.j14 = np.load(bm / "smpl" / "J_regressor_h36m.npy")[H36M_TO_J14].astype(np.float64)
            if self.model.dense_head is not None:  # template-space interpolation of surface offsets to all vertices
                sidx = self.model.dense_head.surface_idx.cpu().numpy()
                tmpl = SMPL(bm / "smpl" / "SMPL_NEUTRAL.pkl").v_template.numpy().astype(np.float64)
                d = np.linalg.norm(tmpl[:, None] - tmpl[sidx][None], axis=-1)
                nn_ = np.argsort(d, 1)[:, :8]
                wgt = np.exp(-(np.take_along_axis(d, nn_, 1) / 0.04) ** 2) + 1e-9
                self.interp = (sidx, nn_, wgt / wgt.sum(1, keepdims=True))
                self.interp_W = None  # the same interpolation as one sparse (6890, S) matrix: ~10x faster per person
                try:
                    from scipy.sparse import csr_matrix

                    self.interp_W = csr_matrix((self.interp[2].ravel(), (np.repeat(np.arange(len(nn_)), nn_.shape[1]),
                                                                        nn_.ravel())), shape=(len(nn_), len(sidx)))
                except ImportError:
                    pass
        self.points_map = None
        if self.model.dense_head is not None and (bm / "smpl" / "dense512_to_mesh.npz").exists():
            z = np.load(bm / "smpl" / "dense512_to_mesh.npz")
            if np.array_equal(z["idx"], self.model.dense_head.surface_idx.cpu().numpy()):
                self.points_map = z["A"].astype(np.float64)
        self.fitter = self.body = self.mesh_fit_idx = self.dense_fit = None
        self.fit_pos = None  # positions of the fitter's input points in the dense head's surface list (None: all)
        absolute = self.model.dense_head is not None and not int(self.model.dense_head.residual)
        if self.profile == "mesh":  # FAB-H's mesh fit: every "auto" choice as for a checkpoint without the dense head
            absolute = False
        if smpl_fitter == "deployed" and self.bundle_defaults.get("smpl_fitter"):
            smpl_fitter = self.bundle_defaults["smpl_fitter"]
        if smpl_fitter == "deployed":
            smpl_fitter = DEPLOYED_FITTERS["absolute" if absolute else "residual"]
        self.smpl_fitter_path = smpl_fitter  # the resolved fitter ("" = no SMPL fit)
        if smpl_fitter and self.model.dense_head is not None and self.profile != "mesh":  # SMPL fitted to the dense surface (one pass)
            from bench.body_models import SMPL
            from fab.fast_fit import GraphFit
            from fab.smpl_fitter import load_fitter

            self.body = SMPL(bm / "smpl" / "SMPL_NEUTRAL.pkl").to(device).eval()
            fnet = load_fitter(smpl_fitter, device)
            sidx_all = self.model.dense_head.surface_idx.cpu().numpy()
            fidx = sidx_all if getattr(fnet, "idx", None) is None else np.asarray(torch.as_tensor(fnet.idx).cpu())
            if not np.array_equal(fidx, sidx_all):  # a head with more points (--dense_extra): fit the fitter's subset
                pos_of = {int(v): i for i, v in enumerate(sidx_all)}
                self.fit_pos = np.array([pos_of[int(v)] for v in fidx])
            fit_sidx = self.model.dense_head.surface_idx if self.fit_pos is None else torch.as_tensor(fidx)
            self.fitter = GraphFit(fnet, self.body, fit_sidx,
                                   graph=str(device).startswith("cuda"))  # one CUDA graph per person count
            if dense_fit:  # refined fit (one-pass + Gauss-Newton), one CUDA graph per person count
                from fab.gn_fit import DenseFit, GraphDenseFit

                reg23 = paths.require(reg23 or paths.home() / "extras" / "mhr23_regressor.npz", "the 23-joint regressor (dense_fit)",
                                      "It is not part of the download; docs/usage.md explains how to build it.")
                self.reg23 = torch.from_numpy(np.load(reg23)["reg"]).float().to(device)
                self.dense_fit = GraphDenseFit(DenseFit(self.fitter.fit, self.body, self.model.dense_head.surface_idx,
                                                        self.reg23, fit_pos=self.fit_pos), graph=str(device).startswith("cuda"))
        elif smpl_fitter and self.to_smpl is not None:  # FAB-H's mesh fit (no dense head, or profile "mesh"): the mesh's own surface points
            from bench.body_models import SMPL
            from fab.dense_head import symmetric_subset
            from fab.fast_fit import GraphFit
            from fab.smpl_fitter import load_fitter

            self.body = SMPL(bm / "smpl" / "SMPL_NEUTRAL.pkl").to(device).eval()
            self.mesh_fit_idx = symmetric_subset(self.body.v_template.cpu().numpy(), 512)[0]
            self.fitter = GraphFit(load_fitter(smpl_fitter, device), self.body, self.mesh_fit_idx,
                                   graph=str(device).startswith("cuda"))
        self.foot = None
        if foot_refine > 0 and self.fitter is not None:  # feet re-solved on the MHR mesh's feet (foot_refine.py, graphed)
            from fab.foot_refine import GraphFootRefine

            self.foot = GraphFootRefine(self.body, passes=foot_refine, mode=foot_refine_mode,
                                        gate=foot_refine_gate,  # gate 10: feet within 10 deg of the mesh's kept (no sole loss)
                                        sole_w=foot_sole_w)  # > 0: plantar vertices x (1 + w) in the Kabsch (the sole plane)
        # absolute dense heads: the feet re-solved after the Gauss-Newton steps and the folded foot_2d target; residual
        # heads: directly after the one-pass fit
        self.foot_after = absolute
        # foot_target: what the feet are re-solved onto -- "mesh" (the MHR mesh's feet), "dense" (the dense-refined mesh:
        # the points' own feet) or "mix" (their mean); "auto": mix for an absolute head trained with the foot terms
        # (MOYO toe 67.8 -> 56.2 mm, heel->toe 20.4 -> 15.7 deg vs mesh), else mesh
        feet_trained = any(str(s).startswith(("foot_kp=", "foot_dir=")) for s in (self.body_args.get("w") or []))
        self.foot_target = (("mix" if absolute and feet_trained else "mesh") if foot_target == "auto" else foot_target)
        if foot_2d == "auto":  # foot_2d folded into the refine for absolute heads trained with the foot terms
            foot_2d = 1.0 if absolute and feet_trained else 0.0
            if foot_2d_mode == "auto" and foot_2d > 0:
                foot_2d_mode, foot_2d_iters = "fold", 1
        if foot_2d_mode == "auto":
            foot_2d_mode = "fold" if absolute else "step"
        # foot_2d (0 = off): after the last foot refine, each in-crop foot turned (this fraction of the angle) until its
        # projected heel -> toe lies on the dense head's 2D one (fab/foot2d.py; feet more than foot_2d_max deg
        # off kept; x P(visible) of a visibility head when foot_2d_vis)
        # foot_2d_mode "step": a step after the last foot refine (+ a second refine solve); "fold": the foot refine's
        # target feet turned onto the dense 2D first, then the one refine solve (no extra solve; absolute heads)
        self.foot_2d, self.foot_2d_max, self.foot_2d_vis = foot_2d, foot_2d_max, foot_2d_vis
        self.foot_2d_mode = foot_2d_mode
        foot_2d_iters = int(foot_2d_iters)
        self.f2d_sets = self.f2d_graph = None
        if foot_2d > 0 and self.foot is not None:  # foot_2d_iters / foot_2d_min: the step's fixed-point steps / only
            # feet more than this many deg off the dense 2D (cheaper / conservative variants); one CUDA graph
            from fab.foot2d import GraphFoot2D, foot_vertex_sets

            self.f2d_sets = foot_vertex_sets(self.body)
            self.f2d_graph = GraphFoot2D(self.f2d_sets, foot_2d_max, foot_2d_iters, foot_2d_min)
        self.kin = None
        if self.model.g1_head is not None:
            from fab.retarget.g1 import G1Kinematics
            from fab.retarget.g1_files import urdf

            self.kin = G1Kinematics(paths.require(urdf(), "the Unitree G1 URDF (the G1 head's joint limits)",
                                                  "Run scripts/download_g1.sh.")).to(device)
        self._tracks = None

    def _prepare_gpu(self, image, boxes, K, img_t=None):
        """prepare_batch (no masks) with the crop on the GPU: the crop geometry from the same transforms (no image),
        cv2.warpAffine's bilinear sampling (dst px -> inverse affine -> src px, zero outside) by grid_sample, rounded
        to uint8 levels as cv2 does, / 255 as ToTensor -> the batch dict predict() reads (img on the device)."""
        dev = self.device
        H, W = image.shape[:2]
        infos = [self.geo(dict(bbox=np.asarray(bx, np.float32), bbox_format="xyxy")) for bx in boxes]
        A = np.stack([np.asarray(r["affine_trans"], np.float64) for r in infos])  # (n, 2, 3) image -> crop
        Ai = np.linalg.inv(np.concatenate([A, np.tile([[[0.0, 0.0, 1.0]]], (len(A), 1, 1))], 1))[:, :2]
        S = 512
        if getattr(self, "_crop_pix", None) is None:
            yy, xx = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32),
                                    torch.arange(S, device=dev, dtype=torch.float32), indexing="ij")
            self._crop_pix = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(-1, 3)  # (S*S, 3)
        n = len(A)
        f = lambda k: torch.from_numpy(np.stack([np.asarray(r[k], np.float32) for r in infos]))[None]  # noqa: E731
        cpu = {"bbox_center": f("bbox_center"), "bbox_scale": f("bbox_scale"),
               "ori_img_size": torch.tensor([[[W, H]] * n], dtype=torch.float32), "affine_trans": f("affine_trans"),
               "cam_int": torch.from_numpy(np.asarray(K, np.float32))[None]}
        # one asynchronous host -> device copy of all of it (+ the inverse affines), not six blocking ones
        flat = torch.cat([torch.from_numpy(Ai.astype(np.float32)).reshape(-1)] + [cpu[k].reshape(-1) for k in
                         ("bbox_center", "bbox_scale", "ori_img_size", "cam_int", "affine_trans")])
        g = self._h2d("geo", flat)
        cut = np.cumsum([0, n * 6, n * 2, n * 2, n * 2, 9, n * 6])
        Ai_t = g[cut[0]:cut[1]].view(n, 2, 3)
        dev_geo = (g[cut[1]:cut[2]].view(n, 2), g[cut[2]:cut[3]].view(n, 2), g[cut[3]:cut[4]].view(n, 2),
                   g[cut[4]:cut[5]].view(1, 3, 3), g[cut[5]:cut[6]].view(n, 2, 3))
        src = torch.einsum("nij,pj->npi", Ai_t, self._crop_pix)  # (n, S*S, 2) source px
        grid = torch.stack([2 * src[..., 0] / (W - 1) - 1, 2 * src[..., 1] / (H - 1) - 1], -1).reshape(len(A), S, S, 2)
        img = img_t if img_t is not None else torch.from_numpy(np.ascontiguousarray(image)).to(dev, non_blocking=True)
        img = img.permute(2, 0, 1)[None].float().expand(len(A), -1, -1, -1)
        crop = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        crop = crop.round_().clamp_(0, 255) / 255.0
        return {"img": crop[None], **cpu, "dev_geo": dev_geo}  # dev_geo: the same values on the device

    def _h2d(self, key, arr):
        """numpy array / CPU tensor -> device tensor through a cached page-locked buffer per key: an asynchronous copy
        (the buffer is refilled only after its previous copy finished); the same values."""
        t = torch.from_numpy(np.ascontiguousarray(arr)) if isinstance(arr, np.ndarray) else arr.contiguous()
        if not str(self.device).startswith("cuda"):
            return t.to(self.device)
        pins = self.__dict__.setdefault("_pins", {})
        buf, ev = pins.get(key, (None, None))
        if ev is not None:
            ev.synchronize()
        if buf is None or buf.numel() < t.numel() or buf.dtype != t.dtype:
            buf = torch.empty(max(t.numel(), 1), dtype=t.dtype, pin_memory=True)
        hb = buf[: t.numel()].view(t.shape)
        hb.copy_(t)
        out = hb.to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        pins[key] = (buf, ev)
        return out

    def _fold_target(self, ftm_t, o, x, i):
        """foot_2d folded into the foot refine: the refine's target mesh (1, 6890, 3; mesh frame) with each in-crop foot
        turned about its ankle until its projected heel -> toe lies on the dense head's 2D (foot2d_target through the
        MHR placement cam_t; x P(visible) of a visibility head when foot_2d_vis) -> the turned target (mesh frame).
        Everything from the forward's GPU outputs (o, the inputs x), one CUDA graph (GraphFoldTarget)."""
        if getattr(self, "_fold_graph", None) is None:
            from fab.foot2d import GraphFoldTarget

            self._fold_graph = GraphFoldTarget(self.body.J_regressor.float(), self.f2d_sets, self.foot_2d_max,
                                               self.f2d_graph.args[2], self.model.size, self.foot_2d)
        pv = o["vis_logits"][i : i + 1].float() if self.foot_2d_vis and "vis_logits" in o else None
        return self._fold_graph(ftm_t, o["pred_cam_t"][i : i + 1].float(), o["dense_uv"][i : i + 1, :70].float(),
                                x[4][i : i + 1].float(), x[5][i : i + 1].float(), pv)

    def _foot2d(self, R, betas, v_cam, uv70, Kc, o, i):
        """foot_2d's step on one person's fit (v_cam camera frame; uv70 crop px; Kc the crop camera) -> R, v_cam."""
        from fab.foot2d import foot2d_step

        pv = None
        if self.foot_2d_vis and "vis_logits" in o:
            pv = torch.softmax(o["vis_logits"][i : i + 1].float(), -1)[:, :70, 0]
        uv = torch.from_numpy(np.asarray(uv70, np.float32)[None]).to(self.device)
        R, v = foot2d_step(self.foot, self.body, R.float(), betas.float(), v_cam.float(), uv, Kc, self.model.size,
                           self.foot_2d, pv, self.foot_2d_max, self.f2d_sets, target_fn=self.f2d_graph)
        return R.clone(), v.clone()

    @torch.inference_mode()  # as predict: the GPU path's cached buffers (the uploads, the detector graph) are inference
    def detect(self, image):  # tensors once predict made them; outside inference mode their in-place refill fails
        return self._detect(image)[0]

    def _init_detector(self, detector):
        """self.detector for the path / engine `detector`: fab.yolo's YOLODetector (detector_backend "fab", a .pt
        checkpoint) or ultralytics' YOLO (detector_backend "ultralytics", or a TensorRT engine)."""
        if self.detector_backend == "fab" and not str(detector).endswith(".engine"):
            from fab.yolo import YOLODetector

            self.detector = YOLODetector(detector, self.device)
            return
        self.detector_backend = "ultralytics"
        from ultralytics import YOLO

        self.detector = YOLO(detector)

    def _detect(self, image):
        """-> (xyxy boxes, the image uploaded to the GPU or None): det_gpu once the predictor exists, else the host path."""
        if getattr(self, "detector_backend", "ultralytics") == "fab":
            return self._detect_fab(image)
        if getattr(self, "det_gpu", False) and self.detector.predictor is not None:
            try:
                return self._detect_gpu(image)
            except Exception as err:  # an ultralytics version without these internals: the host path from now on
                loud_warning(f"FABPredictor: GPU detector pre / post OFF for this predictor from now on (the host path: the same "
                             f"boxes, slower): {type(err).__name__}: {err}")
                self.det_gpu = False
        out = self.detector.predict(np.ascontiguousarray(image[..., ::-1]), classes=[0], conf=self.det_conf, imgsz=640,
                                    verbose=False)[0]
        return out.boxes.xyxy.float().cpu().numpy(), None

    @torch.no_grad()
    def _detect_fab(self, image):
        """detect() with fab.yolo (detector_backend "fab"): _detect_gpu's letterbox (cv2.INTER_LINEAR resize of the RGB
        image on the host, the 114 padding and / 255 on the device), the network, the one-class NMS and scale_boxes
        -> the ultralytics path's boxes, and (CUDA) the full image uploaded once for the GPU crop."""
        img = self._h2d("image", image) if str(self.device).startswith("cuda") else None  # asynchronous, as _detect_gpu
        return self.detector.detect(image, conf=self.det_conf, h2d=self._h2d), img

    @torch.no_grad()
    def _detect_gpu(self, image):
        """detect() without ultralytics' host pre-processing (det_gpu): its LetterBox (its own output size; cv2.INTER_LINEAR
        resize of the RGB image on the host, the 114 padding and / 255 on the GPU), then the predictor's own inference and
        postprocess (its NMS and scale_boxes; the original image only for its shape) -> the host path's boxes, and the full
        image uploaded once for the GPU crop."""
        pr = self.detector.predictor
        H, W = image.shape[:2]
        img = self._h2d("image", image)  # asynchronous (page-locked); the resize below runs meanwhile
        lb = getattr(self, "_lb_shape", {})
        if (H, W) not in lb:  # the letterbox's output size as the predictor builds it (rect / stride padding for dynamic
            # engines), once per image size from its own pre_transform on an empty image of that size
            lb[(H, W)] = pr.pre_transform([np.zeros((H, W, 3), np.uint8)])[0].shape[:2]
            self._lb_shape = lb
        h2, w2 = lb[(H, W)]
        r = min(640 / H, 640 / W)
        nw, nh = int(round(W * r)), int(round(H * r))
        # the resize stays cv2.INTER_LINEAR on the host (its fixed-point result exactly; ~1 ms) but on the RGB image: the
        # host path's full-resolution RGB -> BGR copy (~9 ms at 1080p) is what this skips; per-channel resizing commutes
        # with the channel order, so the network input is the host path's, bit for bit
        import cv2

        small = image if (nh, nw) == (H, W) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = self._h2d("small", small).permute(2, 0, 1)[None].float()
        dw, dh = (w2 - nw) / 2, (h2 - nh) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        im = torch.full((1, 3, h2, w2), 114.0, device=self.device)
        im[:, :, top : top + nh, left : left + nw] = x
        im = im / 255.0
        im = im.half() if getattr(pr.model, "fp16", False) else im
        preds = self._det_infer(im)
        if getattr(self, "det_nms1", True):  # a one-class detector: the same boxes with two host syncs instead of five
            boxes = self._nms_one_class(preds, im, H, W)
            if boxes is not None:
                return boxes, img
            if not getattr(self, "_nms1_warned", False):  # ultralytics' own postprocess for this output (the same boxes)
                self._nms1_warned = True
                loud_warning(f"FABPredictor: one-class NMS not applicable to this detector output ({type(preds).__name__}): "
                             "ultralytics' postprocess from here on when it recurs")
        res = pr.postprocess(preds, im, [np.empty((H, W, 3), np.uint8)])
        return res[0].boxes.xyxy.float().cpu().numpy(), img

    def _det_infer(self, im):
        """pr.inference(im), for ultralytics' TensorRT backend with a dynamic engine (which it runs with the host-synchronous
        execute_v2) as one CUDA graph per input shape: set_input_shape, the graph's own input / output buffers, execute_async_v3
        captured. Anything else, or det_graph off:
        pr.inference."""
        pr = self.detector.predictor
        m = pr.model
        be = getattr(m, "backend", None)
        if not (getattr(self, "det_graph", True) and be is not None and type(be).__name__ == "TensorRTBackend"
                and getattr(be, "dynamic", False) and getattr(be, "is_trt10", False) and getattr(be, "graph", None) is None
                and not getattr(m, "nhwc", False) and im.shape[0] == 1 and len(getattr(be, "output_names", [])) == 1):
            return pr.inference(im)
        if be.fp16 and im.dtype != torch.float16:  # as AutoBackend.forward
            im = im.half()
        key = tuple(im.shape)
        graphs = self.__dict__.setdefault("_det_graphs", {})
        if key not in graphs:
            try:
                ctx, name = be.context, be.output_names[0]
                if not ctx.set_input_shape(be.input, key):
                    raise ValueError(f"input size {key} is outside the TensorRT optimization profile")
                x = torch.empty(key, dtype=im.dtype, device=im.device)
                y = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=be.bindings[name].data.dtype, device=im.device)
                ctx.set_tensor_address(be.input, x.data_ptr())
                ctx.set_tensor_address(name, y.data_ptr())
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    x.copy_(im)
                    for _ in range(2):  # TensorRT allocates on its first run
                        if not ctx.execute_async_v3(st.cuda_stream):
                            raise RuntimeError("TensorRT inference execution failed")
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                        raise RuntimeError("TensorRT inference execution failed")
                graphs[key] = (g, x, y)
                # ultralytics' own call (the host path) re-sets the input shape next time: the context's is ours now
                be.bindings[be.input] = be.bindings[be.input]._replace(shape=())
            except Exception as e:  # noqa: BLE001
                loud_warning(f"FABPredictor: detector CUDA graph OFF for this predictor from now on ({type(e).__name__}: "
                             f"{str(e)[:200]})")
                self.det_graph = False
                be.bindings[be.input] = be.bindings[be.input]._replace(shape=())  # its next call re-sets the shape
                return pr.inference(im)
        g, x, y = graphs[key]
        x.copy_(im)
        g.replay()
        return y

    def _nms_one_class(self, preds, im, H, W):
        """ultralytics' DetectionPredictor.postprocess (nms.non_max_suppression, then scale_boxes and Results.boxes.xyxy)
        for a one-class detector with its own ops in the same order (the confidence candidates, xywh2xyxy, nms_boxes,
        max_det, scale_boxes), without its per-image loop, the best-class / class re-filters (they keep every candidate
        of a one-class model: the class offset is 0) and the Results objects -> the same xyxy boxes (float32 numpy;
        bit-identical on the check frames), or None where its other branches apply (end-to-end / several classes /
        another class filter / > max_nms candidates)."""
        from ultralytics.utils import ops
        from ultralytics.utils.nms import nms_boxes

        pr = self.detector.predictor
        a = pr.args
        p = preds[0] if isinstance(preds, (list, tuple)) else preds
        classes = None if a.classes is None else list(a.classes)
        if (not torch.is_tensor(p) or p.ndim != 3 or p.shape[0] != 1 or p.shape[1] != 5 or a.task != "detect"
                or getattr(pr.model, "end2end", False) or classes not in (None, [0]) or p.shape[-1] > 30000):
            return None
        x = p[0].transpose(0, 1)[p[0, 4] > a.conf]  # (n, 5) candidates (one host sync)
        if not len(x):
            return np.zeros((0, 4), np.float32)
        x[:, :4] = ops.xywh2xyxy(x[:, :4])
        x = x.to(torch.promote_types(x.dtype, torch.float32))  # its cat with the float class column promotes (fp16 engine)
        i = nms_boxes(x[:, :4], x[:, 4], a.iou)[: a.max_det]
        b = ops.scale_boxes(im.shape[2:], x[i, :4], (H, W, 3))
        return b.float().cpu().numpy()

    def _engine_path(self, fab_ckpt):
        """Where fab-build-trt puts a bundle's TensorRT body engine on this GPU:
        <engines dir>/<model folder name>/body-m<2.5D mix>.engine (batch engines: body-m<mix>-b<N>.engine)."""
        dh = self.model.dense_head
        mix = float(getattr(dh, "p25d_mix", 0.0)) if dh is not None else 0.0
        return paths.engines_dir(self.device) / Path(fab_ckpt).resolve().parent.name / f"body-m{mix:g}.engine"

    def _batch_engines(self, body_batch, body_engine, fab_ckpt):
        """body_batch: "auto" (the default: with the auto engine, <its path stem>-b<N>.engine, N = 2..8, those
        that exist and verify for this checkpoint; none built next to an engine = the per-person engine as before), "" (none)
        or a list of engine paths (must load) -> {N: TensorRTBody of a fixed N-person engine}."""
        from fab.tensorrt import TensorRTBody

        if not body_batch or (body_batch == "auto" and body_engine != "auto"):
            return {}
        if body_batch == "auto":
            base = self._engine_path(fab_ckpt)
            paths, strict = [base.with_name(f"{base.stem}-b{n}.engine") for n in range(2, 9)], False
        else:
            paths, strict = [Path(x) for x in body_batch], True
        out = {}
        for path in paths:
            if not path.exists() and not strict:
                continue
            try:
                eng = TensorRTBody(path, fab_ckpt)
                out[int(eng.shapes["img"][0])] = eng
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                print(f"FABPredictor: batch engine {path} not used: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return out

    def _engine(self, body_engine, fab_ckpt, fast):
        """body_engine: a path (must load), "" (PyTorch) or "auto" (the default: the bundle's engine for this GPU and
        2.5D mix when fast=True, tensorrt imports and the engine verifies for this checkpoint; PyTorch otherwise).
        -> (TensorRTBody or None, the reason when PyTorch is used)."""
        if not body_engine:
            return None, ""
        auto = body_engine == "auto"
        path = self._engine_path(fab_ckpt) if auto else Path(body_engine)
        if auto and not fast:
            return None, "auto engine only on the fast path"
        if auto and not path.exists():
            return None, f"no engine {path}"
        if auto and self.mask_extent:  # the fixed engine interface has no prompt-box input
            return None, "the mask head reads the prompt box (+extent): PyTorch body"
        try:
            import json

            from fab.tensorrt import TensorRTBody

            meta = json.loads(Path(str(path) + ".json").read_text())
            dh = self.model.dense_head
            mix = float(getattr(dh, "p25d_mix", 0.0)) if dh is not None else 0.0
            if meta.get("dense_25d_mix") is not None and abs(float(meta["dense_25d_mix"]) - mix) > 1e-6:
                raise ValueError(f"engine 2.5D mix {meta['dense_25d_mix']} != {mix}")
            return TensorRTBody(path, fab_ckpt), ""
        except Exception as e:  # noqa: BLE001
            if not auto:
                raise
            return None, f"engine {path} not used: {type(e).__name__}: {str(e)[:120]}"

    def _orient_to_mesh(self, R, v, mesh):
        """fit_orient_mesh: the rotation dR (Kabsch) of the fit's pelvis-relative J24 onto the MHR mesh's (SMPL topology,
        same camera axes), taken fit_orient_mesh of the way (axis-angle x it), applied about the fit's pelvis to its
        vertices and global rotation -> R, v. Graph-capturable."""
        from bench.body_models import _rodrigues
        from fab.fast_fit import kabsch_S
        from fab.smpl_fit import rotmat_to_aa

        J = self.body.J_regressor.float()
        jf, jm = torch.einsum("jv,bvc->bjc", J, v.float()), torch.einsum("jv,bvc->bjc", J, mesh.float())
        S = torch.einsum("bni,bnj->bij", jf - jf[:, :1], jm - jm[:, :1])
        dR = kabsch_S(S, 8)  # near-aligned (~1 deg): 8 squarings (= 24 to 0 deg on 0-10 deg problems)
        if self.fit_orient_mesh != 1.0:
            dR = _rodrigues(rotmat_to_aa(dR) * self.fit_orient_mesh)
        dR = dR.to(v.dtype)
        p = jf[:, :1].to(v.dtype)
        v = p + torch.einsum("bij,bnj->bni", dR, v - p)
        R = torch.cat([(dR @ R[:, 0].to(dR.dtype))[:, None].to(R.dtype), R[:, 1:]], 1)
        return R, v

    def _ray_foot_target(self, ftm, v0, t):
        """fit_gn_ray < 1: per foot (the foot refine's vertex sets, ankle-relative as its Kabsch) the target = the fit +
        the mismatch with its component along the ray to the fit's ankle (camera frame: + t) x sqrt(fit_gn_ray)."""
        J = self.body.J_regressor.float()
        jt, j0 = torch.einsum("jv,bvc->bjc", J, ftm), torch.einsum("jv,bvc->bjc", J, v0)
        c_ = 1.0 - self.fit_gn_ray ** 0.5
        out = ftm.clone()
        for a, ix in self.foot.refine.feet:
            d = j0[:, a] + t
            d = (d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6))[:, None]
            x = v0[:, ix] - j0[:, a : a + 1]
            e = ftm[:, ix] - jt[:, a : a + 1] - x
            out[:, ix] = jt[:, a : a + 1] + x + e - c_ * (e * d).sum(-1, keepdim=True) * d
        return out

    def _gn_refine(self, R, betas, v, surf, j14, temporal=None, ray=None):
        """fit_gn_iters Gauss-Newton iterations (fab/gn_fit.FastGN) from a fit (R, betas, v) to the dense surface
        points surf (B, S, 3) + joints j14 (B, 14, 3) in the same frame: Geman-McClure IRLS weights from the previous
        iterate (fit_gn_robust mm), a prior toward the input fit's joint rotations / betas (fit_gn_anchor) and, in video
        mode, toward the previous frame's fit of the same person (temporal = (Rp, bp, w, w_b), weights per person,
        fit_temporal_root x w on the global rotation) -> R, betas, v. Graph-capturable (no host reads; GraphFitChain)."""
        if getattr(self, "_fgn", None) is None:
            from fab.gn_fit import FastGN

            self._fgn = FastGN(self.body, self.model.dense_head.surface_idx, self.j14, iters=1, lam=1e-3,
                               w_root=self.fit_gn_root)
            self._fgn_sidx = self.model.dense_head.surface_idx.to(surf.device)
            self._fgn_j14 = torch.as_tensor(self.j14, dtype=torch.float32, device=surf.device)
            part = self.body.weights.argmax(1)[self._fgn_sidx.to(self.body.weights.device)].to(surf.device)
            core = torch.cat([torch.isin(part, torch.tensor([0, 3, 6, 9], device=surf.device)),
                              torch.isin(torch.arange(14, device=surf.device), torch.tensor([2, 3, 8, 9, 12], device=surf.device))])
            self._fgn_cw = 1.0 + (self.fit_gn_core - 1.0) * core.float()  # (S + 14,) point weights
        gn = self._fgn
        X = torch.cat([surf.float(), j14.float()], 1)
        R0, b0 = R.float().clone(), betas.float().clone()
        w = self._fgn_cw.expand(X.shape[:2]).clone()
        s2 = self.fit_gn_robust ** 2
        R, betas = R.float(), betas.float()
        for it in range(self.fit_gn_iters):
            tk = {} if temporal is None else dict(R1=temporal[0], b1=temporal[1], w1=temporal[2], w1_b=temporal[3],
                                                   w1_root=self.fit_temporal_root)
            if ray is not None:  # anisotropic (fit_gn_ray): ray = the points' unit camera rays, X's order
                tk.update(ray=ray, w_ray=self.fit_gn_ray)
            R, betas, t3, _ = gn(R, betas, X, w, R0=R0, b0=b0, w_anchor=self.fit_gn_anchor, w_anchor_b=self.fit_gn_anchor,
                                 **tk)
            if self.fit_gn_robust > 0 and it + 1 < self.fit_gn_iters:  # the next iterate's IRLS weights (the support
                # vertices only: the points' vertices + the J14 regressor's)
                vs, _ = gn.smpl(R, betas, full=False)
                P = torch.cat([vs[:, gn.vpos], torch.einsum("ns,bsc->bnc", gn.reg, vs)], 1)
                e_ = P + t3[:, None] - X
                if ray is not None and self.fit_gn_ray != 1.0:  # the same anisotropic residual
                    e_ = e_ - (1.0 - self.fit_gn_ray ** 0.5) * (e_ * ray).sum(-1, keepdim=True) * ray
                w = (s2 / (s2 + (e_ * 1000).pow(2).sum(-1))) ** 2 * self._fgn_cw
        vv, _ = gn.smpl(R, betas, full=True)
        return R, betas, vv + t3[:, None]

    def _chain_ok(self):
        """GraphFitChain applies: the absolute-head path (one-pass fitter + Gauss-Newton + foot refine onto mesh / dense /
        mix, fold or no foot_2d) on the GPU."""
        if getattr(self, "_chain_ok_", None) is None:
            dh = self.model.dense_head
            self._chain_ok_ = bool(self.fit_graph and str(self.device).startswith("cuda") and dh is not None
                                   and not int(dh.residual) and self.fitter is not None and self.dense_fit is None
                                   and self.mesh_fit_idx is None and self.foot is not None
                                   and self.foot_after and self.interp is not None and self.to_smpl is not None
                                   and not self.place_incrop and self.foot_target in ("mesh", "dense", "mix")
                                   and (self.f2d_sets is None or self.foot_2d_mode == "fold"))
        return self._chain_ok_

    def _place_points(self, X, uv_crop, Ai, K):
        """Closed-form translation of root-relative points X (N, 3) onto their crop-px 2D uv_crop (N, 2) (Ai: crop ->
        image); with place_incrop the points within 2 % of the crop edge are left out (>= 10 kept, else all)."""
        uv_img = uv_crop @ Ai[:2, :2].T + Ai[:2, 2]
        if self.place_incrop:
            S = float(self.model.size)
            inc = ((uv_crop > 0.02 * S) & (uv_crop < 0.98 * S)).all(1)
            if inc.sum() >= 10:
                X, uv_img = X[inc], uv_img[inc]
        return closed_form_translation(X, uv_img, np.asarray(K, np.float64))

    def _temporal(self, prev, n):
        """prev (per box: the previous frame's output dict of the same person, or None) -> the Gauss-Newton's temporal
        prior inputs (Rp, bp, w, w_b) on the device, or None when the prior is off."""
        if self.fit_temporal <= 0 and self.fit_temporal_betas <= 0:
            return None
        from fab.gn_fit import so3_exp

        dev = self.device
        aa, bp, w = torch.zeros(n, 72), torch.zeros(n, 10), torch.zeros(n, 2)
        for i, q in enumerate(list(prev or [])[:n]):
            if q is not None and "smpl_pose" in q:
                aa[i] = torch.as_tensor(np.asarray(q["smpl_pose"], np.float32))
                bp[i] = torch.as_tensor(np.asarray(q["smpl_betas"], np.float32)[:10])
                w[i] = torch.tensor([self.fit_temporal, self.fit_temporal_betas])
        aa, bp, w = aa.to(dev), bp.to(dev), w.to(dev)
        return so3_exp(aa.view(n, 24, 3)), bp, w[:, 0].contiguous(), w[:, 1].contiguous()

    @torch.inference_mode()
    def predict(self, image, K=None, boxes=None, neg_boxes=None, prev=None):
        """image (H, W, 3) RGB uint8; K (3, 3) or None; boxes (N, 4) xyxy or None (detector).
        neg_boxes (M, 4) xyxy or None: the negative box prompt of a train_fab --neg_prompt_p model (eval_hmr --mask_prompt
        neg): +1 in each person's own box, -1 in the other boxes (the other inputs + neg_boxes), 0 in both; eager forward."""
        from fab.data.utils.prepare_batch import prepare_batch
        from fab.retarget.g1 import rot6d_to_matrix

        H, W = image.shape[:2]
        img_t = None
        if boxes is None:
            if self.detector is None:
                raise ValueError("no boxes and no detector")
            boxes, img_t = self._detect(image)  # (det_gpu: the image already on the GPU, reused by the GPU crop)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if not len(boxes):
            return []
        if K is None:
            K = default_K(H, W)
        mfull = None
        if neg_boxes is not None:
            nb = np.asarray(neg_boxes, np.float32).reshape(-1, 4)
            rect = lambda bb: tuple(np.clip(np.round(bb), 0, [W, H, W, H]).astype(int))  # noqa: E731
            mfull = np.zeros((len(boxes), H, W, 1), np.float32)
            for i in range(len(boxes)):
                own, oth = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
                x1, y1, x2, y2 = rect(boxes[i])
                own[y1:y2, x1:x2] = 1
                for j, ob in enumerate(list(boxes) + list(nb)):
                    if j != i:
                        x1, y1, x2, y2 = rect(ob)
                        oth[y1:y2, x1:x2] = 1
                mfull[i, :, :, 0] = own - oth
        if mfull is None and self.gpu_crop:
            b = self._prepare_gpu(image, boxes, K, img_t)
        elif mfull is None:
            b = prepare_batch(image, self.transform, boxes, None, None, cam_int=torch.from_numpy(np.asarray(K)).float()[None],
                              include_masks=False)  # FAB-D reads no masks; the empty full-image masks cost ~25 ms
        else:
            b = prepare_batch(image, self.transform, boxes, mfull, np.ones(len(boxes), np.float32),
                              cam_int=torch.from_numpy(np.asarray(K)).float()[None])
        n = len(boxes)
        dev = self.device
        if "dev_geo" in b:  # the GPU crop's geometry, already on the device (one copy)
            c_, s_, o_, k_, a_ = b["dev_geo"]
            x = (b["img"][0], c_, s_, o_, k_.expand(n, -1, -1), a_)
        else:
            x = (b["img"][0].to(dev), b["bbox_center"][0].to(dev), b["bbox_scale"][0].to(dev), b["ori_img_size"][0].to(dev),
                 b["cam_int"].expand(n, -1, -1).to(dev), b["affine_trans"][0].to(dev))
        bx = ()
        if self.mask_extent:  # the input boxes in crop units (the square crop drops their aspect)
            A_ = b["affine_trans"][0].numpy().astype(np.float64)
            c4 = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [0, 3]], boxes[:, [2, 3]]], 1).astype(np.float64)
            cc = (np.einsum("bij,bkj->bki", A_[:, :, :2], c4) + A_[:, None, :, 2]) / self.model.size
            bx = (torch.from_numpy(np.concatenate([cc.min(1), cc.max(1)], 1).astype(np.float32)).to(dev),)
        if self.body_graph is not None and mfull is None:
            o = self.body_graph(*x, *bx)
            if "part_logits" in o or "mask_logits" in o:  # a TensorRT body with the seg heads (logits)
                dh = self.model.dense_head  # FABPredictor(parts=False / mask=False): the engine's logits dropped too
                off = {"part_logits"} if dh.part_emb is None else set()
                off |= {"mask_logits"} if dh.mask_dconv is None else set()
                if off:
                    o = {k: v for k, v in o.items() if k not in off}
                if ("part_logits" in o or "mask_logits" in o) and "part_map" not in o and "mask_map" not in o \
                        and self.model.map_outputs:  # the crop maps on the GPU, as the PyTorch forward's
                    from fab.seg_maps import crop_maps_gpu

                    pm, mm = crop_maps_gpu(o.get("part_logits"), o.get("mask_logits"))
                    o = {**o, **({"part_map": pm} if pm is not None else {}), **({"mask_map": mm} if mm is not None else {})}
        else:
            kw = {"mask_box": bx[0]} if bx else {}
            if mfull is not None:  # neg_boxes: the signed box prompt (eval_hmr's mask path)
                kw.update(mask=b["mask"][0].to(dev), mask_score=b["mask_score"][0].to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.model(*x[:5], affine=x[5], **kw)
        f32 = lambda x: x.float().cpu().numpy()  # noqa: E731
        smpl_t = self.to_smpl(o["pred_vertices"].float()) if self.to_smpl is not None else None
        # every per-frame network output the host reads, in one device -> host copy (one sync instead of ~10)
        dev_out = {"verts": o["pred_vertices"], "kp3d": o["pred_keypoints_3d"], "kp2d": o["pred_keypoints_2d"],
                   "cam_t": o["pred_cam_t"], "smpl": smpl_t, "dense_uv": o.get("dense_uv"), "dense_xyz": o.get("dense_xyz")}
        if "vis_logits" in o:
            dev_out["vis"] = torch.softmax(o["vis_logits"].float(), -1)
            if "vis_logits_rendered" in o:
                dev_out["vis_r"] = torch.softmax(o["vis_logits_rendered"].float(), -1)
        if "contact_logits" in o:
            dev_out["contact"] = torch.sigmoid(o["contact_logits"].float())
        if "g1_q" in o and self.kin is not None:
            dev_out["g1_q"] = o["g1_q"].float().clamp(self.kin.lower, self.kin.upper)
            dev_out["g1_R"] = rot6d_to_matrix(o["g1_rot6d"].float())
        ch = None
        if self._chain_ok() and mfull is None and smpl_t is not None:  # the fit stage in one graph, read with the rest
            if getattr(self, "_chain", None) is None:
                self._chain = GraphFitChain(self)
            ch = self._chain(o["dense_xyz"], smpl_t, o["pred_cam_t"], o["dense_uv"], x[4], x[5], o.get("vis_logits"),
                             self._temporal(prev, n) if self.fit_gn_iters > 0 else None)
            dev_out["chain"] = ch["packed"]
        host = d2h(dev_out)
        verts, kp3d, kp2d, cam_t = host["verts"], host["kp3d"], host["kp2d"], host["cam_t"]
        people = [{"box": boxes[i], "cam_t": cam_t[i], "mesh_mhr": verts[i] + cam_t[i], "keypoints3d": kp3d[i] + cam_t[i],
                   "keypoints2d_projected": kp2d[i, :, :2], "backend": self.backend} for i in range(n)]
        smpl = host.get("smpl")
        if o.get("mask_logits") is not None:  # mask quality for ranking (as SAM's stability score): IoU of the masks
            ml = o["mask_logits"].float()[:, 0]  # thresholded at logit +1 and -1; and the mean pixel confidence
            hi, lo = (ml > 1).flatten(1).sum(1), (ml > -1).flatten(1).sum(1)
            conf = (2 * torch.sigmoid(ml) - 1).abs().flatten(1).mean(1)
            sc = torch.stack([hi / lo.clamp_min(1), conf], 1).cpu().numpy()
            for i, pp in enumerate(people):
                pp["mask_score"], pp["mask_conf"] = float(sc[i, 0]), float(sc[i, 1])
        from fab.seg_maps import stack_maps

        parts_map = mask_map = rois = None
        if ("part_map" in o or "mask_map" in o) and self.crop_maps:  # one D2H copy of the crop maps, no warp
            mp = stack_maps([o[k] for k in ("part_map", "mask_map") if k in o], 1).cpu().numpy()
            A_ = b["affine_trans"][0].numpy().astype(np.float32)
            for i, pp in enumerate(people):  # image px -> crop px (self.model.size); map cell = crop px * 256 / size
                pp["crop_affine"] = A_[i]
                for c, nm in enumerate(nm for nm, k in (("parts", "part_map"), ("mask", "mask_map")) if k in o):
                    pp[f"{nm}_crop"] = mp[i, c]
        elif ("part_map" in o or "mask_map" in o) and self.gpu_paste:
            # crop maps made inside the forward (map_outputs), warped into each crop's image region on the GPU, one D2H
            from fab.seg_maps import paste_roi_gpu

            roi_names = [nm for nm, k in (("parts", "part_map"), ("mask", "mask_map")) if k in o]
            mp = stack_maps([o[k] for k in ("part_map", "mask_map") if k in o], -1)
            A_ = b["affine_trans"][0].numpy().astype(np.float64)
            g = [paste_roi_gpu(mp[i], A_[i], self.model.size, H, W) for i in range(n)]
            flat = torch.cat([r.reshape(-1) for r, _ in g]).cpu().numpy()
            cuts = np.cumsum([0] + [r.numel() for r, _ in g])
            rois = [(flat[cuts[i]:cuts[i + 1]].reshape(tuple(r.shape)), box) for i, (r, box) in enumerate(g)]
        elif "part_map" in o or "mask_map" in o:  # crop maps made inside the forward (map_outputs): one D2H copy
            maps = stack_maps([o[k] for k in ("part_map", "mask_map") if k in o], 1).cpu().numpy()
            parts_map = maps[:, 0] if "part_map" in o else None
            mask_map = maps[:, -1] if "mask_map" in o else None
        elif o.get("part_logits") is not None or o.get("mask_logits") is not None:
            from fab.seg_maps import crop_maps

            parts_map, mask_map = crop_maps(o.get("part_logits"), o.get("mask_logits"))
        if "dense_uv" in o:
            A = b["affine_trans"][0].numpy().astype(np.float64)
            uv = host["dense_uv"] * self.model.size
            xyz = host["dense_xyz"]
            for i, pp in enumerate(people):
                Ai = np.linalg.inv(np.vstack([A[i], [0, 0, 1]]))
                pp["keypoints2d"] = (uv[i, :70] @ Ai[:2, :2].T + Ai[:2, 2]).astype(np.float32)
                pp["dense_surface2d"] = (uv[i, 84:] @ Ai[:2, :2].T + Ai[:2, 2]).astype(np.float32)  # the head's own 2D
                pp["joints2d_h36m14"] = (uv[i, 70:84] @ Ai[:2, :2].T + Ai[:2, 2]).astype(np.float32)  # (its 14 joints)
                j14, surf = xyz[i, 70:84].astype(np.float64), xyz[i, 84:].astype(np.float64)
                if smpl is not None and int(self.model.dense_head.residual):  # offsets on the mesh's own points
                    sidx, nn_, wgt = self.interp
                    # the losses are hip-aligned, so the offsets carry an arbitrary common shift: remove the joints'
                    # hip offset (as bench/eval_hmr.py does), which keeps the result in the mesh's frame
                    shift = j14[[2, 3]].mean(0)
                    j14 = j14 - shift + self.j14 @ smpl[i]
                    off = surf - shift
                    surf = off + smpl[i][sidx]
                    dv = self.interp_W @ off if self.interp_W is not None else (off[nn_] * wgt[..., None]).sum(1)
                    pp["mesh_smpl_dense"] = (smpl[i] + dv + cam_t[i]).astype(np.float32)
                pp["joints_h36m14"] = (j14 - j14[[2, 3]].mean(0)).astype(np.float32)
                pp["dense_surface"] = surf.astype(np.float32)
                if smpl is not None:  # camera frame: anchored at the mesh's hips, as the foot step below and bench/eval_hmr.py do
                    anch = surf if int(self.model.dense_head.residual) else \
                        surf - (j14[[2, 3]].mean(0) - (self.j14 @ smpl[i])[[2, 3]].mean(0))
                    pp["dense_surface_cam"] = (anch + cam_t[i]).astype(np.float32)
                ftm = smpl[i] if smpl is not None else None  # the foot refine's target (foot_target)
                if smpl is not None and self.foot_target in ("mix", "dense") and ch is None:  # the dense points' feet
                    sidx_, nn2, wgt2 = self.interp
                    sm = surf if int(self.model.dense_head.residual) else \
                        surf - (j14[[2, 3]].mean(0) - (self.j14 @ smpl[i])[[2, 3]].mean(0))  # at the mesh's hips
                    dd = sm - smpl[i][sidx_]
                    dm = smpl[i] + (self.interp_W @ dd if self.interp_W is not None else
                                    (dd[nn2] * wgt2[..., None]).sum(1))  # the dense-refined mesh
                    ftm = dm if self.foot_target == "dense" else 0.5 * (smpl[i] + dm)
                if self.points_map is not None:
                    X = np.concatenate([j14, surf])
                    t = self._place_points(X, uv[i, 70:], Ai, K)
                    c = surf.mean(0)
                    pp["mesh_smpl_points"] = (self.points_map @ (surf - c) + c + t).astype(np.float32)
                if self.fitter is not None and self.dense_fit is not None:
                    from fab.gn_fit import crop_camera
                    from fab.smpl_fit import rotmat_to_aa

                    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)[None]).to(dev)  # noqa: E731
                    body23 = list(range(21)) + [41, 62]
                    mo = xyz[i, :70][body23].astype(np.float64)
                    kp = mo - mo[[9, 10]].mean(0) + f32(self.reg23).astype(np.float64) @ smpl[i]
                    sf = surf
                    if not int(self.model.dense_head.residual):  # absolute heads: root-relative points, put at the
                        # mesh's hips (J14 hips; the MHR keypoints at the mesh's regressed MHR hips; densefit_study.Ctx)
                        sf = surf - (j14[[2, 3]].mean(0) - (self.j14 @ smpl[i])[[2, 3]].mean(0))
                        kp = mo - mo[[9, 10]].mean(0) + (f32(self.reg23).astype(np.float64) @ smpl[i])[[9, 10]].mean(0)
                    Kc = crop_camera(t(K), t(A[i]))
                    R, betas, v = self.dense_fit(t(sf), t(smpl[i]), t(kp), t(uv[i, body23]), Kc)
                    if self.foot is not None:
                        R, v = (x.clone() for x in self.foot(R, betas, v, t(ftm)))
                    if self.f2d_sets is not None:  # foot_2d: each visible foot onto the dense 2D heel -> toe
                        R, v = self._foot2d(R, betas, v, uv[i, :70], Kc, o, i)
                    pp["smpl_pose"] = rotmat_to_aa(R[0].float().cpu()).numpy().reshape(72)  # axis-angle, camera frame
                    pp["smpl_betas"] = f32(betas[0])
                    pp["mesh_smpl_fit"] = f32(v[0]).astype(np.float32)  # camera frame, placed by the 2D
                elif ch is not None:  # the fit stage from GraphFitChain (same steps as the branch below)
                    c, nb = host["chain"][i], ch["b0"].shape[1]
                    pp["smpl_pose"] = c[:72].copy()
                    pp["smpl_betas"] = c[72 : 72 + nb].copy()
                    vf = c[72 + nb : 72 + nb + 6890 * 3].reshape(6890, 3)
                    pp["mesh_smpl_fit"] = (vf + c[-3:]).astype(np.float32)
                elif self.fitter is not None and self.mesh_fit_idx is None:  # (profile "mesh": the mesh fit below)
                    from fab.smpl_fit import rotmat_to_aa

                    pts = torch.from_numpy(surf[None]).float().to(dev)
                    R, betas, v = self.fitter(pts if self.fit_pos is None else pts[:, self.fit_pos])  # placed at the points' centroid
                    t_ray = ray = None
                    if self.fit_gn_ray != 1.0 and not int(self.model.dense_head.residual):  # as GraphFitChain
                        t_ray = place_points_t(o["dense_xyz"][i : i + 1, 70:].float(),
                                               o["dense_uv"][i : i + 1, 70:].float() * self.model.size,
                                               x[5][i : i + 1].float(), x[4][i : i + 1].float()).float()
                        Xc = torch.cat([pts, torch.from_numpy(j14[None]).float().to(dev)], 1) + t_ray[:, None]
                        ray = Xc / Xc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    if self.fit_gn_iters > 0 and not int(self.model.dense_head.residual):
                        tp = self._temporal(prev, n)
                        R, betas, v = self._gn_refine(R, betas, v, pts, torch.from_numpy(j14[None]).float().to(dev),
                                                      None if tp is None else tuple(t[i : i + 1] for t in tp), ray=ray)
                    if self.fit_orient_mesh > 0 and smpl is not None:  # as GraphFitChain
                        R, v = self._orient_to_mesh(R, v, torch.from_numpy(smpl[i][None]).float().to(dev))
                    if self.foot is not None and smpl is not None and not self.foot_after:  # residual heads
                        R, v = (x.clone() for x in self.foot(R, betas, v, torch.from_numpy(ftm[None]).float().to(dev)))
                    fold = self.f2d_sets is not None and self.foot_2d_mode == "fold" and smpl is not None \
                        and self.foot_after
                    ftm_t = None if smpl is None else torch.from_numpy(ftm[None]).float().to(dev)
                    if fold:  # foot_2d folded into the refine: its target feet turned onto the dense 2D
                        ftm_t = self._fold_target(ftm_t, o, x, i)
                    if t_ray is not None and ftm_t is not None and self.foot is not None:
                        ftm_t = self._ray_foot_target(ftm_t, v.float(), t_ray)
                    if self.foot is not None and smpl is not None and self.foot_after:  # absolute heads
                        R, v = (x.clone() for x in self.foot(R, betas, v, ftm_t))
                    if self.f2d_sets is not None and smpl is not None and not fold:  # foot_2d step: each visible foot
                        from fab.gn_fit import crop_camera

                        tf = cam_t[i] if int(self.model.dense_head.residual) else closed_form_translation(
                            np.concatenate([j14, surf]), uv[i, 70:] @ Ai[:2, :2].T + Ai[:2, 2], np.asarray(K, np.float64))
                        tv = torch.from_numpy(np.asarray(tf, np.float32)).to(dev)
                        Kc = crop_camera(torch.from_numpy(np.asarray(K, np.float32)[None]).to(dev),
                                         torch.from_numpy(A[i].astype(np.float32)[None]).to(dev))
                        R, v = self._foot2d(R, betas, v + tv, uv[i, :70], Kc, o, i)
                        v = v - tv
                    pp["smpl_pose"] = rotmat_to_aa(R[0].float().cpu()).numpy().reshape(72)  # axis-angle, camera frame
                    pp["smpl_betas"] = f32(betas[0])
                    t_fit = cam_t[i]
                    if not int(self.model.dense_head.residual):  # absolute points: place by their own 2D, not the MHR cam_t
                        t_fit = self._place_points(np.concatenate([j14, surf]), uv[i, 70:], Ai, K)
                    pp["mesh_smpl_fit"] = (f32(v[0]) + t_fit).astype(np.float32)
                if rois is not None or parts_map is not None or mask_map is not None:  # crop maps -> the crop's region
                    from fab.seg_maps import full_map, paste_roi

                    if rois is not None:
                        names, (roi, box) = roi_names, rois[i]
                    else:
                        names = [n for n, mp in (("parts", parts_map), ("mask", mask_map)) if mp is not None]
                        roi, box = paste_roi(np.stack([mp[i] for mp in (parts_map, mask_map) if mp is not None], -1),
                                             A[i], self.model.size, H, W)  # one warp for both maps
                    pp["roi"] = np.asarray(box)
                    for c, nm in enumerate(names):  # mask: the visible person (clothing, hair)
                        pp[f"{nm}_roi"] = roi[..., c]
                        if self.full_maps:
                            pp[nm] = full_map(roi[..., c], box, H, W)
        if smpl is not None:
            for i, pp in enumerate(people):
                pp["mesh_smpl"] = smpl[i] + cam_t[i]
        if self.mesh_fit_idx is not None and smpl is not None:  # FAB-H: one-pass SMPL fit to its own mesh
            from fab.smpl_fit import rotmat_to_aa

            R, betas, v = self.fitter(torch.from_numpy(smpl[:, self.mesh_fit_idx]).float().to(dev))
            if self.foot is not None:
                R, v = (x.clone() for x in self.foot(R, betas, v, torch.from_numpy(smpl).float().to(dev)))
            for i, pp in enumerate(people):
                pp["smpl_pose"] = rotmat_to_aa(R[i].float().cpu()).numpy().reshape(72)
                pp["smpl_betas"] = f32(betas[i])
                pp["mesh_smpl_fit"] = (f32(v[i]) + cam_t[i]).astype(np.float32)
        if "vis_logits" in o:  # (84, 3) probabilities visible / occluded / outside the crop: 70 MHR keypoints + J14
            pv = host["vis"]  # human-annotation semantics when the head has them
            pr = host.get("vis_r")
            for i, pp in enumerate(people):
                pp["keypoint_visibility"] = pv[i]
                if pr is not None:  # the rendered-label (geometric self / person / object occlusion) classifier
                    pp["keypoint_visibility_rendered"] = pr[i]
        if "contact_logits" in o:  # l/r heel, l/r toe, l/r hand, l/r knee (fab/contact_regions.CONTACT_NAMES)
            c = host["contact"]
            for i, pp in enumerate(people):
                pp["contact"] = c[i]
        if "g1_q" in o and self.kin is not None:
            q, R = host["g1_q"], host["g1_R"]
            for i, pp in enumerate(people):
                pp["g1_q"], pp["g1_root"] = q[i], R[i]
        return people

    def predict_video(self, frames, K=None):
        """Generator over frames: detector every track_every frames, otherwise each tracked person's box from its
        previous-frame keypoints (x track_scale); people are matched to tracks by box IoU."""
        tracks, last = [], []
        for t, image in enumerate(frames):
            H, W = image.shape[:2]
            if t % self.track_every == 0 or not tracks:
                boxes = self.detect(image) if self.detector is not None else np.zeros((0, 4), np.float32)
                prev = [self._match(b, last) for b in boxes]  # the previous frame's same person (box IoU), or None
            else:
                boxes = np.stack(tracks)
                prev = last  # tracked boxes keep the previous frame's order
            people = self.predict(image, K, boxes, prev=prev)
            last = people
            tracks = []
            for pp in people:
                kp = pp["keypoints2d_projected"][BODY_KP]
                c, wh = (kp.min(0) + kp.max(0)) / 2, (kp.max(0) - kp.min(0)) * self.track_scale
                tracks.append(np.concatenate([c - wh / 2, c + wh / 2]).clip(0, [W - 1, H - 1, W - 1, H - 1]).astype(np.float32))
            yield people

    @staticmethod
    def _match(box, people, min_iou=0.3):
        """The person of people whose box overlaps box most (IoU >= min_iou), or None."""
        best, bi = None, min_iou
        for q in people:
            b = q["box"]
            w = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
            h = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
            u = (box[2] - box[0]) * (box[3] - box[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
            iou = w * h / u if u > 0 else 0.0
            if iou >= bi:
                best, bi = q, iou
        return best

    SMOOTH = {"g1_q": "vec", "g1_root": "rot", "smpl_pose": "aa", "smpl_betas": "vec", "cam_t": "vec",
              "mesh_smpl_fit": "vec", "mesh_mhr": "vec", "joints_h36m14": "vec", "keypoints3d": "vec", "keypoints2d": "vec"}

    def predict_video_smoothed(self, frames, K=None, lag=3, sigma=2.5, min_iou=0.3):
        """predict_video with a causal fixed-lag smoother per person track (fab/retarget/smooth.LagSmoother;
        lag 3 / sigma 2.5 = 100 ms at 30 fps, lag 2 / sigma 1.5 for lower latency): yields (frame index, people) for
        frame t - lag while frame t is processed, then the last frames at the end of the stream. People are linked
        across frames by box IoU and get a "track" id; the SMOOTH outputs are replaced by their smoothed values, the
        others are the frame's own."""
        from collections import deque

        from fab.retarget.smooth import LagSmoother

        def iou(a, b):
            w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
            return w * h / u if u > 0 else 0.0

        tracks, out, pending, next_id = {}, {}, deque(), 0

        def release(tid, tr, smoothed):
            t0, pp = tr["t"].popleft(), tr["pp"].popleft()
            merged = dict(pp)
            merged.update(smoothed)
            merged["track"] = tid
            out[t0].append(merged)

        for t, people in enumerate(self.predict_video(frames, K)):
            pending.append(t)
            out[t] = []
            seen = set()
            for pp in people:
                best, best_iou = None, min_iou
                for tid, tr in tracks.items():
                    o = iou(pp["box"], tr["box"]) if tid not in seen else 0.0
                    if o > best_iou:
                        best, best_iou = tid, o
                if best is None:
                    best, next_id = next_id, next_id + 1
                    tracks[best] = {"sm": LagSmoother(lag, sigma, self.SMOOTH), "t": deque(), "pp": deque()}
                tr = tracks[best]
                seen.add(best)
                tr["box"] = pp["box"]
                tr["t"].append(t)
                tr["pp"].append(pp)
                s = tr["sm"].push({k: pp[k] for k in self.SMOOTH if k in pp})
                if s is not None:
                    release(best, tr, s)
            for tid in [k for k in tracks if k not in seen]:  # lost track: release its pending frames
                tr = tracks.pop(tid)
                for s in tr["sm"].flush():
                    release(tid, tr, s)
            while pending and pending[0] <= t - lag:
                f = pending.popleft()
                yield f, out.pop(f)
        for tid, tr in tracks.items():
            for s in tr["sm"].flush():
                release(tid, tr, s)
        while pending:
            f = pending.popleft()
            yield f, out.pop(f)
