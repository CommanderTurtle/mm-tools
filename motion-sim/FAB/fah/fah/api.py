"""FAH models: model (fah-s / fah-b / fah / fah-h) x mode (image / video / world) -> the arguments of the inference
pipeline (bench.methods.fah.FAHMethod), with the weights read from a local release directory (default
$FAH_HOME/weights, FAH_HOME = ~/.cache/fah):

    <weights>/{fah-s,fah-b,fah,fah-h}/{model.pt,config.json}
    <weights>/shared/{detector.pt,mesh_fitter.pt,video_refiner.pt,camera_motion.pt,mhr_fitter.pt}
    <weights>/trt/{<model>-pose-fp16.engine,detector-640-fp16.engine}   (optional, fah-build-trt)
    <weights>/coreml/{<model>-pose.mlpackage,detector-640.mlpackage}    (optional, Apple silicon, fah-build-coreml)

plus two files of SAM 3D Body (model_config.yaml from its repository; assets/mhr_model.pt, the MHR model, from Meta's MHR
release; fah.download_sam3d, default $FAH_HOME/sam-3d-body-dinov3) and MANO (MANO_DIR, else $FAH_HOME/mano).
"""
import json
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # the pipeline lives in the repository's bench/ and train/ modules (editable install)
    sys.path.insert(0, REPO)

MODELS = ("fah-s", "fah-b", "fah", "fah-h")
MODES = ("image", "video", "world")
# opt-in video identity rules (bench.methods.fah side_track / edge_score + edge_track): FAH(model, "video", **IDENTITY)
# or --identity; off by default
IDENTITY = dict(side_track=True, edge_score=0.5, edge_track=True)
MESHES = ("none", "mano", "mhr", "both")
# hand-object contact (contact=True): a vertex / finger is in contact at probability >= CONTACT_THRESH (the released heads
# carry a logit offset fitted on the training pool's validation split that makes 0.5 F1-optimal), a hand at contact_score > 0.5;
# the video contact model's contact_video uses the same threshold (its own validation-picked one mapped to 0.5)
CONTACT_THRESH = 0.5
# opt-in contact flag smoothing (contact_smooth=2; docs/usage.md "Contact"): the video contact flag's hysteresis on the mean of this
# and the previous frame's flag probability, with per-model thresholds (h_on, h_off) picked for that rule on validation; the
# default (contact_smooth=1) is the shipped rule with the thresholds in contact_video.pt
CONTACT_SMOOTH_H = {"fah-s": (0.5, 0.5), "fah-b": (0.85, 0.1), "fah": (0.7, 0.15), "fah-h": (0.9, 0.1)}
FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))
DEFAULT_WEIGHTS = os.environ.get("FAH_WEIGHTS", os.path.join(FAH_HOME, "weights"))
DEFAULT_SAM3D = os.environ.get("FAH_SAM3D", os.path.join(FAH_HOME, "sam-3d-body-dinov3"))
SAM3D_REPO = "facebook/sam-3d-body-dinov3"
SAM3D_FILES = ("model_config.yaml", "assets/mhr_model.pt")
# assets/mhr_model.pt = the MHR model of Meta's Momentum Human Rig release v1.0.1 (Apache-2.0), byte-identical to the copy in
# the SAM 3D Body repository; only model_config.yaml comes from the gated, SAM-licensed repository
MHR_ASSETS = "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
MHR_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"

# settings on top of each model's config.json (the weights stay immutable): image keys for every mode, video keys on top
# for video / world. Identity NMS by wrist distance (pair_wrist), the MANO stage relative to the target wrist
# (mesh_fit_centre + fd step: float32-stable meshes) and, in video, the video detector (fah.video.TrackProposals proposals
# for tracked hands the detector misses)
SETTINGS = dict(image={"pair_wrist": 0.5, "mesh_fit_centre": True, "mesh_fit_fd_eps": 1e-2},
                video={"vd_propose": True, "vd_gate_fill": True, "vd_min_inside": 0.6, "vd_min_hits": 10})
# the hand detector (YOLO-seg): <weights>/shared/detector.pt, its exports <weights>/trt/detector-640-fp16.engine (TensorRT)
# and <weights>/coreml/detector-640.mlpackage (Core ML)
DETECTOR = "detector"

# execution settings per device type (no effect on the model; the CUDA ones are what all reported speeds use)
EXECUTION = {
    "cuda": {"dtype": "bfloat16", "cuda_graph": True, "fast_det": True, "det_half": True, "gpu_frame": True,
             "fast_batch": True, "tf32": True},
    "mps": {"dtype": "float32", "fast_det": True, "det_half": True, "gpu_frame": True, "fast_batch": True},  # fp16 overflows in ViT-L; bf16 can abort in Metal matmul (macOS 15 / PyTorch 2.15)
    "cpu": {"dtype": "float32", "fast_det": True},
}


def download_sam3d(dest=DEFAULT_SAM3D):
    """The two SAM 3D Body files the pipeline needs: the model config from the Hugging Face hub (the repository is gated
    under the SAM License: accept it on huggingface.co/facebook/sam-3d-body-dinov3 and log in with `hf auth login`
    first) and the MHR model from Meta's MHR release (Apache-2.0; fetch_mhr_model). The SAM 3D Body network weights
    (model.ckpt) are not needed: every FAH model.pt holds all of its weights."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(SAM3D_REPO, SAM3D_FILES[0], local_dir=dest)
    fetch_mhr_model(os.path.join(dest, SAM3D_FILES[1]))
    return dest


def fetch_mhr_model(path, url=MHR_ASSETS, sha256=MHR_SHA256):
    """assets/mhr_model.pt from the MHR release archive (about 200 MB; only this member is kept), checked by sha256;
    an existing file with the right hash is kept."""
    import hashlib
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    def digest(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()

    if os.path.exists(path) and digest(path) == sha256:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path) or ".") as tmp:
        archive = os.path.join(tmp, "assets.zip")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z, z.open("assets/mhr_model.pt") as src:
            with open(os.path.join(tmp, "mhr_model.pt"), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
        got = digest(os.path.join(tmp, "mhr_model.pt"))
        assert got == sha256, f"{url}: mhr_model.pt sha256 {got} != {sha256}"
        os.replace(os.path.join(tmp, "mhr_model.pt"), path)
    return path


def pick_device(device="auto"):
    from .utils import pick_device as pick

    return pick(device)


def camera_head(weights=DEFAULT_WEIGHTS):
    """Camera-motion head of the world camera (bench/world_camhead.py --ckpt; docs/world.md)."""
    return os.path.join(weights, "shared", "camera_motion.pt")


def mhr_fitter(weights=DEFAULT_WEIGHTS):
    """Learned initialiser of the MHR-Hand fit (train/mhr_fitter.py; fah.mhr_fit.MHRHandFitter init_net)."""
    return os.path.join(weights, "shared", "mhr_fitter.pt")


def trt_dir(weights=DEFAULT_WEIGHTS):
    """Where fah-build-trt writes the TensorRT engines."""
    return os.path.join(weights, "trt")


def detector_weights(weights=DEFAULT_WEIGHTS):
    """The hand detector, <weights>/shared/detector.pt (what fah-build-trt / fah-build-coreml export)."""
    return os.path.join(weights, "shared", f"{DETECTOR}.pt")


def coreml_noparts(pose):
    """The part-free twin of a Core ML pose export (no part head, no part_logits output: faster while the part maps are off)."""
    return pose[: -len(".mlpackage")] + "-noparts.mlpackage"


def coreml_paths(model="fah", weights=DEFAULT_WEIGHTS, parts=True):
    """Core ML exports of fah-build-coreml (Apple silicon): (pose network, detector). parts=False: the
    part-free pose network (<model>-pose-noparts.mlpackage) if it is built, else the full one."""
    d = os.path.join(weights, "coreml")
    pose = os.path.join(d, f"{model}-pose.mlpackage")
    if not parts and os.path.exists(coreml_noparts(pose)):
        pose = coreml_noparts(pose)
    return pose, os.path.join(d, f"{DETECTOR}-640.mlpackage")


def relaunch_once(module, argv):
    """Run `python -m <module> <argv>` in a child process and start it once more if it aborts (SIGABRT, exit 134) before
    its model has loaded (the child calls loaded()): on macOS, Core ML's first GPU predictions of the FAH / FAH-H
    multifunction pose packages hit an MPSGraph assertion now and then (MPSGraphTensorData.mm:223, about 1 launch in 10;
    not seen with FAH-S). Returns the child's exit code (signals as 128 + the signal number)."""
    import signal
    import subprocess
    import tempfile

    for attempt in range(2):
        with tempfile.TemporaryDirectory(prefix="fah_") as d:
            mark = os.path.join(d, "loaded")
            try:
                rc = subprocess.call([sys.executable, "-m", module, *argv], env=dict(os.environ, FAH_LOADED_MARK=mark))
            except KeyboardInterrupt:  # Ctrl-C reaches the child too
                return 130
            if rc not in (-signal.SIGABRT, 128 + signal.SIGABRT) or os.path.exists(mark) or attempt:
                return 128 - rc if rc < 0 else rc
        print("Core ML aborted while the model was loading (an MPSGraph assertion of macOS): starting once more", file=sys.stderr,
              flush=True)


def loaded():
    """The model has loaded (relaunch_once does not retry an abort from now on)."""
    if os.environ.get("FAH_LOADED_MARK"):
        open(os.environ["FAH_LOADED_MARK"], "w").close()


def mesh_kind(mesh):
    mesh = {"hmr": "mhr"}.get(mesh, mesh)
    assert mesh in MESHES, f"mesh must be one of {MESHES} (or hmr = mhr)"
    return mesh


def method_args(model="fah", mode="image", weights=DEFAULT_WEIGHTS, device="auto", sam3d=DEFAULT_SAM3D,
                trt_dir="", mesh="mano", tips=True, coreml=False, contact=False, parts=False, contact_video=None,
                contact_smooth=1, **overrides):
    """FAHMethod keyword arguments of a model and mode. mesh: none (no parametric mesh fit) | mano | mhr (MHR-Hand: the hand
    of SAM 3D Body's Momentum Human Rig, fah.mhr_fit; alias hmr) | both. tips: the mask-guided fingertip step. trt_dir:
    TensorRT engines built on this GPU (fah-build-trt -> <trt_dir>/<model>-pose-fp16.engine,
    <trt_dir>/detector-640-fp16.engine); used when present (CUDA only). coreml: the Core ML pose network + detector of
    fah-build-coreml (<weights>/coreml, macOS; with parts=False the part-free pose export, coreml_paths); MPS / PyTorch
    when they are missing. contact: the hand-object contact head (<weights>/<model>/contact.pt; PyTorch / TensorRT pose
    path, not Core ML). contact_video: with contact, also the video contact model (<weights>/<model>/contact_video.pt,
    fah.contact_video; online over the frames of a clip, reset() per clip); None = in video / world mode only, True = also
    in image mode (the frames of one video processed one by one), False = image contact only. contact_smooth: 2 = the
    video contact flag on the mean of two consecutive frames' probabilities (CONTACT_SMOOTH_H), 1 = the shipped rule.
    parts: the part maps (parts_roi; off by default: the quoted speeds are without them; the other outputs are the same
    either way)."""
    assert model in MODELS, f"model must be one of {MODELS}"
    assert mode in MODES, f"mode must be one of {MODES}"
    assert contact_smooth in (1, 2), "contact_smooth must be 1 or 2"
    mesh = mesh_kind(mesh)
    d = os.path.join(weights, model)
    assert os.path.exists(os.path.join(d, "config.json")), \
        f"FAH weights not found in {weights} (layout: fah/api.py docstring / README); pass weights= or set FAH_WEIGHTS"
    cfg = json.load(open(os.path.join(d, "config.json")))
    path = lambda k: os.path.normpath(os.path.join(d, cfg[k]))
    device = pick_device(device)
    for f in SAM3D_FILES:
        assert os.path.exists(os.path.join(sam3d, f)), f"{os.path.join(sam3d, f)} missing: run fah.download_sam3d('{sam3d}')"
    args = dict(checkpoint=os.path.join(sam3d, "model.ckpt"), base_weights=False, device=device,
                **EXECUTION[device.split(":")[0]],
                finetune_ckpt=os.path.join(d, cfg["model"]),
                detector_weights=path("detector"), mesh_fit_lmf=path("mesh_fitter"), **cfg["image"])
    args.update(SETTINGS["image"])
    if mode in ("video", "world"):  # world = the video model; its hands are placed with a camera trajectory
        args.update(refiner=path("video_refiner"), **cfg["video"], **SETTINGS["video"])
    if mesh in ("none", "mhr"):
        args["mesh_fit"] = 0
    if mesh in ("mhr", "both"):  # MHR-Hand fit: closed-form init, the learned initialiser, 1 damped Gauss-Newton step
        if os.path.exists(mhr_fitter(weights)):
            args.update(mhr_fit=1, mhr_fit_init=mhr_fitter(weights))
        else:
            warnings.warn(f"{mhr_fitter(weights)} missing: MHR-Hand fit from its closed-form init (2 steps)")
            args["mhr_fit"] = 2
    if not tips:
        args["tip_extend"] = 0.0
    apple = {"mesh_fit_device": "cpu", "mesh_fit_jac": "fdb", **({"refiner_device": "cpu"} if "refiner" in args else {})}
    if device.startswith("mps"):  # Apple silicon: the mesh stage and the video refiner on the CPU (launch-bound on MPS), batched fd Jacobian
        args.update(apple)
    if coreml:
        pose, det = coreml_paths(model, weights, parts=bool(parts))  # part maps off: the part-free pose export
        if sys.platform != "darwin":
            warnings.warn("coreml: Core ML runs on macOS only; using PyTorch")
        elif not (os.path.exists(pose) and os.path.exists(det)):
            warnings.warn(f"coreml: {pose} / {det} missing (build them with fah-build-coreml --backbone ...); using PyTorch on {device}")
        else:  # both networks in Core ML: the PyTorch modules stay on the CPU (no MPS memory, faster load; same outputs)
            args.update(pose_coreml=pose, det_coreml=det, device="cpu", **apple)
    if contact:  # trained on the frozen network: pose outputs unchanged
        path, video = os.path.join(d, "contact.pt"), os.path.join(d, "contact_video.pt")
        if not os.path.exists(path):
            warnings.warn(f"contact: {path} missing (no contact head for {model}); no contact output")
        elif "pose_coreml" in args:
            warnings.warn("contact: the Core ML exports have no contact head; no contact output (PyTorch / MPS has it)")
        else:
            args["contact_ckpt"] = path
            if (mode != "image") if contact_video is None else contact_video:
                if os.path.exists(video):
                    args["contact_video"] = video
                    if contact_smooth == 2:
                        args.update(contact_smooth=2, contact_h=CONTACT_SMOOTH_H[model])
                else:
                    warnings.warn(f"contact: {video} missing (no video contact model for {model}); image contact only")
    if trt_dir and device.startswith("cuda"):
        pose, det = os.path.join(trt_dir, f"{model}-pose-fp16.engine"), os.path.join(trt_dir, f"{DETECTOR}-{args['det_size']}-fp16.engine")
        if os.path.exists(pose):
            args["pose_trt"] = pose
        if os.path.exists(det):
            args["det_trt"] = det
    args["parts"] = bool(parts)
    args.update(overrides)
    return args


class FAH:
    """Hands in one image / the next frame of a video: FAH(model, mode)(image_bgr, K) -> list of hands, each
    {"is_right", "score", "bbox" (4,) xyxy px, "joints3d" (21, 3) camera frame m, "joints2d" (21, 2) px,
    "verts3d" (778, 3) fitted MANO mesh (camera frame m; mesh="mano" / "both"), "verts3d_dense" (778, 3) the dense surface,
    "verts3d_mhr" (2445, 3) fitted MHR-Hand mesh (faces_mhr), "verts3d_mhr_mano" (778, 3) its MANO-topology map,
    "joints3d_mhr_fit" (21, 3) its MANO-convention joints, "joints3d_mhr_rig" (21, 3) the rig's own hand keypoints,
    "mhr_params" (SAM 3D Body / MHR parameters, docs/mhr_hand.md; left hands: the right rig fitted to the mirrored hand,
    mirrored=True) (mesh="mhr" / "both"),
    "surface2d_head" / "surface3d_head" / "surface_visible" dense points, "parts_roi" (x0, y0, part map; parts=True),
    "mask_det" (instance mask RLE, with det_masks=True; det_masks="crop": "mask_det_crop" = (x0, y0, box crop bool) instead, no
    full-frame mask or RLE, for live use; bench.seg.rle paste_crop / encode_crop), "contact" (778,) per-vertex hand-object contact probability,
    "contact_dense" (400,) on the dense points, "contact_score" (hand), "contact_fingers" (6,) palm, thumb .. pinky
    (with contact=True; CONTACT_THRESH), "contact_video" (778,) the video contact model's per-vertex probability,
    "contact_score_video" (hand), "contact_flag_video" (hand in contact, with hysteresis) (contact=True in video / world
    mode, or contact_video=True; FAH), "track_id" (video / world mode: the video detector's track of the hand; refiner gap
    fills keep their track's)}; joints in OpenPose order (wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4,
    pinky 1-4). K: 3x3 camera intrinsics (default: focal = image diagonal). Video / world: call
    reset() before each new clip, then frame by frame (the refiner and the filters are causal). World: the video
    model's hands; place them with fah.world.place and per-frame T_world_cam (docs/world.md; the timed world
    configuration runs with mesh="none")."""

    def __init__(self, model="fah", mode="image", weights=DEFAULT_WEIGHTS, device="auto", sam3d=DEFAULT_SAM3D,
                 trt_dir="", mano_dir=None, fps=30.0, mesh="mano", tips=True, coreml=False, contact=False,
                 parts=False, contact_video=None, **overrides):
        if mano_dir:
            os.environ["MANO_DIR"] = mano_dir
        elif "MANO_DIR" not in os.environ and os.path.isdir(os.path.join(FAH_HOME, "mano")):
            os.environ["MANO_DIR"] = os.path.join(FAH_HOME, "mano")
        from bench.methods.fah import FAHMethod

        self.model, self.mode, self.mesh, self.name = model, mode, mesh_kind(mesh), f"{model}-{mode}"
        self.args = method_args(model, mode, weights, device, sam3d, trt_dir, mesh, tips, coreml, contact, parts,
                                contact_video=contact_video, fps=fps, **overrides)
        self.method = FAHMethod(**self.args)
        self.camera_head = camera_head(weights)
        self.reset()

    def reset(self):
        self.method.reset()

    def __call__(self, image_bgr, K=None):
        if K is None:
            from .estimator import default_cam_int

            K = default_cam_int(*image_bgr.shape[:2])
        hands = self.method(image_bgr, np.asarray(K, np.float32))
        if self.mesh in ("none", "mhr"):  # no MANO fit: the network's dense surface is verts3d_dense, as with mesh="mano"
            for h in hands:
                if h.get("verts3d") is not None and not h.get("mesh_param"):
                    h["verts3d_dense"] = h.pop("verts3d")
        return hands

    @property
    def faces(self):
        """MANO faces (right-hand winding) of verts3d / verts3d_dense."""
        return self.method._mano_faces()

    @property
    def faces_mhr(self):
        """MHR-Hand faces of verts3d_mhr (right-hand winding; left hands: faces_mhr[:, ::-1])."""
        return self.method.est.faces


def load(model="fah", mode="image", **kwargs):
    return FAH(model, mode, **kwargs)
