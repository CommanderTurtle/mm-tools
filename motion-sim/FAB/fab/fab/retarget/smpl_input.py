"""fab-retarget's standalone SMPL input: an SMPL motion from any source (motion capture, AMASS, other estimators) read as the
body fab-retarget retargets (scripts/fab_retarget.py smpl_inputs; docs/retarget.md). A file <clip>.npz holds
  parameters  poses (T, 72) axis-angle, or global_orient (T, 3) + body_pose (T, 69); trans (T, 3); optionally betas ((n,) or
              (T, n); the first 10, missing ones 0) and gender (neutral, male or female; default neutral);
              SMPL-H (AMASS: poses (T, 156), betas (16,)) and body_pose (T, 63): the root and the 21 body joints (pose dims
              0-65), SMPL's hand joints 22 / 23 at zero, the first 10 betas (AMASS's dmpls are not read)
  or vertices verts (T, 6890, 3), SMPL topology;
and fps (or AMASS's mocap_framerate), optionally frames (T,). Parameters are posed with the SMPL model of the file's gender
($FAB_HOME/body_models/smpl/SMPL_<GENDER>.pkl, scripts/download_body_models.sh); the joints are that model's J_regressor on
the vertices, as fab-world's. Metres; up = "y" turns a y-up world z-up first: (x, y, z) -> (x, -z, y).
"""
import numpy as np

from fab import paths

Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # y-up -> z-up
GENDERS = ("neutral", "male", "female")
_MODELS = {}


def kind(files, mode=None):
    """'params' / 'verts' for a standalone SMPL file's keys, else None: a FAB reconstruction (verts with contacts), a fab-world
    output (pose, not poses), a skeleton file, or (mode skeleton) any file with joints, read as a skeleton file."""
    files = set(files)
    if mode == "skeleton" and "joints" in files:
        return None
    if "poses" in files or {"global_orient", "body_pose"} <= files:
        return "params"
    if "verts" in files and "contacts" not in files:
        return "verts"
    return None


def _text(x):
    x = np.asarray(x).reshape(-1)
    x = x[0] if len(x) else ""
    return (x.decode() if isinstance(x, bytes) else str(x)).strip().lower()


def _model(gender):
    from bench.body_models import SMPL

    f = paths.smpl_dir() / f"SMPL_{gender.upper()}.pkl"
    if str(f) not in _MODELS:
        assert f.exists(), f"no SMPL model {f} (scripts/download_body_models.sh)"
        _MODELS[str(f)] = SMPL(f, num_betas=10).double()
    return _MODELS[str(f)], f


def pose72(z):
    """(T, 72) SMPL axis-angle from poses or global_orient + body_pose, and the rule that applied."""
    if "poses" in z:
        P = np.asarray(z["poses"], np.float64)
        P = P.reshape(len(P), -1)
        if P.shape[1] == 72:
            return P, "poses (T, 72): SMPL"
        assert P.shape[1] == 156, f"poses must be (T, 72) SMPL or (T, 156) SMPL-H, not {P.shape}"
        return np.concatenate([P[:, :66], np.zeros((len(P), 6))], 1), \
            "poses (T, 156): SMPL-H; dims 0-65 (root + 21 body joints), SMPL hand joints 22 / 23 zero"
    go = np.asarray(z["global_orient"], np.float64).reshape(-1, 3)
    bp = np.asarray(z["body_pose"], np.float64).reshape(len(go), -1)
    if bp.shape[1] == 69:
        return np.concatenate([go, bp], 1), "global_orient (T, 3) + body_pose (T, 69): SMPL"
    assert bp.shape[1] == 63, f"body_pose must be (T, 69) or (T, 63), not {bp.shape}"
    return np.concatenate([go, bp, np.zeros((len(go), 6))], 1), \
        "global_orient (T, 3) + body_pose (T, 63): 21 body joints, SMPL hand joints 22 / 23 zero"


def load(src, up="z"):
    """A standalone SMPL file -> {verts (T, 6890, 3), joints (T, 24, 3), fps, frames, info}, z-up, metres, as in the file
    (no grounding)."""
    import torch

    z = np.load(src, allow_pickle=True)
    k = kind(z.files)
    assert k, f"{src}: not a standalone SMPL file (poses, global_orient + body_pose, or verts; docs/retarget.md)"
    assert up in ("y", "z"), up
    fk = "fps" if "fps" in z.files else "mocap_framerate"
    assert fk in z.files, f"{src}: no fps (or mocap_framerate)"
    gender = _text(z["gender"]) if "gender" in z.files else "neutral"
    assert gender in GENDERS, f"{src}: gender {gender!r}, not one of {GENDERS}"
    smpl, mf = _model(gender)
    info = {"kind": k, "model": str(mf), "gender": gender, "up": up}
    if k == "params":
        P, info["poses"] = pose72(z)
        T = len(P)
        assert "trans" in z.files, f"{src}: no trans"
        tr = np.asarray(z["trans"], np.float64).reshape(T, 3)
        b = np.asarray(z["betas"], np.float64) if "betas" in z.files else np.zeros(10)
        b = b.reshape(-1, b.shape[-1]) if b.ndim else np.zeros((1, 10))
        assert len(b) in (1, T), f"{src}: betas {b.shape} for {T} frames"
        B = np.zeros((len(b), 10))
        B[:, : min(10, b.shape[1])] = b[:, :10]
        B = np.broadcast_to(B, (T, 10)).copy()
        info["betas"] = f"{b.shape[1]} given, the first {min(10, b.shape[1])} used" + (" (per frame)" if len(b) > 1 else "")
        V = []
        with torch.no_grad():
            for i in range(0, T, 256):
                t = lambda x: torch.as_tensor(np.ascontiguousarray(x[i: i + 256]), dtype=torch.float64)  # noqa: E731
                V.append(smpl(t(P[:, :3]), t(P[:, 3:]), t(B), t(tr))[0].numpy())
        V = np.concatenate(V)
    else:
        V = np.asarray(z["verts"], np.float64)
        assert V.ndim == 3 and V.shape[1:] == (6890, 3), f"{src}: verts must be (T, 6890, 3), not {V.shape}"
    if up == "y":
        V = V @ Y_UP.T
    J = np.einsum("jv,tvc->tjc", smpl.J_regressor.numpy(), V)
    frames = np.asarray(z["frames"], int) if "frames" in z.files else np.arange(len(V))
    return {"verts": V, "joints": J, "fps": float(z[fk]), "frames": frames, "info": info}


def foot_floor_shift(V):
    """The z shift that puts the lowest SMPL foot vertex over the whole clip at z = 0 (foot parts: fab.retarget.contact
    HUMAN_PARTS lfoot / rfoot, the neutral model's skinning-weight argmax, as the solver's contact rule)."""
    from fab.retarget.contact import HUMAN_PARTS

    part = _model("neutral")[0].weights.argmax(1).numpy()
    idx = np.flatnonzero(np.isin(part, HUMAN_PARTS["lfoot"] + HUMAN_PARTS["rfoot"]))
    return -float(V[:, idx, 2].min())
