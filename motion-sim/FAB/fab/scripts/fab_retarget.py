"""FAB human motion -> Unitree G1 reference motion with the released retargeter (fab-retarget).

Input: a fab-world output (<out>/<clip>.npz, with its scene reconstruction in <out>/work/joint/recon/<clip>.npz +
<clip>_hfield.npz), or such a recon <clip>.npz next to its _hfield.npz, or a standalone SMPL file (SMPL / SMPL-H parameters or
vertices, fab/retarget/smpl_input.py; --up y for y-up data): its world is <clip>_hfield.npz next to it (the motion as given) or
else a flat floor at z = 0 with the motion's lowest foot vertex over the clip on it, used as is (steps 1-2), in every mode (mesh
by default); --mode skeleton also takes a skeleton file (below). SMPL and skeleton files need no FAB weights (points mode
builds its correspondence once from FAB-L's).
Steps (fab/retarget/release_config.py holds the configuration):
  1. bench.videomimic.scene_retarget on the reconstruction's level-snapped heightfield (<out>/g/solve_pre): the scene
     objective, foot contacts (4 cm / 0.15 m/s: the mesh rule, or its joint rule in skeleton mode), the input smoother (sigma
     8 frames at 50 fps);
  2. bench.stance_pin on that solution (<out>/g/p_pre): stance runs pinned to the terrain, 4.5 cm / frame root-step cap;
  3. the clip's simulation package on a 2 cm grid (bench.scene_r2s_simpkg build; <out>/stage/clips/<clip>);
  4. the terrain clean along the pinned reference's G1 link paths (scripts/g1_terrain_clean_pkg.py: rise cap near the link
     paths, despike / sliver / pit removal, reliability + structure maps; <out>/clean/<clip>);
  5. the same solve on the clean terrain (<out>/g/solve) and 6. the same pin on it (<out>/g/p)
-> <out>/<clip>_g1.npz (g1_dof, g1_root_pos, g1_root_quat_xyzw, 50 fps) and <out>/<clip>_g1.json (inputs and settings).
A clip the clean step cannot take (a reconstruction heightfield without valid cells) keeps step 2's reference on the
level-snapped terrain; <clip>_g1.json says so. --no_clean: steps 1 and 2 only. Finished steps in --out are reused: give
each mode (and each input) its own --out.

    fab-retarget out/fab_world/clip.npz [--out DIR] [--no_clean] [--mode points|mesh|skeleton]

Three modes, one solver and configuration:
  points (default)  the mesh mode plus the solver's dense-point term (fab/retarget/release_config.py W_PTS) on FAB's dense
                    points (fab-world's FAB stage, work/ours/fab/<clip>_raw.npz, registered on the reconstruction's body by
                    fab/retarget/dense_inputs.py; the correspondence $FAB_HOME/extras/dense_corr_g1.npz, built on first use by
                    fab.retarget.dense_corr). An input without dense points gives the term the body's own vertices at the same
                    ids, with a WARNING naming the cause; <clip>_g1.json records both.
  mesh              the joints and the mesh fitted to the points: contacts, the seated pelvis and the root start from the mesh.
  skeleton          the joints alone: those three from joint rules (fab/retarget/skeleton.py). Besides fab-world outputs it
                    takes a skeleton file: <clip>.npz with joints (T, >= 24, 3) in SMPL-24 order (metres, z-up world), fps
                    (resampled to 50) and optionally frames; its world is <clip>_hfield.npz next to it (x0, y0, res, height)
                    or else a flat floor at z = 0. A given world is used as is: no level snap, no terrain clean (steps 1-2).
On CUDA the solver's and the stance pin's loops run as CUDA graphs (fab/retarget/cuda_graph.py), bitwise the plain loops;
--no_graph runs the plain loops. Needs MuJoCo and the G1 robot description (scripts/download_g1.sh). The stance pin depends
on which clips it solves together (--batch_frames); this command pins each clip alone.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[1]


def _inproc(cmd, env=None):
    """A [python, -m, module, args] or [python, script.py, args] step in this process: the module's main() (the script) with the
    same argv, env and working directory: one interpreter / CUDA start-up for the whole chain, the same computation. -> its
    exit code."""
    import importlib

    cmd = [str(x) for x in cmd]
    keep = {k: os.environ.get(k) for k in (env or {})}
    argv, cwd = sys.argv, os.getcwd()
    os.environ.update(env or {})
    os.environ.setdefault("MUJOCO_GL", "egl")
    script = cmd[1] != "-m"
    sys.argv = [cmd[1]] + cmd[2:] if script else [cmd[2]] + cmd[3:]
    os.chdir(R_)
    try:
        if script:
            import runpy

            runpy.run_path(str(R_ / cmd[1]), run_name="__main__")
        else:
            importlib.import_module(cmd[2]).main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:  # noqa: BLE001  (as a failed subprocess: the traceback, then the caller's rc handling)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        sys.argv = argv
        os.chdir(cwd)
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run(cmd, env=None):
    cmd = [str(x) for x in cmd]
    print(" ".join(cmd), flush=True)
    rc = _inproc(cmd, env)
    if rc:
        raise SystemExit(f"[fab_retarget] failed ({rc}): {' '.join(cmd)}")


def skeleton_inputs(src, c, out):
    """--mode skeleton on a skeleton file: a recon copy under <out>/sk_recon (joints, fps, frames, no contacts) and its world
    (<clip>_hfield.npz next to the file, else a flat floor at z = 0 on a 4 cm grid over the pelvis path + 2 m)."""
    z = np.load(src, allow_pickle=True)
    J = np.asarray(z["joints"], np.float64)
    assert J.ndim == 3 and J.shape[1] >= 24 and J.shape[2] == 3, \
        f"{src}: joints must be (T, >= 24, 3) in SMPL-24 order (docs/retarget.md), not {J.shape}"
    assert "fps" in z.files, f"{src}: no fps"
    dr = out / "sk_recon"
    dr.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(z["frames"], int) if "frames" in z.files else np.arange(len(J))
    np.savez(dr / f"{c}.npz", joints=J[:, :24], fps=float(z["fps"]), frames=frames, contacts="{}")
    hf = src.with_name(f"{c}_hfield.npz")
    if hf.exists():
        h = np.load(hf, allow_pickle=True)
        world = {k: h[k] for k in ("x0", "y0", "res", "height")}
        src_w = str(hf)
    else:
        res, lo, hi = 0.04, J[:, 0, :2].min(0) - 2.0, J[:, 0, :2].max(0) + 2.0
        nx, ny = int(np.ceil((hi[0] - lo[0]) / res)) + 1, int(np.ceil((hi[1] - lo[1]) / res)) + 1
        world = {"x0": float(lo[0]), "y0": float(lo[1]), "res": res, "height": np.zeros((ny, nx), np.float32)}
        src_w = "flat floor at z = 0"
    np.savez(dr / f"{c}_hfield.npz", **world)
    print(f"[fab_retarget] {c}: skeleton file {len(J)} frames at {float(z['fps']):g} fps; world: {src_w}", flush=True)
    return dr, {"skeleton_file": str(src), "world": src_w}


def smpl_inputs(src, c, out, up="z"):
    """A standalone SMPL file (parameters or vertices; fab/retarget/smpl_input.py) as a recon copy under <out>/smpl_recon
    (joints, verts, fps, frames, valid, no contacts: the solver labels them on the terrain) and its world: <clip>_hfield.npz
    next to the file, the motion as given; else a flat floor at z = 0 on a 4 cm grid over the pelvis path + 2 m, the motion
    shifted in z so that its lowest foot vertex over the clip is at z = 0."""
    from fab.retarget import smpl_input as SI

    m = SI.load(src, up)
    V, J = m["verts"], m["joints"]
    hf = src.with_name(f"{c}_hfield.npz")
    if hf.exists():
        h = np.load(hf, allow_pickle=True)
        world, src_w, dz = {k: h[k] for k in ("x0", "y0", "res", "height")}, str(hf), 0.0
    else:
        dz = SI.foot_floor_shift(V)
        V[..., 2] += dz
        J[..., 2] += dz
        res, lo, hi = 0.04, J[:, 0, :2].min(0) - 2.0, J[:, 0, :2].max(0) + 2.0
        nx, ny = int(np.ceil((hi[0] - lo[0]) / res)) + 1, int(np.ceil((hi[1] - lo[1]) / res)) + 1
        world = {"x0": float(lo[0]), "y0": float(lo[1]), "res": res, "height": np.zeros((ny, nx), np.float32)}
        src_w = "flat floor at z = 0"
    dr = out / "smpl_recon"
    dr.mkdir(parents=True, exist_ok=True)
    np.savez(dr / f"{c}.npz", joints=J.astype(np.float32), verts=V.astype(np.float32), fps=m["fps"], frames=m["frames"],
             valid=np.ones(len(V), bool), contacts="{}")
    np.savez(dr / f"{c}_hfield.npz", **world)
    info = {"smpl_file": str(src), **m["info"], "world": src_w,
            "ground_dz_m": dz if not hf.exists() else "none (a given world)"}
    print(f"[fab_retarget] {c}: SMPL file ({m['info']['kind']}, {m['info']['gender']}, up {up}) {len(V)} frames at {m['fps']:g} fps; "
          f"world: {src_w}" + ("" if hf.exists() else f", lowest foot vertex to z = 0 (dz {dz:+.3f} m)"), flush=True)
    return dr, info


def dense_inputs(c, rd, out, standalone=False):
    """Points mode: the reconstruction with FAB's registered dense points (a copy under <out>/d_recon; fab-world's files are
    not changed) and what applied; standalone (an SMPL file): no dense points."""
    from fab.retarget import dense_corr, dense_inputs as DI

    corr = dense_corr.default_path()
    if not corr.exists() or dense_corr.stale(corr):  # (re)built for the installed G1 description
        from fab import paths

        ck = paths.model_file("fab-l")
        if not ck.exists():  # the correspondence's dense-point ids are FAB-L's
            raise SystemExit(f"[fab_retarget] points mode needs the dense-point correspondence {corr}, built once from FAB-L's "
                             f"weights ({ck}: not found). Install the weights (README.md) or use --mode mesh or skeleton"
                             + (" (mesh is an SMPL file's default)" if standalone else ""))
        dense_corr.make(out=corr)
    sidx = np.load(corr)["surface_idx"]
    work = None if standalone else rd.parent.parent  # fab-world's <out>/work (work/joint/recon)
    dr = out / "d_recon"
    dr.mkdir(parents=True, exist_ok=True)
    for f in rd.glob(f"{c}*"):
        if f.name != f"{c}.npz" and not (dr / f.name).exists():
            (dr / f.name).symlink_to(f.resolve())
    z = dict(np.load(rd / f"{c}.npz", allow_pickle=True))
    if standalone:
        reg, why = None, "a standalone SMPL file (no FAB dense points)"
    else:
        raw = work / "ours" / "fab" / f"{c}_raw.npz"
        reg = DI.register_points(rd / f"{c}.npz", raw, sidx)
        why = None if reg else DI.points_unavailable(raw, sidx)
    np.savez(dr / f"{c}.npz", **z, **(reg or {}))
    info = {"work": None if standalone else str(work), "correspondence": str(corr),
            "points": "FAB dense points registered on the body" if reg else "the body's own vertices at the dense ids (no dense points)",
            "pts_info": json.loads(reg["pts_info"]) if reg else None, "no_points_cause": why}
    print(f"[fab_retarget] {c}: points mode: points = {info['points']}", flush=True)
    if why:  # docs/troubleshooting.md
        print(f"[fab_retarget] WARNING {c}: points mode without FAB's dense points: {why}. The dense-point term uses the body's "
              "own vertices at the same ids instead", flush=True)
    return dr, info


def default_mode(src):
    """--mode when none is given: points; mesh for a standalone SMPL file (it has no FAB dense points)."""
    from fab.retarget import smpl_input as SI

    return "mesh" if SI.kind(np.load(src, allow_pickle=True).files) else "points"


def scene(a, c, out, extra_cfg=None):
    """The fab-world reconstruction (work/joint/recon/<clip>.npz + _hfield.npz) through the scene-terrain chain."""
    from fab.retarget import release_config as C
    from fab.retarget import smpl_input as SI

    src = Path(a.input).resolve()
    files = np.load(src, allow_pickle=True).files
    rd = src.parent if "verts" in files else src.parent / "work" / "joint" / "recon"  # export -> its recon
    sk_info = smpl_info = None
    up = getattr(a, "up", "z")
    if SI.kind(files, a.mode):  # a standalone SMPL file with its world
        if getattr(a, "ground_aware", False):
            raise SystemExit("[fab_retarget] --ground_aware needs fab-world's foot contacts; a standalone SMPL file has none")
        rd, smpl_info = smpl_inputs(src, c, out, up)
    elif up != "z":
        raise SystemExit("[fab_retarget] --up applies to standalone SMPL files only (docs/retarget.md)")
    elif a.mode == "skeleton" and not (rd / f"{c}.npz").exists():  # a skeleton file with its world
        rd, sk_info = skeleton_inputs(src, c, out)
    assert (rd / f"{c}.npz").exists() and (rd / f"{c}_hfield.npz").exists(), \
        f"no recon + hfield for {c} under {rd}" + ("" if a.mode == "skeleton" else " (a joints-only file needs --mode skeleton)")
    sig = C.SIGMA_IN
    run_dir, empty = out / "g", out / "g" / "_no_head_files"
    empty.mkdir(parents=True, exist_ok=True)
    ga = None
    if getattr(a, "ground_aware", False):  # fab-world's video contact labels (fab/retarget/release_config.py ground_aware_cfg)
        ga = Path(getattr(a, "ground_work", None) or rd.parent.parent) / "contacts" / f"{c}__0.contact.npz"
        if not ga.exists():
            raise SystemExit(f"[fab_retarget] --ground_aware needs fab-world's foot contacts, {ga} (fab-world writes them to "
                             "<out>/work/contacts); give a fab-world output, or drop --ground_aware")
    dense_info = None
    if a.mode == "points":
        rd, dense_info = dense_inputs(c, rd, out, standalone=smpl_info is not None)
        cfg = C.solve_cfg_dp()
    else:
        cfg = C.solve_cfg()
    if extra_cfg:  # fab-retarget-ee's config (scripts/fab_retarget_ee.py); fab-retarget passes none
        cfg = dict(extra_cfg)
    pin_cfg = C.PIN
    if ga is not None:
        cfg, pin_cfg = C.ground_aware_cfg(cfg, ga.parent.resolve()), C.ground_aware_pin(C.PIN)
    if a.no_graph:  # the plain solver / pin loops (the CUDA-graph default is bitwise the same, faster)
        cfg = dict(cfg, graph=0)
    given = sk_info is not None or smpl_info is not None
    level = [] if given else ["--clean"]  # a reconstruction's heightfield is level-snapped; a given world is used as is
    mode = ["--mode", "skeleton"] if a.mode == "skeleton" else []

    def solve(tag, terrain=None):
        run([sys.executable, "-m", "bench.videomimic.scene_retarget", "--recon", rd, "--only", c, "--out", run_dir / tag] + level + mode
            + ["--sigma_in", str(sig), "--cfg", json.dumps(cfg)] + (["--terrain_solve", terrain[0], "--pkg", terrain[1]] if terrain else []))

    def pin(tag, out_tag, terrain=None):
        run([sys.executable, "-m", "bench.stance_pin", "--run", run_dir, "--tag", tag, "--out_tag", out_tag, "--recon", rd, "--only", c,
             "--contacts_dir", empty, "--cfg", json.dumps(pin_cfg), "--batch_frames", "1500", "--gate"]
            + (["--terrain_clean", terrain[0], "--pkg", terrain[1]] if terrain else []), env={"PIN_GRAPH": "0"} if a.no_graph else None)

    info = {"chain": "scene", "retargeter": "fab", "mode": a.mode, "recon": str(rd), "cfg": cfg, "sigma_in": sig, "pin": pin_cfg, "cuda_graphs": not a.no_graph}
    if ga is not None:
        info["ground_aware"] = str(ga)
    if dense_info is not None:
        info["dense"] = dense_info
    if sk_info is not None:
        info["skeleton_input"] = sk_info
    if smpl_info is not None:
        info["smpl_input"] = smpl_info
    solve("solve_pre")  # 1. on the reconstruction's own (level-snapped) heightfield
    pin("solve_pre", "p_pre")  # 2.
    if a.no_clean or given:
        return run_dir / "p_pre" / f"{c}_motion.npz", dict(info, clean="off (--no_clean)" if a.no_clean else "off (a given world)")
    stage, clean = out / "stage" / "clips", out / "clean"
    wk = Path(dense_info["work"]) if dense_info else rd.parent.parent  # fab-world's work dir
    contacts = wk / "contacts" if (wk / "contacts").exists() else empty  # fab-world's work/contacts
    rc = _inproc([sys.executable, "-m", "bench.scene_r2s_simpkg", "build", "--run", str(run_dir), "--tags", "p_pre",  # 3.
                  "--recon", str(rd), "--clip", c, "--out", str(stage), "--contacts", str(contacts)])
    if rc == 0 and (stage / c / "package.json").exists():
        from bench.scene_r2s_simpkg import hfield_2cm

        hfield_2cm(stage / c, "edge")
        rc = _inproc([sys.executable, "scripts/g1_terrain_clean_pkg.py", "--v", str(out / "stage"), "--out", str(clean), "--only", c,  # 4.
                      "--ref", "p_pre", "--slivers", "3", "--sliver_obj_h", "0.15", "--pit_close", "1", "--workers", "1"])
    cf = clean / c / "terrain_hfield_2cm_clean.npz"
    if not cf.exists():  # a reconstruction heightfield without valid cells (or no package) has no clean terrain: step 2's reference
        print(f"[fab_retarget] {c}: no clean terrain (rc {rc}); keeping the reference on the level-snapped terrain", flush=True)
        return run_dir / "p_pre" / f"{c}_motion.npz", dict(info, clean=f"failed (rc {rc}): level-snapped terrain only")
    info.update(clean=str(cf), clean_flags="--slivers 3 --sliver_obj_h 0.15 --pit_close 1 --ref p_pre")
    solve("solve", (clean, stage))  # 5.
    s1, s2 = (float(np.load(run_dir / t / f"{c}_motion.npz", allow_pickle=True)["scale"]) for t in ("solve_pre", "solve"))
    assert abs(s1 - s2) < 1e-6, f"the re-solve's scale {s2} differs from the clean package's {s1}"
    pin("solve", "p", (clean, stage))  # 6.
    return run_dir / "p" / f"{c}_motion.npz", info


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="fab-world's <out>/<clip>.npz (or a recon <clip>.npz next to its _hfield.npz), a standalone SMPL "
                   "file (parameters or vertices); --mode skeleton: also a skeleton file (docs/retarget.md)")
    p.add_argument("--out", default="output/fab_retarget")
    p.add_argument("--no_clean", action="store_true", help="no terrain clean (solve + pin on the reconstruction's heightfield)")
    p.add_argument("--mode", default=None, choices=("points", "mesh", "skeleton"),
                   help="points (default; mesh for a standalone SMPL file): FAB's predicted dense points (the mesh mode plus the dense-point term, "
                   "fab/retarget/release_config.py W_PTS); mesh: the mesh fitted to them; skeleton: the joints alone "
                   "(docs/retarget.md: the modes and the skeleton input)")
    p.add_argument("--release", default=None, choices=("Dp", "G"), help=argparse.SUPPRESS)  # hidden alias of --mode points / mesh
    p.add_argument("--ground_aware", action="store_true", help="opt-in: the feet's contacts from fab-world's video contact model "
                   "(work/contacts), labelled feet held at the floor height with sliding allowed, and no flight when the person's "
                   "foot is near the floor (docs/retarget.md: ground-aware references)")
    p.add_argument("--up", default="z", choices=("z", "y"), help="a standalone SMPL file's up axis (y: (x, y, z) -> (x, -z, y); "
                   "docs/retarget.md)")
    p.add_argument("--no_graph", action="store_true", help="the plain solver / pin loops instead of CUDA graphs (bitwise the same "
                   "result, slower)")
    a = p.parse_args()
    alias = {"Dp": "points", "G": "mesh"}
    if a.mode and a.release and alias[a.release] != a.mode:
        p.error(f"--release {a.release} is --mode {alias[a.release]}, not --mode {a.mode}")
    a.mode = a.mode or alias.get(a.release)
    c = Path(a.input).stem
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(R_))
    a.mode = a.mode or default_mode(a.input)
    f, info = scene(a, c, out)
    assert f.exists(), f"no output {f}"
    shutil.copy(f, out / f"{c}_g1.npz")
    json.dump({"input": str(Path(a.input).resolve()), "output": str(f), **info}, open(out / f"{c}_g1.json", "w"), indent=1, default=str)
    print(f"[fab_retarget] {c}: {info['chain']} -> {out / f'{c}_g1.npz'}")
    z = np.load(out / f"{c}_g1.npz", allow_pickle=True)
    lab = json.loads(str(z["contacts"])) if "contacts" in z.files else {}
    if not any(np.any(v) for v in lab.values()):  # docs/troubleshooting.md
        print(f"[fab_retarget] WARNING {c}: no contact frames (no support part within the contact rule's height / speed of "
              "the terrain on any frame): the reference is not pinned to the terrain and can float; check fab-world's "
              "scene and contacts", flush=True)


if __name__ == "__main__":
    main()
