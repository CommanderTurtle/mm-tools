"""Multi-view stereo depth with ViPE's cameras held fixed (COLMAP PatchMatch). The monocular (UniDepth) depth is
range-biased and inconsistent across frames on stairs even with accurate cameras, so: COLMAP patch_match_stereo (geometric consistency on,
3 PatchMatch iterations instead of COLMAP's 5) on every --stride-th dense frame, images at --size long side, ViPE's K and cam -> world
poses written as a fixed COLMAP model, sources = the keyframes at +-1, 2, 4, 8 keyframe steps (patch-match.cfg; no SfM points),
depth range [--dmin, --dmax]. Per keyframe: the geometric depth map, kept where >= --cons_min neighbour keyframes (+- --cons_steps) see the same surface within
--cons_rel and with a camera baseline >= --min_parallax x the depth (COLMAP's geometric maps are dense, and identical views
from a static camera agree on any depth; this is stereo_fusion's consistency test plus a parallax floor), at the ViPE dense
resolution
(2x2 block median of valid pixels),
people (ViPE's dynamic mask + the segmenter's people + our projected body) invalid; where MVS has no depth, ViPE's depth x the
frame's median MVS / ViPE ratio on the overlap (one rule for every clip); frames between keyframes come back as ViPE depth
(x the nearest keyframe's ratio) only where that keyframe has < --fill_share MVS (stretches without parallax stay the ViPE run).
-> a ViPE-style dir:
  <out>/dense/<clip>.npz   the keyframes (+ those ViPE frames): depth (f16), dyn, K, frames, down, full_hw, phrases + mvs_share (N,) (share of
                           static pixels from MVS), mvs_ratio (N,) (the fallback's scale), mvs = the settings
  <out>/{cams,...}         symlinks to the ViPE run
    python -m bench.scene_recon.mvs_depth --vipe <vipe out> --recon <world out>/recon --list clips.json \
        --only <clip> --seg_dir <seg dir> --out <dir> [--colmap colmap]
COLMAP: --colmap, else $FAB_COLMAP, else $FAB_ENVS/colmap/bin/colmap (scripts/colmap_setup.sh).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fab import paths

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read_colmap_array(path):
    with open(path, "rb") as f:
        w, h, c = np.genfromtxt(f, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        f.seek(0)
        n = 0
        while n < 3:
            if f.read(1) == b"&":
                n += 1
        a = np.fromfile(f, np.float32)
    return np.transpose(a.reshape((w, h, c), order="F"), (1, 0, 2)).squeeze()


def block_median(d, k):
    h, w = d.shape[0] // k, d.shape[1] // k
    b = d[: h * k, : w * k].reshape(h, k, w, k).transpose(0, 2, 1, 3).reshape(h, w, k * k)
    b = np.where(b > 0, b, np.nan)
    with np.errstate(all="ignore"):
        m = np.nanmedian(b, axis=2)
    return np.nan_to_num(m, nan=0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vipe", required=True)
    p.add_argument("--recon", required=True, help="recon dir with the body (verts, T_world_cam) for the body masks")
    p.add_argument("--list", required=True)
    p.add_argument("--only", required=True)
    p.add_argument("--seg_dir", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--colmap", default=os.environ.get("FAB_COLMAP", str(paths.envs_dir() / "colmap" / "bin" / "colmap")))
    p.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "mvs"))
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--size", type=int, default=960)
    p.add_argument("--steps", default="1,2,4,8")
    p.add_argument("--dmin", type=float, default=0.3)
    p.add_argument("--dmax", type=float, default=20.0)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--geom", type=int, default=1,
                   help="1: COLMAP photometric + geometric pass + filter; 0: photometric pass only (our consistency filter replaces both)")
    p.add_argument("--cons_steps", default="1,2", help="keyframe offsets checked for the consistency filter")
    p.add_argument("--cons_rel", type=float, default=0.02, help="a neighbour agrees if its depth there is within this relative error")
    p.add_argument("--cons_min", type=int, default=2, help="keep a pixel seen consistently by >= this many neighbours")
    p.add_argument("--reuse", type=int, default=0, help="1: keep an existing workspace's depth maps (re-filter only)")
    p.add_argument("--fill_share", type=float, default=0.10,
                   help="frames between keyframes are kept (ViPE depth x ratio) where the nearest keyframe's MVS share is below this")
    p.add_argument("--min_parallax", type=float, default=0.03,
                   help="per-pixel triangulation validity: a neighbour counts only if the camera baseline is >= this x the pixel's "
                        "depth (0.03 ~ a 1.7 deg triangulation angle, the usual minimum; COLMAP's triangulator defaults to 1.5 deg). "
                        "Lower values let noisy head-camera pixels through. No parallax: "
                        "no stereo, the ViPE depth (a static camera falls back everywhere)")
    a = p.parse_args()
    from bench.body_models import SMPL
    from bench.scene_recon.fuse_render import person_masks, seg_union

    faces = SMPL(paths.smpl_dir() / "SMPL_NEUTRAL.pkl").faces.cpu().numpy()
    out = Path(a.out)
    (out / "dense").mkdir(parents=True, exist_ok=True)
    for sub in ("cams", "depth", "pose", "mask", "rgb", "intrinsics", "vipe"):
        if (Path(a.vipe) / sub).exists() and not (out / sub).exists():
            os.symlink(Path(a.vipe) / sub, out / sub)
    L = json.loads(Path(a.list).read_text())
    steps = [int(s) for s in a.steps.split(",")]
    # --geom 0: COLMAP's photometric pass only (its geometric pass and filter off); our consistency filter below is the check
    kind, geom = ("geometric", "true") if a.geom else ("photometric", "false")
    for name in a.only.split(","):
        dst = out / "dense" / f"{name}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        D = dict(np.load(Path(a.vipe) / "dense" / f"{name}.npz"))
        T = np.load(Path(a.vipe) / "cams" / f"{name}.npz")["T_world_cam"].astype(np.float64)
        fr = D["frames"].astype(int)
        h, w = D["depth"].shape[1:]
        ks = [k for k in range(0, len(fr), a.stride) if fr[k] < len(T)]
        if a.max_frames:
            ks = ks[: a.max_frames]
        imgs = L[name]["images"]
        H0, W0 = cv2.imread(imgs[int(fr[ks[0]])]).shape[:2]
        sc = a.size / max(H0, W0)
        W2, H2 = int(round(W0 * sc / 2)) * 2, int(round(H0 * sc / 2)) * 2
        wd = Path(a.work) / name
        names = [f"f{int(fr[k]):05d}.jpg" for k in ks]
        cons0 = [int(x) for x in a.cons_steps.split(",")]
        bmax = max((float(np.linalg.norm(T[int(fr[ks[i]])][:3, 3] - T[int(fr[ks[i + o]])][:3, 3]))
                    for i in range(len(ks)) for o in cons0 if i + o < len(ks)), default=0.0)
        zmed = float(np.median(D["depth"][ks][D["depth"][ks] > 0].astype(np.float32)))
        no_parallax = bmax < a.min_parallax * zmed  # not even the best pair passes the floor at the median depth: all fallback
        if no_parallax:  # nothing to triangulate: the ViPE run unchanged, every frame
            print(f"[mvs] {name}: max neighbour baseline {bmax * 100:.2f} cm < {a.min_parallax} x median depth {zmed:.2f} m: "
                  f"no parallax, the ViPE depth unchanged", flush=True)
            D.update(mvs_share=np.zeros(len(fr), np.float32), mvs_ratio=np.ones(len(fr), np.float32),
                     mvs=f"no parallax (max neighbour baseline {bmax:.4f} m < {a.min_parallax} x {zmed:.2f} m): ViPE depth unchanged")
            np.savez(dst, **D)
            continue
        elif not (a.reuse and all((wd / "dense" / "stereo" / "depth_maps" / f"{nm}.{kind}.bin").exists() for nm in names)):
            shutil.rmtree(wd, ignore_errors=True)
            (wd / "images").mkdir(parents=True)
            (wd / "sparse").mkdir()
            cams, ims = [], []
            for i, k in enumerate(ks):
                f = int(fr[k])
                im = cv2.resize(cv2.imread(imgs[f]), (W2, H2), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(wd / "images" / f"f{f:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                K = D["K"][k].astype(np.float64).copy()
                K[0] *= W2 / w
                K[1] *= H2 / h
                cams.append(f"{i + 1} PINHOLE {W2} {H2} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}")
                Rcw = T[f][:3, :3].T
                tcw = -Rcw @ T[f][:3, 3]
                q = Rotation.from_matrix(Rcw).as_quat()  # x y z w
                ims.append(f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {tcw[0]:.9f} {tcw[1]:.9f} {tcw[2]:.9f} {i + 1} "
                           f"f{f:05d}.jpg\n\n")
            (wd / "sparse" / "cameras.txt").write_text("\n".join(cams) + "\n")
            (wd / "sparse" / "images.txt").write_text("".join(ims))
            (wd / "sparse" / "points3D.txt").write_text("")
            log = open(wd / "colmap.log", "w")
            subprocess.run([a.colmap, "image_undistorter", "--image_path", str(wd / "images"), "--input_path", str(wd / "sparse"),
                            "--output_path", str(wd / "dense"), "--output_type", "COLMAP"], check=True, stdout=log, stderr=log)
            cfg = []
            for i, nm in enumerate(names):
                src = [names[i + o] for st in steps for o in (st, -st) if 0 <= i + o < len(names)]
                cfg += [nm, ", ".join(src)]
            (wd / "dense" / "stereo" / "patch-match.cfg").write_text("\n".join(cfg) + "\n")
            subprocess.run([a.colmap, "patch_match_stereo", "--workspace_path", str(wd / "dense"), "--workspace_format", "COLMAP",
                            "--PatchMatchStereo.geom_consistency", geom, "--PatchMatchStereo.depth_min", str(a.dmin),
                            "--PatchMatchStereo.depth_max", str(a.dmax), "--PatchMatchStereo.max_image_size", str(a.size),
                            "--PatchMatchStereo.filter", geom, "--PatchMatchStereo.num_iterations", "3"],  # COLMAP: 5
                           check=True, stdout=log, stderr=log)
        t_mvs = time.time() - t0
        # people masks at the dense resolution
        dyn = seg_union(D["dyn"], a.seg_dir, name) if a.seg_dir else D["dyn"]
        r = np.load(Path(a.recon) / f"{name}.npz")
        body = person_masks(r["verts"].astype(np.float32), r["T_world_cam"].astype(np.float64),
                            D["K"][np.clip(np.arange(len(r["T_world_cam"])), 0, len(D["K"]) - 1)].astype(np.float64), (h, w), faces, 4)
        G = []
        for i, k in enumerate(ks):
            gp = wd / "dense" / "stereo" / "depth_maps" / f"{names[i]}.{kind}.bin"
            m = read_colmap_array(gp) if gp.exists() else np.zeros((H2, W2), np.float32)
            m = np.where((m > a.dmin) & (m < a.dmax), m, 0.0).astype(np.float32)
            G.append(m)
        # COLMAP's geometric maps are dense; keep a pixel only if >= cons_min neighbour keyframes see the same surface
        vv2, uu2 = np.mgrid[0:H2, 0:W2]
        Kc = []
        for k in ks:
            K = D["K"][k].astype(np.float64).copy()
            K[0] *= W2 / w
            K[1] *= H2 / h
            Kc.append(K)
        cons = [int(x) for x in a.cons_steps.split(",")]
        F = []
        for i, k in enumerate(ks):
            z = G[i]
            m = z > 0
            cnt = np.zeros((H2, W2), np.int16)
            X = np.stack([(uu2[m] - Kc[i][0, 2]) / Kc[i][0, 0] * z[m], (vv2[m] - Kc[i][1, 2]) / Kc[i][1, 1] * z[m], z[m]], 1)
            Xw = X @ T[int(fr[k])][:3, :3].T + T[int(fr[k])][:3, 3]
            for o in [o for c in cons for o in (c, -c)]:
                j = i + o
                if not 0 <= j < len(ks):
                    continue
                Tj = np.linalg.inv(T[int(fr[ks[j]])])
                Xj = Xw @ Tj[:3, :3].T + Tj[:3, 3]
                zj = Xj[:, 2]
                ok = zj > 0.05
                u = np.round(Kc[j][0, 0] * Xj[:, 0] / np.maximum(zj, 1e-6) + Kc[j][0, 2]).astype(int)
                v = np.round(Kc[j][1, 1] * Xj[:, 1] / np.maximum(zj, 1e-6) + Kc[j][1, 2]).astype(int)
                ok &= (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
                zo = np.zeros(len(zj), np.float32)
                zo[ok] = G[j][v[ok], u[ok]]
                base = float(np.linalg.norm(T[int(fr[k])][:3, 3] - T[int(fr[ks[j]])][:3, 3]))
                agree = ok & (zo > 0) & (np.abs(zo - zj) < a.cons_rel * zj) & (base >= a.min_parallax * z[m])
                c1 = np.zeros(int(m.sum()), np.int16)
                c1[agree] = 1
                cnt[m] += c1
            F.append(np.where(cnt >= a.cons_min, z, 0.0).astype(np.float32))
        dep, share, ratio, mmask = [], [], [], []
        k2 = max(1, int(round(H2 / h)))
        for i, k in enumerate(ks):
            f = int(fr[k])
            md = block_median(F[i], k2)
            md = cv2.resize(md, (w, h), interpolation=cv2.INTER_NEAREST) if md.shape != (h, w) else md
            ppl = dyn[k] | (body[f] if f < len(body) else False)
            vz = D["depth"][k].astype(np.float32)
            ok = (md > 0) & ~ppl
            both = ok & (vz > 0)
            st = ~ppl & (vz > 0)
            sh = float(ok[st].mean()) if st.any() else 0.0
            # the fallback's scale from MVS only where MVS covers >= fill_share of the frame (a few matched pixels give no scale)
            rt = float(np.median(md[both] / vz[both])) if sh >= a.fill_share and both.sum() > 200 else 1.0
            d = np.where(ok, md, vz * rt)
            dep.append(np.clip(d, 0, a.dmax).astype(np.float16))
            mmask.append(ok)
            share.append(sh)
            ratio.append(rt)
        # frames between keyframes: where the nearest keyframe has (almost) no MVS (no parallax there), ViPE's own frames come
        # back (x that keyframe's ratio), so a stretch without parallax is the ViPE run as before; elsewhere keyframes only
        kidx = np.array(ks)
        out_k, out_d, out_s, out_r, out_m = list(ks), list(dep), list(share), list(ratio), list(mmask)
        kset = set(ks)
        for k in range(len(fr)):
            if k in kset or fr[k] >= len(T):
                continue
            i = int(np.abs(kidx - k).argmin())
            if share[i] < a.fill_share:
                out_k.append(k)
                out_d.append(np.clip(D["depth"][k].astype(np.float32) * ratio[i], 0, a.dmax).astype(np.float16))
                out_s.append(0.0)
                out_r.append(ratio[i])
                out_m.append(np.zeros((h, w), bool))
        o = np.argsort(out_k)
        ks2 = [out_k[j] for j in o]
        res = {kk: (v[ks2] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(fr) else v) for kk, v in D.items()}
        res.update(depth=np.stack([out_d[j] for j in o]), mvs_share=np.array([out_s[j] for j in o], np.float32),
                   mvs_ratio=np.array([out_r[j] for j in o], np.float32),
                   mvs_mask=np.stack([out_m[j] for j in o]),  # per pixel: the MVS depth (True) or the ViPE fallback
                   mvs=f"COLMAP patch_match_stereo ({'geom consistency + filter' if a.geom else 'photometric only'}), ViPE cameras fixed, stride {a.stride}, {W2}x{H2}, "
                       f"sources +-{a.steps} keyframes, depth [{a.dmin}, {a.dmax}], consistency >= {a.cons_min} of +-{a.cons_steps} within "
                       f"{a.cons_rel} at baseline >= {a.min_parallax} x depth; fallback ViPE x per-frame ratio")
        np.savez(dst, **res)
        print(f"[mvs] {name}: {len(ks)} keyframes + {len(ks2) - len(ks)} ViPE frames kept (no MVS nearby), MVS {t_mvs:.0f} s, "
              f"total {time.time() - t0:.0f} s; MVS share of static pixels "
              f"median {np.median(share):.2f} (p10 {np.percentile(share, 10):.2f}); fallback ratio median {np.median(ratio):.3f} "
              f"(p10 {np.percentile(ratio, 10):.3f}, p90 {np.percentile(ratio, 90):.3f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
