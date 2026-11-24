"""fah-video: one video -> every FAH output per frame (a results pickle) and one render of all of them: a grid of panels
(default 2 x 3: input | 2D joints | MANO mesh / part map + instance masks | per-vertex contact | palm-side contact per hand,
or the world panel with --world) over the per-hand contact timeline, H.264 without audio, plus a poster still.

    fah-video clip.mp4 -o out/clip.mp4                        # FAH-L, video mode, mesh fit, parts, masks, contact
    fah-video clip.mp4 -o out/clip.mp4 --model fah-s --panels input,joints,mesh,contact
    fah-video clip.mp4 --no-render --results out/clip.pkl     # results only
    fah-video clip.mp4 -o out/clip.mp4 --world realtime       # + hands in world space (camera model, docs/world.md)
    fah-video clip.mp4 -o out/clip.mp4 --load out/clip.pkl    # render saved results again (no model run)

The model runs as fah.FAH(model, "video", mesh="mano", parts=True, det_masks=True, contact=True): the causal video
model with its video detector, the mesh fit, the part maps, the detector's instance masks, the contact head and the
video contact model, frame by frame in order. Results (pickle): video, fps, size (W, H), K, model, faces (MANO,
right hand), frames[i] = {index, latency (s), pred: one dict per hand with the fah.FAH outputs (is_right, score, bbox,
joints2d / joints3d, verts3d, dense surface, mask_det RLE, contact*, track_id, ...; parts_roi = (x0, y0, PNG bytes of the
part label map)), filled = a refiner gap fill (score 0, no contact / parts / mask)}; --world: + T_world_cam (N, 4, 4) and
joints3d_world / verts3d_world per hand. Panels show the hands with score >= --min_score (the fah CLI rule). Rendering
needs ffmpeg on the PATH.
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

LABELS = {"fah-s": "FAH-S", "fah-b": "FAH-B", "fah": "FAH-L", "fah-h": "FAH-H"}
BACKBONES = {"s": "fah-s", "b": "fah-b", "l": "fah", "h": "fah-h"}
PANELS = ("input", "joints", "mesh", "parts", "contact", "palm", "world")
KEEP = ("is_right", "score", "bbox", "joints2d", "joints3d", "verts3d", "verts3d_dense", "mesh_param", "tip_ext", "joints2d_head",
        "joints3d_head", "surface2d_head", "surface3d_head", "surface_visible", "mask_rle", "mask_det", "contact", "contact_dense",
        "contact_score", "contact_fingers", "contact_k", "contact_score_k", "contact_video", "contact_score_video",
        "contact_flag_video", "track_id", "verts3d_mhr", "verts3d_mhr_mano", "joints3d_mhr_fit", "joints3d_mhr_rig", "mhr_params")
G = {}  # render state, shared with the forked workers


def parser():
    from .api import DEFAULT_SAM3D, DEFAULT_WEIGHTS

    p = argparse.ArgumentParser(prog="fah-video", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("-o", "--out", default="", help="render (.mp4; default <video name>_fah.mp4 next to the results)")
    p.add_argument("--results", default="", help="results pickle (default: the render path with .pkl)")
    p.add_argument("--load", default="", help="render this results pickle instead of running the model")
    p.add_argument("--no-render", dest="render", action="store_false", help="results only")
    p.add_argument("--model", default="fah", choices=list(LABELS) + list(BACKBONES),
                   help="fah-s / fah-b / fah (FAH-L, default) / fah-h (or s / b / l / h)")
    p.add_argument("--panels", default="", help=f"comma-separated tiles, 3 per row, from {', '.join(PANELS)} (default "
                   "input,joints,mesh,parts,contact,palm; with --world the last is world)")
    p.add_argument("--world", default="", choices=["", "realtime", "offline"],
                   help="+ hands in world space: realtime = the causal camera model; offline = + masked DROID-SLAM and Metric3D "
                        "(the offline extra, not real time); both need their environments (docs/world.md)")
    p.add_argument("--min-score", dest="min_score", type=float, default=0.3, help="display threshold (the results keep every hand)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--mesh", default="mano", choices=["mano", "both"], help="both: + the MHR-Hand mesh in the results")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--sam3d", default=DEFAULT_SAM3D)
    p.add_argument("--mano_dir", default=None, help="MANO_RIGHT.pkl / MANO_LEFT.pkl (default: $MANO_DIR)")
    p.add_argument("--trt", action="store_true", help="the TensorRT engines in <weights>/trt (fah-build-trt)")
    p.add_argument("--trt_dir", default="", help="TensorRT engines in this folder")
    p.add_argument("--focal", type=float, default=0, help="focal length in px (default: the image diagonal)")
    p.add_argument("--fov", type=float, default=0, help="horizontal field of view in degrees (instead of --focal)")
    p.add_argument("--K", default="", help="fx,fy,cx,cy (instead of --focal / --fov)")
    p.add_argument("--max_frames", type=int, default=0, help="stop after this many frames (0: all)")
    p.add_argument("--identity", action="store_true", help="the opt-in video identity rules (fah.api.IDENTITY)")
    p.add_argument("--contact_smooth", type=int, default=1, choices=(1, 2), help="2 = the contact flag on the mean of 2 "
                   "consecutive frames (fewer switches, about +1 frame latency; fah.api.CONTACT_SMOOTH_H)")
    p.add_argument("--fps", type=float, default=0, help="frame rate of the video model's filters and the render (default: the video's)")
    r = p.add_argument_group("render")
    r.add_argument("--tile", type=int, default=640, help="tile width (px)")
    r.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better)")
    r.add_argument("--poster", type=float, default=-1, help="poster still at this time (s; default the first frame)")
    r.add_argument("--jobs", type=int, default=0, help="render processes (default: CPU count, at most 32; 1 on macOS)")
    r.add_argument("--tmp", default="", help="scratch folder for the frames (default: a temporary one, removed)")
    r.add_argument("--no_names", action="store_true", help="generic panel titles and legend (no model / method names)")
    w = p.add_argument_group("world (docs/world.md)")
    t = os.environ.get("THIRDPARTY", "thirdparty")
    w.add_argument("--dpvo_dir", default=os.path.join(t, "DPVO"))
    w.add_argument("--dpvo_python", default=sys.executable)
    w.add_argument("--depth_python", default=sys.executable)
    w.add_argument("--droid_dir", default=os.path.join(t, "DROID-SLAM"))
    w.add_argument("--droid_weights", default="")
    w.add_argument("--metric3d_dir", default=os.path.join(t, "Metric3D"))
    w.add_argument("--metric3d_weights", default="")
    w.add_argument("--slam_python", default=sys.executable)
    return p


def intrinsics(a, h, w):
    from . import default_cam_int, fov_to_cam_int

    if a.K:
        fx, fy, cx, cy = (float(x) for x in a.K.split(","))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    if a.focal > 0:
        return np.array([[a.focal, 0, w / 2], [0, a.focal, h / 2], [0, 0, 1]], np.float32)
    if a.fov > 0:
        return fov_to_cam_int(a.fov, h, w)
    return default_cam_int(h, w)


def frames_of(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    i = 0
    while max_frames <= 0 or i < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        yield i, img
        i += 1
    cap.release()


def export(h):
    """One hand of fah.FAH as stored in the results (parts_roi as a PNG)."""
    o = {k: h[k] for k in KEEP if k in h}
    if "parts_roi" in h:
        x0, y0, lab = h["parts_roi"]
        o["parts_roi"] = (int(x0), int(y0), cv2.imencode(".png", lab)[1].tobytes())
    o["filled"] = float(h.get("score", 1.0)) == 0.0
    return o


def run(a):
    from .api import FAH, IDENTITY, trt_dir

    cap = cv2.VideoCapture(a.video)
    fps, W, H = a.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    model = FAH(a.model, "video", weights=a.weights, sam3d=a.sam3d, device=a.device, mano_dir=a.mano_dir, fps=fps, mesh=a.mesh,
                trt_dir=a.trt_dir or (trt_dir(a.weights) if a.trt else ""), parts=True, contact=True, det_masks=True,
                contact_smooth=a.contact_smooth, **(IDENTITY if a.identity else {}))
    model.reset()
    K = intrinsics(a, H, W)
    name = os.path.splitext(os.path.basename(a.video))[0]
    work = os.path.join(os.path.dirname(os.path.abspath(a.results)), f"{name}_world")
    frames, raw = [], []
    t0 = time.perf_counter()
    for i, img in frames_of(a.video, a.max_frames):
        t = time.perf_counter()
        hands = model(img, K)
        frames.append(dict(index=i, pred=[export(h) for h in hands], latency=time.perf_counter() - t))
        if a.world:  # the camera stages read the frames and the full hands (docs/world.md)
            raw.append(hands)
            d = os.path.join(work, "frames", name)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lat = np.median([f["latency"] for f in frames[10:] or frames] or [0])  # short clips: every frame
    print(f"{name}: {len(frames)} frames, median {1 / max(lat, 1e-9):.1f} fps ({LABELS[a.model]}; "
          f"{len(frames) / max(time.perf_counter() - t0, 1e-9):.1f} fps incl. decode)", flush=True)
    res = dict(video=os.path.basename(a.video), fps=fps, size=(W, H), K=K, model=a.model, faces=np.asarray(model.faces),
               frames=frames, **({"identity": True} if a.identity else {}), **({"contact_smooth": 2} if a.contact_smooth == 2 else {}))
    if a.world:
        res.update(cameras(a, model, work, name, raw, K, (H, W)))
    return res


def cameras(a, model, work, name, raw, K, hw):
    """The fah CLI's world stage: T_world_cam per frame (the camera model or the offline camera) and the hands placed."""
    from .world import estimate_cameras, load_cameras

    d = os.path.join(work, "frames", name)
    np.save(os.path.join(d, "K.npy"), np.asarray(K, np.float64))
    open(os.path.join(d, "done"), "w").close()
    os.makedirs(os.path.join(work, "hands"), exist_ok=True)
    with open(os.path.join(work, "hands", f"{name}.pkl"), "wb") as f:
        pickle.dump(dict(clip=name, method=model.name, video=True, frames=[
            dict(key=f"{i:05d}", index=i, K=K, pred=h, image_size=hw) for i, h in enumerate(raw)]), f)
    if a.world == "offline":
        from .world import estimate_cameras_offline

        path = estimate_cameras_offline(
            work, name, model.camera_head, a.dpvo_dir, a.droid_dir, a.droid_weights or os.path.join(a.droid_dir, "droid.pth"),
            a.metric3d_dir, a.metric3d_weights or os.path.join(a.metric3d_dir, "weight", "metric_depth_vit_large_800k.pth"),
            a.dpvo_python, a.depth_python, a.slam_python, os.environ.get("MANO_DIR"))
    else:
        path = estimate_cameras(work, name, model.camera_head, a.dpvo_dir, a.dpvo_python, a.depth_python, os.environ.get("MANO_DIR"))
    cams = load_cameras(path)
    T = np.stack([cams[i] for i in sorted(cams)])
    k = float(np.load(path)["k_ours_over_slam"]) if a.world == "offline" else None
    if a.world == "offline":  # the offline camera's SLAM: its rotation over the clip and its keyframes
        slam = np.load(os.path.join(work, "slam", f"{name}.npz"))
        rot, kf = rotation_extent(slam["T_world_cam"]), int(len(slam["tstamp"]))
    else:
        rot, kf = rotation_extent(T), None
    tr = translation_extent(np.load(os.path.join(work, "cams", f"{name}.npz"))["T_world_cam"])  # the real-time camera (metric)
    static = static_camera(rot, a.world, tr)
    if static:  # docs/world.md: world mode adds nothing on a static camera
        print(f"fah-video: warning: the camera looks static ({rot:.1f} deg of rotation, {100 * tr:.0f} cm of translation over the "
              "clip): world mode adds nothing here and can drift; use the camera-frame outputs", file=sys.stderr, flush=True)
    return dict(T_world_cam=T, camera=a.world, static_camera=static, camera_rotation_deg=rot, camera_translation_m=tr,
                **({"slam_keyframes": kf, "k_ours_over_slam": k} if k is not None else {}))


# static-camera warning (docs/world.md): the camera's largest rotation from the first frame (the offline camera's SLAM
# trajectory / the real-time camera model's). Dev clips with camera motion: >= 6.7 deg (HOT3D train, 36 clips), >= 7.3
# (EgoDex part 2, 56), >= 8.3 (EgoStandard dev, 80) offline; >= 6.7 / 7.8 / 8.0 in real time. Tripod clips (7): <= 3.0
# offline, <= 5.3 in real time. The thresholds sit in those gaps. A camera that slides without turning is not static:
# the real-time camera's translation must also stay below STATIC_TRANSLATION (tripod clips <= 0.13 m of drift; every dev
# clip with camera motion turns: those under 10 deg move <= 0.14 m).
STATIC_ROT = {"offline": 5.0, "realtime": 6.0}
STATIC_TRANSLATION = 0.25


def rotation_extent(T):
    """Largest camera rotation (deg) relative to the first frame of a trajectory T (N, 4, 4) (scale-free)."""
    R = np.asarray(T, np.float64)[:, :3, :3]
    R = R[np.isfinite(R).all((1, 2))]
    if not len(R):
        return 0.0
    c = (np.einsum("ij,nij->n", R[0], R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).max())


def translation_extent(T):
    """Largest camera-centre distance (m) from the first frame of a trajectory T (N, 4, 4)."""
    c = np.asarray(T, np.float64)[:, :3, 3]
    c = c[np.isfinite(c).all(1)]
    return float(np.linalg.norm(c - c[0], axis=1).max()) if len(c) else 0.0


def static_camera(rot, camera="offline", translation=None):
    """True when the camera barely turns over the clip (rot < STATIC_ROT[camera] deg) and does not slide either (the
    real-time camera's translation < STATIC_TRANSLATION m, when given). A static camera in front of a person who moves a
    lot can still escape it (a selfie video: 9-17 deg, both cameras follow the person)."""
    return rot < STATIC_ROT[camera] and (translation is None or translation < STATIC_TRANSLATION)


def place_all(res):
    from .world import place

    T = res["T_world_cam"]
    for fr in res["frames"]:
        if fr["index"] < len(T):
            place(fr["pred"], T[fr["index"]])


# ------------------------------------------------------------------------------------------------------------- render
TITLES = dict(input=("Input video", "Input"), joints=("{}: 2D joints", "2D keypoints"), mesh=("{}: MANO mesh", "3D hand mesh"),
              parts=("{}: parts + instance masks", "Hand parts + masks"), contact=("{}: contact", "Contact"),
              palm=("{}: contact, palm side", "Contact (palm side)"), world=("{}: world ({})", "World space ({})"))


def title(panel):
    named, generic = TITLES[panel]
    if G["label"] is None:
        return generic.format(G["camera"]) if panel == "world" else generic
    return named.format(G["label"], G["camera"]) if panel == "world" else named.format(G["label"])


def shown(fr):
    return [h for h in fr["pred"] if h["score"] >= G["min_score"]]


def prob_of(h):
    c = h.get("contact_video")
    return np.asarray(c if c is not None else h["contact"], np.float64)


def palm_tile(hands):
    from .demo_render import bar, contact_colors_ramp, kabsch, render_view

    tw, th = G["tw"], G["th"]
    out = np.full((th, tw, 3), 255, np.uint8)
    q = min(tw // 2, th - 34)
    for k, side in enumerate((True, False)):
        x0, y0 = k * (tw // 2) + (tw // 2 - q) // 2, 34 + (th - 34 - q) // 2
        hs = [h for h in hands if bool(h["is_right"]) == side and h.get("verts3d") is not None and h.get("contact") is not None]
        nm = "right" if side else "left"
        if not hs or side not in G["view"]:
            cv2.putText(out, f"{nm}: not detected", (x0 + 10, y0 + q // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
            continue
        h = max(hs, key=lambda h: h["score"])
        v = np.asarray(h["verts3d"], np.float64)
        pal, rest = G["palm"][side], G["rest"][side]
        x = (v - v[pal].mean(0)) @ kabsch(v[pal], rest[pal]).T
        out[y0:y0 + q, x0:x0 + q] = render_view(x, contact_colors_ramp(prob_of(h)), G["faces"][side], G["view"][side], G["span"][side], q)
        cv2.putText(out, nm, (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    return bar(out, title("palm"))


def render_frame(t):
    from .demo_render import bar, contact_colors_ramp, joints_and_parts, render_contact_smooth
    from .vis import LEFT_COLOR, RIGHT_COLOR

    img = cv2.imread(os.path.join(G["tmp"], f"in{t:06d}.jpg"))
    hands = shown(G["frames"][t])
    tw, th, want = G["tw"], G["th"], G["panels"]
    K = np.asarray(G["K"], np.float64).copy()
    K[:2] *= tw / img.shape[1]
    small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    rs = lambda x: cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)
    tiles = {}
    if "joints" in want or "parts" in want:
        j2d, pm = joints_and_parts(img, hands, tw)
        tiles.update(joints=bar(rs(j2d), title("joints")), parts=bar(rs(pm), title("parts")))
    if "mesh" in want:
        meshes = [(np.asarray(h["verts3d"], np.float64), np.tile(RIGHT_COLOR if h["is_right"] else LEFT_COLOR, (778, 1))) for h in hands
                  if h.get("verts3d") is not None]
        tiles["mesh"] = bar(render_contact_smooth(small, meshes, G["faces"][True], K) if meshes else small.copy(), title("mesh"))
    if "contact" in want:
        cont = [(np.asarray(h["verts3d"], np.float64), contact_colors_ramp(prob_of(h))) for h in hands
                if h.get("verts3d") is not None and h.get("contact") is not None]
        tiles["contact"] = bar(render_contact_smooth(small, cont, G["faces"][True], K) if cont else small.copy(), title("contact"))
    if "palm" in want:
        tiles["palm"] = palm_tile(hands)
    if "world" in want:
        tiles["world"] = bar(G["world"].draw(t), title("world"))
    tiles["input"] = bar(small, title("input"))
    row = [tiles[k] for k in want] + [np.full((th, tw, 3), 255, np.uint8)] * (-len(want) % G["cols"])
    grid = np.vstack([np.hstack(row[i:i + G["cols"]]) for i in range(0, len(row), G["cols"])])
    if G["strip"] is not None:
        st = G["strip"].copy()
        cv2.line(st, (G["X"](t), 4), (G["X"](t), 230), (40, 40, 220), 2)
        grid = np.vstack([grid, st])
    cv2.imwrite(os.path.join(G["tmp"], f"out{t:06d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return t


def render(res, a, out):
    from bench.common import ManoLayer

    from .api import FAH_HOME
    from .demo_render import WorldTile, best_view, kabsch, timeline, wrists

    if shutil.which("ffmpeg") is None:
        raise SystemExit("fah-video: rendering needs ffmpeg on the PATH (or --no-render)")
    frames = res["frames"]
    n = len(frames)
    fps = float(res.get("fps") or 30.0)
    panels = [p for p in (a.panels or ("input,joints,mesh,parts,contact," + ("world" if "T_world_cam" in res else "palm"))).split(",") if p]
    bad = [p for p in panels if p not in PANELS]
    if bad or ("world" in panels and "T_world_cam" not in res):
        raise SystemExit(f"fah-video: unknown panels {bad} (from {PANELS})" if bad else "fah-video: the world panel needs --world")
    tmp = a.tmp or tempfile.mkdtemp(prefix="fah_video_")
    os.makedirs(tmp, exist_ok=True)
    W = H = 0
    for i, img in frames_of(a.video, n):
        H, W = img.shape[:2]
        cv2.imwrite(os.path.join(tmp, f"in{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    md = a.mano_dir or os.environ.get("MANO_DIR") or os.path.join(FAH_HOME, "mano")
    mano = ManoLayer(md)
    rest = {s: mano(s, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))[1] for s in (True, False)}
    palm = {s: np.nonzero(mano.layers[s].lbs_weights.numpy().argmax(1) == 0)[0] for s in (True, False)}
    span = {s: np.ptp(rest[s], 0).max() * 1.2 for s in (True, False)}
    faces = {s: mano.faces[s] for s in (True, False)}
    tw = a.tile
    K = res.get("K")
    if K is None:  # results without intrinsics: the release default
        from . import default_cam_int

        K = default_cam_int(H, W)
    G.update(frames=frames, tmp=tmp, K=K, min_score=a.min_score, label=None if a.no_names else LABELS.get(res.get("model", "fah"), "FAH"), tw=tw,
             th=int(round(tw * H / W / 2)) * 2, rest=rest, palm=palm, span=span, faces=faces, panels=panels, cols=min(3, len(panels)))
    # timeline: per side the best shown hand per frame; palm view fixed per clip (the frame with the most contact)
    pred, flag, flag_img = (np.full((n, 2), np.nan) for _ in range(3))
    view, best = {}, {}
    for t, fr in enumerate(frames):
        for c, side in enumerate((True, False)):
            hs = [h for h in shown(fr) if bool(h["is_right"]) == side and h.get("contact") is not None]
            if not hs:
                continue
            h = max(hs, key=lambda h: h["score"])
            vid = h.get("contact_video") is not None
            pred[t, c] = h["contact_score_video"] if vid else h["contact_score"]
            flag[t, c] = float(h["contact_flag_video"]) if vid else float(h["contact_score"] > 0.5)
            flag_img[t, c] = float(h["contact_score"] > 0.5)
            if h.get("verts3d") is not None and (side not in best or prob_of(h).sum() > best[side][0]):
                best[side] = (prob_of(h).sum(), h)
    for side, (_, h) in best.items():
        v = np.asarray(h["verts3d"], np.float64)
        x = (v - v[palm[side]].mean(0)) @ kabsch(v[palm[side]], rest[side][palm[side]]).T
        view[side] = best_view(x, prob_of(h), faces[side])
    G["view"] = view
    G["camera"] = "offline camera" if res.get("camera") == "offline" else "camera model, real time"
    G["world"] = WorldTile(np.asarray(res["T_world_cam"], np.float64), wrists(frames, a.min_score), (tw, G["th"]), "",
                           every=60) if "world" in panels else None  # the camera is named in the tile title
    G["strip"], G["X"] = timeline(pred, flag, np.full((n, 2), np.nan), G["cols"] * tw, fps, flag_img=flag_img, names=not a.no_names) \
        if {"contact", "palm"} & set(panels) else (None, None)
    jobs = a.jobs or (1 if sys.platform == "darwin" else min(os.cpu_count() or 1, 32))
    if jobs > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(jobs) as pool:
            for k, _ in enumerate(pool.imap_unordered(render_frame, range(n), chunksize=4)):
                if k % 500 == 0:
                    print(f"render {k} / {n}", flush=True)
    else:
        for t in range(n):
            render_frame(t)
    poster = min(range(n), key=lambda t: abs(frames[t]["index"] - a.poster * fps)) if a.poster >= 0 else 0
    cv2.imwrite(os.path.splitext(out)[0] + "_poster.jpg", cv2.imread(os.path.join(tmp, f"out{poster:06d}.jpg")), [cv2.IMWRITE_JPEG_QUALITY, 92])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "out%06d.jpg"), "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", "-map_metadata", "-1", out], check=True)
    if not a.tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{n} frames -> {out}", flush=True)


def main(argv=None):
    a = parser().parse_args(argv)
    a.model = BACKBONES.get(a.model, a.model)
    assert os.path.exists(a.video), f"{a.video} not found"
    stem = os.path.splitext(os.path.basename(a.video))[0]
    a.out = a.out or os.path.join(os.path.dirname(a.results) if a.results else ".", f"{stem}_fah.mp4")
    a.results = a.results or os.path.splitext(a.out)[0] + ".pkl"
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.load:
        res = pickle.load(open(a.load, "rb"))
    else:
        res = run(a)
        if "T_world_cam" in res:
            place_all(res)
        with open(a.results, "wb") as f:
            pickle.dump(res, f)
        print(f"results -> {a.results}", flush=True)
    if a.render:
        render(res, a, a.out)


if __name__ == "__main__":
    main()
