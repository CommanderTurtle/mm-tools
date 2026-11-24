"""The G1 tracking clip sets (FLAT_CLIPS: the flat-floor test set with ground-truth SMPL; CLIPS, DEV_CLIPS and
FLAT_DEV_CLIPS: further clips used while designing the retargeter; the 25 test clips are not held out from that design
either) and the RICH / MOYO clip loaders used by
bench/sonic_motion.py (load_sequence). Dataset roots under $FAB_DATA (default $FAB_HOME/data): 3dpw/, emdb/, rich/,
moyo_clips/.

  name -> dataset, sequence, first frame, frames, input fps, category
  3DPW-test / EMDB  the datasets' own videos (30 fps)
  RICH val          data/rich/images/val/<seq>/<cam>: every 3rd frame of the 30 fps capture (10 fps, half resolution),
                    K / extrinsics from the scan calibration, SMPL fits from data/rich/smpl_fits/<split>
  MOYO val          data/moyo_clips/<name>: one camera's contiguous 30 fps frames + female SMPL fits
"""

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from fab import paths

DATA = Path(os.environ.get("FAB_DATA", paths.home() / "data")).expanduser()
# name: (dataset, sequence, start, frames, fps, category)
CLIPS = {
    "3dpw_downtown_runForBus_00": ("3dpw", "downtown_runForBus_00", 0, 600, 30, "walk / run"),
    "3dpw_downtown_walking_00": ("3dpw", "downtown_walking_00", 0, 600, 30, "walk / run"),
    "emdb_P3_31_outdoor_workout": ("emdb", "P3/31_outdoor_workout", 0, 600, 30, "workout"),
    "emdb_P5_42_indoor_dancing": ("emdb", "P5/42_indoor_dancing", 0, 600, 30, "dance"),
    "emdb_P6_49_outdoor_big_stairs_down": ("emdb", "P6/49_outdoor_big_stairs_down", 0, 600, 30, "stairs"),
    "emdb_P6_51_outdoor_dancing": ("emdb", "P6/51_outdoor_dancing", 0, 600, 30, "dance"),
    "emdb_P8_68_outdoor_handstand": ("emdb", "P8/68_outdoor_handstand", 1000, 600, 30, "inverted"),
    "3dpw_downtown_sitOnStairs_00": ("3dpw", "downtown_sitOnStairs_00", 0, 600, 30, "sit / ground"),
    "3dpw_downtown_upstairs_00": ("3dpw", "downtown_upstairs_00", 0, 600, 30, "stairs"),
    "3dpw_outdoors_fencing_01": ("3dpw", "outdoors_fencing_01", 0, 600, 30, "lunge / balance"),
    "3dpw_flat_packBags_00": ("3dpw", "flat_packBags_00", 0, 600, 30, "sit / ground"),
    "emdb_P8_69_outdoor_cartwheel": ("emdb", "P8/69_outdoor_cartwheel", 0, 600, 30, "inverted"),
    "emdb_P5_44_indoor_rom": ("emdb", "P5/44_indoor_rom", 0, 600, 30, "workout"),
    "emdb_P6_50_outdoor_workout": ("emdb", "P6/50_outdoor_workout", 0, 600, 30, "workout"),
    "emdb_P3_32_outdoor_soccer_warmup_a": ("emdb", "P3/32_outdoor_soccer_warmup_a", 0, 600, 30, "workout"),
    "emdb_P1_14_outdoor_climb": ("emdb", "P1/14_outdoor_climb", 0, 600, 30, "lunge / balance"),
    "emdb_P8_67_outdoor_workout_stretch": ("emdb", "P8/67_outdoor_workout_stretch", 0, 600, 30, "workout"),
    "rich_Pavallion_000_yoga1": ("rich", "Pavallion_000_yoga1", 0, 200, 10, "yoga / extreme"),
    "rich_ParkingLot1_005_pushup1": ("rich", "ParkingLot1_005_pushup1", 0, 200, 10, "sit / ground"),
    "rich_Pavallion_013_sidebalancerun": ("rich", "Pavallion_013_sidebalancerun", 0, 200, 10, "lunge / balance"),
    "rich_ParkingLot1_007_burpee2": ("rich", "ParkingLot1_007_burpee2", 0, 200, 10, "sit / ground"),
    "moyo_headstand": ("moyo", "moyo_headstand", 0, 600, 30, "inverted"),
    "moyo_upward_plank": ("moyo", "moyo_upward_plank", 0, 600, 30, "yoga / extreme"),
    "moyo_warrior3": ("moyo", "moyo_warrior3", 0, 600, 30, "yoga / extreme"),
    "moyo_supta_virasana": ("moyo", "moyo_supta_virasana", 0, 600, 30, "sit / ground"),
}
# development clips (disjoint from CLIPS): EMDB outside EMDB-1 and not in CLIPS, 3DPW train, RICH train, MOYO train
DEV_CLIPS = {
    "emdb_P1_15_outdoor_rom": ("emdb", "P1/15_outdoor_rom", 0, 600, 30, "workout"),
    "emdb_P3_28_outdoor_walk_lunges": ("emdb", "P3/28_outdoor_walk_lunges", 0, 600, 30, "lunge / balance"),
    "emdb_P3_29_outdoor_stairs_up": ("emdb", "P3/29_outdoor_stairs_up", 0, 600, 30, "stairs"),
    "emdb_P5_41_indoor_jogging_workout": ("emdb", "P5/41_indoor_jogging_workout", 0, 600, 30, "workout"),
    "3dpw_outdoors_climbing_00": ("3dpw", "outdoors_climbing_00", 0, 600, 30, "lunge / balance"),
    "3dpw_courtyard_capoeira_00": ("3dpw", "courtyard_capoeira_00", 0, 435, 30, "dance"),
    "3dpw_courtyard_laceShoe_00": ("3dpw", "courtyard_laceShoe_00", 0, 600, 30, "sit / ground"),
    "rich_ParkingLot2_016_pushup1": ("rich", "ParkingLot2_016_pushup1", 0, 200, 10, "sit / ground"),
    "rich_Pavallion_000_plankjack": ("rich", "Pavallion_000_plankjack", 0, 200, 10, "sit / ground"),
    "rich_ParkingLot2_015_burpeejump2": ("rich", "ParkingLot2_015_burpeejump2", 0, 200, 10, "sit / ground"),
    "rich_Pavallion_000_yoga2": ("rich", "Pavallion_000_yoga2", 0, 200, 10, "yoga / extreme"),
}
# the retargeter's test / dev clips are flat-floor only (no stairs, slopes, chairs, benches, walls, planters, climbing,
# carried objects). Excluded from the flat sets, with the reason (-> the terrain evaluation):
SCENE_CLIPS = {
    "emdb_P6_49_outdoor_big_stairs_down": "stairs down", "3dpw_downtown_sitOnStairs_00": "sits on stairs",
    "3dpw_downtown_upstairs_00": "stairs up", "emdb_P1_14_outdoor_climb": "climbs a wall / rocks",
    "emdb_P3_31_outdoor_workout": "dips / sits on a planter", "3dpw_flat_packBags_00": "handles bags on a table / sofa",
    "emdb_P6_50_outdoor_workout": "push-ups / steps on a bench", "emdb_P8_67_outdoor_workout_stretch": "stretches on a bench",
    "emdb_P8_68_outdoor_handstand": "handstand against a wall (feet on the wall)",
    "emdb_P3_29_outdoor_stairs_up": "stairs up (dev)", "3dpw_outdoors_climbing_00": "climbs rocks (dev)",
    "3dpw_courtyard_laceShoe_00": "sits on a bench (dev)",
}
# flat-floor test set (25; 3DPW-test, EMDB, RICH val, MOYO val): locomotion 6, dance 2 (all the flat dance there is),
# sports / dynamic 5, inverted 2, floor work 6, yoga / extreme 4
FLAT_CLIPS = {
    "3dpw_downtown_walking_00": CLIPS["3dpw_downtown_walking_00"],
    "3dpw_downtown_runForBus_00": CLIPS["3dpw_downtown_runForBus_00"],
    "3dpw_downtown_windowShopping_00": ("3dpw", "downtown_windowShopping_00", 0, 600, 30, "walk / run"),
    "3dpw_downtown_crossStreets_00": ("3dpw", "downtown_crossStreets_00", 0, 580, 30, "walk / run"),
    "emdb_P8_65_outdoor_walk_straight": ("emdb", "P8/65_outdoor_walk_straight", 0, 600, 30, "walk / run"),
    "emdb_P5_40_indoor_walk_big_circle": ("emdb", "P5/40_indoor_walk_big_circle", 0, 600, 30, "walk / run"),
    "emdb_P5_42_indoor_dancing": CLIPS["emdb_P5_42_indoor_dancing"],
    "emdb_P6_51_outdoor_dancing": CLIPS["emdb_P6_51_outdoor_dancing"],
    "3dpw_outdoors_fencing_01": CLIPS["3dpw_outdoors_fencing_01"],
    "emdb_P3_32_outdoor_soccer_warmup_a": CLIPS["emdb_P3_32_outdoor_soccer_warmup_a"],
    "emdb_P3_33_outdoor_soccer_warmup_b": ("emdb", "P3/33_outdoor_soccer_warmup_b", 0, 600, 30, "workout"),
    "emdb_P6_47_outdoor_run_slalom": ("emdb", "P6/47_outdoor_run_slalom", 0, 600, 30, "walk / run"),
    "emdb_P5_44_indoor_rom": CLIPS["emdb_P5_44_indoor_rom"],
    "emdb_P8_69_outdoor_cartwheel": CLIPS["emdb_P8_69_outdoor_cartwheel"],
    "moyo_headstand": CLIPS["moyo_headstand"],
    "rich_Pavallion_000_yoga1": CLIPS["rich_Pavallion_000_yoga1"],
    "rich_ParkingLot1_005_pushup1": CLIPS["rich_ParkingLot1_005_pushup1"],
    "rich_Pavallion_013_sidebalancerun": CLIPS["rich_Pavallion_013_sidebalancerun"],
    "rich_ParkingLot1_007_burpee2": CLIPS["rich_ParkingLot1_007_burpee2"],
    "rich_ParkingLot1_005_burpeejump1": ("rich", "ParkingLot1_005_burpeejump1", 0, 200, 10, "sit / ground"),
    "rich_ParkingLot1_004_pushup1": ("rich", "ParkingLot1_004_pushup1", 0, 200, 10, "sit / ground"),
    "rich_Pavallion_018_yoga1": ("rich", "Pavallion_018_yoga1", 0, 200, 10, "yoga / extreme"),
    "moyo_upward_plank": CLIPS["moyo_upward_plank"],
    "moyo_warrior3": CLIPS["moyo_warrior3"],
    "moyo_supta_virasana": CLIPS["moyo_supta_virasana"],
}
# flat-floor development clips (11; disjoint from FLAT_CLIPS: EMDB outside EMDB-1, 3DPW train, RICH train)
FLAT_DEV_CLIPS = {
    "emdb_P1_15_outdoor_rom": DEV_CLIPS["emdb_P1_15_outdoor_rom"],
    "emdb_P3_28_outdoor_walk_lunges": DEV_CLIPS["emdb_P3_28_outdoor_walk_lunges"],
    "emdb_P5_41_indoor_jogging_workout": DEV_CLIPS["emdb_P5_41_indoor_jogging_workout"],
    "emdb_P1_13_outdoor_long_walk": ("emdb", "P1/13_outdoor_long_walk", 0, 600, 30, "walk / run"),
    "3dpw_courtyard_capoeira_00": DEV_CLIPS["3dpw_courtyard_capoeira_00"],
    "3dpw_courtyard_dancing_01": ("3dpw", "courtyard_dancing_01", 0, 350, 30, "dance"),
    "3dpw_courtyard_rangeOfMotions_00": ("3dpw", "courtyard_rangeOfMotions_00", 0, 600, 30, "workout"),
    "rich_ParkingLot2_016_pushup1": DEV_CLIPS["rich_ParkingLot2_016_pushup1"],
    "rich_Pavallion_000_plankjack": DEV_CLIPS["rich_Pavallion_000_plankjack"],
    "rich_ParkingLot2_015_burpeejump2": DEV_CLIPS["rich_ParkingLot2_015_burpeejump2"],
    "rich_Pavallion_000_yoga2": DEV_CLIPS["rich_Pavallion_000_yoga2"],
}
FLAT_ALL = {**FLAT_CLIPS, **FLAT_DEV_CLIPS}
ALL_CLIPS = {**CLIPS, **DEV_CLIPS, **FLAT_CLIPS, **FLAT_DEV_CLIPS}
MOYO_SEQS = {"moyo_headstand": "220926_yogi_body_hands_03596_Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
             "moyo_upward_plank": "220926_yogi_body_hands_03596_Upward_Plank_Pose_or_Purvottanasana_-a",
             "moyo_warrior3": "220926_yogi_body_hands_03596_Warrior_III_Pose_or_Virabhadrasana_III_-b",
             "moyo_supta_virasana": "220926_yogi_body_hands_03596_Supta_Virasana_Vajrasana-b"}


def clip_source(name):
    """-> (dataset, sequence) for bench.sonic_motion.load_sequence."""
    if name in ALL_CLIPS:
        ds, seq = ALL_CLIPS[name][:2]
        return ds, (f"{seq}/{rich_camera(seq)}" if ds == "rich" else seq)
    if name.startswith("emdb_"):
        subj, seq = name[5:].split("_", 1)
        return "emdb", f"{subj}/{seq}"
    return "3dpw", name[5:]


def root_of(ds):
    return {"3dpw": DATA / "3dpw", "emdb": DATA / "emdb", "rich": DATA / "rich", "moyo": DATA / "moyo_clips"}[ds]


def rich_split(seq):
    return "val" if (DATA / "rich" / "images" / "val" / seq).is_dir() else "train"


@lru_cache(None)
def _rich_camera_params(seq, cam):
    """K (scaled to the stored half-resolution frames) and world -> camera (4, 4)."""
    raise NotImplementedError("reading RICH camera files is not part of this release")

    sizes = pickle.load(open(DATA / "rich" / "images" / f"{rich_split(seq)}_sizes.pkl", "rb"))
    Rc, tc, K0, _ = read_camera(DATA / "rich" / "scan_calibration" / seq.split("_")[0] / "calibration" /
                                f"{int(cam[4:]):03d}.xml")
    K = K0.copy()
    K[:2] *= sizes[(seq, cam[4:])][2]
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = Rc, tc
    return K, E, sizes[(seq, cam[4:])]


@lru_cache(None)
def rich_camera(seq):
    """The camera where the projected GT body is inside the frame most often, then the tallest median box."""
    fits = np.load(DATA / "rich" / "smpl_fits" / rich_split(seq) / f"{seq}.npz")
    sel = np.arange(0, len(fits["frame"]), 10)
    best = None
    sizes = pickle.load(open(DATA / "rich" / "images" / f"{rich_split(seq)}_sizes.pkl", "rb"))
    for cam in sorted(p.name for p in (DATA / "rich" / "images" / rich_split(seq) / seq).glob("cam_*")):
        if (seq, cam[4:]) not in sizes or not (DATA / "rich" / "scan_calibration" / seq.split("_")[0] / "calibration" /
                                               f"{int(cam[4:]):03d}.xml").exists():
            continue
        K, E, (H0, W0, s) = _rich_camera_params(seq, cam)
        b = _project_boxes(fits["pose"][sel], np.median(fits["betas"], 0), fits["transl"][sel],
                           np.repeat(E[None], len(sel), 0), K, "neutral")
        inside = ((b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= W0 * s) & (b[:, 3] <= H0 * s)).mean()
        key = (round(inside, 2), np.median(b[:, 3] - b[:, 1]))
        if best is None or key > best[0]:
            best = (key, cam)
    return best[1]


def _rich_meta(seq_cam):
    seq, cam = seq_cam.split("/")
    K, E, _ = _rich_camera_params(seq, cam)
    imgs = sorted((DATA / "rich" / "images" / rich_split(seq) / seq / cam).glob("*.jpg"))
    frames = np.array([int(p.stem) for p in imgs])
    fits = np.load(DATA / "rich" / "smpl_fits" / rich_split(seq) / f"{seq}.npz")
    row = {int(f): i for i, f in enumerate(fits["frame"])}
    idx = np.array([row.get(int(f), -1) for f in frames])
    return seq, imgs, frames, K, E, fits, idx


def _project_boxes(pose, betas, transl, ext, K, gender):
    import torch

    from bench.body_models import SMPL

    body = SMPL(paths.smpl_dir() / f"SMPL_{gender.upper()}.pkl").eval()
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))  # noqa: E731
    b = t(betas).reshape(1, -1)[:, :10].expand(len(pose), -1)
    with torch.no_grad():
        v, _ = body(t(pose[:, :3]), t(pose[:, 3:]), b, t(transl))
    v = v.numpy().astype(np.float64)
    vc = np.einsum("nij,nvj->nvi", ext[:, :3, :3], v) + ext[:, None, :3, 3]
    uv = vc @ K.T
    uv = uv[..., :2] / np.maximum(uv[..., 2:3], 1e-6)
    return np.concatenate([uv.min(1), uv.max(1)], 1)


def load_sequence(ds, root, seq):
    """RICH / MOYO counterpart of bench.sonic_motion.load_sequence: image paths, K, world -> camera (N, 4, 4), GT SMPL
    global orientations (dataset up axis only) and the subject's GT box per frame."""
    if ds == "rich":
        _, imgs, frames, K, E, fits, idx = _rich_meta(seq)
        ok = idx >= 0
        pose = np.zeros((len(frames), 72))
        pose[ok] = fits["pose"][idx[ok]]
        transl = np.zeros((len(frames), 3))
        transl[ok] = fits["transl"][idx[ok]]
        ext = np.repeat(E[None], len(frames), 0)
        boxes = np.full((len(frames), 4), np.nan)
        boxes[ok] = _project_boxes(pose[ok], np.median(fits["betas"], 0), transl[ok], ext[ok], K, "neutral")
        return [str(p) for p in imgs], K, ext, pose[:, :3], boxes
    m = np.load(Path(root) / seq / "meta.npz")
    imgs = [str(Path(root) / seq / "images" / f"{i:05d}.jpg") for i in range(len(m["frames"]))]
    return imgs, m["K"].astype(np.float64), m["ext"].astype(np.float64), m["gt_pose"][:, :3].astype(np.float64), \
        m["boxes"].astype(np.float64)


def gt_sequence(name, frames):
    """RICH / MOYO ground-truth sequence: (gender, pose (N, 72) world, betas (10),
    trans (N, 3) world, extrinsics (N, 4, 4), valid (N,))."""
    ds, seq = clip_source(name)
    if ds == "rich":
        _, _, fr, K, E, fits, idx = _rich_meta(seq)
        idx = idx[frames]
        ok = idx >= 0
        pose = np.zeros((len(idx), 72), np.float64)
        pose[ok] = fits["pose"][idx[ok]]
        trans = np.zeros((len(idx), 3), np.float64)
        trans[ok] = fits["transl"][idx[ok]]
        return "neutral", pose, np.median(fits["betas"], 0)[:10], trans, np.repeat(E[None], len(idx), 0), ok
    m = np.load(root_of("moyo") / seq / "meta.npz")
    valid = m["fit_err_mm"][frames] < 25.0
    return (str(m["gt_gender"]), m["gt_pose"][frames].astype(np.float64), m["gt_betas"][:10].astype(np.float64),
            m["gt_transl"][frames].astype(np.float64), m["ext"][frames].astype(np.float64), valid)
