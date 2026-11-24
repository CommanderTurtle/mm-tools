"""Inputs of scripts/reproduce_g1_refs.sh from your own fab-world runs (docs/reproduce.md, "G1 references"). Reads only your
own dataset copies; nothing is downloaded.
    python scripts/g1_refs_inputs.py frames <clip> <dir> [--src <dataset copy>]
        the clip's frames in <dir>/frames/ and the fab-world command for them. 3DPW / EMDB clips (datasets under $FAB_DATA,
        default $FAB_HOME/data; bench/sonic_clips.py): the frames linked from the dataset and its subject box per frame in
        <dir>/subject_boxes.npz. RICH / MOYO clips (SOURCES below; this release has no loader for them): the frames the
        paper used, read from --src, and no subject boxes:
          RICH  --src the extracted RICH JPEG release (JPG_images/val.tar.gz: val/<seq>/cam_XX/<frame>_XX.jpeg): every 3rd
                frame, first..last, of one camera, at half resolution (cv2.INTER_AREA, JPEG quality 93)
          MOYO  --src the MOYO val image zip of the sequence (<seq>.zip) or its extracted directory (.../Cam_<cam>_<frame>.jpg):
                every frame, first..last, of one camera, as stored
    python scripts/g1_refs_inputs.py collect <fab-world out> <inputs dir> <clip>
        <fab-world out>/<clip>.npz and <fab-world out>/work/ours/fab/<clip>_raw.npz into <inputs dir>
"""
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.sonic_clips import ALL_CLIPS, clip_source, root_of  # noqa: E402

# clip: (dataset, split, sequence, camera, first frame, last frame) as the paper's inputs used them (dataset frame numbers)
SOURCES = {
    "rich_ParkingLot1_004_pushup1": ("rich", "val", "ParkingLot1_004_pushup1", 1, 6, 303),
    "rich_ParkingLot1_005_burpeejump1": ("rich", "val", "ParkingLot1_005_burpeejump1", 6, 6, 204),
    "rich_ParkingLot1_005_pushup1": ("rich", "val", "ParkingLot1_005_pushup1", 6, 6, 453),
    "rich_ParkingLot1_007_burpee2": ("rich", "val", "ParkingLot1_007_burpee2", 6, 6, 306),
    "rich_Pavallion_000_yoga1": ("rich", "val", "Pavallion_000_yoga1", 5, 6, 603),
    "rich_Pavallion_013_sidebalancerun": ("rich", "val", "Pavallion_013_sidebalancerun", 6, 6, 603),
    "rich_Pavallion_018_yoga1": ("rich", "val", "Pavallion_018_yoga1", 6, 6, 603),
    "moyo_headstand": ("moyo", "val", "220926_yogi_body_hands_03596_Supported_Headstand_pose_or_Salamba_Sirsasana_-a", 1, 129, 728),
    "moyo_supta_virasana": ("moyo", "val", "220926_yogi_body_hands_03596_Supta_Virasana_Vajrasana-b", 1, 307, 906),
    "moyo_upward_plank": ("moyo", "val", "220926_yogi_body_hands_03596_Upward_Plank_Pose_or_Purvottanasana_-a", 1, 176, 775),
    "moyo_warrior3": ("moyo", "val", "220926_yogi_body_hands_03596_Warrior_III_Pose_or_Virabhadrasana_III_-b", 3, 1, 587),
}
MOYO_IMG = re.compile(r"Cam_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.I)
SRC_HELP = {"rich": "the extracted RICH JPEG release (JPG_images/val.tar.gz: val/<seq>/cam_XX/<frame>_XX.jpeg)",
            "moyo": "the MOYO val image zip of the sequence (<seq>.zip) or its extracted directory"}


def rich_frames(src, split, seq, cam, first, last, fd):
    import cv2

    cd = Path(src) / split / seq / f"cam_{cam:02d}"
    for k, f in enumerate(range(first, last + 1, 3)):
        hits = sorted(cd.glob(f"{f:05d}_*.jp*g"))
        if not hits:
            sys.exit(f"{cd}: frame {f} not found (--src: {SRC_HELP['rich']})")
        img = cv2.resize(cv2.imread(str(hits[0])), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fd / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])


def moyo_frames(src, cam, first, last, fd):
    src = Path(src)
    if src.is_file():
        z = zipfile.ZipFile(src)
        names = {(int(m.group(1)), int(m.group(2))): n for n in z.namelist() for m in [MOYO_IMG.search(n)] if m}
        read = z.read
    else:
        names = {(int(m.group(1)), int(m.group(2))): p for p in src.rglob("*") for m in [MOYO_IMG.search(p.name)] if m}
        read = Path.read_bytes
    for k, f in enumerate(range(first, last + 1)):
        if (cam, f) not in names:
            sys.exit(f"{src}: Cam_{cam}_{f} not found (--src: {SRC_HELP['moyo']})")
        (fd / f"{k:05d}.jpg").write_bytes(read(names[(cam, f)]))


def frames(clip, d, src=None):
    ds, seq, start, n, fps, _ = ALL_CLIPS[clip]
    d = Path(d)
    fd = d / "frames"
    if clip in SOURCES:
        ds, split, sseq, cam, first, last = SOURCES[clip]
        if not src:
            sys.exit(f"{clip}: pass --src, {SRC_HELP[ds]}")
        fd.mkdir(parents=True, exist_ok=True)
        if ds == "rich":
            rich_frames(src, split, sseq, cam, first, last, fd)
        else:
            moyo_frames(src, cam, first, last, fd)
        print(f"{clip}: {len(list(fd.glob('*.jpg')))} frames ({ds} {split} {sseq}, camera {cam}, frames {first}..{last}) -> {fd}")
        print(f"fab-world {fd} --name {clip} --fps {fps} --out <fab-world out>")
        return
    from bench.sonic_motion import load_sequence

    _, src_seq = clip_source(clip)
    images, _, _, _, boxes = load_sequence(ds, str(root_of(ds)), src_seq)
    sel = np.arange(start, min(len(images), start + n))
    fd.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(sel):
        dst = fd / f"{k:05d}{Path(images[i]).suffix}"
        if not dst.exists():
            os.symlink(os.path.abspath(images[i]), dst)
    np.savez(d / "subject_boxes.npz", box=np.asarray(boxes, np.float64)[sel])
    print(f"{clip}: {len(sel)} frames ({ds} {seq}, from frame {start}, {fps} fps) -> {fd}")
    print(f"fab-world {fd} --name {clip} --fps {fps} --subject_boxes {d / 'subject_boxes.npz'} --out <fab-world out>")


def collect(out, inp, clip):
    out, inp = Path(out), Path(inp)
    inp.mkdir(parents=True, exist_ok=True)
    for src, dst in ((out / f"{clip}.npz", inp / f"{clip}.npz"), (out / "work" / "ours" / "fab" / f"{clip}_raw.npz", inp / f"{clip}_raw.npz")):
        if not src.exists():
            sys.exit(f"{src} not found: run fab-world on the clip's frames first (python scripts/g1_refs_inputs.py frames {clip} <dir>)")
        shutil.copy(src, dst)
    print(f"{clip}: inputs in {inp}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["frames"] and (len(a) == 3 or (len(a) == 5 and a[3] == "--src")):
        frames(a[1], a[2], a[4] if len(a) == 5 else None)
    elif a[:1] == ["collect"] and len(a) == 4:
        collect(*a[1:4])
    else:
        sys.exit(__doc__)
