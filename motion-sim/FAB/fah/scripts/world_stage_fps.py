"""Stage and pipeline speeds from timestamped logs: how the paper times the SLAM rows and the offline camera, whose stages
run as separate commands (each in its own environment).
  stamp  prefix every line of stdin with the epoch time (as moreutils `ts %.s`), and end with the command's wall:
             <stage command> 2>&1 | python scripts/world_stage_fps.py stamp > <stage>.log
  fps    a stage's clip walls = the gaps between its consecutive "[<tag>] <clip>: N frames" lines; the first clip is a
         warm-up (its gap holds the model loading) and is dropped; stage fps = frames / sum of the remaining walls.
         Several --stages: pipeline fps = the timed frames / the sum of all stage walls, plus the whole-command walls of
         --whole logs (short CPU steps that do not print per-clip lines: the "[stamp] wall" line(s) that stamp writes), each
         times its optional ":<factor>" (e.g. 0.8 = 4 / 5 when the step also processed the warm-up clip)
             python scripts/world_stage_fps.py fps --stages hawor.log slam.log [--whole calibrate.log:0.8 offline.log]
The paper: bench/splits/egostandard_timing5.json (warm-up clip first, 4 timed EgoStandard test clips of 900 frames),
2 rounds on an idle GPU, each cell the slower round (docs/reproduce.md).
"""
import argparse
import re
import sys
import time

LINE = re.compile(r"^(\d+(?:\.\d+)?) \[(\S+)\] (\S+): (\d+) frames")


def stamp():
    t0 = time.time()
    print(f"{t0:.6f} [stamp] start", flush=True)
    for line in sys.stdin:
        sys.stdout.write(f"{time.time():.6f} {line}")
        sys.stdout.flush()
    print(f"{time.time():.6f} [stamp] wall {time.time() - t0:.3f}", flush=True)


def stage(path):
    """-> (timed frames, wall s, clips) of one stage log (first clip dropped)."""
    ev = [(float(m.group(1)), m.group(3), int(m.group(4))) for m in map(LINE.match, open(path, errors="ignore")) if m]
    ev.sort()
    assert len(ev) >= 2, f"{path}: needs a warm-up clip and at least one timed clip"
    return sum(n for _, _, n in ev[1:]), ev[-1][0] - ev[0][0], [c for _, c, _ in ev[1:]]


def whole(path):
    """Whole-command wall(s) of a log: the sum of its "[stamp] wall <s>" (or "WALL <s>") lines."""
    w = [float(m.group(1)) for m in (re.search(r"(?:\[stamp\] wall|^WALL) ([\d.]+)", l) for l in open(path, errors="ignore")) if m]
    assert w, f"{path}: no wall line"
    return sum(w)


def fps(args):
    frames, wall = None, 0.0
    for p in args.stages:
        n, w, clips = stage(p)
        print(f"{p}: {n} frames in {w:.1f} s ({len(clips)} clips) = {n / w:.2f} fps")
        assert frames in (None, n), f"{p}: {n} timed frames, other stages {frames}"
        frames, wall = n, wall + w
    for spec in args.whole:
        p, f = (spec.rsplit(":", 1) + ["1"])[:2] if re.search(r":[\d.]+$", spec) else (spec, "1")
        w = whole(p) * float(f)
        print(f"{p}: {w:.1f} s (whole command x {float(f):g})")
        wall += w
    if len(args.stages) > 1 or args.whole:
        print(f"pipeline: {frames} frames in {wall:.1f} s = {frames / wall:.2f} fps")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stamp")
    a = sub.add_parser("fps")
    a.add_argument("--stages", nargs="+", required=True)
    a.add_argument("--whole", nargs="*", default=[], help="<log>[:<factor>] ...")
    args = p.parse_args()
    stamp() if args.cmd == "stamp" else fps(args)
