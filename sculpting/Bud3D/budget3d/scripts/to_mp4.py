"""Post-process a gallery folder into H.264 MP4s (plays in VS Code's preview and every browser).

    python scripts/to_mp4.py --out gallery

For every finished object:
  videos/<obj>_bNNN.mp4   each budget's orbit, transcoded from the webm (CRF 16, yuv420p, faststart)
  grids/<obj>.mp4         input + all budgets in one 2x6 grid at full resolution, frames in lock-step
then rebuilds index.html / gallery.md / manifest.json so they point at the MP4s.  Idempotent:
anything already converted is skipped, so it can run repeatedly while a sweep is still adding
objects.  TO_MP4_WORKERS sets the number of parallel ffmpeg processes (default 1).
"""
import argparse
import math
import os
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _ffmpeg():
    """An ffmpeg with libx264: $FFMPEG, else imageio-ffmpeg's bundled binary (pip install imageio-ffmpeg), else PATH."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


FFMPEG = _ffmpeg()
WORKERS = int(os.environ.get("TO_MP4_WORKERS", "1"))
X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def newer(dst, src):
    return os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src) and os.path.getsize(dst) > 0


def tmpname(dst):
    # unique per process and thread: several jobs may convert the same folder at once
    return f"{dst[:-4]}.{os.getpid()}-{threading.get_ident()}.tmp.mp4"


def transcode(src, dst):
    tmp = tmpname(dst)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, *X264, tmp], check=True)
    os.replace(tmp, dst)


def read_frames(path, size):
    raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def grid_mp4(o, out, fps, size, dst):
    """input + every budget, 2 rows x 6, same layout as the GIF but full resolution."""
    runs = o["runs"]
    clips = [read_frames(os.path.join(out, r["video"]), size) for r in runs]
    n = min(len(c) for c in clips)
    cols, lab, head = 6, 40, 46
    rows = math.ceil((len(runs) + 1) / cols)
    W, H = cols * size, head + rows * (size + lab)
    H += H % 2
    base = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(base)
    d.text((14, 10), f"{o['name']}   (input, then 100% -> 10% of 8192 tokens, no warm-up)", fill=(30, 30, 30), font=font(26))
    inp = Image.new("RGB", (size, size), "white")
    im = Image.open(os.path.join(out, o["input"])).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    inp.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    base.paste(inp, (0, head))
    labels = ["input"] + [f"{round(r['budget'] * 100)}%  ·  {r['tokens']} tok" for r in runs]
    pos = [((k % cols) * size, head + (k // cols) * (size + lab)) for k in range(len(labels))]
    for (x, y), t in zip(pos, labels):
        d.text((x + 10, y + size + 7), t, fill=(30, 30, 30), font=font(22))
    canvas = np.asarray(base).copy()
    tmp = tmpname(dst)
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *X264, tmp], stdin=subprocess.PIPE)
    for i in range(n):
        for (x, y), c in zip(pos[1:], clips):
            canvas[y:y + size, x:x + size] = c[i]
        p.stdin.write(canvas.tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {dst}"
    os.replace(tmp, dst)


def one_pass(out):
    import gallery
    m = gallery.rebuild_index(out)          # picks up every object finished so far
    fps, size = m.get("fps", 24), m.get("size", 384)
    def convert(o):
        made = 0
        for r in o["runs"]:
            src = os.path.join(out, r["video"])
            dst = src[:-5] + ".mp4"
            if os.path.exists(src) and not newer(dst, src):
                transcode(src, dst)
                made += 1
        g = os.path.join(out, "grids", f"{o['name']}.mp4")
        newest = max(os.path.getmtime(os.path.join(out, r["video"])) for r in o["runs"])
        if not (os.path.exists(g) and os.path.getmtime(g) >= newest):
            grid_mp4(o, out, fps, size, g)
            made += 1
        return made
    # each conversion is an ffmpeg subprocess, so threads give real parallelism
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:
        made = sum(ex.map(convert, m["objects"]))
    gallery.rebuild_index(out)              # last word on index.html: the MP4-aware page
    return len(m["objects"]), made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="a gallery folder written by scripts/gallery.py")
    a = ap.parse_args()
    n, made = one_pass(a.out)
    print(f"[to_mp4] {n} objects, {made} new MP4s", flush=True)


if __name__ == "__main__":
    main()
