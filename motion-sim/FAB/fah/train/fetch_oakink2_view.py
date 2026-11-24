"""Fetch one camera view of OakInk2 sequences from HF (kelvin34501/OakInk-v2) without downloading the whole tars.

Each data/<seq>.tar stores the 4 views as contiguous per-serial directories (order varies per sequence), so only the
byte range of the wanted serial is needed (~25% of the tar): 16 coarse + ~2 x 10 binary-search HTTP range probes (1 MB
each, scanned for valid ustar headers) locate the first and the last member of the view, then that range is streamed
to <raw>/<view>/<seq>.tar (+ the two zero end blocks, so it is a plain tar of that view). The anno_preview pkl (needed
for cam_def: serial -> view) is downloaded whole to <raw>/anno_preview/. Every step leaves a .done marker; sequences
with a full data/<seq>.tar.done are skipped (the pool builder reads those directly).

    python -m train.fetch_oakink2_view --raw $DATA_ROOT/oakink2_raw --view egocentric --workers 8
"""
import argparse
import json
import os
import tarfile
import urllib.request
from multiprocessing import Pool

REPO = "https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/main"
WIN = 1 << 20


def resolve(path):
    """Presigned CDN URL of a repo file (the resolve endpoint redirects), so probes skip the redirect."""
    req = urllib.request.Request(f"{REPO}/{path}", method="HEAD")
    return urllib.request.urlopen(req, timeout=60).url


def get_range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    return urllib.request.urlopen(req, timeout=120).read()


def headers_in(buf, base):
    """[(absolute offset, member name, size)] of the valid ustar headers in buf (base = absolute offset of buf)."""
    out = []
    for o in range((-base) % 512, len(buf) - 511, 512):
        h = buf[o:o + 512]
        if h[257:262] != b"ustar":
            continue
        try:
            chk = int(h[148:156].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            continue
        if chk != sum(h[:148]) + 256 + sum(h[156:]):
            continue
        name = h[:100].split(b"\0")[0].decode(errors="replace")
        prefix = h[345:500].split(b"\0")[0].decode(errors="replace")
        size = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
        out.append((base + o, f"{prefix}/{name}" if prefix else name, size))
    return out


def serial_at(url, off, size, seq):
    """(serial of the first member header at or after off, its offset); None past the last member."""
    for win in (WIN, 4 * WIN):
        a = max(0, off - off % 512)
        hs = [h for h in headers_in(get_range(url, a, min(a + win, size)), a) if h[1].startswith(seq + "/")]
        hs = [h for h in hs if len(h[1].split("/")) > 1 and h[1].split("/")[1]]
        if hs:
            return hs[0][1].split("/")[1], hs[0][0]
        if a + win >= size:
            return None
    raise RuntimeError(f"no tar header near {off} in {seq}")


def view_range(url, size, seq, serial):
    """[start, end) byte range of the members of one serial directory (blocks are contiguous; past the last member and
    the end sentinel count as 'not this view')."""
    n = 16
    offs = [k * size // n for k in range(n)] + [size]
    ser = [serial_at(url, o, size, seq) for o in offs[:-1]] + [None]
    in_view = [s is not None and s[0] == serial for s in ser]
    hit = [k for k, v in enumerate(in_view) if v]
    if not hit:
        raise RuntimeError(f"{seq}: {serial} not found by coarse probes")

    def boundary(lo, hi, into_view):
        """First member header in (lo, hi] that enters (into_view) or leaves the view; lo / hi straddle it."""
        while hi - lo > WIN:
            mid = (lo + hi) // 2
            s = serial_at(url, mid, size, seq)
            if (s is not None and s[0] == serial) == into_view:
                hi = mid
            else:
                lo = mid
        a = lo - lo % 512
        for off, name, _ in headers_in(get_range(url, a, min(hi + 4 * WIN, size)), a):
            if name.startswith(seq + "/") and len(name.split("/")) > 1 and name.split("/")[1] and \
                    (name.split("/")[1] == serial) == into_view:
                return off
        if not into_view and hi + 4 * WIN >= size:
            return size  # the view runs to the end of the archive
        raise RuntimeError(f"{seq}: boundary not found in [{lo}, {hi})")

    k0, k1 = hit[0], hit[-1]
    start = ser[0][1] if k0 == 0 else boundary(offs[k0 - 1], offs[k0], True)
    end = boundary(offs[k1], offs[k1 + 1], False)
    return start, end


def fetch(arg):
    raw, view, seq, size, anno_size = arg
    if os.path.exists(os.path.join(raw, "data", f"{seq}.tar.done")) or os.path.exists(os.path.join(raw, view, f"{seq}.tar.done")):
        return seq, "skip"
    from train.build_oakink2_crops import load_anno

    ap = os.path.join(raw, "anno_preview", f"{seq}.pkl")
    if not os.path.exists(ap + ".done"):
        data = urllib.request.urlopen(f"{REPO}/anno_preview/{seq}.pkl", timeout=600).read()
        assert len(data) == anno_size, (seq, len(data), anno_size)
        open(ap + ".part", "wb").write(data)
        os.replace(ap + ".part", ap)
        open(ap + ".done", "w").close()
    serial = [k for k, v in load_anno(ap, tensors=False)["cam_def"].items() if v == view][0]
    url = resolve(f"data/{seq}.tar")
    start, end = view_range(url, size, seq, serial)
    out = os.path.join(raw, view, f"{seq}.tar")
    with open(out + ".part", "wb") as f:
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            while True:
                b = r.read(1 << 24)
                if not b:
                    break
                f.write(b)
        if end != size:
            f.write(b"\0" * 1024)  # tar end-of-archive blocks
    with tarfile.open(out + ".part") as t:
        names = [m.name for m in t.getmembers() if m.isfile()]
    assert names and all(n.split("/")[1] == serial for n in names), f"{seq}: foreign members in the {view} range"
    os.replace(out + ".part", out)
    json.dump(dict(serial=serial, start=start, end=end, members=len(names)), open(out + ".done", "w"))
    return seq, f"{len(names)} frames, {(end - start) / 1e9:.2f} GB of {size / 1e9:.2f}"


def main(args):
    os.makedirs(os.path.join(args.raw, args.view), exist_ok=True)
    os.makedirs(os.path.join(args.raw, "anno_preview"), exist_ok=True)
    tree = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/datasets/kelvin34501/OakInk-v2/tree/main/data", timeout=60).read())
    anno = {x["path"].split("/")[1][:-4]: x["size"] for x in json.loads(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/kelvin34501/OakInk-v2/tree/main/anno_preview", timeout=60).read())}
    jobs = sorted((args.raw, args.view, x["path"].split("/")[1][:-4], x["size"], anno[x["path"].split("/")[1][:-4]]) for x in tree)
    with Pool(args.workers) as pool:
        for k, (seq, msg) in enumerate(pool.imap_unordered(fetch, jobs)):
            print(f"{k + 1}/{len(jobs)} {seq}: {msg}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True)
    p.add_argument("--view", default="egocentric")
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
