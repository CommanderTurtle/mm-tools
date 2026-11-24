"""Release scrub: fail if the tree contains any pattern from a private denylist.

The denylist is kept outside the release tree (it names the very strings that must not ship). Pass it with --denylist or
$FAB_RELEASE_DENYLIST: one Python regular expression per line (case-insensitive), '#' comments, and optional
'allow: <regex>' lines for exact phrases that may stay (e.g. a public citation URL).

Scans every file name, every text file (including notebook JSON with its outputs), the printable strings of binary files,
and the members of zip archives (.npz, .pt, .zip), with the text of .npy string and object arrays decoded (numpy keeps
unicode as UTF-32, which a printable-strings pass misses; object arrays' pickles are read op by op, never unpickled);
skips .gitignore's top-level output/, weights/ and data/ folders (your runs and downloads). Exit status 1 and one line per
hit if anything matches.

    python scripts/release_scrub.py --denylist /path/outside/the/tree/denylist.txt [ROOT]
"""
import argparse
import ast
import io
import os
import pickletools
import re
import sys
import zipfile

import numpy as np

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
TOP_SKIP = {"output", "weights", "data"}  # the top-level run-output / download folders .gitignore leaves out


def load(path):
    deny, allow = [], []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("allow:"):
            allow.append(re.compile(s[len("allow:"):].strip(), re.I))
        else:
            deny.append(re.compile(s, re.I))
    return deny, allow


def strings(data, n=4):
    return [m.decode("ascii", "replace") for m in re.findall(rb"[\x20-\x7e]{%d,}" % n, data)]


IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
VIDEOS = (".mp4", ".m4v", ".mov")


def video_texts(path):
    """An MP4 / MOV file's boxes except the media data ('mdat': the compressed frames, not text; short patterns occur in
    it by chance): the printable strings of every other top-level box (moov with its udta / meta tags, ftyp, free, ...)."""
    data = open(path, "rb").read()
    out, i = [], 0
    while i + 8 <= len(data):
        size, typ = int.from_bytes(data[i:i + 4], "big"), data[i + 4:i + 8]
        hdr = 8
        if size == 1:
            size, hdr = int.from_bytes(data[i + 8:i + 16], "big"), 16
        elif size == 0:
            size = len(data) - i
        if size < hdr or i + size > len(data):  # not a well-formed box list: scan the whole file as before
            return [(path, s) for s in strings(data, 6)]
        if typ != b"mdat":
            out += [(path, s) for s in strings(data[i:i + size], 6)]
        i += size
    return out


def npy_numeric(blob):
    """True for an .npy member whose dtype is numeric (its bytes are numbers, not text)."""
    if not blob.startswith(b"\x93NUMPY"):
        return False
    head = blob[:256].decode("latin1")
    m = re.search(r"'descr':\s*'([^']*)'", head)
    return bool(m) and not any(c in m.group(1) for c in "USO")


def npy_texts(blob):
    """The text of a non-numeric .npy: its printable strings, string arrays decoded (unicode is UTF-32 in .npy) and the
    strings in an object array's pickle (read op by op, never unpickled)."""
    out = strings(blob)
    try:
        f = io.BytesIO(blob)
        major, _ = np.lib.format.read_magic(f)
        n = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        dtype = np.lib.format.descr_to_dtype(ast.literal_eval(f.read(n).decode("latin1" if major < 3 else "utf8"))["descr"])
        if dtype.hasobject:
            out += [a if isinstance(a, str) else a.decode("latin1") for _, a, _ in pickletools.genops(f) if isinstance(a, (str, bytes))]
        else:
            a = np.lib.format.read_array(io.BytesIO(blob), allow_pickle=False)
            out += [str(x) for x in a.ravel()] if a.dtype.kind in "US" else [str(a.tolist())]
    except Exception as e:  # unreadable: report it rather than pass it
        out.append(f"<unreadable .npy: {type(e).__name__}: {e}>")
    return out


def image_texts(path):
    """An image's metadata only (text chunks, EXIF, XMP, comments): its pixel bytes are not text."""
    try:
        from PIL import Image
    except ImportError:
        return [(path, s) for s in strings(open(path, "rb").read(), 8)]
    with Image.open(path) as im:
        info = {k: v for k, v in im.info.items() if isinstance(v, (str, bytes))}
        exif = im.getexif()
    out = [(path, f"{k}={v.decode('latin1', 'replace') if isinstance(v, bytes) else v}") for k, v in info.items()]
    out += [(path, f"exif {k}={v}") for k, v in exif.items() if isinstance(v, (str, bytes))]
    return out


def texts(path):
    """(where, text) pairs to scan for one file: text files whole; zip archives (.npz, .pt) member names, pickles and
    non-numeric arrays (raw tensor / numeric array bytes are skipped); images their metadata; MP4 / MOV files every box but
    the media data; other binaries their printable strings."""
    if path.lower().endswith(IMAGES):
        return image_texts(path)
    if path.lower().endswith(VIDEOS):
        return video_texts(path)
    data = open(path, "rb").read()
    out = []
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                out.append((f"{path}!{name}", name))
                if re.search(r"(^|/)data/\d+$", name):  # a torch tensor storage: raw numbers
                    continue
                blob = z.read(name)
                if name.endswith(".npy"):
                    if not npy_numeric(blob):
                        out += [(f"{path}!{name}", s) for s in npy_texts(blob)]
                    continue
                out += [(f"{path}!{name}", s) for s in strings(blob)]
        return out
    if data.startswith(b"\x93NUMPY"):
        return [] if npy_numeric(data) else [(path, s) for s in npy_texts(data)]
    try:
        return [(path, data.decode("utf-8"))]
    except UnicodeDecodeError:
        return [(path, s) for s in strings(data, 6)]


def scan(root, deny, allow):
    hits = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not (os.path.samefile(d, root) and x in TOP_SKIP)]
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            for where, text in [(rel + " (name)", rel)] + [(os.path.relpath(w, root), t) for w, t in texts(p)]:
                for ln, line in enumerate(text.splitlines(), 1):
                    if any(a.search(line) for a in allow):
                        continue
                    for r in deny:
                        m = r.search(line)
                        if m:
                            hits.append(f"{where}:{ln}: {m.group(0)!r}")
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--denylist", default=os.environ.get("FAB_RELEASE_DENYLIST", ""))
    a = p.parse_args()
    if not a.denylist or not os.path.exists(a.denylist):
        sys.exit("release_scrub: give --denylist (kept outside the release tree) or set FAB_RELEASE_DENYLIST")
    deny, allow = load(a.denylist)
    hits = scan(a.root, deny, allow)
    for h in hits:
        print(h)
    print(f"release_scrub: {len(hits)} hit(s) under {a.root} ({len(deny)} patterns)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
