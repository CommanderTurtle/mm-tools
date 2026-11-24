"""Download the images of a released set from their original hosts (datasets/<set>/images.jsonl) and check each file
against the stored SHA-256 and image size.

Per image, the "get" entries are tried in order: {"zip", "member"} reads one member of a remote zip with HTTP range
requests (HaGRID v2's per-gesture zips are 37-63 GB; only the central directory and the member are fetched), {"url"}
downloads a file, {"commons"} asks the Wikimedia Commons API for the file's thumbnail of the given width, and
{"open_images": "<split>/<image id>"} takes the Open Images copy, which the official Open Images downloader
(downloader.py of https://github.com/openimages/dataset, given with --oi_downloader) fetches for all such ids before
the other downloads. Without --oi_downloader these entries are skipped and the Flickr copies listed next are tried.
Entries with "jpeg" are re-encoded the way the set was made (cv2: shrink to "max_side" with INTER_AREA if larger, JPEG
at that quality), others are kept byte for byte. A file whose bytes differ from the release is kept if its decoded pixels are
the same ("same_pixels", e.g. metadata stripped by the host) or, failing that, its width / height ("size_only": the
labels are pixel coordinates, so they still apply; the report lists these files); anything else is an error.

    python datasets/fetch_images.py --set datasets/shadow_v1 --out $DATA_ROOT/shadow_v1 --oi_downloader downloader.py
-> <out>/images/..., <out>/annotations.json, <out>/fetch_report.json
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

UA = "fah-dataset-fetch/1.0 (research benchmark download)"
_CD = {}


def _open(req, timeout=120, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410) or t == tries - 1:
                raise
            time.sleep(min(float(e.headers.get("Retry-After") or 5 * (t + 1)), 60))
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def _range(url, a, b):
    return _open(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": UA}))[0]


def central_directory(url):
    """{member name: (local header offset, compressed size, method)} of a remote (zip64) archive."""
    if url in _CD:
        return _CD[url]
    size = int(_open(urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA}), 60)[1]["Content-Length"])
    tail = _range(url, size - 65536, size - 1)
    i = tail.rfind(b"PK\x06\x06")
    if i >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    else:
        j = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[j + 12:j + 20])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra = cd[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):  # zip64 extended information: the 0xFFFFFFFF fields, in order
            hid, ln = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                    if flag:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                        r += 8
                    else:
                        vals.append(None)
                csize = vals[1] if vals[1] is not None else csize
                off = vals[2] if vals[2] is not None else off
            q += 4 + ln
        out[name] = (off, csize, method)
        p += 46 + nl + el + cl
    _CD[url] = out
    return out


def zip_member(url, name):
    off, csize, method = central_directory(url)[name]
    head = _range(url, off, off + 30 + 1024 + csize - 1)  # local header + name / extra (< 1 KB) + data
    nl, el = struct.unpack("<HH", head[26:30])
    data = head[30 + nl + el:30 + nl + el + csize]
    if len(data) < csize:
        data += _range(url, off + 30 + nl + el + len(data), off + 30 + nl + el + csize - 1)
    return zlib.decompress(data, -15) if method == zipfile.ZIP_DEFLATED else data


def commons_thumb(title, width):
    q = urllib.parse.urlencode(dict(action="query", format="json", prop="imageinfo", iiprop="url", iiurlwidth=width, titles=title))
    page = next(iter(json.loads(_open(urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}",
                                                             headers={"User-Agent": UA}))[0])["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    return ii.get("thumburl") or ii["url"]


def reencode(data, quality, max_side):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def pixel_md5(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    return None if img is None else hashlib.md5(img.tobytes()).hexdigest()


def size_of(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    return None if img is None else (img.shape[1], img.shape[0])


def open_images(ids, downloader, python, workers, folder):
    """The official Open Images downloader on an id list ("<split>/<image id>" per line) -> <folder>/<image id>.jpg."""
    lst = os.path.join(folder, "image_ids.txt")
    open(lst, "w").write("\n".join(ids) + "\n")
    subprocess.run([python, downloader, lst, "--download_folder", folder, "--num_processes", str(workers)], check=False)


def fetch(rec, out, oi_dir):
    dst = os.path.join(out, rec["file"])
    if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == rec["sha256"]:
        return dict(file=rec["file"], status="exact", note="present")
    fallback, errors = None, []
    for g in rec["get"]:
        try:
            if "open_images" in g:
                if oi_dir is None:
                    raise FileNotFoundError("no --oi_downloader")
                data = open(os.path.join(oi_dir, g["open_images"].split("/")[-1] + ".jpg"), "rb").read()
            elif "zip" in g:
                data = zip_member(g["zip"], g["member"])
            elif "commons" in g:
                data = _open(urllib.request.Request(commons_thumb(g["commons"], g["width"]), headers={"User-Agent": UA}))[0]
            else:
                data = _open(urllib.request.Request(g["url"], headers={"User-Agent": UA}))[0]
            if len(data) < 5000:  # Flickr's "photo unavailable" placeholder
                raise ValueError(f"{len(data)} bytes")
            if "jpeg" in g:
                data = reencode(data, g["jpeg"], g.get("max_side", 1 << 30))
        except Exception as e:  # next source
            errors.append(f"{g}: {e}")
            continue
        if hashlib.sha256(data).hexdigest() == rec["sha256"]:
            fallback = (data, "exact")
            break
        if rec.get("pixels_md5") and pixel_md5(data) == rec["pixels_md5"]:  # e.g. metadata stripped by the host
            fallback = (data, "same_pixels")
        elif fallback is None and size_of(data) == (rec["width"], rec["height"]):
            fallback = (data, "size_only")
    if fallback is None:
        return dict(file=rec["file"], status="failed", errors=errors[-3:])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".tmp", "wb") as f:
        f.write(fallback[0])
    os.replace(dst + ".tmp", dst)
    return dict(file=rec["file"], status=fallback[1])


def main(args):
    recs = [json.loads(l) for l in open(os.path.join(args.set, "images.jsonl"))]
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.set, "annotations.json"), os.path.join(args.out, "annotations.json"))
    zips = sorted({g["zip"] for r in recs for g in r["get"] if "zip" in g})
    for z in zips:  # one central-directory read per archive, before the parallel member reads
        central_directory(z)
    oi_ids = sorted({g["open_images"] for r in recs for g in r["get"] if "open_images" in g})
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        oi_dir = None
        if oi_ids and args.oi_downloader:
            open_images(oi_ids, args.oi_downloader, args.oi_python, args.workers, tmp)
            oi_dir = tmp
        with ThreadPoolExecutor(args.workers) as ex:
            rep = list(ex.map(lambda r: fetch(r, args.out, oi_dir), recs))
    count = {}
    for r in rep:
        count[r["status"]] = count.get(r["status"], 0) + 1
    json.dump(dict(count=count, files=rep), open(os.path.join(args.out, "fetch_report.json"), "w"), indent=1)
    print(f"{len(recs)} images: {count}" + ("  (see fetch_report.json)" if count.get("failed") else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, help="released set dir (annotations.json + images.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--oi_downloader", default="", help="downloader.py of https://github.com/openimages/dataset")
    p.add_argument("--oi_python", default=sys.executable, help="python with the downloader's requirements")
    main(p.parse_args())
