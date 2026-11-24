"""Focal estimates for world mode's focal rule (and a diagnosis against GT intrinsics): per clip of a clips json, the focal of
  vipe      ViPE 1.2's camera (<vipe>/cams/<clip>.npz fx; what the pipeline uses)
  geocalib  GeoCalib (pinhole weights) without a prior, the median over --n evenly spaced frames      (stage geocalib, ViPE env)
  moge      MoGe-2's own intrinsics, the median over the same frames                                  (stage moge, FAB env)
and (report) each one's |log f / f_gt| on the clips with GT intrinsics (<gt>/<clip>.npz K), plus fixed
combinations (median of the three, the geometric mean of GeoCalib and MoGe-2): one constant rule for every clip, no GT input.
    $FAB_ENVS/vipe/bin/python -m bench.scene_recon.focal_compare geocalib --list L --out J
    python -m bench.scene_recon.focal_compare moge --list L --out J
    python -m bench.scene_recon.focal_compare report --list L --out J --vipe <vipe out> --gt <dir of <clip>.npz with K>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FAB_HOME = Path(os.environ.get("FAB_HOME", "~/.cache/fab")).expanduser()  # as fab.paths (the geocalib stage runs in ViPE's env)
FAB_ENVS = Path(os.environ.get("FAB_ENVS", FAB_HOME / "envs")).expanduser()


def frames_of(images, n):
    return np.unique(np.linspace(0, len(images) - 1, n).round().astype(int))


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("geocalib", "moge", "report"))
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True, help="json, one entry per clip (merged across stages)")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--only", default="")
    p.add_argument("--vipe_code", default=os.environ.get("VIPE_DIR", str(FAB_ENVS / "src" / "vipe")))
    p.add_argument("--vipe", default="", help="report: the run's ViPE dir (cams/<clip>.npz)")
    p.add_argument("--gt", default="", help="report: a dir of <clip>.npz with GT K")
    a = p.parse_args()
    clips = json.loads(Path(a.list).read_text())
    names = [c for c in sorted(clips) if not a.only or any(o in c for o in a.only.split(","))]
    res = load(a.out)
    if a.stage in ("geocalib", "moge"):
        import cv2
        import torch

        if a.stage == "geocalib":
            os.environ.setdefault("TORCH_HOME", str(FAB_HOME / "hf-cache" / "vipe" / "torch"))
            sys.path.insert(0, a.vipe_code)
            from vipe.priors.geocalib import GeoCalib

            model = GeoCalib(weights="pinhole").cuda()
        else:
            from moge.model.v2 import MoGeModel

            model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
        for c in names:
            if a.stage in res.get(c, {}):
                continue
            imgs = clips[c]["images"]
            fs = []
            for f in frames_of(imgs, a.n):
                img = cv2.imread(imgs[f])[:, :, ::-1]
                x = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255
                with torch.no_grad():
                    if a.stage == "geocalib":
                        o = model.calibrate(x)
                        if "camera" not in o or not hasattr(o["camera"], "f"):
                            raise KeyError(f"GeoCalib output keys {list(o)}")
                        fs.append(float(o["camera"].f.reshape(-1)[0]))
                    else:
                        fs.append(float(model.infer(x)["intrinsics"][0, 0]) * img.shape[1])
            res.setdefault(c, {})[a.stage] = {"f_median": float(np.median(fs)), "f_all": fs, "width": int(img.shape[1])}
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
            print(a.stage, c, round(float(np.median(fs)), 1), flush=True)
        return
    rows = {}
    for c in names:
        g = Path(a.gt) / f"{c}.npz"
        cp = Path(a.vipe) / "cams" / f"{c}.npz"
        if not g.exists() or not cp.exists():
            continue
        fg = float(np.median(np.load(g)["K"][:, 0, 0]))
        e = {"vipe": float(np.load(cp)["fx"])}
        for k in ("geocalib", "moge"):
            if k in res.get(c, {}):
                e[k] = res[c][k]["f_median"]
        if len(e) == 3:
            e["median3"] = float(np.median([e["vipe"], e["geocalib"], e["moge"]]))
            e["gm_geo_moge"] = float(np.sqrt(e["geocalib"] * e["moge"]))
        rows[c] = {k: float(np.log(v / fg)) for k, v in e.items()}
        rows[c]["f_gt"] = fg
    keys = ("vipe", "geocalib", "moge", "median3", "gm_geo_moge")
    print("| clip | f_gt | " + " | ".join(f"{k} %" for k in keys) + " |\n|---|---|" + "---|" * len(keys))
    for c, r in rows.items():
        print(f"| {c} | {r['f_gt']:.0f} | " + " | ".join(f"{100 * (np.exp(r[k]) - 1):+.1f}" if k in r else "" for k in keys) + " |")
    print("| mean abs log % | | " + " | ".join(f"{100 * np.mean([abs(r[k]) for r in rows.values() if k in r]):.1f}" for k in keys) + " |")
    print("| max abs log % | | " + " | ".join(f"{100 * np.max([abs(r[k]) for r in rows.values() if k in r] or [np.nan]):.1f}" for k in keys) + " |")
    print("| share within 10 % | | " + " | ".join(f"{np.mean([abs(r[k]) < np.log(1.1) for r in rows.values() if k in r]):.2f}" for k in keys) + " |")
    res["_report"] = rows
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
