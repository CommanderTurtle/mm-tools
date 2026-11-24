"""World mode's focal rule (scripts/fab_world.sh): the
focal is unobservable without camera translation, so for a static camera (ViPE's own run: p90 camera-centre spread < 0.5 %
of the median static scene depth) the focal is median(ViPE, GeoCalib, MoGe-2) and ViPE re-runs with K fixed; a moving camera
keeps ViPE's focal. Then the run whose cameras / depth / gravity the pipeline uses (vipe_sel, vipe_mvs, geo_sel links).
On one work dir <work> (fab_world.sh layout: vipe/, geocalib/, vipe_fk/, geocalib_fk/, focal/):
  static  ViPE's run -> focal/<clip>.json {spread_over_depth, static, fx_vipe}
  focal   (static cameras) focal/{geo,moge}_<clip>.json (bench.scene_recon.focal_compare) -> the median -> focal/fixed_<clip>.json
  given   --focal F: a known focal (calibrated camera; opt-in): focal/fixed_<clip>.json, one fixed-K run, no rule
  select  the chosen run (vipe_fk / geocalib_fk for a static camera or a given focal, else vipe / geocalib) -> vipe_sel/,
          vipe_mvs/ (all but dense: MVS writes its own), geo_sel/ symlinks
    python -m bench.scene_recon.focal_rule static --work <work> --clip <clip>
"""

import argparse
import json
import os

import numpy as np

STATIC_SPREAD = 0.005  # p90 camera-centre spread / median static depth below which the camera counts as static


def static(w, c):
    T = np.load(f"{w}/vipe/cams/{c}.npz")["T_world_cam"]
    d = np.load(f"{w}/vipe/dense/{c}.npz")
    dep, dyn = d["depth"], d["dyn"]
    z = float(np.median(dep[~dyn & np.isfinite(dep) & (dep > 0)]))
    cen = T[:, :3, 3]
    ratio = float(np.percentile(np.linalg.norm(cen - np.median(cen, 0), axis=1), 90)) / z
    v = {"spread_over_depth": ratio, "static": bool(ratio < STATIC_SPREAD), "fx_vipe": float(np.load(f"{w}/vipe/cams/{c}.npz")["fx"])}
    json.dump(v, open(f"{w}/focal/{c}.json", "w"))
    print(c, "static" if v["static"] else "moving", round(ratio, 4))
    return v["static"]


def focal(w, c):
    v = json.load(open(f"{w}/focal/{c}.json"))
    cand = [v["fx_vipe"]] + [json.load(open(f"{w}/focal/{k}_{c}.json"))[c][s]["f_median"] for k, s in (("geo", "geocalib"), ("moge", "moge"))]
    v["focal_candidates"] = cand
    v["focal"] = float(np.median(cand))
    json.dump(v, open(f"{w}/focal/{c}.json", "w"))
    json.dump({c: v["focal"]}, open(f"{w}/focal/fixed_{c}.json", "w"))
    print(c, "focal", [round(x) for x in cand], "->", round(v["focal"]))


def given(w, c, f):
    json.dump({c: f}, open(f"{w}/focal/fixed_{c}.json", "w"))
    json.dump({"focal": f, "K": "given (--focal)", "static": None}, open(f"{w}/focal/{c}.json", "w"))


def select(w, c):
    v = json.load(open(f"{w}/focal/{c}.json"))
    src, gs = ("vipe_fk", "geocalib_fk") if v.get("K") or v["static"] else ("vipe", "geocalib")
    for root in ("vipe_sel", "vipe_mvs"):
        for sub in os.listdir(f"{w}/{src}"):
            if not os.path.isdir(f"{w}/{src}/{sub}") or sub == "logs" or (root == "vipe_mvs" and sub == "dense"):
                continue
            os.makedirs(f"{w}/{root}/{sub}", exist_ok=True)
            for f in os.listdir(f"{w}/{src}/{sub}"):
                if f.startswith(c + ".") and not os.path.lexists(f"{w}/{root}/{sub}/{f}"):
                    os.symlink(os.path.realpath(f"{w}/{src}/{sub}/{f}"), f"{w}/{root}/{sub}/{f}")
    os.makedirs(f"{w}/geo_sel", exist_ok=True)
    if not os.path.lexists(f"{w}/geo_sel/{c}.npz"):
        os.symlink(f"{w}/{gs}/{c}.npz", f"{w}/geo_sel/{c}.npz")
    v["vipe"], v["geo"] = src, gs
    json.dump(v, open(f"{w}/focal/{c}.json", "w"))
    print(c, "cameras:", src)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("static", "focal", "given", "select"))
    p.add_argument("--work", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--focal", type=float, default=0.0, help="given: the known focal (px)")
    a = p.parse_args()
    os.makedirs(f"{a.work}/focal", exist_ok=True)
    if a.stage == "static":
        raise SystemExit(0 if static(a.work, a.clip) else 10)  # exit 0: static (the rule applies), 10: moving (1: an error)
    {"focal": lambda: focal(a.work, a.clip), "given": lambda: given(a.work, a.clip, a.focal),
     "select": lambda: select(a.work, a.clip)}[a.stage]()


if __name__ == "__main__":
    main()
