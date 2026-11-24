"""Joint ranges of G1 robot models vs Unitree G1 Model 15 (fab/retarget/g1_model15.py: the ranges of Unitree's
g1_29dof_mode_15.urdf). URDF <limit>, MJCF joint range (compiler angle respected), or a Python dict file.
    python bench/g1_limits_audit.py name=path [name=path ...] [--tol 1e-4] [--out audit.json]"""
import argparse, ast, json, math, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path


def model15():
    from fab.retarget.g1_model15 import RANGES

    return dict(RANGES)


def from_urdf(p):
    out = {}
    for j in ET.parse(p).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "continuous") and lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def from_mjcf(p):
    root = ET.parse(p).getroot()
    comp = root.find("compiler")
    deg = comp is None or comp.get("angle", "degree") == "degree"
    cls_range = {}
    for d in root.iter("default"):
        jd = d.find("joint")
        if jd is not None and jd.get("range"):
            cls_range[d.get("class", "main")] = jd.get("range")
    out = {}

    def walk(el, cls):
        for ch in el:
            c = ch.get("childclass", cls) if ch.tag == "body" else cls
            if ch.tag == "joint" and ch.get("type", "hinge") == "hinge" and ch.get("name"):
                r = ch.get("range") or cls_range.get(ch.get("class", c))
                if r:
                    lo, hi = map(float, r.split())
                    out[ch.get("name")] = (math.radians(lo), math.radians(hi)) if deg else (lo, hi)
            walk(ch, c)
    wb = root.find("worldbody")
    walk(wb if wb is not None else root, "main")
    return out, ("degree" if deg else "radian")


def from_pydict(p):
    src = open(p).read()
    lo = hi = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.keyword) and node.arg in ("joint_pos_lower_limit", "joint_pos_upper_limit"):
            d = ast.literal_eval(node.value)
            lo, hi = (d, hi) if node.arg.endswith("lower_limit") else (lo, d)
    return {k: (lo[k], hi[k]) for k in lo}


def load(p):
    p = str(p)
    if p.endswith(".urdf"):
        return from_urdf(p), "urdf"
    if p.endswith(".xml"):
        r, unit = from_mjcf(p)
        return r, f"mjcf ({unit})"
    return from_pydict(p), "py dict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = model15()
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        rng, kind = load(path)
        miss = [j for j in ref if j not in rng]
        mm = []
        for j, (lo, hi) in ref.items():
            if j in rng:
                l2, h2 = rng[j]
                if abs(l2 - lo) > a.tol or abs(h2 - hi) > a.tol:
                    mm.append({"joint": j, "m15": [lo, hi], "model": [l2, h2],
                               "d_lo_deg": round(math.degrees(l2 - lo), 3), "d_hi_deg": round(math.degrees(h2 - hi), 3)})
        res[name] = {"path": path, "kind": kind, "n_joints": len(rng), "missing": miss, "mismatch": mm}
        print(f"== {name}: {kind}, {len(rng)} joints, {len(ref) - len(miss)}/29 Model-15 joints present, "
              f"{len(mm)} range mismatches (tol {a.tol} rad)  [{path}]")
        for j in miss:
            print(f"   missing {j}")
        for m in mm:
            print(f"   {m['joint']:28s} M15 [{m['m15'][0]:+.5f}, {m['m15'][1]:+.5f}]  model [{m['model'][0]:+.5f}, "
                  f"{m['model'][1]:+.5f}]  d_lo {m['d_lo_deg']:+.2f} deg  d_hi {m['d_hi_deg']:+.2f} deg")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
