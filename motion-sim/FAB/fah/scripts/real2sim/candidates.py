"""Candidate table of the real-to-sim page study: per clip the strict page rule (sim.py page_success) and a disclosed secondary
rule (same terms, "follows" within 8 cm), the failing term, how much the contact optimisation moved the hand (contact.py:
per-frame rigid shift of FAH's raw hand, = vertex displacement) and the controller's finger-target shifts, the DexYCB GT checks
(eval_gt.json) and the stage diagnosis from the GT-object / GT-hand replays (sim_diag_gtobj / sim_diag_gtgt): object stage if
FAH + GT object passes where the full pipeline fails, hand / contact stage if only GT hand + GT object passes, replay / rule if
neither does -> markdown on stdout, json (--json).

    python scripts/real2sim/candidates.py --root $R2S --candidates drill,mug,... --json candidates.json
"""
import argparse
import json
import os

import numpy as np

RUN = "sim_fah_recon_contact_ctrl_hold_pen"


def page(d):
    f = os.path.join(d, "metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


def secondary(p, follow=80.0):
    """The strict rule with "follows" relaxed to `follow` mm (every other term unchanged)."""
    if p is None or not p["applicable"]:
        return False
    return bool(p["lift_mm"] >= 50 and p["held_frac"] >= 0.8 and p["held_s"] >= 0.5 and p["end_held"] and p["pen95_mm"] <= 10
                and p["follow_mm"] is not None and p["follow_mm"] <= follow)


def failing_term(p):
    if p is None:
        return "no run"
    if not p["applicable"]:
        return "n/a: the video track lifts < 5 cm"
    for ok, name in ((p["lift_mm"] >= 50, "not lifted"), (p["held_frac"] >= 0.8 and p["held_s"] >= 0.5, "not held"),
                     (p["end_held"], "dropped"), (p["pen95_mm"] <= 10, "penetration"),
                     (p["follow_mm"] is not None and p["follow_mm"] <= 50, "follows > 5 cm")):
        if not ok:
            return name
    return ""


def contact_shift(c, run=RUN):
    """mean / max displacement of the hand by the contact optimisation over the frames where it moved it, mm."""
    f = os.path.join(c, "contact_fah_recon_shape.npz" if "nohold" not in run else "contact_fah_recon_shape_track_nohold.npz")
    if not os.path.exists(f):
        return None, None
    dl = np.linalg.norm(np.load(f, allow_pickle=True)["delta"], axis=1)
    dl = dl[dl > 1e-6]
    return (round(float(dl.mean()) * 1e3, 1), round(float(dl.max()) * 1e3, 1)) if len(dl) else (0.0, 0.0)


def main(args):
    rows = []
    for n in args.candidates.split(","):
        c = os.path.join(args.root, n)
        clip = json.load(open(os.path.join(c, "clip.json")))
        m = page(os.path.join(c, args.run))
        p = m["page"] if m else None
        seg = os.path.exists(os.path.join(c, "masks.npz"))
        if seg and not os.path.exists(os.path.join(c, "recon_s3do", "layout.json")):
            on = np.nonzero(np.load(os.path.join(c, "masks.npz"))["obj"].any((1, 2)))[0]
            seg = False
        ev = json.load(open(os.path.join(c, "eval_gt.json"))) if os.path.exists(os.path.join(c, "eval_gt.json")) else {}
        go, gg = page(os.path.join(c, "sim_diag_gtobj" + args.diag)), page(os.path.join(c, "sim_diag_gtgt" + args.diag))
        go, gg = go and go["page"]["success"], gg and gg["page"]["success"]
        strict = bool(p and p["success"])
        if not seg and os.path.exists(os.path.join(c, "masks.npz")) and not len(on):
            stage = "segmentation (SAM 3 masks empty after propagation)"
        elif not seg and os.path.exists(os.path.join(c, "masks.npz")):
            stage = f"segmentation (SAM 3 object mask only on frames {on[0]}-{on[-1]}, none in the reconstruction window)"
        elif not seg:
            stage = "segmentation (SAM 3 found no object)"
        elif strict:
            stage = ""
        elif ev and go:
            stage = "object (reconstruction / track): FAH + GT object passes"
        elif ev and gg:
            stage = "hand / contact: only GT hand + GT object passes"
        elif ev and go is not None:
            stage = "replay / rule: GT hand + GT object fails too"
        else:
            stage = "egocentric, no GT object: " + failing_term(p)
        cm, cx = contact_shift(c, args.run)
        rows.append(dict(candidate=n, source=clip["source"], object=clip.get("object_ycb", clip.get("object")), segmented=seg,
                         strict=strict, secondary=secondary(p), failing=failing_term(p) if seg else "segmentation", stage=stage,
                         contact_shift_mm_mean=cm, contact_shift_mm_max=cx,
                         ctrl_shift_mm_median=m.get("ctrl_shift_mm_median") if m else None, ctrl_shift_mm_max=m.get("ctrl_shift_mm_max") if m else None,
                         page=p, gtobj_strict=go, gtgt_strict=gg,
                         gt=dict((k, ev.get(k)) for k in ("shape_chamfer_mm", "scale_err_pct", "obj_centroid_err_mm_median",
                                                          "hand_mpjpe_cam_mm_raw", "wrist_obj_rel_err_mm_raw")) if ev else None))
    hdr = ["candidate", "source", "strict", "8 cm", "failing term", "stage", "contact shift mean / max (mm)", "controller max (mm)",
           "FAH+GT obj", "GT+GT", "GT: shape CD / centre err / hand MPJPE / wrist-obj (mm)"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        g = r["gt"] or {}
        print("| " + " | ".join(str(x) for x in [
            r["candidate"], r["source"].replace("DexYCB ", "").replace("HOT3D ", "HOT3D "), "PASS" if r["strict"] else "fail",
            "PASS" if r["secondary"] else "fail", r["failing"], r["stage"], f'{r["contact_shift_mm_mean"]} / {r["contact_shift_mm_max"]}',
            r["ctrl_shift_mm_max"], {True: "pass", False: "fail", None: "-"}[r["gtobj_strict"]], {True: "pass", False: "fail", None: "-"}[r["gtgt_strict"]],
            " / ".join(str(g.get(k)) for k in ("shape_chamfer_mm", "obj_centroid_err_mm_median", "hand_mpjpe_cam_mm_raw", "wrist_obj_rel_err_mm_raw"))
            if g else "-"]) + " |")
    ns, n2 = sum(r["strict"] for r in rows), sum(r["secondary"] for r in rows)
    print(f"\nstrict {ns} / {len(rows)}, secondary (8 cm) {n2} / {len(rows)}")
    if args.split:  # rates per split group x source x seen by FAH in training
        sp = json.load(open(args.split))
        seen = set(sp.get("fah_seen", []))
        for g in [k for k in ("dev", "eval_old", "eval_new") if k in sp]:
            for src in ("DexYCB", "HOT3D"):
                for sn in (True, False):
                    rs = [r for r in rows if r["candidate"] in sp[g] and r["source"].startswith(src) and (r["candidate"] in seen) == sn]
                    if rs:
                        print(f"{g:9s} {src:6s} {'seen by FAH' if sn else 'unseen':11s}: strict {sum(r['strict'] for r in rs)} / {len(rs)}, "
                              f"8 cm {sum(r['secondary'] for r in rs)} / {len(rs)}, FAH + GT object "
                              f"{sum(bool(r['gtobj_strict']) for r in rs)} / {sum(r['gtobj_strict'] is not None for r in rs)}, GT + GT "
                              f"{sum(bool(r['gtgt_strict']) for r in rs)} / {sum(r['gtgt_strict'] is not None for r in rs)}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--json", default="")
    p.add_argument("--run", default=RUN, help="replay dir in each candidate")
    p.add_argument("--diag", default="", help="suffix of the sim_diag_gtobj / sim_diag_gtgt dirs")
    p.add_argument("--split", default="", help="split json (bench/splits/real2sim.json): rates per group, source, seen by FAH")
    main(p.parse_args())
