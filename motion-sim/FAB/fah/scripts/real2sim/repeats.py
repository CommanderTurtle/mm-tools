"""Repeat-aware rates of the real-to-sim study: the replay is not bit-reproducible (CoACD), so each clip is replayed
--runs times per controller (CoACD seeds 0 / 1 / 2); per clip the strict / 8 cm pass counts and the GT diagnostics' counts, a pass
= >= --need of the repeats, and the rates per split group x source x seen by FAH in training -> markdown on stdout, json (--json).

    python scripts/real2sim/repeats.py --root $R2S --split bench/splits/real2sim.json \
        --runs sim_cart,sim_cart_r2,sim_cart_r3 --diags _cart,_cart_r2,_cart_r3 --json /tmp/rep_cart.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.real2sim.candidates import failing_term, page, secondary  # noqa: E402


def counts(c, runs):
    ps = [page(os.path.join(c, r)) for r in runs]
    ps = [m["page"] for m in ps if m]
    return (sum(p["success"] for p in ps), sum(secondary(p) for p in ps), len(ps), [failing_term(p) or "PASS" for p in ps])


def main(args):
    sp = json.load(open(args.split))
    seen = set(sp["fah_seen"])
    runs, diags = args.runs.split(","), args.diags.split(",")
    rows = []
    for g in args.groups.split(","):
        for n in sp[g]:
            c = os.path.join(args.root, n)
            clip = json.load(open(os.path.join(c, "clip.json")))
            seg = os.path.exists(os.path.join(c, "recon_s3do", "layout.json"))
            s, s8, k, terms = counts(c, runs)
            go = counts(c, ["sim_diag_gtobj" + d for d in diags]) if os.path.exists(os.path.join(c, "gt.npz")) else None
            gg = counts(c, ["sim_diag_gtgt" + d for d in diags]) if os.path.exists(os.path.join(c, "gt.npz")) else None
            rows.append(dict(candidate=n, group=g, source=clip["source"].split()[0], object=clip.get("object"), seen=n in seen,
                             segmented=seg, runs=k, strict_n=s, secondary_n=s8, terms=terms, strict=k > 0 and s >= args.need,
                             secondary=k > 0 and s8 >= args.need, all_pass=k == len(runs) and s == k,
                             gtobj_n=None if not go or not go[2] else go[0], gtobj_runs=go[2] if go else 0,
                             gtgt_n=None if not gg or not gg[2] else gg[0], gtgt_runs=gg[2] if gg else 0))
    print("| clip | group | source | seen | strict passes | 8 cm passes | terms | FAH + GT obj | GT + GT |\n|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['candidate']} | {r['group']} | {r['source']} | {'yes' if r['seen'] else 'no'} | "
              f"{r['strict_n']} / {r['runs']} | {r['secondary_n']} / {r['runs']} | {', '.join(r['terms']) if r['segmented'] else 'segmentation'} | "
              f"{r_n(r, 'gtobj')} | {r_n(r, 'gtgt')} |")
    maj = lambda r, k: r[k + "_n"] is not None and r[k + "_n"] >= args.need
    print()
    for name, sel in [("all", lambda r: True)] + [(f"{src} {'seen' if sn else 'unseen'}", (lambda src, sn: lambda r: r["source"] == src and r["seen"] == sn)(src, sn))
                                                  for src in ("DexYCB", "HOT3D") for sn in (True, False)] + \
            [(f"{g}", (lambda g: lambda r: r["group"] == g)(g)) for g in args.groups.split(",")]:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        d = [r for r in rs if r["gtobj_n"] is not None]
        print(f"{name:14s}: strict {sum(r['strict'] for r in rs)} / {len(rs)}, 8 cm {sum(r['secondary'] for r in rs)} / {len(rs)}, "
              f"all {len(runs)} repeats {sum(r['all_pass'] for r in rs)} / {len(rs)}"
              + (f"; FAH + GT object {sum(maj(r, 'gtobj') for r in d)} / {len(d)}, GT + GT {sum(maj(r, 'gtgt') for r in d)} / {len(d)}" if d else ""))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


def r_n(r, k):
    return "-" if r[k + "_n"] is None else f"{r[k + '_n']} / {r[k + '_runs']}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--groups", default="eval_old,eval_new")
    p.add_argument("--runs", required=True, help="comma-separated replay dirs, one per repeat")
    p.add_argument("--diags", default="", help="comma-separated suffixes of the sim_diag_gtobj / sim_diag_gtgt dirs, one per repeat")
    p.add_argument("--need", type=int, default=2, help="passes needed of the repeats")
    p.add_argument("--json", default="")
    main(p.parse_args())
