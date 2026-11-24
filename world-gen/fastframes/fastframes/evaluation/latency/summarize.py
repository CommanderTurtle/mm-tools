#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Summarize latency from component JSON records.

    python summarize.py RESULTS_DIR [--output summary.json]

Step latency sums separately measured component medians. Records are matched by
settings; the fastest match is used and all candidates are listed. Baseline DiT
records prefer the upstream profiler, falling back to the native baseline mode.
If only the final-timestep DiT record exists, FastFrames counts it twice.
Three-latent baseline chunks use three single-latent VAE times; direct chunk
measurements are reported separately. See docs/inference.md#profiling for formulas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models import MODEL_DEFAULTS, tuned_split_kv

FINAL_T = {m: d["timestep"] for m, d in MODEL_DEFAULTS.items()}
FIRST_T = 1000.0


# Paper values (ms unless noted).
PAPER = {
    "table1": {"CausVid": 662, "Self-Forcing": 717, "Causal Forcing++": 272,
               "FastFrames-S": 36, "FastFrames-S (8 GPU)": 14, "FastFrames-L": 158, "FastFrames-L (8 GPU)": 45},
    "fig2": {"CF++ DiT evaluation": 41.6, "CF++ VAE": 147, "CF++ total": 271.8,
             "FF-S DiT evaluation": 16.4, "FF-S KV refinement": 2, "FF-S pixel decoder": 1, "FF-S total": 35.8},
    "sec3.3": {"KV refinement 1.3B": 2, "KV refinement 14B": 6},
    "sec3.4": {"1.3B DiT baseline": 41.6, "1.3B DiT optimized": 16.4, "14B DiT baseline": 89.2,
               "14B DiT optimized": 75.6, "split-KV reduction (%)": 13, "1.3B DiT CP8": 5.8, "14B DiT CP8": 20.3},
    "fig3": {"VAE decoder": 147, "ConvNeXt pixel decoder": 0.9, "VAE decoder params (M)": 73.3,
             "ConvNeXt decoder params (M)": 63.9},
    "table2": {  # B200 measurements
        "FF-S 1 GPU DiT 480p": 16, "FF-S 1 GPU DiT 720p": 60, "FF-S 1 GPU KV 480p": 2, "FF-S 1 GPU KV 720p": 5,
        "FF-S 1 GPU VAE 480p": 147, "FF-S 1 GPU VAE 720p": 327, "FF-S 1 GPU decoder 480p": 1,
        "FF-S 1 GPU decoder 720p": 2, "FF-S 1 GPU total 480p": 36, "FF-S 1 GPU total 720p": 127,
        "FF-L 8 GPU DiT 480p": 20, "FF-L 8 GPU DiT 720p": 54, "FF-L 8 GPU KV 480p": 3, "FF-L 8 GPU KV 720p": 7,
        "FF-L 8 GPU decoder 480p": 1, "FF-L 8 GPU decoder 720p": 1, "FF-L 8 GPU total 480p": 45,
        "FF-L 8 GPU total 720p": 116},
}


class Records:
    def __init__(self, root):
        self.items = []
        for path in sorted(Path(root).rglob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and "component" in record and "latency_ms" in record:
                self.items.append((path.name, record))

    def find(self, component, *, model=None, resolution="480p", gpus=1, **settings):
        """Fastest matching record -> ``(latency_ms, source, candidates)`` or ``None``."""
        found = []
        for name, r in self.items:
            if r["component"] != component or r.get("resolution") != resolution or r.get("gpus") != gpus:
                continue
            if model is not None and r.get("model") != model:
                continue
            s = r.get("settings", {})
            if all(_match(s.get(k), v) for k, v in settings.items()):
                found.append((r["latency_ms"], name, r))
        if not found:
            return None
        found.sort(key=lambda x: x[0])
        return found[0][0], found[0][1], [(n, round(v, 4)) for v, n, _ in found], found[0][2]


def _match(value, want):
    if callable(want):
        return want(value)
    if isinstance(want, float):
        return value is not None and abs(float(value) - want) < 1e-3
    return value == want


class Summary:
    def __init__(self, records):
        self.r, self.rows, self.sources = records, [], {}

    def get(self, key, component, **query):
        hit = self.r.find(component, **query)
        if hit is None:
            return None
        self.sources[key] = {"used": hit[1], "candidates": hit[2]}
        return hit[0]

    def dit(self, model, res, gpus, t=None, **extra):
        query = dict(model=model, resolution=res, gpus=gpus, mode="optimized", chunk=1,
                     split_kv=tuned_split_kv(model, res, gpus))
        if t is not None:
            query["timestep"] = t
        query.update(extra)
        key = f"dit {model} {res} {gpus}gpu " + " ".join(f"{k}={v}" for k, v in extra.items() if not callable(v))
        return self.get(f"{key} t={t}", "dit", **query)

    def ff_parts(self, model, res, gpus):
        final = self.dit(model, res, gpus, t=FINAL_T[model])
        first = self.dit(model, res, gpus, t=FIRST_T)
        kv = self.get(f"kv {model} {res} {gpus}gpu", "kv-refine", model=model, resolution=res, gpus=gpus)
        dec = self.get(f"decoder {model} {res} {gpus}gpu", "pixel-decoder", model=model, resolution=res, gpus=gpus)
        dits = [final, first if first is not None else final]
        total = None if None in (final, kv, dec) else sum(dits) + kv + dec
        mean_dit = None if final is None else sum(dits) / 2
        return dict(dit_final_t=final, dit_t1000=first, dit_mean=mean_dit, kv=kv, decoder=dec, total=total)

    def vae(self, res, chunk=1):
        return self.get(f"vae {res} chunk{chunk}", "vae", resolution=res, gpus=1,
                        latents_per_call=lambda v: (v or 1) == chunk)

    def baseline_dit(self, chunk, mode=None):
        """Upstream-code record (``baselines.py``) when present, else ``--mode baseline``."""
        for m in ([mode] if mode else ["upstream", "baseline"]):
            value = self.get(f"dit 1.3b {m} chunk{chunk}", "dit", model="1.3b", resolution="480p", gpus=1,
                             mode=m, chunk=chunk)
            if value is not None:
                return value
        return None

    def add(self, group, name, measured, formula=""):
        paper = PAPER[group].get(name)
        rel = None if measured is None or paper is None else 100.0 * (measured - paper) / paper
        self.rows.append(dict(group=group, quantity=name, paper=paper, measured=measured,
                              relative_diff_percent=rel, formula=formula))


def build(records):
    s = Summary(records)
    ffs1, ffs8 = s.ff_parts("1.3b", "480p", 1), s.ff_parts("1.3b", "480p", 8)
    ffl1, ffl8 = s.ff_parts("14b", "480p", 1), s.ff_parts("14b", "480p", 8)
    ffs1_720, ffl8_720 = s.ff_parts("1.3b", "720p", 1), s.ff_parts("14b", "720p", 8)
    vae1, vae3 = s.vae("480p", 1), s.vae("480p", 3)
    base1, base3 = s.baseline_dit(1), s.baseline_dit(3)
    vae3_paper = None if vae1 is None else 3 * vae1

    def lin(*terms):
        return None if any(t is None for _, t in terms) else sum(c * t for c, t in terms)

    # Table 1
    s.add("table1", "CausVid", lin((4, base3), (1, vae3_paper)), "4 x DiT(3 latents) + 3 x VAE(1 latent)")
    s.add("table1", "Self-Forcing", lin((5, base3), (1, vae3_paper)), "5 x DiT(3 latents) + 3 x VAE(1 latent)")
    s.add("table1", "Causal Forcing++", lin((3, base1), (1, vae1)), "3 x DiT(1 latent) + VAE(1 latent)")
    for name, parts in (("FastFrames-S", ffs1), ("FastFrames-S (8 GPU)", ffs8),
                        ("FastFrames-L", ffl1), ("FastFrames-L (8 GPU)", ffl8)):
        s.add("table1", name, parts["total"], "DiT(t=1000) + DiT(t_final) + KV refinement + pixel decoder")
    extra = {"CausVid (direct 3-latent VAE)": lin((4, base3), (1, vae3)),
             "Self-Forcing (direct 3-latent VAE)": lin((5, base3), (1, vae3))}
    native1, native3 = s.baseline_dit(1, "baseline"), s.baseline_dit(3, "baseline")
    extra.update({"baseline DiT, fastframes.wan, 1 latent": native1,
                  "baseline DiT, fastframes.wan, 3 latents": native3,
                  "Causal Forcing++ (fastframes.wan DiT)": lin((3, native1), (1, vae1)),
                  "Self-Forcing (fastframes.wan DiT)": lin((5, native3), (1, vae3_paper)),
                  "CausVid (fastframes.wan DiT)": lin((4, native3), (1, vae3_paper))})
    # Fig. 2
    s.add("fig2", "CF++ DiT evaluation", base1)
    s.add("fig2", "CF++ VAE", vae1)
    s.add("fig2", "CF++ total", lin((3, base1), (1, vae1)), "3 x DiT + VAE")
    s.add("fig2", "FF-S DiT evaluation", ffs1["dit_final_t"])
    s.add("fig2", "FF-S KV refinement", ffs1["kv"])
    s.add("fig2", "FF-S pixel decoder", ffs1["decoder"])
    s.add("fig2", "FF-S total", ffs1["total"], "2 x DiT + KV + decoder")
    # Sec. 3.3 / 3.4
    s.add("sec3.3", "KV refinement 1.3B", ffs1["kv"])
    s.add("sec3.3", "KV refinement 14B", ffl1["kv"])
    s.add("sec3.4", "1.3B DiT baseline", base1)
    s.add("sec3.4", "1.3B DiT optimized", ffs1["dit_final_t"])
    base14 = s.get("dit 14b baseline", "dit", model="14b", resolution="480p", gpus=1, mode="baseline", chunk=1)
    s.add("sec3.4", "14B DiT baseline", base14)
    s.add("sec3.4", "14B DiT optimized", ffl1["dit_final_t"])
    unsplit = s.dit("1.3b", "480p", 1, t=FINAL_T["1.3b"], split_kv=1)
    reduction = None if None in (unsplit, ffs1["dit_final_t"]) else 100 * (1 - ffs1["dit_final_t"] / unsplit)
    s.add("sec3.4", "split-KV reduction (%)", reduction, "1 - DiT(split 5) / DiT(split 1)")
    s.add("sec3.4", "1.3B DiT CP8", ffs8["dit_final_t"])
    s.add("sec3.4", "14B DiT CP8", ffl8["dit_final_t"])
    # Fig. 3
    conv_rec = records.find("pixel-decoder", model="1.3b", resolution="480p", gpus=1)
    vae_rec = records.find("vae", resolution="480p", gpus=1, latents_per_call=lambda v: (v or 1) == 1)
    s.add("fig3", "VAE decoder", vae1)
    s.add("fig3", "ConvNeXt pixel decoder", ffs1["decoder"])
    for name, rec in (("VAE decoder params (M)", vae_rec), ("ConvNeXt decoder params (M)", conv_rec)):
        settings = {} if rec is None else rec[3].get("settings", {})
        count = settings.get("decoder_parameters", settings.get("parameters"))
        s.add("fig3", name, None if count is None else count / 1e6)
    # Table 2 (B200)
    vae720 = s.vae("720p", 1)
    for res, ffs, ffl, vae in (("480p", ffs1, ffl8, vae1), ("720p", ffs1_720, ffl8_720, vae720)):
        s.add("table2", f"FF-S 1 GPU DiT {res}", ffs["dit_mean"], "mean of the two DiT evaluations")
        s.add("table2", f"FF-S 1 GPU KV {res}", ffs["kv"])
        s.add("table2", f"FF-S 1 GPU VAE {res}", vae)
        s.add("table2", f"FF-S 1 GPU decoder {res}", ffs["decoder"])
        s.add("table2", f"FF-S 1 GPU total {res}", ffs["total"])
        s.add("table2", f"FF-L 8 GPU DiT {res}", ffl["dit_mean"], "mean of the two DiT evaluations")
        s.add("table2", f"FF-L 8 GPU KV {res}", ffl["kv"])
        s.add("table2", f"FF-L 8 GPU decoder {res}", ffl["decoder"])
        s.add("table2", f"FF-L 8 GPU total {res}", ffl["total"])
    parts = {"FF-S 1 GPU 480p": ffs1, "FF-S 8 GPU 480p": ffs8, "FF-L 1 GPU 480p": ffl1, "FF-L 8 GPU 480p": ffl8,
             "FF-S 1 GPU 720p": ffs1_720, "FF-L 8 GPU 720p": ffl8_720,
             "baseline DiT 1.3B 1 latent": base1, "baseline DiT 1.3B 3 latents": base3,
             "VAE 1 latent 480p": vae1, "VAE 3 latents 480p": vae3, "VAE 1 latent 720p": vae720}
    return dict(rows=s.rows, components=parts, additional=extra, sources=s.sources)


def _fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="directory of component JSON records (searched recursively)")
    p.add_argument("--output", help="write the assembled summary as JSON")
    args = p.parse_args()
    summary = build(Records(args.results))
    group = None
    for row in summary["rows"]:
        if row["group"] != group:
            group = row["group"]
            print(f"\n{group}\n{'quantity':42s} {'paper':>8s} {'measured':>9s} {'diff %':>7s}  formula")
        print(f"{row['quantity']:42s} {_fmt(row['paper']):>8s} {_fmt(row['measured']):>9s} "
              f"{_fmt(row['relative_diff_percent'], 1):>7s}  {row['formula']}")
    print("\nadditional")
    for name, value in summary["additional"].items():
        print(f"{name:42s} {'':8s} {_fmt(value):>9s}")
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
