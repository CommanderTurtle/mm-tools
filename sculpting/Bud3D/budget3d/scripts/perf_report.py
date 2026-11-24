"""Speed / memory report for a budget-sweep gallery (needs the per-run "perf" records gallery.py writes).

    python scripts/perf_report.py --out gallery [--smi nvidia_smi.csv] [--keep_warmup] [--write_to DIR]

Unit of measurement: one sampling call = one batch of B objects at one budget.  Every budget of a batch runs on the same GPU with the same inputs and noise, so speedups are
*paired*: for each batch, t(100%) / t(budget), then summarised over batches.  The first batch of each
shard is dropped by default (CUDA / cuBLAS warm-up lands in its 100% call).

Writes perf_report.md and perf_report.json into <out> (or --write_to).
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict


def summ(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "mean": st.mean(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0,
            "median": st.median(xs), "p10": q(0.1), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def smi_summary(path):
    """nvidia-smi CSV (index, utilization.gpu, memory.used, power.draw) -> per-GPU mean/peak."""
    per = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                g = row["index"]
                per[g]["util"].append(float(row["utilization.gpu [%]"].split()[0]))
                per[g]["mem"].append(float(row["memory.used [MiB]"].split()[0]))
                per[g]["power"].append(float(row["power.draw [W]"].split()[0]))
            except (KeyError, ValueError, IndexError):
                continue
    return {g: {"samples": len(v["mem"]), "util_mean": st.mean(v["util"]), "mem_mean_mib": st.mean(v["mem"]),
                "mem_peak_mib": max(v["mem"]), "power_mean_w": st.mean(v["power"]), "power_peak_w": max(v["power"])}
            for g, v in sorted(per.items()) if v["mem"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smi", default=None, help="nvidia-smi CSV logged during the run")
    ap.add_argument("--keep_warmup", action="store_true", help="keep each shard's first batch")
    ap.add_argument("--write_to", default=None, help="folder for the two report files (default: --out)")
    a = ap.parse_args()

    objs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.out, "objects", "*.json")))]
    runs = [(o["name"], r) for o in objs for r in o["runs"] if "perf" in r]
    if not runs:
        raise SystemExit(f"no perf records in {a.out}/objects (gallery.py predates the instrumentation?)")
    settings = json.load(open(os.path.join(a.out, "manifest.json"))) if os.path.exists(
        os.path.join(a.out, "manifest.json")) else {}

    # one record per sampling call: key = (shard, batch_index, budget)
    calls = {}
    for name, r in runs:
        p = r["perf"]
        calls.setdefault((p["shard"], p["batch_index"], r["budget"]), (r, p))
    first = {}
    for (sh, bi, _b) in calls:
        first[sh] = min(first.get(sh, bi), bi)
    dropped = 0
    if not a.keep_warmup:
        n0 = len(calls)
        calls = {k: v for k, v in calls.items() if k[1] != first[k[0]]}
        dropped = n0 - len(calls)
    # only the main batch size: a shard's last, smaller batch is less efficient per object (and, with
    # torch.compile, recompiles for its new shape), so mixing it in would skew both time and memory
    from collections import Counter
    main_b = Counter(p["batch"] for _, (r, p) in calls.items()).most_common(1)[0][0]
    n1 = len(calls)
    calls = {k: v for k, v in calls.items() if v[1]["batch"] == main_b}
    other_batch = n1 - len(calls)
    budgets = sorted({k[2] for k in calls}, reverse=True)
    by_b = defaultdict(list)
    for k, (r, p) in calls.items():
        by_b[k[2]].append((k, r, p))
    t100 = {(k[0], k[1]): p["sample_batch_s"] / p["batch"] for k, (r, p) in calls.items() if k[2] == max(budgets)}

    # per-object stages (decode, render) are measured per object, not per call
    per_obj = defaultdict(list)
    for name, r in runs:
        p = r["perf"]
        if (a.keep_warmup or p["batch_index"] != first.get(p["shard"])) and p["batch"] == main_b:
            per_obj[r["budget"]].append(p)

    rows = []
    for b in budgets:
        recs = by_b[b]
        s_obj = [p["sample_batch_s"] / p["batch"] for _, _, p in recs]
        steps = recs[0][2]["steps"]
        paired = [t100[(k[0], k[1])] / (p["sample_batch_s"] / p["batch"]) for k, _, p in recs if (k[0], k[1]) in t100]
        po = per_obj[b]
        dec = [p["decode_s"] for p in po]
        ren = [p["render_s"] for p in po]
        fps = [p["render_fps"] for p in po]
        e2e = [p["sample_batch_s"] / p["batch"] + p["decode_s"] + p["render_s"] for p in po]
        rows.append({
            "budget": b, "tokens": recs[0][1]["tokens"],
            "n4": recs[0][1]["n4"], "n2": recs[0][1]["n2"], "n1": recs[0][1]["n1"],
            "batches": len(recs), "batch_sizes": sorted({p["batch"] for _, _, p in recs}),
            "sample_s_per_obj": summ(s_obj),
            "ms_per_step_per_obj": 1000 * st.mean(s_obj) / steps,
            "throughput_obj_per_min": 60 / st.mean(s_obj),
            "speedup_paired": summ(paired),
            "speedup_of_means": (st.mean([v for v in t100.values()]) / st.mean(s_obj)) if t100 else None,
            "token_ratio": 8192 / recs[0][1]["tokens"],
            "sample_peak_alloc_gb": summ([p["sample_mem"]["peak_alloc_mb"] / 1024 for _, _, p in recs]),
            "sample_peak_reserved_gb": summ([p["sample_mem"]["peak_reserved_mb"] / 1024 for _, _, p in recs]),
            "sample_avg_alloc_gb": summ([p["sample_mem"]["avg_alloc_mb"] / 1024 for _, _, p in recs]),
            "sample_base_alloc_gb": summ([p["sample_mem"]["base_alloc_mb"] / 1024 for _, _, p in recs]),
            "sample_peak_minus_base_per_obj_gb": summ([(p["sample_mem"]["peak_alloc_mb"] - p["sample_mem"]["base_alloc_mb"])
                                                       / 1024 / p["batch"] for _, _, p in recs]),
            "decode_s": summ(dec), "decode_peak_alloc_gb": summ([p["decode_mem"]["peak_alloc_mb"] / 1024 for p in po]),
            "render_s_per_orbit": summ(ren), "render_fps": summ(fps),
            "render_peak_alloc_gb": summ([p["render_mem"]["peak_alloc_mb"] / 1024 for p in po]),
            "end_to_end_s_per_obj": summ(e2e),
        })
    gpus = sorted({p["gpu"] for _, (r, p) in calls.items()})
    rep = {"gallery": os.path.basename(os.path.abspath(a.out)), "objects": len(objs), "objects_with_perf": len({n for n, _ in runs}),
           "sampling_calls": len(calls), "dropped_warmup_calls": dropped, "dropped_other_batch_calls": other_batch, "gpus": gpus,
           "turbo": bool(settings.get("turbo")), "train_step": settings.get("train_step"), "precision": settings.get("precision", "fp32"), "steps": settings.get("steps"), "cfg": settings.get("cfg"),
           "num_gaussians": settings.get("num_gaussians"), "frames": settings.get("frames"), "size": settings.get("size"),
           "per_budget": rows}
    if a.smi and os.path.exists(a.smi):
        rep["nvidia_smi"] = smi_summary(a.smi)
    dst = a.write_to or a.out
    os.makedirs(dst, exist_ok=True)
    json.dump(rep, open(os.path.join(dst, "perf_report.json"), "w"), indent=1)
    guidance = "no guidance (1 pass/step)" if rep["turbo"] or not rep["cfg"] else f"CFG {rep['cfg']} (2 passes/step)"

    f2 = lambda x: f"{x:.2f}"
    L = [f"# Speed and memory: {os.path.basename(os.path.abspath(a.out))}", "",
         f"Checkpoint step {rep['train_step']}, {rep['steps']} Euler steps, {guidance}, {rep['precision']}, "
         f"GPU: {', '.join(gpus)}.  {rep['objects_with_perf']} objects, {rep['sampling_calls']} sampling calls "
         f"(batches of {', '.join(map(str, sorted({s for r in rows for s in r['batch_sizes']})))} objects); "
         f"{dropped} warm-up calls (first batch of each shard) and {other_batch} calls with a smaller final batch excluded.", "",
         "Speedup is paired: t(100%)/t(budget) for the same batch on the same GPU, mean ± std over batches. "
         "Memory is torch's allocator (excludes the ~0.5 GB CUDA context); peak = high-water mark during the "
         "sampling call, avg = time-averaged allocation polled every 20 ms; both include the resident weights "
         "(\"base\").", "",
         "## Sampling (the part the budget changes)", "",
         "| budget | tokens | 4x/2x/1x | s/obj (mean ± std) | median | ms/step/obj | obj/min | speedup | token ratio "
         "| peak alloc GB | avg alloc GB | peak reserved GB | activations GB/obj |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, sp = r["sample_s_per_obj"], r["speedup_paired"]
        L.append(f"| {round(r['budget'] * 100)}% | {r['tokens']} | {r['n4']}/{r['n2']}/{r['n1']} | "
                 f"{f2(s['mean'])} ± {f2(s['std'])} | {f2(s['median'])} | {r['ms_per_step_per_obj']:.0f} | "
                 f"{r['throughput_obj_per_min']:.2f} | {f2(sp.get('mean', float('nan')))}× ± {f2(sp.get('std', 0))} | "
                 f"{f2(r['token_ratio'])}× | {f2(r['sample_peak_alloc_gb']['mean'])} (max {f2(r['sample_peak_alloc_gb']['max'])}) | "
                 f"{f2(r['sample_avg_alloc_gb']['mean'])} | {f2(r['sample_peak_reserved_gb']['max'])} | "
                 f"{f2(r['sample_peak_minus_base_per_obj_gb']['mean'])} |")
    L += ["", "## Decode, render, end to end (per object)", "",
          "| budget | decode s | decode peak GB | render s / orbit | render fps | render peak GB | end-to-end s/obj |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {round(r['budget'] * 100)}% | {f2(r['decode_s']['mean'])} | {f2(r['decode_peak_alloc_gb']['max'])} | "
                 f"{f2(r['render_s_per_orbit']['mean'])} | {f2(r['render_fps']['mean'])} | "
                 f"{f2(r['render_peak_alloc_gb']['max'])} | {f2(r['end_to_end_s_per_obj']['mean'])} |")
    if "nvidia_smi" in rep:
        L += ["", "## Whole-GPU view (nvidia-smi, every 5 s over the whole job, includes CUDA context)", "",
              "| GPU | util % (mean) | memory MiB (mean) | memory MiB (peak) | power W (mean / peak) |", "|---|---|---|---|---|"]
        for g, v in rep["nvidia_smi"].items():
            L.append(f"| {g} | {v['util_mean']:.0f} | {v['mem_mean_mib']:.0f} | {v['mem_peak_mib']:.0f} | "
                     f"{v['power_mean_w']:.0f} / {v['power_peak_w']:.0f} |")
    open(os.path.join(dst, "perf_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
