"""Quality against token budget for a 3D gallery (scripts/gallery.py output): how far each budget's object is from the
same model's 100 % result, and how far the 100 % result is from a reference model's (e.g. the original Idol).

    python scripts/eval_budget.py --gallery gallery_warmup0_final --ref gallery_idol_ref --out eval_idol.json

For every object, the saved latents of every budget are decoded into Gaussians (with the same decoder seed, so the
anchor sampling is shared) and rendered from --views views spread over the object's own orbit (the one fixed at
100 %).  Each budget is compared with the 100 % renders: PSNR and SSIM on the RGB images, and LPIPS (AlexNet) when the
lpips package is importable.  The noise floor is the 100 % result decoded with a different decoder seed: differences
below it are not measurable.  --ref compares the reference gallery's 100 % object, rendered from the same views, with
this gallery's 100 % object.  Writes per-object rows and per-budget means to --out (JSON) and a Markdown table next to it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

import numpy as np
import torch
import torch.nn.functional as F




def ssim(a, b):
    """Mean SSIM of two (N, 3, H, W) images in [0, 1]: 11x11 Gaussian window, sigma 1.5, per channel."""
    g = torch.exp(-(torch.arange(11, device=a.device, dtype=torch.float32) - 5) ** 2 / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :] / g.sum() ** 2).expand(3, 1, 11, 11)
    blur = lambda x: F.conv2d(x, w, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean())


def psnr(a, b):
    return float(10 * torch.log10(1.0 / F.mse_loss(a, b).clamp_min(1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--ref", default=None, help="a gallery of another model whose 100 %% objects are the reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    from budget3d.models.load import decode_gaussians as decode, idol_weights_dir, load_decoders
    from budget3d.render.splat import Splats, orbit_positions, render
    dev = torch.device("cuda")
    dec, dec_gs, cfg = load_decoders(idol_weights_dir(), dev)
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:                                                  # noqa: BLE001
        print(f"[eval] no LPIPS ({type(e).__name__}: {e})", flush=True)
        lp = None
    man = json.load(open(os.path.join(a.gallery, "manifest.json")))
    objs = man["objects"][: a.limit] if a.limit else man["objects"]
    si, sn = (int(v) for v in a.shard.split("/"))
    objs = objs[si::sn]

    def views(latent_path, orbit, seed):
        z = torch.load(latent_path, map_location=dev, weights_only=False)["latent"]
        torch.manual_seed(seed)
        sp = Splats(decode(z, dec, dec_gs, cfg, a.num_gaussians))
        cams = orbit_positions(orbit["yaw"], orbit["pitch"], orbit["radius"], a.views)
        return torch.stack([render(sp, c, orbit["fov"], size=a.size) for c in cams]).permute(0, 3, 1, 2).float().clamp(0, 1)

    def compare(x, y):
        out = {"psnr": psnr(x, y), "ssim": ssim(x, y)}
        if lp is not None:
            with torch.no_grad():
                out["lpips"] = float(lp(x * 2 - 1, y * 2 - 1).mean())
        return out

    rows = []
    with torch.no_grad():
        for k, o in enumerate(objs):
            runs = {round(r["budget"] * 100): r for r in o["runs"]}
            if 100 not in runs:
                continue
            orbit = runs[100]["orbit"]
            lat = lambda g, b: os.path.join(g, "latents", f"{o['name']}_b{b:03d}.pt")
            full = views(lat(a.gallery, 100), orbit, 0)
            row = {"name": o["name"], "floor": compare(views(lat(a.gallery, 100), orbit, 1), full), "budgets": {}}
            for b, r in sorted(runs.items(), reverse=True):
                if b == 100:
                    continue
                row["budgets"][b] = dict(compare(views(lat(a.gallery, b), orbit, 0), full), tokens=r["tokens"])
            if a.ref and os.path.exists(lat(a.ref, 100)):
                row["ref100"] = compare(views(lat(a.ref, 100), orbit, 0), full)
            rows.append(row)
            print(f"[eval] {k + 1}/{len(objs)} {o['name']}: floor PSNR {row['floor']['psnr']:.1f}; "
                  + "  ".join(f"{b}% {v['psnr']:.1f}" for b, v in row["budgets"].items())
                  + (f"; vs ref {row['ref100']['psnr']:.1f}" if "ref100" in row else ""), flush=True)
    keys = ["psnr", "ssim"] + (["lpips"] if lp is not None else [])
    mean = lambda vals: (float(np.mean(vals)), float(np.std(vals)))
    summary = {"objects": len(rows), "views": a.views, "size": a.size,
               "floor": {m: mean([r["floor"][m] for r in rows]) for m in keys}, "budgets": {}}
    for b in sorted({b for r in rows for b in r["budgets"]}, reverse=True):
        summary["budgets"][b] = {m: mean([r["budgets"][b][m] for r in rows if b in r["budgets"]]) for m in keys}
        summary["budgets"][b]["tokens"] = int(np.mean([r["budgets"][b]["tokens"] for r in rows if b in r["budgets"]]))
    if any("ref100" in r for r in rows):
        summary["ref100"] = {m: mean([r["ref100"][m] for r in rows if "ref100" in r]) for m in keys}
    out = a.out if sn == 1 else a.out.replace(".json", f".{si}of{sn}.json")
    json.dump({"summary": summary, "rows": rows, "gallery": a.gallery, "ref": a.ref}, open(out, "w"), indent=1)
    lines = ["| budget | tokens | " + " | ".join(m.upper() + (" ↓" if m == "lpips" else " ↑") for m in keys) + " |",
             "|---:|---:|" + "---:|" * len(keys)]
    for b, v in summary["budgets"].items():
        lines.append(f"| {b} % | {v['tokens']} | " + " | ".join(f"{v[m][0]:.3f}" if m != "psnr" else f"{v[m][0]:.2f}" for m in keys) + " |")
    lines.append("| noise floor (100 %, other decoder seed) | 8192 | " + " | ".join(
        f"{summary['floor'][m][0]:.3f}" if m != "psnr" else f"{summary['floor'][m][0]:.2f}" for m in keys) + " |")
    if "ref100" in summary:
        lines.append("| reference model at 100 % | 8192 | " + " | ".join(
            f"{summary['ref100'][m][0]:.3f}" if m != "psnr" else f"{summary['ref100'][m][0]:.2f}" for m in keys) + " |")
    open(out.replace(".json", ".md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
