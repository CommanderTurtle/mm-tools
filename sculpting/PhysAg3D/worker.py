"""A single stage in an owned process group. No model survives a stage boundary."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from state import ROOT, SOURCE, TRELLIS, PIN, atomic, inside, read, native_python
from lifecycle import own, release
from state import LANE_STAMP


def copy_local(source, target):
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Attempt files must be local, not symlinks")
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "presentation", "validated.json", "__pycache__", "*.partial"
        ),
    )


def progress(attempt, stage, message):
    atomic(
        attempt / "progress.json",
        dict(stage=stage, message=message, updated=time.time()),
    )


def generate(case, attempt, config):
    models = ROOT / "models"
    weights = (models / "TRELLIS.2-4B").resolve()
    local = attempt / "model-config"
    local.mkdir()
    manifest = read(weights / "pipeline.json")
    for key, value in manifest["args"]["models"].items():
        prefix = (
            "ckpts/ss_dec_conv3d_16l8_fp16"
            if key == "sparse_structure_decoder"
            else value
        )
        file = (weights / prefix).resolve()
        if (
            not file.is_relative_to(weights)
            or not Path(str(file) + ".safetensors").is_file()
        ):
            raise ValueError("Missing local model prefix: " + prefix)
        manifest["args"]["models"][key] = str(file)
    atomic(local / "pipeline.json", manifest)
    inputs = attempt / "input"
    inputs.mkdir()
    shutil.copy2(case / "input/source.png", inputs / "source_image.png")
    runner = SOURCE / "method/trellis2-native-refine/scripts/run_trellis.py"
    args = [
        str(native_python()),
        str(runner),
        "--route",
        "native",
        "--input",
        str(inputs),
        "--output",
        str(attempt / "output"),
        "--runtime",
        str(TRELLIS),
        "--model",
        str(local),
        "--dino",
        str((models / "deps/camenduru--dinov3-vitl16-pretrain-lvd1689m").resolve()),
        "--rembg",
        str((models / "deps/ZhengPeng7--BiRefNet").resolve()),
        "--gpu",
        str(config["device"]),
        "--resolution",
        str(config["resolution"]),
        "--texture-size",
        str(config["texture"]),
        "--steps",
        str(config["steps"]),
        "--seed",
        str(config["seed"]),
        "--min-free-gb",
        str(config["min_free_gb"]),
    ]
    env = dict(
        os.environ,
        ATTN_BACKEND=PIN["attention"],
        SPARSE_ATTN_BACKEND=PIN["sparse_attention"],
        SPARSE_CONV_BACKEND=PIN["sparse_convolution"],
        PYTHONPATH=str(TRELLIS),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    atomic(
        attempt / "invocation.json",
        {
            "argv": args,
            "source": LANE_STAMP,
            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        },
    )
    progress(
        attempt,
        "native",
        "Generating the textured candidate; the native process exits afterwards",
    )
    subprocess.run(args, cwd=TRELLIS, env=env, check=True)
    report = read(attempt / "output/run.json", {})
    if report.get("status") != "candidate_generated_pending_independent_validation":
        raise ValueError(
            report.get("error", "Native generation did not produce a candidate")
        )
    from delivery import glb_info

    glb_info(attempt / "output/candidate_yup.glb")
    atomic(attempt / "result.json", {"status": "complete"})


def agent(case, attempt, meta):
    phase, config = meta["phase"], meta["settings"]
    work = attempt / "work"
    work.mkdir()
    previous = meta.get("resume_from")
    if previous:
        copy_local(inside(case / "runs", previous) / "work", work)
    elif phase == "refine":
        plan = inside(case / "runs", meta["plan"]) / "work"
        for name in ("asset_plan.json", "scaffold.blend"):
            shutil.copy2(inside(plan, name), work / name)
    (work / "input").mkdir(exist_ok=True)
    shutil.copy2(case / "input/source.png", work / "input/source.png")
    if phase == "refine":
        native = inside(case / "runs", meta["native"]) / "output"
        target = work / "native"
        target.mkdir(exist_ok=True)
        for name in (
            "candidate_yup.glb",
            "native_raw.npz",
            "condition_processed.png",
            "run.json",
        ):
            shutil.copy2(inside(native, name), target / name)
        # A prior attempt's completion record must never satisfy a new run.
        (work / "delivery.json").unlink(missing_ok=True)
        (work / "validated.json").unlink(missing_ok=True)
    runtime = dict(
        stage=phase,
        python=sys.executable,
        blender=meta["blender"],
        source_reference=str(SOURCE / "method"),
        source_revision=LANE_STAMP,
        model=config["model"],
        minutes=config["minutes"],
        attempt=str(work),
        feedback=config["feedback"],
        native_attempt=meta.get("native"),
    )
    atomic(work / "runtime.json", runtime)
    sessions = attempt / "session"
    sessions.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("PHYS_OMP", "omp"),
        "--cwd",
        str(work),
        "--session-dir",
        str(sessions),
        "--model",
        config["model"],
        "--config",
        str(ROOT / "omp.yml"),
        "--mode",
        "json",
        "--print",
        "--no-title",
        "--no-pty",
        "--no-lsp",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-context-files",
        "--no-prewalk",
        "--tools",
        "read,write,edit,bash,grep,glob",
        "--wait-async",
        "--max-time",
        str(config["minutes"] * 60),
        "--append-system-prompt",
        str(ROOT / "prompts/contract.md"),
    ]
    if config["resume"] and previous:
        prior = sorted(
            (inside(case / "runs", previous) / "session").glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if prior:
            command.extend(("--fork", str(prior[-1])))
    command.append("@" + str(work / "input/source.png"))
    prompt = (
        f"Perform the {phase} stage now in {work}. Read runtime.json and follow the stage contract. "
        "Prior session paths may point to an older attempt: use only the current attempt for writes. "
        "Finish the requested artifacts, then stop; do not start a different stage.\nUser refinement:\n"
        + config["feedback"]
    )
    atomic(attempt / "invocation.json", {"argv": command, "runtime": runtime})
    progress(attempt, phase, "OMP is working on the saved attempt")
    env = dict(
        os.environ,
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        MUJOCO_GL="egl",
        OTEL_SDK_DISABLED="true",
    )
    subprocess.run(command, input=prompt, text=True, cwd=work, env=env, check=True)
    if phase == "plan":
        plan = read(work / "asset_plan.json")
        if (
            not isinstance(plan, dict)
            or not plan
            or not (work / "scaffold.blend").is_file()
            or not (work / "scaffold.blend").stat().st_size
        ):
            raise ValueError(
                "Planning needs asset_plan.json and a nonempty scaffold.blend"
            )
        atomic(attempt / "result.json", {"status": "complete"})
    else:
        from delivery import package

        progress(
            attempt, "package", "Checking artifact paths and packaging the local viewer"
        )
        data = package(work)
        atomic(
            attempt / "result.json",
            {"status": "complete" if data["status"] == "delivered" else "partial"},
        )


def main():
    case, attempt = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    own(int(sys.argv[3]))
    meta = read(attempt / "run.json")
    if meta["phase"] == "native":
        generate(case, attempt, meta["settings"])
    elif meta["phase"] == "package":
        from delivery import package

        work = attempt / "work"
        copy_local(inside(case / "runs", meta["resume_from"]) / "work", work)
        data = package(work)
        atomic(
            attempt / "result.json",
            {"status": "complete" if data["status"] == "delivered" else "partial"},
        )
    else:
        agent(case, attempt, meta)
    progress(
        attempt,
        "saved",
        "Stage saved; owned processes and GPU allocations are released",
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if len(sys.argv) > 2:
            atomic(
                Path(sys.argv[2]) / "error.json",
                {"error": str(error) or type(error).__name__},
            )
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        release()
