import random
import subprocess
import sys
import time
import traceback
import zipfile

from state import (
    NATIVE,
    PIN,
    atomic,
    options,
    read,
    require_models,
    run_path,
    transcript_mode,
)


def main():
    job = run_path(sys.argv[1])
    cfg = options(read(job / "run.json")["settings"])
    model_path = require_models(cfg["model"])
    sys.path.insert(0, str(NATIVE))
    import numpy as np
    import soundfile as sf
    import torch

    def stage(message):
        if (job / "cancel").exists():
            raise InterruptedError("Cancelled")
        atomic(job / "progress.json", dict(message=message))
        print(message, flush=True)

    references, exports = [], []
    stage("Preparing reference recordings")
    for index, _ in enumerate(cfg["speakers"], 1):
        target = job / f"reference-{index}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(job / f"upload-{index}"),
                "-t",
                "30.1",
                "-vn",
                "-c:a",
                "pcm_f32le",
                str(target),
            ],
            check=True,
        )
        info = sf.info(target)
        if not 0.1 <= info.duration <= 30:
            raise ValueError(
                f"Speaker {index}: use 0.1–30 seconds of reference audio (3–10 recommended)"
            )
        references.append(str(target))
        exports.append(target.name)
    stage(f"Loading Ming3 {cfg['model']} · BF16")
    seed = cfg["seed"]
    if seed is None:
        seed = (
            1895
            if cfg["model"] == "0.5B"
            else 1235
            if cfg["task"] == "podcast"
            else 1234
        )
    elif seed == -1:
        seed = random.SystemRandom().randrange(2**31)
    common = dict(
        text=cfg["text"],
        sigma=cfg["sigma"],
        temperature=cfg["temperature"],
        max_length=cfg["max_length"],
        raw_text=cfg["raw"],
        trim_silence=cfg["trim"],
    )
    transcript = (
        [s["transcript"] for s in cfg["speakers"]] if transcript_mode(cfg) else None
    )
    task = cfg["task"]
    stopped = None
    started = time.monotonic()
    if cfg["model"] == "tiny":
        from ming_tiny import MingTinyTTS

        engine = MingTinyTTS.from_pretrained(
            str(model_path), text_normalization=cfg["normalization"]
        )
        args = dict(
            common,
            task=task,
            seed=seed,
            cfg_scale=cfg["cfg"],
            flow_steps=cfg["flow_steps"],
        )
        if references:
            args["prompt_wav"] = references if task == "podcast" else references[0]
        if transcript:
            args["prompt_text"] = transcript if task == "podcast" else transcript[0]
        if task in ("seed", "multilingual", "paralinguistic"):
            args["target_language"] = cfg["language"]
        for field, modes in {
            "style": ("voice_design",),
            "instruction": ("freeform",),
            "ip": ("ip",),
            "emotion": ("emotion",),
            "dialect": ("dialect",),
            "speed": ("basic",),
            "pitch": ("basic",),
            "volume": ("basic",),
        }.items():
            if task in modes:
                args[field] = cfg[field]
        if task == "podcast":
            args.update(
                min_generation_frames=cfg["min_frames"],
                max_generation_frames=cfg["max_frames"],
            )
        else:
            args.update(
                min_decode_steps=cfg["min_decode_steps"],
                max_decode_steps=cfg["max_decode_steps"],
            )
        args["stop_threshold"] = cfg["stop_threshold"]
        stage("Synthesizing dialogue" if task == "podcast" else "Synthesizing speech")
        output = engine.generate(**args)
        waveform = output.waveform
        stopped = bool(output.stopped)
    else:
        from ming_tts import MingTTS

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        engine = MingTTS(
            str(model_path),
            text_normalization=cfg["normalization"],
            cache_prompts=False,
        )
        args = dict(
            common,
            task="TTS" if task in ("seed", "paralinguistic") else task.upper(),
            cfg=cfg["cfg"],
            max_decode_steps=cfg["max_decode_steps"] or 1000,
            use_spk_emb=bool(references),
            use_zero_spk_emb=not references,
        )
        if references:
            args["prompt_wav_path"] = references[0]
        if transcript and transcript[0]:
            args["prompt_text"] = transcript[0]
        if task in ("seed", "multilingual", "paralinguistic"):
            args["language_id"] = cfg["language"]
        controls = {
            "voice_design": ["style"],
            "ip": ["ip"],
            "emotion": ["emotion"],
            "dialect": ["dialect"],
            "basic": ["speed", "pitch", "volume"],
        }
        if task in controls:
            args["controls"] = {k: cfg[k] for k in controls[task]}
        if task == "freeform":
            args["instruction"] = cfg["instruction"]
        stage("Synthesizing speech")
        waveform = engine.synthesize(**args)
    sample_rate = engine.sample_rate
    samples = waveform.detach().float().cpu().numpy().reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Native generation returned empty or nonfinite audio")
    stage("Writing audio")
    sf.write(job / "speech.wav", samples, sample_rate, subtype="FLOAT")
    sf.write(job / "speech.flac", samples, sample_rate, subtype="PCM_24")
    peaks = [
        round(float(np.abs(part).max()), 5)
        for part in np.array_split(samples, min(1800, samples.size))
    ]
    result = dict(
        sample_rate=sample_rate,
        duration=samples.size / sample_rate,
        seed=seed,
        elapsed=time.monotonic() - started,
        stopped=stopped,
        peaks=peaks,
        warning="Decode budget reached; the ending may be incomplete"
        if stopped is False
        else None,
        references=exports,
        settings=cfg,
        source=PIN,
    )
    atomic(job / "manifest.json", {k: v for k, v in result.items() if k != "peaks"})
    with zipfile.ZipFile(
        job / "result.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for name in [
            "speech.wav",
            "speech.flac",
            "manifest.json",
            "run.json",
            *exports,
        ]:
            archive.write(job / name, name)
    atomic(job / "result.json", result)
    stage("Complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic(run_path(sys.argv[1]) / "error.json", dict(error=str(error)))
        traceback.print_exc()
        sys.exit(1)
