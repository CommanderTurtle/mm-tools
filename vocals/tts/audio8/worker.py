import random
import subprocess
import sys
import time
import traceback
import zipfile

from state import MODELS, PIN, atomic, options, read, require_models, run_path


def main(job):
    require_models()
    cfg, texts = options(read(job / "run.json")["settings"])

    def progress(stage, **fields):
        atomic(job / "progress.json", dict(stage=stage, **fields))

    progress("Preparing reference")
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoProcessor, StoppingCriteria

    reference = job / "reference.wav"
    if cfg["voice"] != "none":
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(job / "upload"),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-t",
                "15.1",
                "-c:a",
                "pcm_f32le",
                str(reference),
            ],
            check=True,
            timeout=60,
        )
        info = sf.info(reference)
        if not 3 <= info.duration <= 15:
            raise ValueError("Use a reference recording between 3 and 15 seconds")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    progress("Loading Realistic Air · BF16")
    processor = AutoProcessor.from_pretrained(
        str(MODELS), trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            str(MODELS),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )
    sample_rate = int(model.config.codec_sample_rate)
    generator = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    records = []

    class FrameProgress(StoppingCriteria):
        def __init__(self, index, count, budget, retry):
            self.steps = 0
            self.index, self.count, self.budget, self.retry = (
                index,
                count,
                budget,
                retry,
            )

        def __call__(self, input_ids, scores, **kwargs):
            self.steps += 1
            if self.steps == 1 or self.steps % 16 == 0:
                progress(
                    "Generating",
                    first=self.index + 1,
                    count=self.count,
                    total=len(texts),
                    frames=self.steps,
                    budget=self.budget,
                    retry=self.retry,
                )
            return False

    def synthesize(batch, index, budget, retry=False):
        values = dict(text=batch, return_tensors="pt")
        if cfg["voice"] != "none":
            values.update(
                reference_audio=[reference] * len(batch),
                reference_text=[cfg["transcript"]] * len(batch),
            )
        inputs = {key: value.to("cuda") for key, value in processor(**values).items()}
        return model.generate(
            **inputs,
            max_new_tokens=budget,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            do_sample=cfg["sample"],
            generator=generator,
            return_dict_in_generate=True,
            stopping_criteria=[FrameProgress(index, len(batch), budget, retry)],
        )

    started = time.monotonic()
    for first in range(0, len(texts), cfg["batch_size"]):
        batch = texts[first : first + cfg["batch_size"]]
        output = synthesize(batch, first, cfg["frames"])
        for offset, text in enumerate(batch):
            index = first + offset
            finished = bool(output.finished[offset])
            codes = output.codes[offset, :, : int(output.code_lengths[offset])]
            retried = False
            if not finished and cfg["retry_frames"] > cfg["frames"]:
                retry = synthesize([text], index, cfg["retry_frames"], retry=True)
                codes = retry.codes[0, :, : int(retry.code_lengths[0])]
                finished, retried = bool(retry.finished[0]), True
            progress("Decoding", first=index + 1, total=len(texts))
            waveforms, lengths = model.decode_audio(codes)
            audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
            if not len(audio):
                raise ValueError(
                    f"Paragraph {index + 1} produced no audio; change the text or reference"
                )
            stem = f"take-{index + 1:03d}"
            sf.write(job / f"{stem}.wav", audio, sample_rate, subtype="FLOAT")
            sf.write(job / f"{stem}.flac", audio, sample_rate, subtype="PCM_24")
            if cfg["codes"]:
                np.save(job / f"{stem}.npy", codes.cpu().numpy())
            bins = min(1800, len(audio))
            peaks = [float(np.max(np.abs(x))) for x in np.array_split(audio, bins)]
            records.append(
                dict(
                    stem=stem,
                    text=text,
                    seconds=len(audio) / sample_rate,
                    sample_rate=sample_rate,
                    frames=int(codes.shape[1]),
                    peaks=peaks,
                    finished=finished,
                    retried=retried,
                    codes=cfg["codes"],
                )
            )
            atomic(job / "partial.json", records)
    progress("Exporting")
    result = dict(takes=records, elapsed=time.monotonic() - started, source=PIN)
    atomic(
        job / "manifest.json",
        {
            **result,
            "takes": [{k: v for k, v in r.items() if k != "peaks"} for r in records],
        },
    )
    with zipfile.ZipFile(job / "result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(job.iterdir()):
            if path.suffix in (".wav", ".flac", ".npy") or path.name in (
                "manifest.json",
                "run.json",
            ):
                archive.write(path, path.name)
    atomic(job / "result.json", result)
    progress("Complete", total=len(texts))


if __name__ == "__main__":
    job = run_path(sys.argv[1])
    try:
        main(job)
    except Exception as error:
        atomic(job / "error.json", dict(error=str(error)))
        traceback.print_exc()
        raise SystemExit(1) from None
