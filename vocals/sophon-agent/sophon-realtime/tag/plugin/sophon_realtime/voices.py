"""Voice prompts for the Tag build.

voices/<name>.pt    -- precomputed flow conditioning (tools/mk_voices.py)
voices/<name>.wav   -- zero-shot clone: a 3-10 s clip of one speaker (.flac also works). The flow conditioning (speech tokens,
                       mel features, speaker embedding) is computed on first use with the same recipe as the official CosyVoice3
                       frontend and cached; a changed file is re-read, a removed one is forgotten.
Lookup order: "<voice>", "<voice>_<lang>", then the default voice for the language."""
import os
import re
import threading

import numpy as np
import torch

HOP = 10                                                    # flow prompt tokens are cut to a multiple of the first-chunk hop
MAX_REF_S = float(os.environ.get("SOPHON_MAX_REF_SECONDS", "12"))
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")             # voice names come from clients: no paths
EXTS = (".pt", ".wav", ".flac")
_CACHE: dict[str, tuple[tuple[str, float], dict]] = {}
_BAD: set[tuple[str, float]] = set()
_LOCK = threading.Lock()
_FE: dict = {}


def candidates(name: str, lang: str) -> list[str]:
    name = (name or "").lower().strip()
    if name and not NAME.fullmatch(name):
        name = ""
    out = [name, f"{name}_{lang}"] if name else []
    return out + [f"chelsie_{lang}", "vivian_zh" if lang == "zh" else "ryan_en", "vivian_zh"]


def _frontend(talker_dir: str) -> dict:
    if not _FE:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.intra_op_num_threads = 2
        cpu = ["CPUExecutionProvider"]
        _FE["tok"] = ort.InferenceSession(os.path.join(talker_dir, "speech_tokenizer_v3.onnx"), sess_options=o, providers=cpu)
        _FE["spk"] = ort.InferenceSession(os.path.join(talker_dir, "campplus.onnx"), sess_options=o, providers=cpu)
    return _FE


def from_clip(path: str, talker_dir: str) -> dict:
    """Flow conditioning for a reference clip, as the official CosyVoice3 frontend computes it (frontend_zero_shot)."""
    import soundfile as sf
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from vllm_omni.model_executor.models.cosyvoice3.utils import log_mel_spectrogram, mel_spectrogram
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(np.ascontiguousarray(y.mean(1)))[None]
    if sr < 16000:
        raise ValueError(f"sample rate {sr} Hz is below 16 kHz")
    w16 = torchaudio.functional.resample(w, sr, 16000) if sr != 16000 else w
    w16 = w16[:, : int(16000 * MAX_REF_S)]
    if w16.shape[1] < 16000 * 2:
        raise ValueError("reference clip is shorter than 2 s")
    fe = _frontend(talker_dir)
    mel = log_mel_spectrogram(w16, n_mels=128)
    tok = fe["tok"].run(None, {fe["tok"].get_inputs()[0].name: mel.numpy(), fe["tok"].get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)})[0]
    tok = torch.tensor(tok.reshape(1, -1), dtype=torch.int32)
    w24 = torchaudio.functional.resample(w16, 16000, 24000)
    feat = mel_spectrogram(w24, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)
    feat = feat.squeeze(0).transpose(0, 1)[None]                                            # [1, T, 80]
    fb = kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000); fb = fb - fb.mean(0, keepdim=True)
    emb = torch.tensor(fe["spk"].run(None, {fe["spk"].get_inputs()[0].name: fb[None].numpy()})[0].reshape(1, -1), dtype=torch.float32)
    n = min(feat.shape[1] // 2, tok.shape[1]) // HOP * HOP
    if n < HOP:
        raise ValueError("reference clip too short after tokenization")
    return {"speech_token": tok[:, :n], "speech_feat": feat[:, : 2 * n].float(), "embedding": emb, "hop": HOP}


def _get(n: str, p: str, ext: str, key: tuple, talker_dir: str, log=None):
    """Cached conditioning for one voice file; None if the file cannot be used."""
    hit = _CACHE.get(n)
    if hit is not None and hit[0] == key:
        return hit[1]
    if key in _BAD:
        return None
    with _LOCK:
        hit = _CACHE.get(n)
        if hit is not None and hit[0] == key:
            return hit[1]
        try:
            d = torch.load(p, map_location="cpu", weights_only=True) if ext == ".pt" else from_clip(p, talker_dir)
        except Exception as e:  # noqa: BLE001  -- a bad voice file must not take the stage down
            _BAD.add(key)
            if log: log.error("sophon tag: cannot load voice %s: %s", p, e)
            return None
        _CACHE[n] = (key, d)
    if log and ext != ".pt":
        log.info("sophon tag: cloned voice %r from %s (%d prompt tokens)", n, p, d["speech_token"].shape[1])
    return d


def load(vdir: str, name: str, lang: str, talker_dir: str, log=None) -> tuple[str, dict]:
    """Return (resolved name, conditioning dict). Never raises for a missing or bad custom voice: falls back to the default."""
    for n in candidates(name, lang):
        for ext in EXTS:
            p = os.path.join(vdir, n + ext)
            try:
                key = (p, os.path.getmtime(p))
            except OSError:
                continue
            d = _get(n, p, ext, key, talker_dir, log)
            if d is not None:
                return n, d
    raise ValueError(f"sophon tag: no voice found in {vdir}")


def start_background(vdir: str, talker_dir: str, log=None, every: float = 5.0) -> None:
    """Prepare clip voices ahead of time, so a request never waits for it: load the frontend once, then keep scanning the voices
    folder and compute every new or changed clip that has no .pt of the same name. Runs in the process that serves the voices."""
    import time

    def run():
        try:
            _frontend(talker_dir)
        except Exception as e:  # noqa: BLE001
            if log: log.warning("sophon tag: voice cloning frontend unavailable: %s", e)
            return
        while True:
            try:
                for f in sorted(os.listdir(vdir)):
                    n, ext = os.path.splitext(f)
                    if ext not in (".wav", ".flac") or not NAME.fullmatch(n) or os.path.exists(os.path.join(vdir, n + ".pt")):
                        continue
                    p = os.path.join(vdir, f)
                    _get(n, p, ext, (p, os.path.getmtime(p)), talker_dir, log)
            except Exception as e:  # noqa: BLE001
                if log: log.warning("sophon tag: voice scan failed: %s", e)
            time.sleep(every)

    threading.Thread(target=run, name="sophon-voices", daemon=True).start()
