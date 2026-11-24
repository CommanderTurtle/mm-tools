"""Zero-shot voice cloning for the E2E build (added in 1.3.0).

voices/<name>.wav   -- a 3-12 s clip of one speaker (.flac also works); prepared on first use and cached, re-read when the file changes
voices/<name>.pt    -- the same conditioning precomputed (keys speech_token, speech_feat, embedding, as Tag's voices/*.pt)

The decoder conditions on a reference prompt: the clip's speech tokens (as the decoder's own input), its mel frames and its CAM++ speaker
embedding. They are computed with the same recipe as Tag's voice files (the official CosyVoice3 frontend), which needs two small models
from the Tag release, `speech_tokenizer_v3.onnx` and `campplus.onnx` (install.sh downloads them to model/clone_frontend/).
Built-in voice names (chelsie, brand_*) and anything not found fall back to the built-in voice.

The talker stage knows the request's voice, the decoder stage does not (stage payloads carry only the frames), so the talker leaves the
voice name in a small per-request file that the decoder reads when the request's first chunk arrives."""
import os
import re
import tempfile
import threading
import time

import numpy as np
import torch

NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")             # voice names come from clients: no paths
BUILTIN = {"", "chelsie", "brand", "brand_zh", "brand_en", "brand_ja", "default"}
EXTS = (".pt", ".wav", ".flac")
MAX_REF_S = float(os.environ.get("SOPHON_MAX_REF_SECONDS", "12"))
SHM = os.environ.get("SOPHON_E2E_VOICE_SHM") or os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(),
                                                              f"sophon_e2e_voice_{os.getuid()}")
_LOCK = threading.Lock()
_FE: dict = {}
_REF: dict = {}          # name -> ((path, mtime), conditioning dict)
_PROMPT: dict = {}       # (name, lang, mtime) -> (mu, mel, spk)
_BAD: set = set()
_N = [0]


def voices_dir(model_dir: str | None = None) -> str:
    return os.environ.get("SOPHON_VOICES") or os.path.join(model_dir or os.environ.get("SOPHON_MODEL_DIR") or os.environ.get("SOPHON_MODEL") or "model", "voices")


def frontend_dir(model_dir: str | None = None) -> str:
    return os.environ.get("SOPHON_CLONE_FRONTEND") or os.path.join(model_dir or os.environ.get("SOPHON_MODEL_DIR") or os.environ.get("SOPHON_MODEL") or "model", "clone_frontend", "talker")


# ---- talker stage -> decoder stage: which voice a request uses
def _key(rid) -> str:
    return str(rid).replace("/", "_")


def put_voice(rid, name) -> None:
    name = str(name or "").lower().strip()
    if name in BUILTIN or not NAME.fullmatch(name):
        return
    try:
        os.makedirs(SHM, exist_ok=True)
        with open(os.path.join(SHM, _key(rid)), "w") as f:
            f.write(name)
        _N[0] += 1
        if _N[0] % 200 == 0:                                 # files of requests the decoder never finished
            now = time.time()
            for fn in os.listdir(SHM):
                p = os.path.join(SHM, fn)
                if now - os.path.getmtime(p) > 600:
                    os.remove(p)
    except OSError:
        pass


def take_voice(*rids) -> str | None:
    for r in rids:
        if r in (None, "", "None"):
            continue
        try:
            with open(os.path.join(SHM, _key(r))) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def drop_voice(*rids) -> None:
    for r in rids:
        if r in (None, "", "None"):
            continue
        try:
            os.remove(os.path.join(SHM, _key(r)))
        except OSError:
            pass


# ---- reference clip -> conditioning (same recipe as Tag's voices.from_clip)
def _frontend(fdir: str) -> dict:
    if not _FE:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.intra_op_num_threads = 2
        cpu = ["CPUExecutionProvider"]
        _FE["tok"] = ort.InferenceSession(os.path.join(fdir, "speech_tokenizer_v3.onnx"), sess_options=o, providers=cpu)
        _FE["spk"] = ort.InferenceSession(os.path.join(fdir, "campplus.onnx"), sess_options=o, providers=cpu)
    return _FE


def from_clip(path: str, fdir: str) -> dict:
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
    fe = _frontend(fdir)
    mel = log_mel_spectrogram(w16, n_mels=128)
    tok = fe["tok"].run(None, {fe["tok"].get_inputs()[0].name: mel.numpy(), fe["tok"].get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)})[0]
    tok = torch.tensor(tok.reshape(1, -1), dtype=torch.int32)
    w24 = torchaudio.functional.resample(w16, 16000, 24000)
    feat = mel_spectrogram(w24, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)
    feat = feat.squeeze(0).transpose(0, 1)[None]                                            # [1, T, 80]
    fb = kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000); fb = fb - fb.mean(0, keepdim=True)
    emb = torch.tensor(fe["spk"].run(None, {fe["spk"].get_inputs()[0].name: fb[None].numpy()})[0].reshape(1, -1), dtype=torch.float32)
    n = min(feat.shape[1] // 2, tok.shape[1])
    return {"speech_token": tok[:, :n], "speech_feat": feat[:, : 2 * n].float(), "embedding": emb}


def _reference(name: str, model_dir: str | None, log=None):
    vdir = voices_dir(model_dir)
    for ext in EXTS:
        p = os.path.join(vdir, name + ext)
        if not os.path.isfile(p):
            continue
        key = (p, os.path.getmtime(p))
        hit = _REF.get(name)
        if hit is not None and hit[0] == key:
            return key, hit[1]
        if key in _BAD:
            return None
        try:
            d = torch.load(p, map_location="cpu", weights_only=True) if ext == ".pt" else from_clip(p, frontend_dir(model_dir))
            d = {k: d[k] for k in ("speech_token", "speech_feat", "embedding")}
        except Exception as e:  # noqa: BLE001  -- a bad file must not take the stage down
            _BAD.add(key)
            if log: log.warning("E2E voice %r: cannot use %s (%r), using the built-in voice", name, p, e)
            return None
        _REF[name] = (key, d)
        return key, d
    return None


def prompt(core, name: str, lang: str, dev, model_dir: str | None = None, log=None):
    """(prompt mu, prompt mel, projected speaker vector) for a cloned voice, or None for the built-in voice.
    The prompt has the same length as the built-in prompt for the language, so the decoder's captured graphs are reused."""
    name = str(name or "").lower().strip()
    if name in BUILTIN or not NAME.fullmatch(name):
        return None
    with _LOCK:
        r = _reference(name, model_dir, log)
        if r is None:
            return None
        key, d = r
        P = int(core.prompts.get(lang, core.prompts["zh"])[0].shape[2])                    # frames (50 Hz)
        pk = (name, lang, key)
        if pk in _PROMPT:
            return _PROMPT[pk]
        ntok = P // 2
        if d["speech_token"].shape[1] < ntok or d["speech_feat"].shape[1] < 2 * ntok:
            if log: log.warning("E2E voice %r: reference shorter than %.1f s, using the built-in voice", name, P / 50)
            _PROMPT[pk] = None
            return None
        with torch.no_grad():
            tok = d["speech_token"][:, :ntok].long().to(dev)
            mu = core.m2.pla(core.m2.emb(tok)).repeat_interleave(2, dim=1).transpose(1, 2).float().contiguous()
            mel = d["speech_feat"][:, :2 * ntok].transpose(1, 2).to(dev).float().contiguous()
            spk = core.aff(torch.nn.functional.normalize(d["embedding"].reshape(1, -1).to(dev).float(), dim=-1))
        if len(_PROMPT) > 256:
            _PROMPT.clear()
        _PROMPT[pk] = (mu, mel, spk)
        if log: log.info("E2E cloned voice %s (%s): %d prompt frames", name, lang, 2 * ntok)
        return _PROMPT[pk]
