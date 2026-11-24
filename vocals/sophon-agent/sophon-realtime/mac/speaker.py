"""Speaker embeddings (WeSpeaker ResNet34, ONNX): tells whether two stretches of speech come from the same voice, so that a
television or a bystander is not taken for the person who woke the device. Input is 16 kHz mono float32 audio; the model wants
Kaldi-style 80-bin log mel filterbank features with the mean removed, computed here with numpy."""
import numpy as np, onnxruntime as ort

SR = 16000
MIN_S = 0.8                     # shorter speech gives an embedding too noisy to compare
MAX_S = 6.0                     # the model's cost grows with length and a voice is settled well before this
_WIN, _HOP, _NFFT, _MELS = 400, 160, 512, 80


def _mel_bank():
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(20.0), mel(SR / 2)
    c = lo + (hi - lo) * np.arange(_MELS + 2) / (_MELS + 1)
    f = np.arange(_NFFT // 2) * SR / _NFFT; m = mel(f)
    bank = np.zeros((_MELS, _NFFT // 2 + 1), np.float32)
    for i in range(_MELS):
        l, ce, r = c[i], c[i + 1], c[i + 2]
        bank[i, : _NFFT // 2] = np.maximum(0.0, np.minimum((m - l) / (ce - l), (r - m) / (r - ce)))
    return bank


_BANK = _mel_bank()
_WINDOW = np.power(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / (_WIN - 1)), 0.85).astype(np.float32)   # Kaldi's "povey" window


def features(x):
    """80-bin log mel filterbank frames (T x 80) of 16 kHz audio in [-1, 1], with the per-utterance mean removed."""
    x = np.asarray(x, np.float32) * 32768.0
    n = 1 + (len(x) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    fr = x[idx]; fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1) * _WINDOW
    p = np.abs(np.fft.rfft(fr, _NFFT, axis=1)) ** 2
    f = np.log(np.maximum(p @ _BANK.T, np.finfo(np.float32).eps)).astype(np.float32)
    return f - f.mean(axis=0, keepdims=True)


class Speaker:
    def __init__(self, path):
        self.net = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def embed(self, x):
        """Unit-length voice embedding of the speech in x, or None when there is too little of it to compare."""
        if len(x) < MIN_S * SR: return None
        e = self.net.run(None, {"feats": features(x[: int(MAX_S * SR)])[None]})[0][0]
        return e / np.linalg.norm(e)
