"""Loads the Sophon-Realtime E2E speaking engine modules (sophon_expressive) inside a vLLM-Omni stage process and builds the
half of the engine each stage needs: TalkerCore (talker + CUDA-Graph engine) or DecoderCore (adapter + DiT + HiFT + voice)."""
import os
import sys
import time

import torch

_T = None
CUDNN_BENCH = os.environ.get("SOPHON_E2E_CUDNN_BENCHMARK", "1") == "1"
# largest batch the adapter / decoder graphs are captured for; start.sh sets 8 for a 4B model (8 sessions), which roughly halves their memory
DEC_MAXB = int(os.environ.get("SOPHON_E2E_DEC_MAXB", "16"))
BKS = tuple(b for b in (1, 2, 4, 8, 16) if b <= DEC_MAXB)


def engine_module():
    global _T
    if _T is None:
        # The speaking engine is the sophon_expressive package at the root of the sophon-realtime repository (this plugin lives in e2e/plugin/).
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [os.environ.get("SOPHON_E2E_SRC"), here, os.path.dirname(os.path.dirname(here))]
        src = next((c for c in cands if c and os.path.isfile(os.path.join(c, "sophon_expressive", "engine220.py"))), None)
        if src is None:
            raise RuntimeError("sophon_realtime (E2E): cannot find the sophon_expressive engine package; run from a full checkout of the sophon-realtime repository or set SOPHON_E2E_SRC to the directory that contains sophon_expressive/")
        for p in (src, os.path.join(src, "sophon_expressive")):
            if p not in sys.path:
                sys.path.insert(0, p)
        if not os.environ.get("SOPHON_MODEL_DIR"):
            from .env import model_dir
            d = model_dir()
            if not d or not os.path.isdir(os.path.join(d, "talker220")):
                raise RuntimeError("sophon_realtime (E2E): cannot find the model directory; set SOPHON_MODEL to the Sophon-Realtime-32B-E2E directory")
            os.environ["SOPHON_MODEL_DIR"] = d
        from sophon_expressive import paths as _P
        if not os.path.isdir(_P.base_dir()):          # vLLM-Omni layout: the thinker lives in thinker/ (there is no base/)
            _P.base_dir = lambda: os.path.join(_P.model_dir(), "thinker")
        import engine220 as T
        _T = T
        if os.environ.get("SOPHON_E2E_SHARED_POOL", "1") == "1":       # all captured graphs of this process share one memory pool (they never replay concurrently)
            _G = torch.cuda.graph; _pool = torch.cuda.graph_pool_handle()

            class _Shared(_G):
                def __init__(self, cuda_graph, pool=None, stream=None, **kw):
                    super().__init__(cuda_graph, pool=pool if pool is not None else _pool, stream=stream, **kw)
            torch.cuda.graph = _Shared
    return _T


def _mem(tag, log):
    if log: log.info("E2E mem %s (cudnn.benchmark=%s): allocated %.1f GB reserved %.1f GB", tag, torch.backends.cudnn.benchmark, torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30)


class _f32:
    """vLLM builds models with its own default dtype (bf16); the engine modules must be created in float32 as in the standalone engine."""
    def __enter__(self):
        self.old = torch.get_default_dtype(); torch.set_default_dtype(torch.float32)
        self.bm = torch.backends.cudnn.benchmark; torch.backends.cudnn.benchmark = False     # benchmark mode at serving time means a pause whenever a new conv shape shows up
        self.tf = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)      # same numerics as the standalone engine (PyTorch defaults)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        self.dc = torch.device("cpu"); self.dc.__enter__()      # vLLM builds models inside `with torch.device("cuda")`; the engine expects CPU as the default device (e.g. its fixed noise table)
        self.sdp = torch.backends.cuda.cudnn_sdp_enabled(); torch.backends.cuda.enable_cudnn_sdp(True)   # vLLM turns the cuDNN attention kernel off on import; the engine was built and tested with it
    def __exit__(self, *a):
        torch.set_default_dtype(self.old); torch.backends.cudnn.benchmark = self.bm
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.tf
        torch.backends.cuda.enable_cudnn_sdp(self.sdp)
        self.dc.__exit__(None, None, None)


class TalkerCore:
    def __init__(self, slots: int, log=None):
        T = engine_module(); S = T.S; dev = "cuda"; t0 = time.time()
        if log: log.info("E2E ambient flags: matmul.allow_tf32=%s cudnn.allow_tf32=%s cudnn.benchmark=%s default_dtype=%s autocast=%s", torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.get_default_dtype(), torch.is_autocast_enabled())
        with _f32(), torch.inference_mode(False), torch.no_grad():
            self.m, self.tok = S.load(os.environ["SOPHON_MODEL_DIR"], dev); self.IM = self.tok.convert_tokens_to_ids("<|im_end|>")
            _mem("talker loaded", log)
            self.eng = T.FG.Engine(self.m, slots); self.eng.done.fill_(True)
            _mem("engine buffers", log)
            pn = int(os.environ.get("S220_PREP_PAD", "0")); nw = int(os.environ.get("S220_PREP_WARM", "1024" if pn > 0 else "320"))
            if nw > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for n in (range(pn, nw + 1, pn) if pn > 0 else range(1, nw + 1)):
                        for rep in range(2):
                            ids = torch.full((1, n), 1000, device=dev); ip = torch.zeros_like(ids, dtype=torch.bool)
                            if pn > 0 and rep: ip[:, -3:] = True
                            self.m.cond = None; self.m.memory(ids, ip, torch.tensor([0], device=dev))
            _mem("prep warm", log)
            self.eng.step(); self.eng.reset(); self.eng.done.fill_(True)       # capture the step graph now
            _mem("step graph", log)
            # adapter (vector frames -> decoder conditioning); the DiT / vocoder weights loaded with it are not kept here
            self.m2, _dec, _aff, _hift, _ = T.E2M.load_all(dev); del _dec, _aff, _hift
            import graphs220 as G
            self.adg = G.GraphFn(lambda x_: self.m2(x_)[1]); D = S.D_V
            torch.backends.cudnn.benchmark = CUDNN_BENCH       # as for the vocoder graphs in DecoderCore: only while capturing at start-up
            try:
                for Tn in (3, 5, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30):
                    for B in (BKS if Tn == 30 else tuple(b for b in (1, 2, 4) if b <= DEC_MAXB)):
                        for _ in range(4): self.adg(torch.randn(B, Tn, D, device=dev))
            finally:
                torch.backends.cudnn.benchmark = False
            _mem("adapter graphs", log); torch.cuda.empty_cache(); _mem("after empty_cache", log)
        self.slots = [None] * slots
        if log: log.info("E2E TalkerCore ready in %.1fs, %d slots, frame dim %d", time.time() - t0, slots, S.D_V)


class DecoderCore:
    def __init__(self, log=None):
        T = engine_module(); EN = T.EN; dev = "cuda"; t0 = time.time()
        from safetensors.torch import load_file
        import threading
        with _f32(), torch.inference_mode(False), torch.no_grad():
            self.m2, self.dec, self.aff, self.hift, _ = T.E2M.load_all(dev); self.dec.inference_cfg_rate = 0.0
            if os.environ.get("S220_HIFT_FOLD", "1") == "1":
                import torch.nn.utils.parametrize as _pz
                for _m in self.hift.modules():
                    if _pz.is_parametrized(_m, "weight"): _pz.remove_parametrizations(_m, "weight")
            V = load_file(os.path.join(os.environ["SOPHON_MODEL_DIR"], "voices", "chelsie.safetensors"))
            self.prompts = {lg: (V[f"prompt_mu.{lg}"].to(dev), V[f"prompt_mel.{lg}"].to(dev)) for lg in ("zh", "en", "ja")}
            self.spk = {lg: V[f"spk.{lg}"].to(dev).float() for lg in ("zh", "en", "ja")}
            self.voices = {f"brand_{lg}": self.spk[lg] for lg in self.spk}; self.lock = threading.Lock()
            if not hasattr(T._ES.Stream, "gen0"): T._ES.Stream.gen0 = T._ES.Stream.gen
            self.pool = []; EN.E2Batch.BK = BKS; self.eb = EN.E2Batch(self)
            r = T.Render(self, "zh")
            _mem("decoder loaded", log)
            for B in EN.E2Batch.BK: EN._kvc_graph(self.eb, r.v2, B, 1 if r.v2.cfg == 0 else 2)
            _mem("kvc graphs", log)
            import graphs220 as G
            self.adg = None
            self.hg = G.HiftGraph(self.hift, max_b=BKS[-1])

            def _mk(B, Tw):
                rs = []
                for _ in range(B):
                    q = T.Render(self, "zh"); q.mel_cum = torch.randn(1, 80, Tw, device=dev) - 4; q.pend = torch.zeros(1, 80, 0, device=dev); rs.append((q, 0, False))
                return rs
            # Capture the vocoder graphs with cuDNN picking the conv algorithms by timing (the choice is frozen into each graph): cuDNN's default
            # choice is 4-10x slower on RTX PRO 6000 (Blackwell) and about equal on H100. Only here, at start-up, never while serving.
            torch.backends.cudnn.benchmark = CUDNN_BENCH
            try:
                self.hg.warm(_mk, Tws=(56,), Bs=BKS); self.hg.warm(_mk, Tws=(16, 32, 48), Bs=BKS)
            finally:
                torch.backends.cudnn.benchmark = False
            _mem("hift graphs", log); torch.cuda.empty_cache(); _mem("after empty_cache", log)
        if log: log.info("E2E DecoderCore ready in %.1fs", time.time() - t0)


# ---- wire format between the two stages: one row per DiT chunk, as non-negative ints below 65536 (valid token ids)
NMEL = 16; ROW = 3 + 2 * 80 * NMEL        # [n mel frames, last flag, language id] + float32 payload as (hi16, lo16) pairs
LANGS = ("zh", "en", "ja")


def pack_row(mu: torch.Tensor, fin: bool, lang: str) -> torch.Tensor:
    """mu [1, 80, n<=16] float -> int64 [ROW]."""
    n = int(mu.shape[2]); x = torch.zeros(80, NMEL, device=mu.device, dtype=torch.float32); x[:, :n] = mu[0].float()
    b = x.reshape(-1).contiguous().view(torch.int32).to(torch.long) & 0xFFFFFFFF
    row = torch.empty(ROW, device=mu.device, dtype=torch.long)
    row[0] = n; row[1] = int(bool(fin)); row[2] = LANGS.index(lang) if lang in LANGS else 0
    row[3::2] = b >> 16; row[4::2] = b & 0xFFFF
    return row


def unpack_row(row: torch.Tensor):
    row = row.to(torch.long); n = int(row[0]); fin = bool(int(row[1])); li = int(row[2])
    b = (row[3::2] << 16) | row[4::2]
    b = torch.where(b >= 2 ** 31, b - 2 ** 32, b).to(torch.int32).contiguous().view(torch.float32)
    return b.reshape(1, 80, NMEL)[:, :, :n].clone(), fin, LANGS[li] if 0 <= li < len(LANGS) else "zh"
