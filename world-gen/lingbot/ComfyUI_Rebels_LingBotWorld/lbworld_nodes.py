"""
ComfyUI_Rebels_LingBotWorld - LingBot-World-v2 14B causal-fast on 8GB VRAM.
by RealRebelAI

Architecture (per house rules and what actually fits):
- The 14B DiT loads ONLY from GGUF (bf16 is 28GB; a 16GB-RAM box cannot hold it).
  Quantized bytes live pinned in system RAM; each layer dequantizes on the fly
  during forward (proven pattern from SCAIL-2 / NL-Diffusion packs).
- Text encoding comes from ComfyUI's native UMT5-XXL path: load your umt5 GGUF
  with the ComfyUI-GGUF CLIPLoader (type wan), CLIPTextEncode as usual, feed the
  CONDITIONING here. Their T5 class is bypassed with a shim so the 11GB .pth is
  never needed.
- VAE is the stock Wan2.1_VAE.pth you already have, loaded through their class
  (correct stride + normalization).
- The vendored `wan` package provides the exact causal-fast loop: 4 distilled
  timesteps per chunk, KV-cache windowing (local_attn_size / sink_size), plucker
  camera embeddings + optional WASD actions from an action folder.

Action folders: put example dirs (poses.npy, intrinsics.npy, action.npy,
wasd_action.npy) under  ComfyUI/input/lingbot_actions/<name>/  - the dropdown
lists them. Copy the repo's examples/ dirs there to start.
"""

import gc
import hashlib
import os
import sys

import numpy as np
import torch

import folder_paths

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACK_DIR, "model_assets")
if PACK_DIR not in sys.path:
    sys.path.insert(0, PACK_DIR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# GGUF streaming machinery (SCAIL-2 / NLD proven pattern)
# ---------------------------------------------------------------------------
def _import_city96_dequant():
    import importlib.util
    for name in ("ComfyUI-GGUF", "ComfyUI-GGUF-main", "gguf"):
        cand = os.path.join(os.path.dirname(PACK_DIR), name, "dequant.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("rebels_lbw_dequant", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("[LBWorld] GPU dequant kernels loaded from {}".format(cand))
            return mod
    raise ImportError(
        "ComfyUI-GGUF not found next to this pack - install it (city96) first.")


_DQ = None


def dequant_bytes(qbytes, qtype, oshape, dtype):
    global _DQ
    if _DQ is None:
        _DQ = _import_city96_dequant()
    return _DQ.dequantize(qbytes, qtype, oshape, dtype=dtype)


class GGUFLinearWrapper(torch.nn.Module):
    """Holds quantized bytes (pinned CPU); dequantizes per forward on the GPU."""
    _reported = False

    def __init__(self, qdata, qtype, oshape, bias=None):
        super().__init__()
        # PLAIN attribute on purpose: register_buffer would make model.to(device)
        # move ALL quantized bytes (11GB+) onto the card. qdata must stay CPU-side;
        # forward() streams it per-layer.
        object.__setattr__(self, "qdata", qdata)
        self.qtype = qtype
        self.oshape = tuple(int(x) for x in oshape)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def forward(self, x):
        if not GGUFLinearWrapper._reported:
            GGUFLinearWrapper._reported = True
            print("[LBWorld] FIRST LINEAR: x.device={} x.dtype={} qdata.device={}".format(
                x.device, x.dtype, self.qdata.device))
        q = self.qdata.to(x.device, non_blocking=True) if self.qdata.device != x.device else self.qdata
        w = dequant_bytes(q, self.qtype, self.oshape, x.dtype)
        b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(x, w, b)
        del w
        if q is not self.qdata:
            del q
        return out


def _gguf_files():
    """ComfyUI's file lists filter by registered extensions and .gguf usually is not
    one of them - so check city96's unet_gguf key AND walk the folders directly."""
    files = set()
    for key in ("unet_gguf", "diffusion_models", "unet"):
        try:
            files.update(f for f in folder_paths.get_filename_list(key) if f.endswith(".gguf"))
        except Exception:
            pass
    for key in ("diffusion_models", "unet"):
        try:
            bases = folder_paths.get_folder_paths(key)
        except Exception:
            continue
        for base in bases:
            if not os.path.isdir(base):
                continue
            for root, _dirs, names in os.walk(base):
                for n in names:
                    if n.endswith(".gguf"):
                        files.add(os.path.relpath(os.path.join(root, n), base))
    return sorted(files) or ["none found"]


def _vae_files():
    try:
        files = folder_paths.get_filename_list("vae")
    except Exception:
        files = []
    return sorted(f for f in files if f.endswith((".pth", ".safetensors"))) or ["none found"]


def _resolve(kind_keys, name):
    for key in kind_keys:
        try:
            p = folder_paths.get_full_path(key, name)
            if p:
                return p
        except Exception:
            pass
    # direct folder walk fallback (matches how _gguf_files finds them)
    for key in kind_keys:
        try:
            bases = folder_paths.get_folder_paths(key)
        except Exception:
            continue
        for base in bases:
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(name)


def _action_dirs():
    base = os.path.join(folder_paths.get_input_directory(), "lingbot_actions")
    if not os.path.isdir(base):
        return ["none found (put dirs in input/lingbot_actions/)"]
    out = [d for d in sorted(os.listdir(base))
           if os.path.isfile(os.path.join(base, d, "poses.npy"))]
    return out or ["none found (put dirs in input/lingbot_actions/)"]


_pinned = {"bytes": 0}


class _T5Shim:
    """Stands in for their T5EncoderModel. Returns pre-encoded ComfyUI embeds."""

    class _M(torch.nn.Module):
        pass

    def __init__(self):
        self.model = self._M()  # their offload code calls .model.to(...)
        self._store = {}

    def set(self, prompt_key, embeds):
        self._store[prompt_key] = embeds

    def __call__(self, prompts, device):
        outs = []
        for p in prompts:
            key = hashlib.sha256(p.encode("utf-8")).hexdigest()
            if key not in self._store:
                raise RuntimeError(
                    "[LBWorld] T5 shim has no embeds for this prompt. Feed BOTH the "
                    "positive and negative CONDITIONING into the sampler.")
            outs.append(self._store[key].to(device))
        return outs


def _cond_to_embed(conditioning):
    """ComfyUI CONDITIONING -> [L, C] tensor their loop expects."""
    t = conditioning[0][0]
    if t.dim() == 3:
        t = t[0]
    return t.contiguous()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class LBWorldLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_gguf": (_gguf_files(),),
                "vae_name": (_vae_files(),),
                # world-memory window: KV cache scales linearly with local+sink.
                # Upstream defaults (18+6) need ~21GB at 480p - datacenter territory.
                "local_attn_size": ("INT", {"default": 6, "min": 1, "max": 64}),
                "sink_size": ("INT", {"default": 2, "min": 0, "max": 32}),
                # pinned RAM speeds up weight streaming but LOCKS Windows commit charge;
                # on a 16GB box with a small pagefile this causes phantom CUDA OOMs.
                "pin_gb": ("INT", {"default": 2, "min": 0, "max": 8}),
            }
        }

    RETURN_TYPES = ("LBWORLD_PIPE",)
    FUNCTION = "load"
    CATEGORY = "Rebels/LingBotWorld"

    def load(self, dit_gguf, vae_name, local_attn_size, sink_size, pin_gb=2):
        import gguf as ggufpkg
        from accelerate import init_empty_weights
        from wan.modules.model_fast import WanModelFast
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.configs.wan_i2v_A14B import i2v_A14B as cfg
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        import wan.image2video as i2v_mod

        # ---- DiT skeleton from bundled config ----
        cfg_json = os.path.join(ASSETS_DIR, "transformer_config.json")
        if os.path.isfile(cfg_json):
            import json
            with open(cfg_json) as f:
                tconf = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        else:
            raise FileNotFoundError(
                "model_assets/transformer_config.json missing - copy the config.json "
                "from LINGBOT_WORLD/transformers/ and rename it.")
        tconf["local_attn_size"] = local_attn_size
        tconf["sink_size"] = sink_size
        print("[LBWorld] building WanModelFast skeleton (14B)")
        with init_empty_weights():
            model = WanModelFast.from_config(tconf)

        # ---- attach GGUF tensors ----
        path = _resolve(("unet_gguf", "diffusion_models", "unet"), dit_gguf)
        print("[LBWorld] loading GGUF: {}".format(path))
        reader = ggufpkg.GGUFReader(path)

        # >4D tensors (patch_embedding) are stored flattened with their true shape
        # in comfy.gguf.orig_shape.<name> metadata - build the restore map
        orig_shapes = {}
        for field in reader.fields.values():
            if field.name.startswith("comfy.gguf.orig_shape."):
                tname = field.name[len("comfy.gguf.orig_shape."):]
                orig_shapes[tname] = [int(field.parts[d][0]) for d in field.data]
        if orig_shapes:
            print("[LBWorld] orig_shape metadata for: {}".format(list(orig_shapes)))

        sd_keys = dict(model.named_parameters())
        sd_keys.update(dict(model.named_buffers()))

        attached, unmatched = 0, []
        modules = dict(model.named_modules())
        F16 = ggufpkg.GGMLQuantizationType.F16
        F32 = ggufpkg.GGMLQuantizationType.F32
        pin_budget = int(pin_gb) * (1024 ** 3)
        from collections import Counter
        qhist = Counter()
        qbytes_by_type = Counter()

        for t in reader.tensors:
            name = str(t.name)
            if name not in sd_keys:
                unmatched.append(name)
                continue
            oshape = orig_shapes.get(name) or [int(d) for d in reversed(t.shape)]
            qtype = t.tensor_type
            qhist[str(qtype).split(".")[-1]] += 1
            qbytes_by_type[str(qtype).split(".")[-1]] += t.data.nbytes
            # from_numpy on the reader's memmap SHARES the file mapping: no RAM copy.
            # Only pinning (below) copies - and only within the pin budget.
            data = torch.from_numpy(t.data)
            mod_name, _, param_name = name.rpartition(".")
            parent = modules.get(mod_name)
            if qtype in (F16, F32):
                w = data.view(torch.float16) if qtype == F16 else data.view(torch.float32)
                w = w.reshape(oshape).to(torch.bfloat16)
                if param_name in dict(parent.named_parameters(recurse=False)):
                    setattr(parent, param_name, torch.nn.Parameter(w, requires_grad=False))
                else:
                    parent._buffers[param_name] = w
            elif isinstance(parent, torch.nn.Linear) and param_name == "weight":
                qb = data
                if _pinned["bytes"] < pin_budget:
                    try:
                        qb = qb.pin_memory()
                        _pinned["bytes"] += qb.numel()
                    except RuntimeError:
                        pass
                bias = parent.bias.detach().to(torch.bfloat16) if parent.bias is not None \
                    and parent.bias.device.type != "meta" else None
                wrapper = GGUFLinearWrapper(qb, qtype, oshape, bias=bias)
                # replace the module in its parent
                gp_name, _, leaf = mod_name.rpartition(".")
                gp = modules.get(gp_name, model)
                setattr(gp, leaf, wrapper)
            else:
                # quantized non-linear (norm/conv/embedding): dequant once to bf16
                w = dequant_bytes(data, qtype, oshape, torch.bfloat16)
                if param_name in dict(parent.named_parameters(recurse=False)):
                    setattr(parent, param_name, torch.nn.Parameter(w, requires_grad=False))
                else:
                    parent._buffers[param_name] = w
            attached += 1

        if _pinned["bytes"]:
            print("[LBWorld] pinned {:.2f} GB of quantized weights (async H2D)".format(
                _pinned["bytes"] / 1e9))
        print("[LBWorld] quant histogram (count / GB): " + ", ".join(
            "{} x{} ({:.2f}GB)".format(k, qhist[k], qbytes_by_type[k] / 1e9)
            for k, _ in qhist.most_common()))
        meta_left = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        print("[LBWorld] {} tensors attached; unmatched-in-model: {}; still-meta params: {}".format(
            attached, len(unmatched), len(meta_left)))
        if unmatched[:5]:
            print("[LBWorld]   unmatched sample: {}".format(unmatched[:5]))
        if meta_left[:5]:
            print("[LBWorld]   meta sample: {}".format(meta_left[:5]))
        model.eval().requires_grad_(False)

        # ---- VAE ----
        vae_path = _resolve(("vae",), vae_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # build the VAE on CPU: loading it straight onto CUDA was the first device
        # allocation and died on Windows commit exhaustion before sampling even began
        vae = Wan2_1_VAE(vae_pth=vae_path, device=torch.device("cpu"))

        # ---- pipeline object, constructed WITHOUT their __init__ (which would
        #      force-load an 11GB T5 .pth and a 28GB DiT we cannot hold) ----
        WanI2V = i2v_mod.WanI2V if hasattr(i2v_mod, "WanI2V") else None
        if WanI2V is None:  # fall back: find the class defined in the module
            for v in vars(i2v_mod).values():
                if isinstance(v, type) and hasattr(v, "_generate_causal_fast"):
                    WanI2V = v
                    break
        pipe = object.__new__(WanI2V)
        pipe.infer_mode = "causal_fast"
        pipe.device = device
        pipe.config = cfg
        pipe.rank = 0
        pipe.t5_cpu = True
        pipe.init_on_cpu = True
        pipe.num_train_timesteps = cfg.num_train_timesteps
        pipe.boundary = cfg.boundary
        pipe.param_dtype = cfg.param_dtype
        pipe.pipe_dtype = torch.bfloat16
        pipe.local_attn_size = local_attn_size
        pipe.sink_size = sink_size
        pipe.text_encoder = _T5Shim()
        pipe.vae_stride = cfg.vae_stride
        pipe.patch_size = cfg.patch_size
        pipe.vae = vae
        pipe.model = model
        pipe.sp_size = 1
        pipe.sample_neg_prompt = cfg.sample_neg_prompt
        pipe.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps, shift=1,
            use_dynamic_shifting=False)
        pipe._t5_cache = {}
        pipe._cross_attn_initialized = False

        return ({"pipe": pipe, "model": model, "device": device},)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
class LBWorldSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lbworld": ("LBWORLD_PIPE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "action_dir": (_action_dirs(),),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["256x448 (lowest VRAM)", "320x544", "384x656", "480x832 (needs tiny window)"],
                               {"default": "256x448 (lowest VRAM)"}),
                "frame_num": ("INT", {"default": 81, "min": 5, "max": 361, "step": 4}),
                "chunk_size": ("INT", {"default": 3, "min": 1, "max": 9}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "action_override": ("STRING", {"forceInput": True}),  # from Action Builder
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/LingBotWorld"

    def sample(self, lbworld, positive, negative, image, action_dir, prompt_text,
               resolution, frame_num, chunk_size, shift, seed, action_override=None):
        if action_override and str(action_override).strip():
            action_dir = str(action_override).strip()
        from PIL import Image as PILImage

        pipe = lbworld["pipe"]
        model = lbworld["model"]
        device = lbworld["device"]

        try:
            import comfy.model_management as mm
            mm.unload_all_models()   # umt5 etc. - conditioning is already computed
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # register embeds with the shim, keyed exactly how their loop will ask
        prompt = prompt_text.strip()
        neg = pipe.sample_neg_prompt
        pipe.text_encoder._store.clear()
        pipe._t5_cache.clear()
        for text, cond in ((prompt, positive), (neg, negative)):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            pipe.text_encoder.set(key, _cond_to_embed(cond))

        base = os.path.join(folder_paths.get_input_directory(), "lingbot_actions")
        action_path = os.path.join(base, action_dir)
        if not os.path.isfile(os.path.join(action_path, "poses.npy")):
            raise FileNotFoundError(
                "action dir has no poses.npy: {} - copy an examples/ dir from the "
                "lingbot-world-v2 repo into input/lingbot_actions/".format(action_path))

        arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil = PILImage.fromarray(arr)

        if (frame_num - 1) % 4 != 0:
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            print("[LBWorld] frame_num adjusted to {} (4n+1)".format(frame_num))

        res_h, res_w = {"256x448 (lowest VRAM)": (256, 448), "320x544": (320, 544),
                        "384x656": (384, 656), "480x832 (needs tiny window)": (480, 832)}[resolution]
        max_area = res_h * res_w

        # ---- KV-cache pre-flight ----
        # The causal cache is 2 (k+v) x 40 layers x window x tokens/frame x 5120 x 2B.
        # At 480x832 with window 18+6 that is ~21GB: it CANNOT fit on 8GB, and letting
        # it spill into WDDM shared memory is the silent multi-minute hang.
        local_ = getattr(pipe, "local_attn_size", 18)
        sink_ = getattr(pipe, "sink_size", 6)
        ar = image.shape[1] / image.shape[2]  # h/w of the input drives their h,w calc
        import math as _m
        lat_h = round(_m.sqrt(max_area * ar) // 8 // 2 * 2)
        lat_w = round(_m.sqrt(max_area / ar) // 8 // 2 * 2)
        tokens = max(int(lat_h * lat_w // 4), 1)
        kv_gb = 2 * 40 * (local_ + sink_) * tokens * 5120 * 2 / 1e9
        free_gb = 0.0
        if torch.cuda.is_available():
            free_b, _tot = torch.cuda.mem_get_info()
            free_gb = free_b / 1e9
        print("[LBWorld] KV-cache estimate: {:.2f} GB (window {}+{}, ~{} tok/frame); "
              "free VRAM {:.2f} GB".format(kv_gb, local_, sink_, tokens, free_gb))
        budget = max(free_gb - 2.5, 0.5)  # leave room for activations + dequant buffers
        if kv_gb > budget:
            need_window = max(int(budget * 1e9 / (2 * 40 * tokens * 5120 * 2)), 1)
            raise RuntimeError(
                "[LBWorld] KV cache ({:.1f} GB) will not fit ({:.1f} GB budget). "
                "Lower the resolution preset, or set local_attn_size+sink_size <= {} "
                "in the Loader (currently {}+{}). Refusing to start rather than "
                "thrash shared memory for minutes and die.".format(
                    kv_gb, budget, need_window, local_, sink_))

        print("[LBWorld] causal_fast: {}x{} frames={} chunk={} shift={} seed={}".format(
            res_w, res_h, frame_num, chunk_size, shift, seed))
        def _move_vae(v, dev):
            try:
                v.model = v.model.to(dev)
                v.mean = v.mean.to(dev)
                v.std = v.std.to(dev)
                v.scale = [v.mean, 1.0 / v.std]
                if hasattr(v, "device"):
                    v.device = dev
            except Exception as e:
                print("[LBWorld] vae move to {} failed: {}".format(dev, e))

        try:
            model.to(device)
            _move_vae(pipe.vae, device)
            with torch.inference_mode():
                video = pipe.generate(
                    prompt,
                    pil,
                    action_path,
                    chunk_size=chunk_size,
                    max_area=max_area,
                    frame_num=frame_num,
                    shift=shift,
                    seed=seed & 0x7FFFFFFF,
                    offload_model=True,
                )
        finally:
            try:
                model.to("cpu")
            except Exception:
                pass
            _move_vae(pipe.vae, torch.device("cpu"))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # their return: tensor [C, N, H, W] in [-1, 1]
        v = video.detach().float().cpu()
        if v.dim() == 4 and v.shape[0] in (3,):
            v = v.permute(1, 2, 3, 0)  # -> [N, H, W, C]
        v = (v.clamp(-1, 1) + 1.0) / 2.0
        return (v.contiguous(),)




# ---------------------------------------------------------------------------
# Action builder: type camera moves, get a valid action folder
# ---------------------------------------------------------------------------
class LBWorldActionBuilder:
    """Writes poses.npy ([N,4,4] c2w, OpenCV convention) + intrinsics.npy into
    input/lingbot_actions/<name>/ so the sampler can use it. In the causal_fast
    path these two files are the ONLY conditioning (wasd is unused upstream).

    Segment lines, one per line:  <move> <frames> <amount>
      moves: forward back left right up down turn_left turn_right tilt_up tilt_down stay
      amount: total meters for translations, total degrees for turns (ignored for stay)
    Example:
      forward 32 2.0
      turn_left 24 45
      forward 24 1.5
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "my_action"}),
                "segments": ("STRING", {"multiline": True,
                                        "default": "forward 32 2.0\nturn_left 24 45\nforward 24 1.5"}),
                "fov_deg": ("FLOAT", {"default": 60.0, "min": 20.0, "max": 120.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("action_dir",)
    FUNCTION = "build"
    CATEGORY = "Rebels/LingBotWorld"

    def build(self, name, segments, fov_deg):
        import math as _m

        def rot(axis, deg):
            a = _m.radians(deg)
            c, s_ = _m.cos(a), _m.sin(a)
            R = np.eye(3)
            i, j = {"y": (0, 2), "x": (1, 2)}[axis]
            R[i, i] = c; R[j, j] = c
            R[i, j] = s_; R[j, i] = -s_
            return R

        # OpenCV camera axes: +x right, +y down, +z forward
        MOVES = {"forward": np.array([0, 0, 1.0]), "back": np.array([0, 0, -1.0]),
                 "left": np.array([-1.0, 0, 0]), "right": np.array([1.0, 0, 0]),
                 "up": np.array([0, -1.0, 0]), "down": np.array([0, 1.0, 0])}
        # OpenCV y points DOWN: negative yaw about y turns the forward axis toward -x (left).
        TURNS = {"turn_left": ("y", -1.0), "turn_right": ("y", 1.0),
                 "tilt_up": ("x", -1.0), "tilt_down": ("x", 1.0)}

        R = np.eye(3)
        t = np.zeros(3)
        poses = [np.eye(4)]
        for line_no, raw in enumerate(segments.strip().splitlines(), 1):
            parts = raw.strip().split()
            if not parts:
                continue
            move = parts[0].lower()
            try:
                nf = int(parts[1])
                amt = float(parts[2]) if len(parts) > 2 else 0.0
            except (IndexError, ValueError):
                raise ValueError("bad segment line {}: '{}' (want: move frames amount)".format(line_no, raw))
            for _ in range(nf):
                if move in MOVES:
                    t = t + R @ (MOVES[move] * (amt / max(nf, 1)))
                elif move in TURNS:
                    ax, sign = TURNS[move]
                    R = R @ rot(ax, sign * amt / max(nf, 1))
                elif move != "stay":
                    raise ValueError("unknown move '{}' on line {}".format(move, line_no))
                p = np.eye(4)
                p[:3, :3] = R
                p[:3, 3] = t
                poses.append(p)

        n = len(poses)
        n4 = ((n - 1) // 4) * 4 + 1
        poses = np.stack(poses[:n4]).astype(np.float64)

        w, h = 832.0, 480.0
        fx = (w / 2.0) / _m.tan(_m.radians(fov_deg) / 2.0)
        # their loader expects [N, 4] flat vectors (fx, fy, cx, cy) - NOT a 3x3 matrix
        # (get_Ks_transformed does Ks.chunk(4, dim=-1))
        K = np.array([[fx, fx, w / 2.0, h / 2.0]]).astype(np.float64)

        base = os.path.join(folder_paths.get_input_directory(), "lingbot_actions", name)
        os.makedirs(base, exist_ok=True)
        np.save(os.path.join(base, "poses.npy"), poses)
        np.save(os.path.join(base, "intrinsics.npy"), K)
        print("[LBWorld] wrote action '{}': {} poses (trimmed to 4n+1), fov {}".format(
            name, n4, fov_deg))
        return (name,)

NODE_CLASS_MAPPINGS = {
    "LBWorldLoader": LBWorldLoader,
    "LBWorldSampler": LBWorldSampler,
    "LBWorldActionBuilder": LBWorldActionBuilder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LBWorldLoader": "LingBot-World Loader (14B GGUF)",
    "LBWorldSampler": "LingBot-World Sampler (causal-fast)",
    "LBWorldActionBuilder": "LingBot-World Action Builder",
}
