# ComfyUI_Rebels_LingBotWorld

Run **LingBot-World-v2 14B (causal-fast)** — an action-controlled world model — on an
**8GB GPU**. Built and tested on an RTX 3070 / 16GB RAM / Windows 11.

The 14B DiT loads **GGUF-only** (quantized bytes stay in system RAM, memmap-backed,
dequantized per layer on the fly). Text encoding uses ComfyUI's native **UMT5-XXL GGUF**
(CLIPLoader GGUF, type `wan`) — the upstream 11GB T5 .pth is never needed. VAE is the
standard `Wan2.1_VAE.pth` you already have.

This is **offline generation**, not real-time play: you author an action track (camera
path + movement), the model renders the video that track produces from your start image.
The model genuinely follows the actions — that's the feature prompt-only video can't do.

## Nodes
- **LingBot-World Loader (14B GGUF)** — DiT GGUF + VAE dropdowns, world-memory window
  (`local_attn_size`/`sink_size`), `pin_gb` for pinned-RAM streaming speed.
- **LingBot-World Sampler (causal-fast)** — image + positive/negative CONDITIONING +
  action folder + resolution preset. 4 distilled steps per chunk. Includes a **KV-cache
  pre-flight** that refuses with corrective numbers instead of hanging when settings
  exceed VRAM.
- **LingBot-World Action Builder** — writes a valid action folder from text.

## Action Builder syntax
One segment per line: `<move> <frames> <amount>`
- `frames` = duration; `amount` = **total** meters (moves) or **total** degrees (turns)
- Fewer frames for the same amount = faster motion
- moves: `forward back left right up down turn_left turn_right tilt_up tilt_down stay`
```
forward 32 2.0     # walk 2 m forward over 32 frames
turn_left 24 45    # rotate 45° left over 24 frames
forward 24 1.5     # then 1.5 m forward, slightly faster
stay 16            # hold 16 frames
```
Or copy the repo's `examples/` dirs into `ComfyUI/input/lingbot_actions/` and pick them
from the dropdown (each needs `poses.npy` + `intrinsics.npy`).

## VRAM math you must respect
The causal KV cache **is** the model's world memory and scales with
`(local_attn_size + sink_size) × resolution`. Upstream defaults (18+6 @ 480×832) need
~21GB — datacenter territory. On 8GB use window **6+2** and:
| Preset | KV cache |
|---|---|
| 256×448 | ~2.9 GB ✔ default |
| 320×544 | ~4.5 GB (tight) |
| 384×656 | ~6.5 GB (smaller window needed) |
| 480×832 | ~10.2 GB (does not fit) |

Smaller window = shorter world memory: scenes stay consistent over shorter horizons.
Favor forward exploration over look-back-and-return shots.

## Setup
1. Unzip into `custom_nodes\`; install city96's **ComfyUI-GGUF** next to it (dequant kernels).
2. `pip install easydict einops ftfy imageio scipy` (ignore upstream requirements.txt).
3. Copy `transformers/config.json` from the HF repo into `model_assets/transformer_config.json`.
4. DiT GGUF → `models/diffusion_models/`; `Wan2.1_VAE.pth` → `models/vae/`;
   UMT5-XXL GGUF via CLIPLoader (GGUF), type `wan`.
5. Example action dirs → `input/lingbot_actions/`.
6. **Windows: set a real pagefile (32GB+).** WDDM backs GPU allocations with system
   commit; a tiny pagefile causes phantom CUDA OOMs with this pipeline.

First run: 256×448, frame_num 21, chunk 1. The negative CLIPTextEncode must contain the
upstream Chinese negative from `wan/configs/wan_i2v_A14B.py` verbatim.

## Credits & license
Model/code: robbyant/lingbot-world-v2 (**CC BY-NC-SA 4.0** — non-commercial, share-alike).
Node code MIT. GGUF tooling: city96. Nodes/quants: RealRebelAI.
