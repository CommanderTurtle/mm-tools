#!/usr/bin/env python3
"""
SVS Single-Sample Inference Script (Standalone)

Generates audio from a JSON label entry. Fully independent of preprocessed
data — reads raw audio for prompt and runs VAE encoding on the fly.

Input JSON format (same as label.json / Opencpop.json):
    [
      {
        "item_name": "2001000001",
        "word": ["感", "受", ...],
        "pitch": [68, 68, ...],
        "note": ["<NOTE_DOT_16>", "<NOTE_DOT_8>", ...],
        "pitch2word": [0, 1, ...],
        "bpm": 89,
        "wav_fn": "segments/wavs/2001000001.wav"
      },
      ...
    ]

Usage:
    # Prompt-conditioned inference (required)
    python scripts/infer_vocalrender_svs_single.py \\
        --ckpt_dir outputs/svs_ft/latest \\
        --json_file Opencpop.json \\
        --item_name 2001000001 \\
        --prompt_audio /path/to/2-8s_prompt.wav
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vocalrender.model.utils import get_in_sample_rate, get_out_sample_rate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_svs_model(ckpt_dir: str, device: str = "cuda"):
    """Load SVS fine-tuned model from checkpoint directory."""
    from transformers import LlamaTokenizerFast

    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "latest").exists():
        ckpt_path = ckpt_path / "latest"

    print(f"[SVS Single] Loading model from: {ckpt_path}", file=sys.stderr)

    with open(ckpt_path / "config.json") as f:
        config_dict = json.load(f)
    architecture = str(config_dict.get("architecture", "voxcpm")).lower()
    if architecture == "voxcpm2":
        from vocalrender.model.voxcpm2 import VoxCPM2Model as ModelClass, VoxCPMConfig as ConfigClass
        from vocalrender.modules.audiovae import AudioVAEV2 as AudioVAEClass
    else:
        from vocalrender.model.voxcpm import VoxCPMModel as ModelClass, VoxCPMConfig as ConfigClass
        from vocalrender.modules.audiovae import AudioVAE as AudioVAEClass

    config = ConfigClass(**config_dict)

    tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_path))
    if len(tokenizer) != config.lm_config.vocab_size:
        config.lm_config.vocab_size = len(tokenizer)

    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAEClass(config=audio_vae_config) if audio_vae_config else AudioVAEClass()

    vae_sf_path = ckpt_path / "audiovae.safetensors"
    vae_pt_path = ckpt_path / "audiovae.pth"

    sf_path = ckpt_path / "model.safetensors"
    pt_path = ckpt_path / "pytorch_model.bin"
    try:
        from safetensors.torch import load_file as load_safetensors
        HAS_SF = True
    except ImportError:
        HAS_SF = False

    if vae_sf_path.exists() and HAS_SF:
        vae_sd = load_safetensors(str(vae_sf_path))
    elif vae_pt_path.exists():
        vae_ckpt = torch.load(str(vae_pt_path), map_location="cpu", weights_only=True)
        vae_sd = vae_ckpt.get("state_dict", vae_ckpt)
    else:
        raise FileNotFoundError(f"AudioVAE weights not found: {vae_sf_path} or {vae_pt_path}")
    audio_vae.load_state_dict(vae_sd)

    if architecture == "voxcpm2":
        model = ModelClass(
            config, tokenizer, audio_vae, lora_config=None,
        )
    else:
        model = ModelClass(config, tokenizer, audio_vae, lora_config=None)

    if sf_path.exists() and HAS_SF:
        model_sd = load_safetensors(str(sf_path))
    elif pt_path.exists():
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        model_sd = ckpt.get("state_dict", ckpt)
    else:
        raise FileNotFoundError(f"Weights not found: {sf_path} or {pt_path}")

    for k, v in vae_sd.items():
        model_sd[f"audio_vae.{k}"] = v
    model.load_state_dict(model_sd, strict=False)

    from vocalrender.model.utils import get_dtype
    model = model.to(get_dtype(config.dtype)).to(device).eval()
    model.device = device
    model.audio_vae = model.audio_vae.to(torch.float32)

    print(f"[SVS Single] Model loaded on {device}", file=sys.stderr)
    return model


# ---------------------------------------------------------------------------
# SVS prompt builder
# ---------------------------------------------------------------------------

def build_svs_prompt_from_entry(
    entry: Dict, model, force_lyrics_only: bool = False,
) -> str:
    """Build SVS prompt string from a JSON label entry.

    Supports full-label (word+pitch+note+pitch2word+bpm) and
    weak-label (word only) entries.

    Args:
        entry: JSON label dict with word, pitch, note, pitch2word, bpm fields.
        model: VoxCPMModel (used to get tokenizer).
        force_lyrics_only: If True, use lyrics-only mode even for fully-annotated
            entries (BPM/pitch/note replaced with ``<SVS_MASK>``).
    """
    from vocalrender.preprocessing import rebuild_svs_prompt

    tokenizer = model.text_tokenizer.tokenizer  # underlying LlamaTokenizerFast
    preprocessor = _get_preprocessor(model)

    prompt = rebuild_svs_prompt(
        sample=entry,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        force_lyrics_only=force_lyrics_only,
    )
    if not prompt:
        raise ValueError(f"Empty SVS prompt for '{entry.get('item_name', '?')}'")
    return prompt


# Cached lightweight preprocessor
_cached_preprocessor = None


def _get_preprocessor(model):
    """Build a lightweight SVSPreprocessor (token maps only, no VAE)."""
    global _cached_preprocessor
    if _cached_preprocessor is not None:
        return _cached_preprocessor

    from vocalrender.preprocessing import create_lightweight_preprocessor

    _cached_preprocessor = create_lightweight_preprocessor(
        model.text_tokenizer.tokenizer,
    )
    return _cached_preprocessor


# ---------------------------------------------------------------------------
# On-the-fly VAE encoding for prompt audio
# ---------------------------------------------------------------------------

def encode_prompt_audio(
    wav_path: str,
    model,
    max_frames: int = 50,
) -> Optional[torch.Tensor]:
    """Load audio from file, encode through VAE, return [T, P, D] latent.

    Args:
        wav_path: Path to .wav file
        model: VoxCPMModel with audio_vae
        max_frames: Max number of latent patches (tokens) to keep

    Returns:
        Tensor [T, P, D] or None on failure
    """
    import soundfile as sf
    import torchaudio

    target_sr = get_in_sample_rate(model)
    patch_size = model.patch_size

    # Load audio
    audio_data, sr = sf.read(wav_path)
    wav = torch.from_numpy(audio_data).float()
    if wav.dim() == 2:
        wav = wav.mean(dim=1)  # stereo → mono

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)

    # VAE encode: [1, 1, T_wav] → [1, D, T'] → [T', D]
    wav_input = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    hop = model.audio_vae.hop_length
    wav_len = wav_input.size(-1)
    patch_len = hop * patch_size
    if wav_len % patch_len != 0:
        pad = patch_len - wav_len % patch_len
        wav_input = nn.functional.pad(wav_input, (0, pad))

    wav_input = wav_input.to(model.device)
    with torch.no_grad():
        z = model.audio_vae.encode(wav_input, target_sr)  # [1, D, T']
        feats = z.transpose(1, 2).cpu()  # [1, T', D]

    # Reshape to patches: [1, T', D] → [T, P, D]
    T_prime = feats.size(1)
    if T_prime % patch_size != 0:
        pad_len = patch_size - T_prime % patch_size
        feats = nn.functional.pad(feats.transpose(1, 2), (0, pad_len)).transpose(1, 2)

    feats = rearrange(feats, "b (t p) c -> b t p c", p=patch_size).squeeze(0)  # [T, P, D]

    # Crop to max_frames
    if feats.shape[0] > max_frames:
        import random
        start = random.Random(42).randint(0, feats.shape[0] - max_frames)
        feats = feats[start : start + max_frames]

    return feats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SVS single-sample inference from JSON label file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--ckpt_dir", required=True,
                        help="SVS model checkpoint directory")
    parser.add_argument("--device", default="cuda")

    # Input
    parser.add_argument("--json_file", required=True,
                        help="JSON label file (e.g. Opencpop.json)")
    parser.add_argument("--item_name", required=True,
                        help="item_name of the entry to synthesize")

    # Prompt audio is required: released checkpoints were trained with
    # prompt_audio_prob=1.0, so prompt-free inference is out-of-distribution.
    parser.add_argument("--prompt_audio", required=True,
                        help="Path to a clean 2-8 second singing prompt wav")
    parser.add_argument("--prompt_max_frames", type=int, default=50,
                        help="Max VAE latent frames for prompt audio")
    parser.add_argument("--lyrics_only", action="store_true",
                        help="Force lyrics-only mode: mask all BPM/pitch/note tokens")

    # Generation
    parser.add_argument("--cfg_value", type=float, default=2.0)
    parser.add_argument("--inference_timesteps", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fsq_temperature", type=float, default=0.0)

    parser.add_argument("--output", default="svs_output.wav")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[SVS Single] Warning: CUDA not available, falling back to CPU",
              file=sys.stderr)
        args.device = "cpu"

    # ---- Load JSON ----
    print(f"[SVS Single] Loading JSON: {args.json_file}", file=sys.stderr)
    with open(args.json_file, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    # Build item_name → entry index for quick lookup
    name_to_entry = {}
    for e in all_entries:
        name = e.get("item_name")
        if name:
            name_to_entry[name] = e

    entry = name_to_entry.get(args.item_name)
    if entry is None:
        print(f"Error: item_name '{args.item_name}' not found", file=sys.stderr)
        sys.exit(1)

    has_score = bool(entry.get("pitch") and entry.get("note"))
    effective_mode = "lyrics-only" if args.lyrics_only else ("full" if has_score else "weak")
    print(f"[SVS Single] Entry: {args.item_name}", file=sys.stderr)
    print(f"  words: {''.join(entry['word'])}", file=sys.stderr)
    print(f"  bpm: {entry.get('bpm', 120)}, label: {effective_mode}",
          file=sys.stderr)

    wav_path = Path(args.prompt_audio)
    if not wav_path.is_file():
        print(f"Error: prompt audio not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Load model ----
    model = load_svs_model(args.ckpt_dir, device=args.device)
    sample_rate = get_out_sample_rate(model)

    # ---- Build SVS prompt ----
    svs_prompt = build_svs_prompt_from_entry(
        entry, model, force_lyrics_only=args.lyrics_only,
    )
    print(f"  prompt: {svs_prompt[:200]}{'...' if len(svs_prompt) > 200 else ''}",
          file=sys.stderr)

    # ---- Required prompt audio (on-the-fly VAE encode) ----
    print(f"[SVS Single] Encoding prompt audio: {wav_path}", file=sys.stderr)
    prompt_audio_feats = encode_prompt_audio(
        str(wav_path), model, max_frames=args.prompt_max_frames,
    )
    if prompt_audio_feats is None or prompt_audio_feats.numel() == 0:
        print("Error: prompt audio encoding failed", file=sys.stderr)
        sys.exit(1)
    print(f"  prompt latent: {prompt_audio_feats.shape}", file=sys.stderr)

    # ---- Generate ----
    print(f"[SVS Single] Generating (cfg={args.cfg_value}, "
          f"steps={args.inference_timesteps}, max_len={args.max_len})...",
          file=sys.stderr)

    gen_kwargs = dict(
        target_texts=[svs_prompt],
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        verbose=True,
        temperature=args.temperature,
        fsq_temperature=args.fsq_temperature,
        prompt_audio_feats=[prompt_audio_feats],
    )

    with torch.no_grad():
        audio_tensors = model.generate_batch(**gen_kwargs)

    if not audio_tensors or audio_tensors[0] is None or audio_tensors[0].numel() == 0:
        print("Error: audio generation failed", file=sys.stderr)
        sys.exit(1)

    # ---- Save ----
    from vocalrender.evaluation.audio_utils import normalize_audio
    import soundfile as sf

    audio_np = normalize_audio(audio_tensors[0].float().numpy().flatten())
    duration = len(audio_np) / sample_rate

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_np, sample_rate)

    print(f"[SVS Single] Saved: {output_path} ({duration:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
