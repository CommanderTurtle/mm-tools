# Vendored model code

Patched copies of the model definitions from
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MIT), which builds on lucidrains' BS-RoFormer implementation (MIT).

Patches applied for Apple Silicon (MPS) inference:

- complex mask math is done in real/imag components — MPS does not support
  complex `scatter_add_` (mel-band variant) or reliable complex multiply
- `torch.istft` is replaced with a manual overlap-add (it depends on
  `aten::unfold_backward`, which is not implemented on MPS)
- the STFT and band-split run in fp32 (fp16 accumulation overflows on loud
  material); the attention stack runs in fp16 for speed
- the mel filter bank is precomputed (`mel_bank_44100_2048_60.npy`) so librosa
  is not a runtime dependency
