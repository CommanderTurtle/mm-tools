#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Download and verify the VBench metric weights into VBENCH_CACHE_DIR (~9.4 GB).
#
#   VBENCH_CACHE_DIR=/path/to/vbench_cache bash download_models.sh           # download missing files
#   VBENCH_CACHE_DIR=/path/to/vbench_cache bash download_models.sh --verify  # only check hashes
#
# Every file is checked against the SHA-256 of the copy used for the paper's
# scores. Existing files with the right hash are skipped.
set -euo pipefail

CACHE="${VBENCH_CACHE_DIR:?set VBENCH_CACHE_DIR}"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
HF=https://huggingface.co
CLIP=https://openaipublic.azureedge.net/clip/models
BERT_REV=86b5e0934494bd15c9632b12f734a8a67f723594

# relative path | sha256 | url
FILES=(
  "clip_model/ViT-B-32.pt|40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af|$CLIP/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
  "clip_model/ViT-L-14.pt|b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836|$CLIP/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
  "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth|2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba|https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"
  "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth|bfee78a03bf806fbc0c216309e8f92792011b7db6556fa860e9d869c1e69fecd|$HF/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth"
  "amt_model/amt-s.pth|07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77|$HF/lalala125/AMT/resolve/main/amt-s.pth"
  "dino_model/dino_vitbase16_pretrain.pth|bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e|https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
  "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth|358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50|https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"
  "ViCLIP/ViClip-InternVid-10M-FLT.pth|7a4d6ad6eac6632db3693f4b97f9f8f6445b65b1e139a6fc6686150522238c56|$HF/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
  "ViCLIP/bpe_simple_vocab_16e6.txt.gz|924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a|https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
  "grit_model/grit_b_densecap_objectdet.pth|53b6e9b3fd948eac55b574c9b6f94ad0743dff46ba449df7ac2d33009ee92ef1|$HF/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth"
  "caption_model/tag2text_swin_14m.pth|4ce96f0ce98f940a6680d567f66a38ccc9ca8c4e638e5f5c5c2e881a0e3502ac|$HF/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth"
  "bert-base-uncased/config.json|7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20|$HF/google-bert/bert-base-uncased/resolve/$BERT_REV/config.json"
  "bert-base-uncased/pytorch_model.bin|097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a|$HF/google-bert/bert-base-uncased/resolve/$BERT_REV/pytorch_model.bin"
  "bert-base-uncased/tokenizer.json|ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98|$HF/google-bert/bert-base-uncased/resolve/$BERT_REV/tokenizer.json"
  "bert-base-uncased/tokenizer_config.json|a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10|$HF/google-bert/bert-base-uncased/resolve/$BERT_REV/tokenizer_config.json"
  "bert-base-uncased/vocab.txt|07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3|$HF/google-bert/bert-base-uncased/resolve/$BERT_REV/vocab.txt"
)
RAFT_SHA=fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1
RAFT_ZIP_URL=https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
DINO_REPO=https://github.com/facebookresearch/dino
DINO_COMMIT=7c446df5b9f45747937fb0d72314eb9f7b66930a
DINO_SOURCES=(  # every file that torch.hub imports from the clone
  "hubconf.py|fa80babf82c71d2123e23b05061eb1991a1fbc9877238f250b2a25023e910f72"
  "vision_transformer.py|b1f998d5f49ab43666b9fc6d007c5f6540c3ead2e8645e144f74045f63ff44d7"
  "utils.py|962a97e1acda1c986dfd921275325e4083f9016dddadcf018f1ccf03fa600eab"
)

sha() { sha256sum "$1" | cut -d' ' -f1; }
ok() { [ -s "$1" ] && [ "$(sha "$1")" = "$2" ]; }
fails=0
report() { echo "[$1] $2"; [ "$1" = ok ] || fails=$((fails + 1)); }

fetch() {  # url dest sha
  local tmp="$2.part"
  mkdir -p "$(dirname "$2")"
  curl -L --fail --retry 3 -o "$tmp" "$1" || { rm -f "$tmp"; return 1; }
  [ "$(sha "$tmp")" = "$3" ] || { echo "hash mismatch for $1" >&2; rm -f "$tmp"; return 1; }
  mv "$tmp" "$2"
}

for entry in "${FILES[@]}"; do
  IFS='|' read -r rel digest url <<< "$entry"
  if ok "$CACHE/$rel" "$digest"; then report ok "$rel"
  elif [ "$VERIFY_ONLY" -eq 1 ]; then report missing "$rel"
  elif fetch "$url" "$CACHE/$rel" "$digest"; then report ok "$rel"
  else report failed "$rel"; fi
done

raft="$CACHE/raft_model/models/raft-things.pth"
if ok "$raft" "$RAFT_SHA"; then report ok raft_model/models/raft-things.pth
elif [ "$VERIFY_ONLY" -eq 1 ]; then report missing raft_model/models/raft-things.pth
else
  mkdir -p "$CACHE/raft_model"
  # extract only the one checkpoint (no archive paths), then verify it
  if curl -L --fail --retry 3 -o "$CACHE/raft_model/models.zip" "$RAFT_ZIP_URL" \
      && unzip -o -q -j -d "$CACHE/raft_model/models" "$CACHE/raft_model/models.zip" models/raft-things.pth \
      && ok "$raft" "$RAFT_SHA"; then
    report ok raft_model/models/raft-things.pth
  else rm -f "$raft"; report failed raft_model/models/raft-things.pth; fi
  rm -f "$CACHE/raft_model/models.zip"
fi

dino="$CACHE/dino_model/facebookresearch_dino_main"
dino_ok() { local entry rel digest; for entry in "${DINO_SOURCES[@]}"; do
  IFS='|' read -r rel digest <<< "$entry"; ok "$dino/$rel" "$digest" || return 1; done; }
if [ "$VERIFY_ONLY" -eq 0 ] && ! dino_ok; then  # missing, stale or partial: fetch the pinned commit again
  rm -rf "$dino" "$dino.tmp"
  { git init -q "$dino.tmp" && git -C "$dino.tmp" fetch -q --depth 1 "$DINO_REPO" "$DINO_COMMIT" \
      && git -C "$dino.tmp" checkout -q FETCH_HEAD && mv "$dino.tmp" "$dino"; } || rm -rf "$dino.tmp"
fi
for entry in "${DINO_SOURCES[@]}"; do
  IFS='|' read -r rel digest <<< "$entry"
  if ok "$dino/$rel" "$digest"; then report ok "dino_model/facebookresearch_dino_main/$rel"
  else report "$([ "$VERIFY_ONLY" -eq 1 ] && echo missing || echo failed)" "dino_model/facebookresearch_dino_main/$rel"; fi
done

# torch.hub loads the DINO weights from TORCH_HOME (score.sh default: $VBENCH_CACHE_DIR/torch).
hub="$CACHE/torch/hub/checkpoints/dino_vitbase16_pretrain.pth"
src="$CACHE/dino_model/dino_vitbase16_pretrain.pth"
if [ "$VERIFY_ONLY" -eq 0 ] && ok "$src" bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e && ! ok "$hub" bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e; then
  mkdir -p "$(dirname "$hub")" && cp "$src" "$hub"
fi
if ok "$hub" bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e; then report ok torch/hub/checkpoints/dino_vitbase16_pretrain.pth
else report missing torch/hub/checkpoints/dino_vitbase16_pretrain.pth; fi

echo "$fails problem(s) in $CACHE"
[ "$fails" -eq 0 ]
