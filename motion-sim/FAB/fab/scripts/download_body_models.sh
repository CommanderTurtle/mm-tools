#!/bin/bash
# Body-model files FAB reads, under $FAB_HOME/body_models (default ~/.cache/fab/body_models); nothing of it is shipped:
#   smpl/SMPL_NEUTRAL.pkl (+ SMPL_MALE.pkl, SMPL_FEMALE.pkl for the benchmarks)  from your SMPL download (registration at
#                          smpl.is.tue.mpg.de; SMPL for Python v1.1.0), copied here under these names
#   smpl/J_regressor_h36m.npy  the H36M joint regressor of SPIN's data release (BSD-3-Clause repository), checked by SHA256
#   mhr/*                  MHR's MHR <-> SMPL conversion assets (tools/mhr_smpl_conversion/assets, Apache-2.0), pinned commit
#   bash scripts/download_body_models.sh /path/to/SMPL_python_v.1.1.0     # the folder you downloaded and unpacked
set -euo pipefail
SRC=${1:?the unpacked SMPL download (SMPL_python_v.1.1.0)}
OUT=${FAB_HOME:-$HOME/.cache/fab}/body_models
SUM=$(command -v sha256sum > /dev/null && echo "sha256sum" || echo "shasum -a 256")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT/smpl" "$OUT/mhr"

# 1. SMPL (your download): neutral / male / female
pick() { find "$SRC" -iname "$1" | head -n 1; }
for g in neutral male female; do
  G=$(echo $g | tr a-z A-Z)
  f=$(pick "SMPL_${G}.pkl"); [ -n "$f" ] || f=$(pick "basicmodel_${g:0:1}_lbs_10_207_0_v1.1.0.pkl")
  [ -n "$f" ] || f=$(pick "basicmodel_${g}_lbs_10_207_0_v1.1.0.pkl")
  if [ -n "$f" ]; then cp "$f" "$OUT/smpl/SMPL_${G}.pkl"; echo "SMPL $g: $f"
    [ $g = neutral ] && ! echo "98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42  $OUT/smpl/SMPL_NEUTRAL.pkl" | $SUM -c - > /dev/null 2>&1 \
      && echo "warning: this SMPL_NEUTRAL.pkl differs from the one the released numbers used (README.md)"
  elif [ $g = neutral ]; then echo "no neutral SMPL model under $SRC"; exit 1
  else echo "SMPL $g: not found (only the benchmarks need it)"; fi
done

# 2. the H36M joint regressor (SPIN's data archive, about 15 MB; only this member is kept)
SPIN_URL=http://visiondata.cis.upenn.edu/spin/data.tar.gz
SHA_J36M=c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0
curl -sSfL "$SPIN_URL" -o "$TMP/spin.tar"
tar xf "$TMP/spin.tar" -C "$TMP" data/J_regressor_h36m.npy
echo "$SHA_J36M  $TMP/data/J_regressor_h36m.npy" | $SUM -c - > /dev/null || { echo "J_regressor_h36m.npy checksum mismatch"; exit 1; }
cp "$TMP/data/J_regressor_h36m.npy" "$OUT/smpl/"

# 3. MHR's conversion assets at a pinned commit
MHR_REPO=https://github.com/facebookresearch/MHR.git
MHR_COMMIT=85fc2cbaf77f606e41346cb791d8ad96a75e1a4b
git init -q "$TMP/MHR"; git -C "$TMP/MHR" remote add origin "$MHR_REPO"
git -C "$TMP/MHR" sparse-checkout set --no-cone tools/mhr_smpl_conversion/assets
git -C "$TMP/MHR" fetch -q --depth 1 origin "$MHR_COMMIT"; git -C "$TMP/MHR" checkout -q FETCH_HEAD
cp "$TMP/MHR/tools/mhr_smpl_conversion/assets/"* "$OUT/mhr/"
(cd "$OUT/mhr" && $SUM -c - <<EOF
60a64a305731dcc1c09f167dfa844246443b3115ae05abebe3813aa64a9ba2ae  head_hand_mask.npz
4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7  mhr2smpl_mapping.npz
d6d023f9f483be73cd4491d0ddf217450627263405c9bc6bf43795dfff809af1  mhr2smplx_mapping.npz
db8ed4fca24a744b1e1f4ed9562f55aff837c9f270debbb34a9d1537a3ac62e1  mhr_face_mask.ply
84f908ad534f470dc5db072a4aeb9ccb4f9aa4484d2bce2558cc1a346df95c98  smpl2mhr_mapping.npz
94742d378ccbd6058ce8aa0221d16726d1e5def45490685c223b2e87c6070f97  smplx2mhr_mapping.npz
76008acafd4468dda802e3ee5523b29fef4de8408cc5e1c65351d4145efc01aa  subsampled_vertex_indices.npy
EOF
)
echo "body models installed in $OUT"
