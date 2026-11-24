#!/bin/bash
# Unitree G1 robot description for fab-retarget and the G1 head, from public upstreams at pinned commits, checked by SHA256,
# installed under $FAB_G1 (default $FAB_HOME/g1, $FAB_HOME default ~/.cache/fab). Nothing of it is shipped with FAB.
#   Unitree's g1_description (unitree_ros, robots/g1_description: g1_29dof.urdf, g1_29dof.xml and its meshes,
#   g1_29dof_rev_1_0.xml), built into FAB's files by scripts/build_g1.py:
#     g1_29dof.urdf      the 29-DoF URDF with the Model 15 effort / velocity limits of 14 joints (fab/retarget/g1_model15.py)
#                        and FAB's frames; kinematics, ranges and inertials as upstream
#     g1_29dof.xml       the 29-DoF MuJoCo model with FAB's frames, and its meshes (unitree_meshes/, OBJ)
#     g1_gate_robot.xml  the stance pin's collision model (Unitree's rev 1.0 model with FAB's collision primitives)
#     scene_g1_gate.xml  FAB's scene of the stance pin's physics checks (that model on a floor plane)
#   g1_mjlab.xml         mjlab's G1 MuJoCo model with collision capsules (the solver's self-collision term) and its meshes
#                        (mujocolab/mjlab, src/mjlab/asset_zoo/robots/unitree_g1/xmls: g1.xml and assets/)
# Licences: Unitree's G1 description BSD-3-Clause (Copyright Unitree Robotics); mjlab Apache-2.0; see THIRD_PARTY_NOTICES.md.
#   bash scripts/download_g1.sh            # needs git and FAB's Python environment (numpy)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${FAB_G1:-${FAB_HOME:-$HOME/.cache/fab}/g1}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
SUM=$(command -v sha256sum > /dev/null && echo "sha256sum" || echo "shasum -a 256")

# ---- pinned upstreams ------------------------------------------------------------------------------------------------------
UNITREE_REPO=https://github.com/unitreerobotics/unitree_ros.git
UNITREE_COMMIT=157ff950351e5f535e03c5d85a16b7d594b3dbcc
G1D=robots/g1_description
MJLAB_REPO=https://github.com/mujocolab/mjlab.git                                 # the collision-capsule MJCF (Apache-2.0)
MJLAB_COMMIT=5e3b2e6cf1f00b37515a6658d1972e47e6059da0
MJLAB_G1=src/mjlab/asset_zoo/robots/unitree_g1/xmls
# SHA256 of the upstream files
SHA_URDF_SRC=e1dc89366bf96aa33ba4ef211884f6d81320ddd2ac6f8b1ad82986e64008be31   # g1_description/g1_29dof.urdf
SHA_MJCF_SRC=dfcfd8b41d0980695d0e75d76fdfd9965f71431a1c67afc31c8dbe730eb66c72   # g1_description/g1_29dof.xml
SHA_REV1_SRC=165fa7a5275745e45fa25e1bc78010c1055d81bb81c013b3c7bc8b9347a3b9c3   # g1_description/g1_29dof_rev_1_0.xml
SHA_MESHES=3f6cb97fb490200971b713105953e3801fa095c8c63d648290fc047f813eca26     # the model's 36 meshes (sha256 of their sorted sha256 list)
SHA_CAPSULE=d6026290f4264198a0e8064f94bca7f20f7d3c72e9f6b48c019531360a9a8395    # mjlab's g1.xml (installed as g1_mjlab.xml)
SHA_CAPSULE_MESHES=10a553c1feae501b27b2e1278a8b7894bfd917cad5ddc236ff2f2c48e057a361  # its 35 meshes (sha256 of their sorted sha256 list)
# SHA256 of the built files (scripts/build_g1.py; Python's XML writer, checked with CPython 3.12)
SHA_URDF=cd80ca7e64a594aaa8009be125e2a67551149fd87ae6f5e235ed42f7b946c495
SHA_MJCF=037f9e611f3540bf407c732e352dfafe2119672380f4778a8763ab8de29b957d
SHA_SCENE=04bd71dae3f8a824b4aa2de3f51bdfcd434ff494c1ba2bde3fe6385d7a2edb23
SHA_GATE=9cea3f18c00b0748c35a59e9271c8d95ef296ca6a0528a1522696ea8e30ea204
# -------------------------------------------------------------------------------------------------------------------------------

fetch() {  # repo commit path... -> $TMP/<repo name> (sparse, blobs fetched on checkout only)
  local repo=$1 commit=$2; shift 2
  local d; d=$(mktemp -d "$TMP/$(basename "$repo" .git).XXXX")
  git init -q "$d"
  git -C "$d" remote add origin "$repo"
  git -C "$d" sparse-checkout set --no-cone "$@"
  git -C "$d" fetch -q --depth 1 --filter=blob:none origin "$commit"
  git -C "$d" checkout -q FETCH_HEAD
  echo "$d"
}

# 1. Unitree's g1_description: the URDF and the MJCF, then the meshes the MJCF names
D=$(fetch "$UNITREE_REPO" "$UNITREE_COMMIT" "$G1D/g1_29dof.urdf" "$G1D/g1_29dof.xml" "$G1D/g1_29dof_rev_1_0.xml")
S=$D/$G1D
printf '%s  %s\n%s  %s\n%s  %s\n' "$SHA_URDF_SRC" "$S/g1_29dof.urdf" "$SHA_MJCF_SRC" "$S/g1_29dof.xml" "$SHA_REV1_SRC" "$S/g1_29dof_rev_1_0.xml" \
  | ${SUM} -c - > /dev/null \
  || { echo "upstream g1_description checksum mismatch"; exit 1; }
MESHES=$(python3 -c "import re, sys; print(' '.join(sorted(set(re.findall(r'file=\"([^\"]+)\"', open(sys.argv[1]).read())))))" "$S/g1_29dof.xml")
git -C "$D" sparse-checkout set --no-cone "$G1D/g1_29dof.urdf" "$G1D/g1_29dof.xml" "$G1D/g1_29dof_rev_1_0.xml" \
  $(for f in $MESHES; do echo "$G1D/meshes/$f"; done)
got=$(cd "$S/meshes" && ${SUM} $MESHES | ${SUM} | cut -d' ' -f1)
[ "$got" = "$SHA_MESHES" ] || { echo "upstream G1 meshes checksum mismatch"; exit 1; }

# 2. FAB's files from it
python3 "$HERE/build_g1.py" --src "$S" --out "$OUT" > "$TMP/build.log" || { cat "$TMP/build.log"; exit 1; }

# 3. the capsule model (mjlab): its g1.xml as g1_mjlab.xml, its meshes in assets/ (the file's meshdir)
D=$(fetch "$MJLAB_REPO" "$MJLAB_COMMIT" "$MJLAB_G1/g1.xml" "$MJLAB_G1/assets")
echo "$SHA_CAPSULE  $D/$MJLAB_G1/g1.xml" | ${SUM} -c - > /dev/null || { echo "upstream mjlab G1 checksum mismatch"; exit 1; }
CMESHES=$(python3 -c "import re, sys; print(' '.join(sorted(set(re.findall(r'file=\"([^\"]+)\"', open(sys.argv[1]).read())))))" "$D/$MJLAB_G1/g1.xml")
got=$(cd "$D/$MJLAB_G1/assets" && ${SUM} $CMESHES | ${SUM} | cut -d' ' -f1)
[ "$got" = "$SHA_CAPSULE_MESHES" ] || { echo "upstream mjlab G1 meshes checksum mismatch"; exit 1; }
mkdir -p "$OUT/assets"
cp "$D/$MJLAB_G1/g1.xml" "$OUT/g1_mjlab.xml"
for f in $CMESHES; do cp "$D/$MJLAB_G1/assets/$f" "$OUT/assets/$f"; done

# 4. checksums
(cd "$OUT" && $SUM -c - <<EOF
$SHA_CAPSULE  g1_mjlab.xml
EOF
)
(cd "$OUT" && $SUM -c --quiet - <<EOF
$SHA_URDF  g1_29dof.urdf
$SHA_MJCF  g1_29dof.xml
$SHA_SCENE  scene_g1_gate.xml
$SHA_GATE  g1_gate_robot.xml
EOF
) 2> /dev/null || echo "WARNING: the built G1 files differ in their bytes from the reference build (another Python XML writer?); inputs verified"
echo "G1 description installed in $OUT"
