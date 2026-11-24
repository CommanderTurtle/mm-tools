#!/bin/bash
# Dyn-HaMR with one camera fix, in a separate tree <tree>-camfix (the checkout stays untouched; _DATA, third-party and
# shim are linked). Use with DYNHAMR_ROOT=<tree>-camfix bash scripts/dynhamr/run.sh.
# Fix: dyn-hamr/data/dataset.py CameraData.load_data ("move first camera to origin", inherited from SLAHMR) does
#     t0 = -cam_t[sidx:sidx+1] + torch.randn(3) * 0.1 ; self.cam_t = cam_t[sidx:eidx] - t0
# i.e. it adds cam_t[sidx] plus a random 0.1-sigma vector to every world-to-camera translation: a constant camera-frame
# offset, not a rigid change of the world frame for a rotating camera. Reported upstream, unresolved: Dyn-HaMR issues #12
# and #25, SLAHMR issue #18. The fix keeps the given world-to-camera poses (self.cam_t = cam_t[sidx:eidx]).
#   bash scripts/dynhamr/camfix.sh $THIRDPARTY/Dyn-HaMR      (or $THIRDPARTY/Dyn-HaMR-98e1622)
set -ex
D=$(cd "${1:?usage: camfix.sh <Dyn-HaMR tree>}" && pwd)
F=$D-camfix
mkdir -p "$F"
rsync -a --delete "$D/dyn-hamr/" "$F/dyn-hamr/"
for x in _DATA third-party shim; do ln -sfn "$D/$x" "$F/$x"; done
P=$F/dyn-hamr/data/dataset.py
grep -q "t0 = -cam_t\[sidx:sidx+1\] + torch.randn(3) \* 0.1" "$P"
sed -i 's/^\(\s*\)t0 = -cam_t\[sidx:sidx+1\] + torch.randn(3) \* 0.1$/\1t0 = torch.zeros(1, 3)  # camfix (scripts\/dynhamr\/camfix.sh)/' "$P"
grep -n "camfix" "$P"
