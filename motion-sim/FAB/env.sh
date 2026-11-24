# SPDX-License-Identifier: AGPL-3.0-only
model_store="$STUDIO/../../models"
export FAB_HOME="${FAB_HOME:-$model_store/fab}"
export FAB_G1="${FAB_G1:-$FAB_HOME/g1}"
shared_sam="$model_store/flowhmr/ckpts/sam-3d-body-dinov3"
shared_soma="$STUDIO/../nvidia-sim/SOMA-X/assets"
if [[ ! -f "$shared_sam/assets/mhr_model.pt" && -f "$FAB_HOME/sam3d/assets/mhr_model.pt" ]]; then shared_sam="$FAB_HOME/sam3d"; fi
if [[ ! -f "$shared_soma/SOMA_neutral.npz" && -f "$FAB_HOME/soma/SOMA_neutral.npz" ]]; then shared_soma="$FAB_HOME/soma"; fi
export FAB_SAM3D="${FAB_SAM3D:-$shared_sam}" SOMA_ASSETS="${SOMA_ASSETS:-$shared_soma}"
export FAH_SAM3D="${FAH_SAM3D:-$FAB_SAM3D}" FAH_WEIGHTS="${FAH_WEIGHTS:-$model_store/fab/hands}" FAB_WEIGHTS="${FAB_WEIGHTS:-$model_store/fab/weights}"
export FAB_INPUT_SMPL_DIR="${FAB_SMPL_DIR:-}" FAB_INPUT_MANO_DIR="${MANO_DIR:-}"
export FAB_SMPL_DIR="$FAB_HOME/body_models/smpl" MANO_DIR="$FAB_HOME/mano-cache/current"
