"""Loss specializations for object-local-up rotation symmetry."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.loss import ObjectPoseWSegLoss
from utils.scene_pose_perception_codec import local_up_equivalent_angle_bins


class ObjectPoseWSegLocalUpLoss(ObjectPoseWSegLoss):
    """Existing det/seg loss with correct object-local-up symmetry.

    Validity BCE, translation/scale CE, Dice, focal, and auxiliary-layer
    aggregation are inherited unchanged.  Only the Euler target equivalence is
    specialized; all four equivalent rotations supervise all three Euler axes.
    """

    def __init__(self, config):
        config = dict(config)
        # Disable the legacy world-Z-only target shift in the parent class.
        config["rotation_z_quarter_turns"] = False
        super().__init__(config)
        self.rotation_local_up_quarter_turns = bool(
            config.get("rotation_local_up_quarter_turns", True)
        )

    def angle_loss_per_object(self, angle_logits, angle_targets):
        batch_size, num_objects, num_axes, num_bins = angle_logits.shape
        if num_axes != 3:
            raise ValueError("Euler supervision requires exactly three axes")
        if not self.rotation_local_up_quarter_turns:
            return super().angle_loss_per_object(angle_logits, angle_targets)
        candidate_targets = local_up_equivalent_angle_bins(
            angle_targets, num_bins=num_bins
        )
        expanded_logits = angle_logits.unsqueeze(-3).expand(
            -1, -1, 4, -1, -1
        )
        candidate_ce = F.cross_entropy(
            expanded_logits.reshape(-1, num_bins),
            candidate_targets.reshape(-1),
            reduction="none",
        ).view(batch_size, num_objects, 4, 3).mean(dim=-1)
        return candidate_ce.min(dim=-1).values


class ScenePosePerceptionLossV2(ObjectPoseWSegLocalUpLoss):
    """Compatibility name for the signed-pose det/seg v2 training path."""
