import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import (
    NUM_BINS,
    EULER_Z_MIN,
    EULER_Z_MAX,
)
from utils.discrete import continue_transform_torch


def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0, weight=None, reduction='mean'):
    """
    Focal loss for binary classification with logits.

    Args:
        logits: Raw logits (before sigmoid)
        targets: Binary targets (0 or 1)
        alpha: Weighting factor for POSITIVE class. Use alpha > 0.5 when positives are rare.
               Default 0.75 means positives get 3x weight compared to negatives.
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Optional per-element weights
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Focal loss value
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # p_t = p if target==1 else 1-p
    p_t = p * targets + (1 - p) * (1 - targets)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting: alpha for positives, (1-alpha) for negatives
    # When positives are rare, use alpha > 0.5 to upweight them
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_weight * focal_weight * ce_loss

    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def masked_scene_mean(loss_per_object, object_valid_masks, eps=1e-6):
    """Average valid object losses per scene, then average scenes.

    With B=1 this is identical to a plain valid-object mean. With B>1 it
    preserves the scene-balanced objective used by B=1 gradient accumulation
    instead of weighting each local batch by object count.
    """

    if loss_per_object.shape != object_valid_masks.shape:
        raise ValueError(
            "loss_per_object and object_valid_masks must have the same shape: "
            f"{tuple(loss_per_object.shape)} vs {tuple(object_valid_masks.shape)}"
        )
    weights = object_valid_masks.to(
        device=loss_per_object.device, dtype=loss_per_object.dtype
    )
    valid_counts = weights.sum(dim=1)
    per_scene = (loss_per_object * weights).sum(dim=1) / (
        valid_counts + eps
    )
    valid_scenes = (valid_counts > 0).to(dtype=loss_per_object.dtype)
    return (per_scene * valid_scenes).sum() / valid_scenes.sum().clamp_min(1.0)


# @torch.compile
def euler_angles_to_matrix(angles):
    """
    Convert Euler angles to rotation matrices.

    Uses extrinsic XYZ convention (static axes), matching trimesh's 'sxyz'.
    R = Rz @ Ry @ Rx

    Args:
        angles: (..., 3) tensor - [x, y, z] radians

    Returns:
        R: (..., 3, 3) tensor - rotation matrices
    """
    x = angles[..., 0]
    y = angles[..., 1]
    z = angles[..., 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # Pure functional form — torch.compile fuses all trig + element-wise
    # ops into a single kernel (the old torch.empty + 9 scatter writes
    # produced 9 separate tiny kernels that couldn't fuse).
    return torch.stack([
        torch.stack([cy * cz,  cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx], dim=-1),
        torch.stack([cy * sz,  sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx], dim=-1),
        torch.stack([   -sy,                  cy * sx,                  cy * cx], dim=-1),
    ], dim=-2)


def rotation_matrix_geodesic_distance(rotation_pred, rotation_gt):
    """Return broadcasted SO(3) geodesic distance in radians."""
    rotation_diff = torch.matmul(rotation_gt.transpose(-1, -2), rotation_pred)
    trace = (
        rotation_diff[..., 0, 0]
        + rotation_diff[..., 1, 1]
        + rotation_diff[..., 2, 2]
    )
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.acos(cos_angle)


def z_quarter_turn_equivalent_rotations(rotation):
    """Return Rz(k*pi/2) @ rotation for k=0,1,2,3."""
    angles = torch.arange(4, device=rotation.device, dtype=rotation.dtype)
    angles = angles * (torch.pi / 2.0)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    quarter_turns = torch.stack(
        [
            torch.stack([cos_angles, -sin_angles, zeros], dim=-1),
            torch.stack([sin_angles, cos_angles, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return torch.matmul(quarter_turns, rotation.unsqueeze(-3))


def z_quarter_turn_angle_bin_targets(angle_targets, num_bins=NUM_BINS):
    """Build four Euler-bin targets differing by quarter turns around Z."""
    if angle_targets.shape[-1] != 3:
        raise ValueError(f"Expected [..., 3] Euler targets, got {angle_targets.shape}")

    angle_targets = angle_targets.long()
    candidates = angle_targets.unsqueeze(-2).expand(*angle_targets.shape[:-1], 4, 3).clone()

    period = float(EULER_Z_MAX - EULER_Z_MIN)
    z_angle = (
        angle_targets[..., 2].float() / float(num_bins - 1) * period
        + float(EULER_Z_MIN)
    )
    offsets = torch.arange(4, device=angle_targets.device, dtype=z_angle.dtype)
    offsets = offsets * (torch.pi / 2.0)
    shifted_z = torch.remainder(
        z_angle.unsqueeze(-1) - float(EULER_Z_MIN) + offsets,
        period,
    ) + float(EULER_Z_MIN)
    shifted_z_bins = torch.floor(
        (shifted_z - float(EULER_Z_MIN)) / period * float(num_bins - 1) + 1e-6
    ).long().clamp_(0, num_bins - 1)

    # Preserve the exact original discrete target for the zero-turn candidate.
    shifted_z_bins[..., 0] = angle_targets[..., 2]
    candidates[..., 2] = shifted_z_bins
    return candidates


def angle_bin_cross_entropy_per_object(
    angle_logits,
    angle_targets,
    z_quarter_turns=False,
):
    """Compute per-object Euler-bin CE, optionally minimized over Z quarter turns."""
    batch_size, num_objects, num_axes, num_bins = angle_logits.shape
    if num_axes != 3:
        raise ValueError(f"Expected three Euler axes, got {num_axes}")

    if not z_quarter_turns:
        angle_ce = F.cross_entropy(
            angle_logits.reshape(-1, num_bins),
            angle_targets.reshape(-1),
            reduction='none',
        ).view(batch_size, num_objects, 3)
        return angle_ce.mean(dim=2)

    candidate_targets = z_quarter_turn_angle_bin_targets(angle_targets, num_bins=num_bins)
    xy_ce = F.cross_entropy(
        angle_logits[..., :2, :].reshape(-1, num_bins),
        angle_targets[..., :2].reshape(-1),
        reduction='none',
    ).view(batch_size, num_objects, 2)
    z_logits = angle_logits[..., 2, :].unsqueeze(-2).expand(-1, -1, 4, -1)
    z_ce = F.cross_entropy(
        z_logits.reshape(-1, num_bins),
        candidate_targets[..., 2].reshape(-1),
        reduction='none',
    ).view(batch_size, num_objects, 4)
    candidate_ce = (xy_ce.sum(dim=-1, keepdim=True) + z_ce) / 3.0
    return candidate_ce.min(dim=-1).values


def geodesic_rotation_loss(angle_pred, angle_gt, z_quarter_turns=False):
    """
    Compute geodesic (angular) distance between two rotations.

    Args:
        angle_pred: (..., 3) tensor - predicted Euler angles
        angle_gt: (..., 3) tensor - ground truth Euler angles

    Returns:
        angle: (...,) tensor - rotation angle in radians between the two rotations
    """
    # Convert to rotation matrices
    R_pred = euler_angles_to_matrix(angle_pred)  # (..., 3, 3)
    R_gt = euler_angles_to_matrix(angle_gt)  # (..., 3, 3)

    if not z_quarter_turns:
        return rotation_matrix_geodesic_distance(R_pred, R_gt)

    equivalent_gt = z_quarter_turn_equivalent_rotations(R_gt)
    candidate_distances = rotation_matrix_geodesic_distance(
        R_pred.unsqueeze(-3),
        equivalent_gt,
    )
    return candidate_distances.min(dim=-1).values

@torch.no_grad()
def latent_loss(
    latent_preds,
    latent_gt,
    occupancy_preds,
    occupancy_gt,
):
    # print("latent_loss:")
    # print(f"latent_preds.shape: {latent_preds.shape}, latent_gt.shape: {latent_gt.shape}")
    # print(f"occupancy_preds.shape: {occupancy_preds.shape}, occupancy_gt.shape: {occupancy_gt.shape}")
    latent_huber = F.huber_loss(latent_preds, latent_gt, reduction='none', delta=1.0)
    latent_loss_per_object = latent_huber.mean(dim=(-1))  # [num_objects]
    loss_latents = latent_loss_per_object.mean()

    occupancy_preds = torch.sign(occupancy_preds)
    hinge_loss = F.relu(1.0 - occupancy_gt * occupancy_preds)
    occupancy_loss_per_object = hinge_loss.mean(dim=(-1))  # [num_objects]
    loss_occupancies = occupancy_loss_per_object.mean()

    return {
        "loss_latents": loss_latents,
        "loss_occupancies": loss_occupancies,
    }

@torch.no_grad()
def ss_latent_loss(
    ss_latent_preds,
    ss_latent_gt,
):
    ss_latent_huber = F.huber_loss(ss_latent_preds, ss_latent_gt, reduction='none', delta=1.0)
    ss_latent_loss_per_object = ss_latent_huber.mean(dim=(-1))  # [num_objects]
    loss_ss_latents = ss_latent_loss_per_object.mean()

    return {
        "loss_ss_latents": loss_ss_latents,
    }

@torch.no_grad()
def feat_loss(
    feat_preds,
    feat_gt,
):
    feat_huber = F.huber_loss(feat_preds, feat_gt, reduction='none', delta=1.0)
    loss_feats = feat_huber.mean(dim=(-1)).mean()  # [1]

    return {
        "loss_feats": loss_feats,
    }

@torch.no_grad()
def pose_l1_loss(
    pos_bin_logits,
    angle_bin_logits,
    scale_bin_logits,
    tgt_pos_bins,
    tgt_angle_bins,
    tgt_scale_bins,
    max_num_objects,
    object_valid_masks,
    rotation_z_quarter_turns=False,
):
    """
    Args:
        pos_bin_logits: (B, MAX_SCENE_OBJECTS, 3, num_bins) float
        angle_bin_logits: (B, MAX_SCENE_OBJECTS, 3, num_bins) float
        scale_bin_logits: (B, MAX_SCENE_OBJECTS, 1, num_bins) float
        tgt_pos_bins: (B, max_num_objects, 3) long - pre-computed position bin targets
        tgt_angle_bins: (B, max_num_objects, 3) long - pre-computed angle bin targets
        tgt_scale_bins: (B, max_num_objects, 1) long - pre-computed scale bin targets
        max_num_objects: int - maximum number of objects in the batch
        object_valid_masks: (B, max_num_objects), bool, True for valid objects

    Returns:
        loss_dict containing:
            l1_translations: L1 loss for position (in continuous space)
            l1_angles: geodesic loss for angles (angle in radians)
            l1_scales: L1 loss for scale (in continuous space)
    """

    B = tgt_pos_bins.shape[0]

    pos_bin_pred = torch.argmax(pos_bin_logits, dim=-1)[:, :max_num_objects]
    angle_bin_pred = torch.argmax(angle_bin_logits, dim=-1)[:, :max_num_objects]
    scale_bin_pred = torch.argmax(scale_bin_logits, dim=-1)[:, :max_num_objects]

    pos_bin_gt = tgt_pos_bins.reshape(B, max_num_objects, 3)
    angle_bin_gt = tgt_angle_bins.reshape(B, max_num_objects, 3)
    scale_bin_gt = tgt_scale_bins.reshape(B, max_num_objects, 1)

    scale_pred, angle_pred, trans_pred = continue_transform_torch(scale_bin_pred, angle_bin_pred, pos_bin_pred)
    scale_gt, angle_gt, trans_gt = continue_transform_torch(scale_bin_gt, angle_bin_gt, pos_bin_gt)

    # L1 loss for translations: [B, num_objects, 3] -> [B, num_objects]
    trans_l1 = torch.abs(trans_pred - trans_gt).mean(dim=2)  # average over xyz
    loss_translations = (trans_l1 * object_valid_masks.float()).sum() / (object_valid_masks.float().sum() + 1e-6)

    # Geodesic loss for angles: [B, num_objects, 3] -> [B, num_objects]
    # Measures actual rotation angle (in radians) between predicted and GT rotations
    rot_geodesic_raw = geodesic_rotation_loss(angle_pred, angle_gt)
    rot_geodesic = geodesic_rotation_loss(
        angle_pred,
        angle_gt,
        z_quarter_turns=rotation_z_quarter_turns,
    )  # [B, max_num_objects]
    loss_rotations = (rot_geodesic * object_valid_masks.float()).sum() / (object_valid_masks.float().sum() + 1e-6)

    # L1 loss for scales: [B, num_objects, 1] -> [B, num_objects]
    scale_l1 = torch.abs(scale_pred - scale_gt).squeeze(-1)  # remove last dim
    loss_scales = (scale_l1 * object_valid_masks.float()).sum() / (object_valid_masks.float().sum() + 1e-6)

    result = {
        "l1_translations": loss_translations,
        "l1_rotations": loss_rotations,
        "l1_scales": loss_scales,
    }
    if rotation_z_quarter_turns:
        result["l1_rotations_raw"] = (
            (rot_geodesic_raw * object_valid_masks.float()).sum()
            / (object_valid_masks.float().sum() + 1e-6)
        )
    return result

# @torch.compile
def dice_cost_matrix(pred_masks, gt_masks, eps=1.0):
    """
    输入:
      pred_masks: (B, K, N) 或 (K, N), 经过 Sigmoid 的概率值
      gt_masks:   (B, M, N) 或 (M, N), 0/1 标签
    输出:
      cost_matrix: (B, K, M) 或 (K, M)
    """
    # -------------------------------------------------------
    # 步骤 2: 计算分子 (Intersection) -> Batch 矩阵乘法
    # (B, K, N) @ (B, N, M) -> (B, K, M)
    # torch.matmul 自动支持 batch 维度的广播
    # -------------------------------------------------------
    # 使用 transpose(-2, -1) 来交换最后两个维度，这样既兼容 (M, N) 也兼容 (B, M, N)
    intersection = torch.matmul(pred_masks, gt_masks.transpose(-2, -1))

    # -------------------------------------------------------
    # 步骤 3: 计算分母 (Union) -> 广播机制
    # -------------------------------------------------------

    # 计算每个预测的总面积:
    # sum(dim=-1) 对最后一个维度 N 求和 -> (B, K)
    # unsqueeze(-1) 在最后增加一维 -> (B, K, 1)
    pred_area = pred_masks.sum(dim=-1).unsqueeze(-1)

    # 计算每个真值的总面积:
    # sum(dim=-1) -> (B, M)
    # unsqueeze(-2) 在倒数第二维增加 -> (B, 1, M)
    gt_area = gt_masks.sum(dim=-1).unsqueeze(-2)

    # 广播相加: (B, K, 1) + (B, 1, M) -> (B, K, M)
    union = pred_area + gt_area

    # -------------------------------------------------------
    # 步骤 4: 计算 Dice Score 并转换为 Cost
    # -------------------------------------------------------
    dice_score = (2.0 * intersection + eps) / (union + eps)

    # Cost 越小越好
    cost_matrix = 1.0 - dice_score

    return cost_matrix # Shape: (B, K, M)


def dice_loss_matched(
    pred_probs,
    gt_masks,
    object_valid_masks,
    eps=1.0,
    square=False,
    context_valid_masks=None,
):
    """
    Input:
      pred_probs: (B, M, N) or (M, N)
      gt_masks:   (B, M, N) or (M, N)
      object_valid_masks: (B, M), bool, True for valid objects
    Output:
      loss: Scalar (average over batch and objects)
    """
    assert pred_probs.shape == gt_masks.shape
    if context_valid_masks is not None:
        if context_valid_masks.shape != (pred_probs.shape[0], pred_probs.shape[2]):
            raise ValueError(
                "context_valid_masks must have shape [B, N], got "
                f"{tuple(context_valid_masks.shape)} for predictions "
                f"{tuple(pred_probs.shape)}"
            )
        context_valid_masks = context_valid_masks.to(
            device=pred_probs.device, dtype=pred_probs.dtype
        )
        pred_probs = pred_probs * context_valid_masks[:, None, :]
        gt_masks = gt_masks * context_valid_masks[:, None, :]


    # Intersection: (B, M, N) -> sum over N -> (B, M)
    intersection = (pred_probs * gt_masks).sum(dim=-1)

    # Union: (B, M, N) -> sum over N -> (B, M)
    if square:
        pred_area = (pred_probs * pred_probs).sum(dim=-1)
        gt_area   = (gt_masks * gt_masks).sum(dim=-1)
    else:
        pred_area = pred_probs.sum(dim=-1)
        gt_area   = gt_masks.sum(dim=-1)

    union = pred_area + gt_area # (B, M)

    # -------------------------------------------------------
    # calculate dice loss
    # -------------------------------------------------------
    # calculate dice score: (B, M)
    dice_score = (2.0 * intersection + eps) / (union + eps)

    # if gt area is small, it is invalid/ghost object, not calculate loss
    valid_mask = torch.logical_and(gt_area > 0.5, object_valid_masks).float()

    # only calculate loss for objects with real voxels
    loss = 1.0 - dice_score
    loss = masked_scene_mean(loss, valid_mask.bool())
    return loss


def sigmoid_focal_cost_matrix(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: Logits: A float tensor of arbitrary shape.
                The predictions for each example.
                Shape: (B, K, N)
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                Shape: (B, M, N)
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    B, K, N = inputs.shape

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    ) # (B, K, N)
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    ) # (B, K, N)
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    cost_matrix = torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum(
        "bnc,bmc->bnm", focal_neg, (1 - targets)
    ) # (B, K, M)

    return cost_matrix / N


def sigmoid_focal_loss_matched(
    inputs,
    targets,
    object_valid_masks,
    alpha: float = 0.25,
    gamma: float = 2,
    context_valid_masks=None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: Logits: A float tensor of arbitrary shape.
                The predictions for each example.
                Shape: (B, M, N)
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                Shape: (B, M, N)
        object_valid_masks: (B, M), bool, True for valid objects
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    B, M, N = inputs.shape

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (B, M, N)
    p_t = prob * targets + (1 - prob) * (1 - targets) # (B, M, N)
    loss = ce_loss * ((1 - p_t) ** gamma) # (B, M, N)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if context_valid_masks is None:
        loss = loss.mean(dim=-1) # (B, M)
    else:
        if context_valid_masks.shape != (B, N):
            raise ValueError(
                "context_valid_masks must have shape [B, N], got "
                f"{tuple(context_valid_masks.shape)} for inputs "
                f"{tuple(inputs.shape)}"
            )
        context_valid_masks = context_valid_masks.to(
            device=inputs.device, dtype=loss.dtype
        )
        valid_counts = context_valid_masks.sum(dim=-1).clamp_min(1.0)
        loss = (
            loss * context_valid_masks[:, None, :]
        ).sum(dim=-1) / valid_counts[:, None] # (B, M)
    loss = masked_scene_mean(loss, object_valid_masks.bool())

    return loss

def spherical_discriminative_loss(features, instance_ids, delta_v=0.1, delta_d=0.5, alpha=1.0, beta=1.0):
    """
    Fully vectorized Center Discriminative Loss for L2-normalized 3D Point Clouds.
    Assumes all instance IDs are valid (>= 0), there are no background points,
    and features lie on a unit hypersphere.

    Args:
        features: Tensor of shape (B, N, C). Must be L2-normalized!
        instance_ids: Tensor of shape (B, N) containing integer IDs (0 to K-1).
        delta_v: Pull margin (variance) for points toward their center.
        delta_d: Push margin (distance) between different instance centers.
        alpha: Weight for the variance (pull) loss.
        beta: Weight for the distance (push) loss.
    """
    B, N, C = features.shape

    # 1. Setup and Dynamic Sizing
    # Find the maximum instance ID to size our one-hot tensor dynamically
    K_max = int(instance_ids.max().item()) + 1
    if K_max <= 0:
        return torch.tensor(0.0, requires_grad=True, device=features.device)

    # Create one-hot mask directly since all IDs are valid: (B, N, K_max)
    one_hot = F.one_hot(instance_ids, num_classes=K_max).float()

    # 2. Calculate Centers
    # Number of points per instance: (B, K_max)
    inst_sizes = one_hot.sum(dim=1)

    # Mask of which instances actually exist in each batch: (B, K_max)
    inst_exists_mask = (inst_sizes > 0)

    # Sum of features per instance via Batch Matrix Mult: (B, K_max, C)
    sum_features = torch.bmm(one_hot.transpose(1, 2), features)

    # Divide by counts to get raw interior centers, clamped to avoid division by zero
    safe_sizes = inst_sizes.clamp(min=1e-6).unsqueeze(-1)
    raw_centers = sum_features / safe_sizes # (B, K_max, C)

    # Project the centers back onto the surface of the unit hypersphere
    centers = F.normalize(raw_centers, p=2, dim=-1)

    # 3. Variance Loss (Pull)
    # Project centers back to points: (B, N, K_max) @ (B, K_max, C) = (B, N, C)
    assigned_centers = torch.bmm(one_hot, centers)

    # Euclidean distance from each point to its assigned spherical center: (B, N)
    pull_dist = torch.norm(features - assigned_centers, dim=-1)

    # Pull penalty per point: (B, N)
    pull_penalty = F.relu(pull_dist - delta_v)**2

    # Sum penalties per instance, then average by instance size
    pull_loss_per_inst = torch.bmm(one_hot.transpose(1, 2), pull_penalty.unsqueeze(-1)).squeeze(-1) # (B, K_max)
    pull_loss_per_inst = pull_loss_per_inst / safe_sizes.squeeze(-1)

    # Mean over all existing instances in the batch
    total_valid_instances = inst_exists_mask.sum().clamp(min=1)
    l_var = (pull_loss_per_inst * inst_exists_mask.float()).sum() / total_valid_instances

    # 4. Distance Loss (Push)
    # Pairwise Euclidean distances between all centers in a batch: (B, K_max, K_max)
    center_dists = torch.cdist(centers, centers, p=2)

    # Push penalty: (B, K_max, K_max)
    push_penalty = F.relu(2 * delta_d - center_dists)**2

    # Masking out invalid comparisons (diagonal and non-existent instances)
    eye = torch.eye(K_max, device=features.device).unsqueeze(0).bool() # (1, K_max, K_max)
    valid_pair_mask = inst_exists_mask.unsqueeze(2) & inst_exists_mask.unsqueeze(1) # (B, K_max, K_max)
    actual_mask = valid_pair_mask & (~eye)

    # Mean over all valid pairs
    total_valid_pairs = actual_mask.sum().clamp(min=1)
    l_dist = (push_penalty * actual_mask.float()).sum() / total_valid_pairs

    return (alpha * l_var) + (beta * l_dist)

class ObjectPoseWSegLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Loss lambdas (weights for each loss component)
        self.lambda_translation = config.get('lambda_translation', 1.0) # ~2.0
        self.lambda_angle = config.get('lambda_angle', 1.0) # ~2.0
        self.lambda_scale = config.get('lambda_scale', 1.0) # ~2.0
        self.lambda_valid = config.get('lambda_valid', 2.0) # ~0.1
        self.lambda_valid_eos = config.get('lambda_valid_eos', 0.5)
        self.lambda_seg_dice = config.get('lambda_seg_dice', 1.0) # ~0.5
        self.lambda_seg_focal = config.get('lambda_seg_focal', 20.0) # ~0.001
        self.rotation_z_quarter_turns = config.get('rotation_z_quarter_turns', False)
        self.learned_background_segmentation = bool(
            config.get('learned_background_segmentation', False)
        )


        print("loss config:")
        print(f"lambda_translation: {self.lambda_translation}")
        print(f"lambda_angle: {self.lambda_angle}")
        print(f"lambda_scale: {self.lambda_scale}")
        print(f"lambda_valid: {self.lambda_valid}")
        print(f"lambda_valid_eos: {self.lambda_valid_eos}")
        print(f"lambda_seg_dice: {self.lambda_seg_dice}")
        print(f"lambda_seg_focal: {self.lambda_seg_focal}")
        print(f"rotation_z_quarter_turns: {self.rotation_z_quarter_turns}")
        print(
            "learned_background_segmentation: "
            f"{self.learned_background_segmentation}"
        )
        print("--------------------------------")

        # Number of bins for discrete tokens
        self.num_bins = NUM_BINS

    def angle_loss_per_object(self, angle_logits, angle_targets):
        """Hook used by isolated pose-domain branches.

        The default calls the historical implementation exactly, so existing
        det/seg configurations retain identical loss arithmetic.
        """

        return angle_bin_cross_entropy_per_object(
            angle_logits,
            angle_targets,
            z_quarter_turns=self.rotation_z_quarter_turns,
        )

    def forward(
        self,
        pos_bin_logits,
        angle_bin_logits,
        scale_bin_logits,
        valid_logits,
        pred_masks_logits,
        gt_pos_bins,
        gt_angle_bins,
        gt_scale_bins,
        gt_masks,
        max_num_objects,
        object_valid_masks,
        background_pred_masks_logits=None,
        background_gt_mask=None,
        include_background_segmentation=True,
        return_per_token_losses=False,
    ):
        """
        Compute loss between resorted predictions and ground truth targets.

        After Hungarian matching and resorting in the decoder:
        - Predictions at indices [0, num_objects-1] correspond to GT objects [0, num_objects-1]
        - Predictions at indices [num_objects, MAX_SCENE_OBJECTS-1] are unmatched (invalid)

        Args:
            pos_bin_logits: (B, MAX_SCENE_OBJECTS, 3, num_bins) float - resorted position predictions
            angle_bin_logits: (B, MAX_SCENE_OBJECTS, 3, num_bins) float - resorted angle predictions
            scale_bin_logits: (B, MAX_SCENE_OBJECTS, 1, num_bins) float - resorted scale predictions
            valid_logits: (B, MAX_SCENE_OBJECTS) float - validity predictions for all slots
            pred_masks_logits: (B, MAX_SCENE_OBJECTS, context_length) float - predicted masks logits

            gt_pos_bins: (B, max_num_objects, 3) long - position bin targets
            gt_angle_bins: (B, max_num_objects, 3) long - angle bin targets
            gt_scale_bins: (B, max_num_objects, 1) long - scale bin targets
            gt_masks: (B, max_num_objects, context_length) float - ground truth masks
            max_num_objects: int - maximum number of objects in the batch
            object_valid_masks: (B, max_num_objects), bool, True for valid objects


        Returns:
            loss_dict containing:
                loss: total weighted loss
                loss_translations: cross entropy for position tokens
                loss_angles: cross entropy for angle tokens
                loss_scales: cross entropy for scale tokens
                loss_valid: BCE loss for valid/invalid prediction
                loss_seg: dice loss for segmentation
        """
        # ==========================================
        # 🛡️ 精度升级核心：强制进入 FP32 上下文
        # 即使外部使用了 BF16/FP16 的 autocast，这里也会被强行拉回到 FP32
        # ==========================================
        device_type = pos_bin_logits.device.type
        with torch.autocast(device_type=device_type, dtype=torch.float32):

            # --- 1. 显式升精度 (Upcast) ---
            # 浮点数预测强制转为 FP32，防止后续计算产生舍入误差或大数吃小数
            pos_bin_logits = pos_bin_logits.float()
            angle_bin_logits = angle_bin_logits.float()
            scale_bin_logits = scale_bin_logits.float()
            valid_logits = valid_logits.float()
            pred_masks_logits = pred_masks_logits.float()
            pred_masks = pred_masks_logits.sigmoid()
            gt_masks = gt_masks.float()

            # --- 2. 确保标签类型安全 ---
            # CrossEntropy 需要的 Target 必须是 LongTensor
            gt_pos_bins = gt_pos_bins.long()
            gt_angle_bins = gt_angle_bins.long()
            gt_scale_bins = gt_scale_bins.long()
            # ==========================================

            B = pos_bin_logits.shape[0]
            num_query_slots = valid_logits.shape[1]
            if max_num_objects > num_query_slots:
                raise ValueError(
                    "ground-truth object count exceeds decoder query slots: "
                    f"max_num_objects={max_num_objects}, "
                    f"num_query_slots={num_query_slots}"
                )

            # Slice predictions to match GT size (first max_num_objects predictions are matched)
            pos_logits = pos_bin_logits[:, :max_num_objects]  # [B, max_num_objects, 3, num_bins]
            angle_logits = angle_bin_logits[:, :max_num_objects]  # [B, max_num_objects, 3, num_bins]
            scale_logits = scale_bin_logits[:, :max_num_objects]  # [B, max_num_objects, 1, num_bins]
            pred_masks = pred_masks[:, :max_num_objects]  # [B, max_num_objects, context_length]
            pred_masks_logits = pred_masks_logits[:, :max_num_objects]  # [B, max_num_objects, context_length]

            # ========== Discrete Losses ==========
            # Targets are already in correct shape
            pos_targets = gt_pos_bins  # [B, max_num_objects, 3]
            angle_targets = gt_angle_bins  # [B, max_num_objects, 3]
            scale_targets = gt_scale_bins  # [B, max_num_objects, 1]

            # Compute cross entropy for translations (position)
            pos_ce = F.cross_entropy(
                pos_logits.reshape(-1, self.num_bins),
                pos_targets.reshape(-1),
                reduction='none',
                # label_smoothing=0.1
            ).view(B, max_num_objects, 3)  # [B, max_num_objects, 3]
            pos_loss_per_object = pos_ce.mean(dim=2)  # [B, max_num_objects]
            loss_translations = masked_scene_mean(
                pos_loss_per_object, object_valid_masks.bool()
            )

            # Compute cross entropy for angles
            angle_loss_per_object = self.angle_loss_per_object(
                angle_logits, angle_targets
            )
            loss_angles = masked_scene_mean(
                angle_loss_per_object, object_valid_masks.bool()
            )

            # Compute cross entropy for scales
            scale_ce = F.cross_entropy(
                scale_logits.reshape(-1, self.num_bins),
                scale_targets.reshape(-1),
                reduction='none',
                # label_smoothing=0.1
            ).view(B, max_num_objects, 1)  # [B, max_num_objects, 1]
            scale_loss_per_object = scale_ce.squeeze(-1)  # [B, max_num_objects]
            loss_scales = masked_scene_mean(
                scale_loss_per_object, object_valid_masks.bool()
            )

            # ========== Valid Loss (BCE) ==========
            # valid_logits: [B, MAX_SCENE_OBJECTS]
            # Target: matched positions [0, max_num_objects-1] with object_valid_masks, rest are invalid (0)

            # Build target: valid for matched objects, invalid for unmatched slots
            # 显式指定 dtype=torch.float32 (最佳实践)
            object_valid_float = object_valid_masks.float() # [B, max_num_objects]
            if max_num_objects < num_query_slots:
                valid_targets = F.pad(
                    object_valid_float,
                    (0, num_query_slots - max_num_objects),
                    value=0.0,
                )
            else:
                valid_targets = object_valid_float

            # Build weight: lambda_valid_eos for invalid slots, 1.0 for valid slots
            # Use where instead of indexing assignment (more GPU-friendly)
            valid_weights = torch.where(
                valid_targets > 0.5, 1.0, float(self.lambda_valid_eos)
            )

            # Valid loss for all slots with per-element weighting
            loss_valid = F.binary_cross_entropy_with_logits(
                valid_logits, # [B, MAX_SCENE_OBJECTS]
                valid_targets, # [B, MAX_SCENE_OBJECTS]
                weight=valid_weights,
                reduction='mean'
            )

            # ========== Segmentation Loss (Dice Loss) ==========
            # 因为之前已经做过了 pred_masks.float() 和 gt_masks.float()
            # 这里进如 Dice loss 的就一定是高精度的 Tensor，累加分母时不再惧怕 BF16 截断
            if (
                self.learned_background_segmentation
                and include_background_segmentation
            ):
                if (
                    background_pred_masks_logits is None
                    or background_gt_mask is None
                ):
                    raise ValueError(
                        "learned background segmentation requires predicted "
                        "and target background masks"
                    )
                background_pred_masks_logits = (
                    background_pred_masks_logits.float()
                )
                background_gt_mask = background_gt_mask.float()
                if background_pred_masks_logits.shape != (
                    background_gt_mask.shape
                ):
                    raise ValueError(
                        "background predicted/GT mask shape mismatch: "
                        f"{background_pred_masks_logits.shape} vs "
                        f"{background_gt_mask.shape}"
                    )
                segmentation_logits = torch.cat(
                    [background_pred_masks_logits, pred_masks_logits], dim=1
                )
                segmentation_targets = torch.cat(
                    [background_gt_mask, gt_masks], dim=1
                )
                segmentation_valid_masks = torch.cat(
                    [
                        torch.ones(
                            (B, 1),
                            dtype=torch.bool,
                            device=object_valid_masks.device,
                        ),
                        object_valid_masks,
                    ],
                    dim=1,
                )
                segmentation_probs = segmentation_logits.sigmoid()
            else:
                segmentation_logits = pred_masks_logits
                segmentation_targets = gt_masks
                segmentation_valid_masks = object_valid_masks
                segmentation_probs = pred_masks
            context_valid_masks = segmentation_targets.bool().any(dim=1)

            loss_seg_dice = dice_loss_matched(
                segmentation_probs,
                segmentation_targets,
                segmentation_valid_masks,
                eps=1.0,
                square=False,
                context_valid_masks=context_valid_masks,
            )
            loss_seg_focal = sigmoid_focal_loss_matched(
                segmentation_logits,
                segmentation_targets,
                segmentation_valid_masks,
                alpha=0.25,
                gamma=2,
                context_valid_masks=context_valid_masks,
            )

            with torch.no_grad():
                loss_seg_dice_original_eps = dice_loss_matched(
                    segmentation_probs,
                    segmentation_targets,
                    segmentation_valid_masks,
                    eps=1e-4,
                    square=True,
                    context_valid_masks=context_valid_masks,
                )

            # Compute valid prediction statistics for monitoring
            with torch.no_grad():
                valid_probs = valid_logits.sigmoid()
                pos_mask = valid_targets == 1
                neg_mask = valid_targets == 0
                valid_pred_pos_mean = valid_probs[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0)
                valid_pred_neg_mean = valid_probs[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0)
                valid_logits_mean = valid_logits.mean()
                valid_logits_std = valid_logits.std()
                num_positives = pos_mask.sum().float()
                num_negatives = neg_mask.sum().float()

                pos_pred_mask = valid_probs > 0.5
                neg_pred_mask = valid_probs < 0.5
                num_positives_pred = pos_pred_mask.sum().float()
                num_negatives_pred = neg_pred_mask.sum().float()
                num_pos_diff = (pos_pred_mask != pos_mask).sum().float()
                num_neg_diff = (neg_pred_mask != neg_mask).sum().float()

                # accuracy and recall
                true_positives = (pos_mask & pos_pred_mask).sum().float()
                false_positives = (~pos_mask & pos_pred_mask).sum().float()
                true_negatives = (~pos_mask & ~pos_pred_mask).sum().float()
                false_negatives = (pos_mask & ~pos_pred_mask).sum().float()
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                recall = true_positives / (true_positives + false_negatives + 1e-6)
                precision = true_positives / (true_positives + false_positives + 1e-6)
                f1_score = 2 * precision * recall / (precision + recall + 1e-6)


            # ========== Total Loss ==========
            loss = (
                self.lambda_translation * loss_translations +
                self.lambda_angle * loss_angles +
                self.lambda_scale * loss_scales +
                self.lambda_valid * loss_valid +
                self.lambda_seg_dice * loss_seg_dice +
                self.lambda_seg_focal * loss_seg_focal
            )

            loss_dict = {
                "loss": loss,
                "loss_translations": loss_translations,
                "loss_angles": loss_angles,
                "loss_scales": loss_scales,
                "loss_valid": loss_valid,
                "loss_seg_dice": loss_seg_dice,
                "loss_seg_focal": loss_seg_focal,
                "loss_seg": loss_seg_dice_original_eps,
            }
            if (
                self.learned_background_segmentation
                and include_background_segmentation
            ):
                with torch.no_grad():
                    background_valid = torch.ones(
                        (B, 1), dtype=torch.bool, device=gt_masks.device
                    )
                    loss_dict["loss_seg_bg_dice"] = dice_loss_matched(
                        segmentation_probs[:, :1],
                        segmentation_targets[:, :1],
                        background_valid,
                        eps=1.0,
                        square=False,
                        context_valid_masks=context_valid_masks,
                    )
                    loss_dict["loss_seg_fg_dice"] = dice_loss_matched(
                        pred_masks,
                        gt_masks,
                        object_valid_masks,
                        eps=1.0,
                        square=False,
                        context_valid_masks=context_valid_masks,
                    )
            if return_per_token_losses:
                loss_dict.update({
                    "valid_pred_pos_mean": valid_pred_pos_mean,
                    "valid_pred_neg_mean": valid_pred_neg_mean,
                    "valid_logits_mean": valid_logits_mean,
                    "valid_logits_std": valid_logits_std,
                    "valid_num_positives": num_positives,
                    "valid_num_negatives": num_negatives,
                    "valid_num_positives_pred": num_positives_pred,
                    "valid_num_negatives_pred": num_negatives_pred,
                    "valid_num_pos_diff": num_pos_diff,
                    "valid_num_neg_diff": num_neg_diff,
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "f1_score": f1_score,
                })
            return loss_dict
