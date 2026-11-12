# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the G1 inference model from the three released checkpoint states."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .inference import motion_inference
from .networks import pose_backbone_network, root_backbone_network
from .rig import G1PlacedPosePreprocess, G1PlacedPosePostprocess
from ..representation import GlobalRootGlobalJoints, Stats
from ..vqvae import VQVAE

INPUT_NAMES = (
    "context_qpos",
    "target_qpos",
    "has_global_root_values",
    "has_local_root_values",
    "has_local_poses",
    "allowed_pred_num_tokens",
)
OUTPUT_NAMES = ("pred_qpos", "pred_num_tokens", "pred_foot_contacts")
CONFIG = {
    "num_inference_step": 1,
    "smooth_root_traj": False,
    "allow_pred_out_of_reach_num_tokens": False,
    "pose_token_sampling_use_argmax": True,
    "skip_ending_target_cond": False,
}


class G1JointSpace(nn.Module):
    """Run G1 inference on encoded inputs, and return motion features and token counts."""

    def __init__(self, inferencer):
        super().__init__()
        self.inferencer = inferencer

    def forward(
        self,
        global_root_values,
        has_global_root_values,
        local_root_values,
        has_local_root_values,
        local_poses,
        has_local_poses,
        allowed_pred_num_tokens,
    ):
        features, tokens = self.inferencer.predict(
            global_root_values,
            has_global_root_values.bool(),
            local_root_values,
            has_local_root_values.bool(),
            local_poses,
            has_local_poses.bool(),
            num_tokens=None,
            allowed_pred_num_tokens=allowed_pred_num_tokens,
            config=CONFIG,
            info={},
        )
        return features, tokens.to(torch.int32)


class BatchedG1(nn.Module):
    """Batched G1 graph that takes and returns joint coordinates.

    It encodes the context and target poses, runs inference, and decodes the result with
    its foot contacts.
    """

    def __init__(self, inferencer):
        super().__init__()
        with np.load(
            Path(__file__).resolve().parents[2] / "assets/g1_model_rig.npz",
            allow_pickle=False,
        ) as data:
            arrays = dict(data)
        indices = inferencer.global_motion_rep.indices
        self.pre = G1PlacedPosePreprocess(arrays, indices)
        self.core = G1JointSpace(inferencer)
        self.post = G1PlacedPosePostprocess(arrays, indices)

    def forward(
        self,
        context_qpos,
        target_qpos,
        has_global_root_values,
        has_local_root_values,
        has_local_poses,
        allowed_pred_num_tokens,
    ):
        joint_inputs = self.pre(
            context_qpos,
            target_qpos,
            has_global_root_values,
            has_local_root_values,
            has_local_poses,
            allowed_pred_num_tokens,
        )
        return self.post(*self.core(*joint_inputs))


class Predictor(nn.Module):
    """One predictor network with its motion representation and settings."""

    def __init__(self, backbone, motion_rep, args, pose_decoder=None):
        super().__init__()
        self.backbone_net = backbone
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep
        self.args = args
        self.supporting_nets = {"pose_net": pose_decoder}
        self.vqvae_model_loaded = True


def build_g1(sources):
    """Load the root predictor, the pose predictor, and the VQ-VAE motion decoder."""
    mean = next(value for path, value in sources.items() if path.endswith("mean.npy"))
    std = next(value for path, value in sources.items() if path.endswith("std.npy"))
    stats = Stats()
    stats.register_from_tensors(
        torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    )
    motion_rep = GlobalRootGlobalJoints(
        fps=30,
        skeleton=SimpleNamespace(nbjoints=34),
        name="g1skel34_dual_root_global_joints",
        stats=stats,
    )
    common = dict(
        min_tokens=6,
        max_tokens=16,
        down_t=2,
        local_pose_feature="joint_positions_and_rotations_and_hip_height",
    )
    pose_args = dict(
        common,
        pose_root_mode="pose",
        pose_vqvae_motion_rep="local",
        cond_root_feature="root_without_hip_height",
        cond_root_feature_is_from_motion_rep="global",
    )
    root_args = dict(
        common,
        pose_root_mode="root",
        local_root_feature="root",
        global_root_feature="root",
        use_hard_num_token_emb_for_root_prediction=True,
    )
    pose_network_args = dict(
        pose_args,
        pose_token_mlp_num_layers=2,
        root_feat_width=256,
        pose_feat_width=640,
        token_length_feat_width=128,
        text_emb_dim=4096,
        text_embeddings=None,
        n_embd=1024,
        n_head=16,
        n_layers=16,
        pose_vqvae=dict(
            nb_code=100000000, code_dim=256, num_heads=8, has_codebook=True
        ),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    root_network_args = dict(
        root_args,
        pose_feat_dim=256,
        local_root_feat_dim=64,
        global_root_feat_dim=64,
        text_emb_dim=4096,
        text_embeddings=None,
        input_feat_mlp_num_layers=2,
        n_embd=512,
        n_head=16,
        n_layers_shared=3,
        n_layers_root_token=3,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        norm="None",
        pose_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    pose_backbone = pose_backbone_network(motion_rep, pose_network_args)
    root_backbone = root_backbone_network(root_network_args, motion_rep)
    vqvae = VQVAE(
        pose_root_mode="pose",
        motion_rep=motion_rep.dual_rep.local_motion_rep,
        feature_mode=[
            "joint_positions_and_rotations_and_hip_height",
            "pose",
            "joint_positions_and_rotations_and_hip_height",
            "root_without_hip_height_without_heading",
        ],
        nb_code=100000000,
        code_dim=256,
        output_emb_width=256,
        down_t=2,
        stride_t=2,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        num_heads=8,
        norm="None",
        cond_fusion_last_layer=False,
    )
    for kind, network, prefix in (
        ("pose", pose_backbone, "backbone_net."),
        ("root", root_backbone, "backbone_net."),
        ("vqvae", vqvae, "pose_net."),
    ):
        state = sources[
            f"motionbricks_{kind}/version_1/checkpoints/model-step=2000000.safetensors"
        ]
        weights = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        network.load_state_dict(weights, strict=True)
    models = dict(
        pose=Predictor(pose_backbone, motion_rep, pose_args, vqvae),
        root=Predictor(root_backbone, motion_rep, root_args),
    )
    return BatchedG1(motion_inference(models, pose_args, device="cpu")).eval()
