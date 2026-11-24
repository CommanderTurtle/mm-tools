# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LicenseRef-SAM-License
# From SAM 3D Body (https://github.com/facebookresearch/sam-3d-body), modified for FAH; licence: third_party_licenses/SAM-License.txt
# Hand-only extraction of SAM 3D Body (sam_3d_body/models/meta_arch/sam3d_body.py).
# Keeps the shared image encoder + the hand decoder / heads and removes the body
# decoder, body heads and body-hand fusion. Module names are kept identical to the
# original so the released SAM 3D Body checkpoints load directly (see build.py).

from functools import partial
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.backbones import create_backbone
from .models.decoders import build_decoder, PromptEncoder
from .models.decoders.prompt_encoder import PositionEmbeddingRandom
from .models.heads import build_head
from .models.modules.camera_embed import CameraEncoder
from .models.modules.transformer import FFN, MLP
from .models.optim.fp16_utils import convert_module_to_f16, convert_to_fp16_safe
from .models.modules.sync_free import dev_index


PROMPT_KEYPOINTS = {"mhr70": {i: i for i in range(70)}}
VIT_BACKBONES = ["vit_hmr", "vit", "vit_b", "vit_l"]
VIT_512_384_BACKBONES = ["vit_hmr_512_384"]


class SAM3DHand(nn.Module):
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def __init__(self, cfg, backbone_dtype=None):
        super().__init__()
        self.cfg = cfg
        self._initialize_model(backbone_dtype)
        self._max_num_person = None

    @property
    def device(self):
        return self.image_mean.device

    def _initialize_model(self, backbone_dtype=None):
        cfg = self.cfg
        dim = cfg.MODEL.DECODER.DIM
        self.register_buffer(
            "image_mean", torch.tensor(cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Image encoder (shared by the body and hand decoders in SAM 3D Body)
        self.backbone = create_backbone(cfg.MODEL.BACKBONE.TYPE, cfg)

        # Hand pose head: MHR head in hand mode (wrist-centric, non-hand params zeroed)
        self.head_pose_hand = build_head(
            cfg, cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera_hand = build_head(
            cfg,
            cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        cond_dim = 3
        init_dim = self.head_pose_hand.npose + self.head_camera_hand.ncam + cond_dim
        self.init_to_token_mhr_hand = nn.Linear(init_dim, dim)
        self.prev_to_token_mhr_hand = nn.Linear(init_dim - cond_dim, dim)

        # Prompt encoder (shared by both decoders in SAM 3D Body)
        self.max_num_clicks = 0
        if cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]
            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None),
            )
            self.prompt_to_token = nn.Linear(self.backbone.embed_dims, dim)

        self.decoder_hand = build_decoder(
            cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        if backbone_dtype is None:
            if cfg.TRAIN.USE_FP16:
                backbone_dtype = (
                    torch.float16
                    if cfg.TRAIN.get("FP16_TYPE", "float16") == "float16"
                    else torch.bfloat16
                )
            else:
                backbone_dtype = torch.float32
        self.backbone_dtype = backbone_dtype
        if backbone_dtype != torch.float32:
            self._set_fp16(self.backbone, backbone_dtype)

        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim, self.backbone.patch_size
        )

        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), dim
        )

        if cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(2, dim)
            # decide if there is left or right hand inside the image
            self.hand_cls_embed = nn.Linear(dim, 2)
            self.bbox_embed = MLP(dim, dim, 4, 3)

        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=dim,
            output_dims=dim,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear_hand = nn.Linear(self.backbone.embed_dims, dim)

        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), dim
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=dim,
            output_dims=dim,
            num_fcs=2,
            add_identity=False,
        )

    def _set_fp16(self, module, fp16_type):
        if hasattr(module, "pos_embed"):
            module.apply(partial(convert_module_to_f16, dtype=fp16_type))
            module.pos_embed.data = module.pos_embed.data.to(fp16_type)
        elif hasattr(module.encoder, "rope_embed"):
            # DINOv3
            module.encoder.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.encoder.rope_embed = module.encoder.rope_embed.to(fp16_type)
        else:
            # DINOv2
            module.encoder.pos_embed.data = module.encoder.pos_embed.data.to(fp16_type)

    # ------------------------------------------------------------------ batch utils
    def _initialize_batch(self, batch: Dict) -> None:
        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"
        if self._max_num_person:
            shape = x.shape
            x = x.view(self._batch_size * self._max_num_person, *shape[2:])
        return x

    def data_preprocess(self, inputs: torch.Tensor, unit_range=None) -> torch.Tensor:
        # unit_range=True: caller guarantees [0, 1] inputs (skips the range check, a host sync)
        if unit_range is not None:
            if not unit_range and self.image_mean.max() <= 1.0:
                inputs = inputs / 255.0
        elif inputs.max() > 1 and self.image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and self.image_mean.max() > 1:
            inputs = inputs * 255.0
        batch_inputs = (inputs - self.image_mean) / self.image_std

        # ViT backbones assume a different aspect ratio as input size
        if self.cfg.MODEL.BACKBONE.TYPE in VIT_BACKBONES:
            batch_inputs = batch_inputs[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in VIT_512_384_BACKBONES:
            batch_inputs = batch_inputs[:, :, :, 64:-64]
        return batch_inputs

    def _full_to_crop(self, batch: Dict, pred_keypoints_2d: torch.Tensor):
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, -1:])], dim=-1
        )
        affine_trans = self._flatten_person(batch["affine_trans"]).to(
            pred_keypoints_2d_cropped
        )
        img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5
        return pred_keypoints_2d_cropped

    def _per_crop_cam_int(self, batch: Dict) -> torch.Tensor:
        return self._flatten_person(
            batch["cam_int"]
            .unsqueeze(1)
            .expand(-1, batch["img"].shape[1], -1, -1)
            .contiguous()
        )

    # ------------------------------------------------------------------ conditions
    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, 0:1]

            cam_int = self._per_crop_cam_int(batch)
            focal_length = cam_int[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = cam_int[:, :2, 2]  # (slices, not list indices: no H2D copy)
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, 0:1], cy - full_img_cxy[:, 1:2], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(-1)
            condition_info[:, 2] = condition_info[:, 2] / focal_length
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        dev = batch["img"].device
        meshgrid_xy = torch.stack(
            torch.meshgrid(torch.arange(W, device=dev), torch.arange(H, device=dev), indexing="xy"), dim=2
        )[None, None].float()  # 1 x 1 x H x W x 2 (x = column), broadcast over B x N
        meshgrid_xy = (
            meshgrid_xy / torch.diagonal(batch["affine_trans"][:, :, None, None, :2, :2], dim1=-2, dim2=-1)
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, :2, 2]
            / torch.diagonal(batch["affine_trans"][:, :, None, None, :2, :2], dim1=-2, dim2=-1)
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, :2, 2]
        )
        meshgrid_xy = (
            meshgrid_xy / torch.diagonal(batch["cam_int"][:, None, None, None, :2, :2], dim1=-2, dim2=-1)
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def _get_mask_prompt(self, batch, image_embeddings):
        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        # no mask prompts: skip the mask convs (their result is discarded). The caller can say so with
        # batch["mask_prompt"] = False, which avoids a host sync (needed for CUDA-graph capture).
        has_mask = batch["mask_prompt"] if "mask_prompt" in batch else bool((mask_score > 0).any())
        if not has_mask:
            no_mask = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            return no_mask.expand(image_embeddings.shape[0], -1, *image_embeddings.shape[2:]).to(image_embeddings)
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr", "vit"]:
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in VIT_512_384_BACKBONES:
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    # ------------------------------------------------------------------ decoder
    def camera_project_hand(self, pose_output: Dict, batch: Dict) -> Dict:
        head_camera = self.head_camera_hand
        pred_cam = pose_output["pred_cam"]
        proj_args = (
            pred_cam,
            self._flatten_person(batch["bbox_center"]),
            self._flatten_person(batch["bbox_scale"])[:, 0],
            self._flatten_person(batch["ori_img_size"]),
            self._per_crop_cam_int(batch),
        )
        use_intrin_center = self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False)

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            *proj_args,
            use_intrin_center=use_intrin_center,
        )
        pose_output.update(cam_out)
        return pose_output

    def forward_decoder_hand(
        self,
        image_embeddings: torch.Tensor,
        keypoints: torch.Tensor,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            keypoints: prompt input, shape (B, N, 3), 3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional previous estimate for refinement, shape (B, 1, C)
            condition_info: CLIFF condition concatenated with the input token, (B, c)
        """
        batch_size = image_embeddings.shape[0]

        init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(dim=1)
        init_camera = self.init_camera_hand.weight.expand(batch_size, -1).unsqueeze(
            dim=1
        )
        init_estimate = torch.cat([init_pose, init_camera], dim=-1)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr_hand(init_input).view(
            batch_size, 1, -1
        )
        num_pose_token = token_embeddings.shape[1]

        if prev_estimate is None:
            prev_estimate = init_estimate
        prev_embeddings = self.prev_to_token_mhr_hand(prev_estimate).view(
            batch_size, 1, -1
        )

        if self.cfg.MODEL.BACKBONE.TYPE in VIT_BACKBONES:
            image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in VIT_512_384_BACKBONES:
            image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
        else:
            image_augment = self.hand_pe_layer(image_embeddings.shape[-2:]).unsqueeze(0)

        image_embeddings = self.ray_cond_emb_hand(
            image_embeddings, batch["ray_cond_hand"]
        )

        prompt_embeddings, prompt_mask = self.prompt_encoder(keypoints=keypoints)
        prompt_embeddings = self.prompt_to_token(prompt_embeddings)

        # Decoder input tokens: [pose, prev_estimate, prompts, (hand boxes), kps2d, kps3d]
        token_embeddings = torch.cat(
            [token_embeddings, prev_embeddings, prompt_embeddings], dim=1
        )
        token_augment = torch.zeros_like(token_embeddings)
        token_augment[:, num_pose_token : num_pose_token + 1] = prev_embeddings
        token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
        token_mask = None

        def _append_tokens(token_embeddings, token_augment, weight):
            token_embeddings = torch.cat(
                [token_embeddings, weight[None, :, :].repeat(batch_size, 1, 1)], dim=1
            )
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )
            return token_embeddings, token_augment

        do_hand_detect_tokens = self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False)
        if do_hand_detect_tokens:
            hand_det_emb_start_idx = token_embeddings.shape[1]
            token_embeddings, token_augment = _append_tokens(
                token_embeddings, token_augment, self.hand_box_embedding.weight
            )

        assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
        kps_emb_start_idx = token_embeddings.shape[1]
        token_embeddings, token_augment = _append_tokens(
            token_embeddings, token_augment, self.keypoint_embedding_hand.weight
        )

        do_kps3d_tokens = self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False)
        if do_kps3d_tokens:
            kps3d_emb_start_idx = token_embeddings.shape[1]
            token_embeddings, token_augment = _append_tokens(
                token_embeddings, token_augment, self.keypoint3d_embedding_hand.weight
            )

        # Intermediate model predictions (used for iterative keypoint token updates)
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            pose_token = tokens[:, 0]
            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # Intermediate layers only need keypoints (fast hand mode skips skinning)
            pose_output = self.head_pose_hand(
                pose_token,
                prev_pose,
                return_vertices=layer_idx == self.decoder_hand.num_active_layers - 1,
            )
            pose_output["pred_cam"] = self.head_camera_hand(pose_token, prev_camera)
            pose_output = self.camera_project_hand(pose_output, batch)
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"]
            )
            return pose_output

        def keypoint_token_update_fn_comb(*args):
            args = self.keypoint_token_update_fn_hand(
                kps_emb_start_idx, image_embeddings, *args
            )
            if do_kps3d_tokens:
                args = self.keypoint3d_token_update_fn_hand(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder_hand(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if do_hand_detect_tokens:
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        return pose_token, pose_output

    def _forward_slim(self, slim, batch, image_embeddings, condition_info, x):
        emb = self.ray_cond_emb_hand(image_embeddings, batch["ray_cond_hand"])
        n = emb.shape[0]
        pose_token = slim(emb, condition_info)
        pose_output = self.head_pose_hand(pose_token, self.init_pose_hand.weight.expand(n, -1), return_vertices=True)
        pose_output["pred_cam"] = self.head_camera_hand(pose_token, self.init_camera_hand.weight.expand(n, -1))
        pose_output = self.camera_project_hand(pose_output, batch)
        pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(batch, pose_output["pred_keypoints_2d"])
        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):  # queries 1-2 play the hand-detect tokens
            pose_output["hand_box"] = self.bbox_embed(slim.last_tokens[:, 1:3]).sigmoid()
            pose_output["hand_logits"] = self.hand_cls_embed(slim.last_tokens[:, 1:3])
        if getattr(slim, "kp_uv", None) is not None:  # returned so DDP sees the keypoint head as used
            pose_output["kp_uv"] = slim.kp_uv
        verify_head = getattr(self, "verify_head", None)
        if verify_head is not None:
            pose_output["verify_logits"] = verify_head(slim.last_tokens[:, 0])
        mask_head = getattr(self, "mask_head", None)
        if mask_head is not None:
            pose_output["mask_logits"] = mask_head(slim.last_image_embedding, emb.shape[-2:], slim.last_tokens[:, 0], x)
        kp_head = getattr(self, "keypoint_head", None)
        if kp_head is not None:
            pose_output["pt_uv"], pose_output["pt_xyz"], _ = kp_head(slim.last_image_embedding, emb.shape[-2:],
                                                                     slim.last_tokens[:, 0], x)
        return pose_output

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == self.decoder_hand.num_active_layers - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions, -0.5 ~ 0.5
        pred_keypoints_2d_cropped = pose_output["pred_keypoints_2d_cropped"].clone()
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        kp_idx = dev_index(self.keypoint_embedding_idxs_hand, pred_keypoints_2d_cropped)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[:, kp_idx]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[:, kp_idx]

        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Mask of keypoints that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Bilinearly sample image features at the projected keypoints (-1 ~ 1)
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in VIT_BACKBONES + VIT_512_384_BACKBONES:
            # 256 x 256 coords -> 256 x 192 (HW) since image_embeddings is 16x12
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == self.decoder_hand.num_active_layers - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Pelvis normalize (pelvis keypoints are zeroed in hand mode, kept for parity)
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, self.pelvis_idx[0] : self.pelvis_idx[0] + 1, :]
                + pred_keypoints_3d[:, self.pelvis_idx[1] : self.pelvis_idx[1] + 1, :]
            )
            / 2
        )
        pred_keypoints_3d = pred_keypoints_3d[:, dev_index(self.keypoint3d_embedding_idxs_hand, pred_keypoints_3d)]

        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    # ------------------------------------------------------------------ forward
    def encode(self, batch: Dict) -> torch.Tensor:
        """Backbone features (B*N, C, h, w) for a batch of crops."""
        self._initialize_batch(batch)
        x = self.data_preprocess(self._flatten_person(batch["img"]), batch.get("img_unit_range"))
        image_embeddings = self.backbone(x.type(self.backbone_dtype))
        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        return image_embeddings.type(x.dtype)

    def forward(self, batch: Dict, keypoints_prompt: Optional[torch.Tensor] = None,
                image_embeddings: Optional[torch.Tensor] = None):
        """Run the hand branch on a batch of right-hand crops (left hands pre-flipped).
        `image_embeddings` (B*N, C, h, w) replaces the backbone (e.g. a distilled student encoder);
        its grid must match the crop resolution / patch size (rays are built from the crop)."""
        self._initialize_batch(batch)
        batch_size, num_crops = batch["img"].shape[:2]

        x = self.data_preprocess(self._flatten_person(batch["img"]), batch.get("img_unit_range"))

        ray_cond = self._flatten_person(self.get_ray_condition(batch))
        if self.cfg.MODEL.BACKBONE.TYPE in VIT_BACKBONES:
            ray_cond = ray_cond[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in VIT_512_384_BACKBONES:
            ray_cond = ray_cond[:, :, :, 64:-64]
        batch["ray_cond_hand"] = ray_cond

        if image_embeddings is None:
            image_embeddings = self.backbone(x.type(self.backbone_dtype))
            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)

        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                image_embeddings = image_embeddings + self._get_mask_prompt(
                    batch, image_embeddings
                )
            else:
                raise NotImplementedError

        condition_info = self._get_decoder_condition(batch)

        slim = getattr(self, "slim_decoder", None)
        if slim is not None:  # CPU-friendly small decoder (fah/models/decoders/slim.py), same output heads
            return self._forward_slim(slim, batch, image_embeddings, condition_info, x)
        if getattr(self, "kp_only", False):  # decoder-free: direct keypoints from the ray-conditioned encoder latent
            emb = self.ray_cond_emb_hand(image_embeddings, batch["ray_cond_hand"])
            tok = emb.flatten(2).transpose(1, 2)
            # The point, depth, verification and mesh heads share this same token.
            pose = self.keypoint_head.pool(torch.cat([tok.float().mean(1), condition_info.float()], -1))
            uv, xyz, _ = self.keypoint_head(tok, emb.shape[-2:], pose, x, cond=condition_info, return_conf=True)
            res = dict(pt_uv=uv, pt_xyz=xyz, kp_dense=self.keypoint_head.dense[None].expand(uv.shape[0], -1),
                       pt_conf=self.keypoint_head.last_conf)  # no second full heatmap softmax
            if getattr(self.keypoint_head, "local_split", False):
                res["pt_uv_local"] = self.keypoint_head.last_uv_local
            if getattr(self.keypoint_head, "vis", None) is not None:
                res["pt_vis"] = self.keypoint_head.last_vis  # visibility logits per point
            if getattr(self.keypoint_head, "part_emb", None) is not None:
                if self.keypoint_head.last_part_logits is not None:  # (B, 17, 4h, 4w): bg, palm, 15 finger segments
                    res["part_logits"] = self.keypoint_head.last_part_logits
            if getattr(self.keypoint_head, "contact_on", False) and getattr(self.keypoint_head, "contact_geo", True) != "external":
                # contact logits (dense points, hand); contact_geo "external": the caller runs contact_from itself
                res["contact_logits"], res["contact_hand"] = self.keypoint_head.last_contact, self.keypoint_head.last_contact_hand
                if getattr(self.keypoint_head, "contact_dists", None) is not None:  # several tolerances (mm): (B, n, K), (B, K)
                    res["contact_logits_k"], res["contact_hand_k"] = self.keypoint_head.last_contact_k, self.keypoint_head.last_contact_hand_k
            if getattr(self.keypoint_head, "depth", None) is not None:
                res["pt_depth"] = self.keypoint_head.depth(pose)[:, 0]
                res["pt_token"] = pose  # per-hand latent (video refiner input)
            if getattr(self, "verify_head", None) is not None:  # [presence, side-consistency] logits from the pooled token
                res["verify_logits"] = self.verify_head(pose)
            if getattr(self, "mano_head", None) is not None:  # parametric mesh from the model's own 2D points
                res["mano_verts"], res["mano_joints"], res["mano_params"] = self.mano_head(
                    pose, uv.detach(), self.keypoint_head.last_pooled)
            return res

        # Initial estimate with a dummy prompt
        if keypoints_prompt is None:
            keypoints_prompt = torch.zeros((batch_size * num_crops, 1, 3), device=x.device, dtype=x.dtype)
            keypoints_prompt[:, :, -1] = -2

        tokens_output, pose_outputs = self.forward_decoder_hand(
            image_embeddings,
            keypoints=keypoints_prompt,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_outputs[-1]

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            # x1, y1, w, h (0 ~ 1 in crop) and left/right presence logits
            pose_output["hand_box"] = self.bbox_embed(tokens_output).sigmoid()
            pose_output["hand_logits"] = self.hand_cls_embed(tokens_output)
        verify_head = getattr(self, "verify_head", None)
        if verify_head is not None:  # [presence, side-consistency] logits
            pose_output["verify_logits"] = verify_head(self.decoder_hand.last_tokens[:, 0])
        mask_head = getattr(self, "mask_head", None)
        if mask_head is not None:  # this crop's hand mask, (B*N, 4h, 4w) logits at 1/4 crop resolution
            dec = self.decoder_hand
            pose_output["mask_logits"] = mask_head(dec.last_image_embedding, image_embeddings.shape[-2:],
                                                   dec.last_tokens[:, 0], x)
        kp_head = getattr(self, "keypoint_head", None)
        if kp_head is not None:  # direct joints + dense surface points: (B*N, P, 2) crop units, (B*N, P, 3) m
            dec = self.decoder_hand
            pose_output["pt_uv"], pose_output["pt_xyz"], _ = kp_head(dec.last_image_embedding, image_embeddings.shape[-2:],
                                                                     dec.last_tokens[:, 0], x)

        return pose_output
