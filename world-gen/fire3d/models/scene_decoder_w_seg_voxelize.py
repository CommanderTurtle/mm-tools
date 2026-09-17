from enum import IntEnum
from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from models.detr_transformer import DetrWSegWReconTransformer, PosTransformerEncoder
from modules.utils import convert_module_to, str_to_dtype
from scipy.optimize import linear_sum_assignment
from utils.constants import MAX_SCENE_OBJECTS, NUM_BINS, POS_MAX
from utils.discrete import continue_transform_torch, get_bin_centers
from utils.loss import (
    euler_angles_to_matrix,
    dice_cost_matrix,
    sigmoid_focal_cost_matrix,
    rotation_matrix_geodesic_distance,
    z_quarter_turn_equivalent_rotations,
)
from utils.scene_pose_perception_codec import (
    local_up_quarter_turn_equivalent_rotations,
)
import numpy as np

# Token type ranges for convenience
VALID_TOKEN = 0
POS_TOKEN_START = 1
POS_TOKEN_END = 3      # inclusive
ANGLE_TOKEN_START = 4
ANGLE_TOKEN_END = 6      # inclusive
SCALE_TOKEN = 7
SEG_TOKEN = 8
TOKENS_PER_OBJECT = 9

class SceneDecoder(nn.Module):
    """
    Detr based Transformer Decoder for 3D Scene Synthesis.

    Full sequence structure: [valid, pos*3, angle*3, scale, seg]xN (bg first, then objects)

    Structure per Object (TOKENS_PER_OBJECT tokens):
    - 1 Valid token (Boolean)
    - 3 Position tokens (Discrete bins)
    - 3 Angle tokens (Discrete bins - Euler angles)
    - 1 Scale token (Discrete bin)
    - 1 Segmentation token (Continuous)

    Token type indices:
    - 0: Valid (Boolean)
    - 1-3: Position (discrete)
    - 4-6: Angle (discrete)
    - 7: Scale (discrete)
    - 8: Segmentation (continuous)
    """

    def __init__(
        self,
        d_model,
        num_attn_heads,
        dim_feedforward,
        num_bins=NUM_BINS,
        max_num_tokens=None,
        max_scene_objects=MAX_SCENE_OBJECTS,
        num_encoder_layers=6,
        d_seg_model=1024,
        nhead_seg=None,
        dim_seg_feedforward=None,
        num_seg_encoder_layers=6,
        seg_pos_input_first=False,
        seg_feat_with_spatial=False,
        seg_feat_use_pose_transformer=False,
        num_decoder_layers=6,
        seg_feat_dim=None,
        dropout=0.1,
        aux=True,
        dtype=None,
        cost_weights: Optional[dict] = None,
        logit_scale: Optional[float] = None,
        rotation_z_quarter_turns: bool = False,
        rotation_local_up_quarter_turns: bool = False,
        learned_background_segmentation: bool = False,
    ):
        """
        Args:
            d_model: int. Dimension of model.
            num_attn_heads: int. Number of attention heads.
            dim_feedforward: int. Dimension of feedforward network.
            num_bins: int. Number of discretized bins for position/angle/scale.
            max_num_tokens: int. Maximum number of tokens in a sequence.
            num_encoder_layers: int. Number of encoder layers.
            num_decoder_layers: int. Number of decoder layers.
            seg_feat_dim: int. Dimension of segmentation feature vectors.
            dropout: float. Dropout rate.
            aux: bool. Whether to use auxiliary loss.
            cost_weights: dict. Cost weights for each loss component.
        """
        super().__init__()
        self.d_model = d_model
        self.d_seg_model = d_seg_model if d_seg_model is not None else d_model
        self.dim_seg_feedforward = dim_seg_feedforward if dim_seg_feedforward is not None else dim_feedforward
        self.nhead_seg = nhead_seg if nhead_seg is not None else num_attn_heads
        self.num_bins = num_bins
        self.max_scene_objects = int(max_scene_objects)
        if self.max_scene_objects < 1:
            raise ValueError("max_scene_objects must be positive")
        if max_num_tokens is None:
            max_num_tokens = self.max_scene_objects * TOKENS_PER_OBJECT
        self.max_num_tokens = int(max_num_tokens)
        min_num_tokens = self.max_scene_objects * TOKENS_PER_OBJECT
        if self.max_num_tokens < min_num_tokens:
            raise ValueError(
                "max_num_tokens must cover every object token: "
                f"got {self.max_num_tokens}, need at least {min_num_tokens}"
            )
        self.aux = aux
        self.seg_feat_dim = seg_feat_dim
        self.seg_feat_with_spatial = seg_feat_with_spatial
        self.seg_feat_use_pose_transformer = seg_feat_use_pose_transformer
        self.rotation_z_quarter_turns = bool(rotation_z_quarter_turns)
        self.rotation_local_up_quarter_turns = bool(
            rotation_local_up_quarter_turns
        )
        if self.rotation_z_quarter_turns and self.rotation_local_up_quarter_turns:
            raise ValueError(
                "rotation_z_quarter_turns and "
                "rotation_local_up_quarter_turns are mutually exclusive"
            )
        self.learned_background_segmentation = bool(
            learned_background_segmentation
        )
        self.cost_weights = cost_weights if cost_weights is not None else {
            "cls": 1.0,
            "pos": 25.0,
            "scl": 25.0,
            "rot": 10.0,
            "seg_dice": 1.0,
            "seg_focal": 20.0,
        }
        # print("cost weights:")
        # print(f"cls: {self.cost_weights['cls']}")
        # print(f"pos: {self.cost_weights['pos']}")
        # print(f"scl: {self.cost_weights['scl']}")
        # print(f"rot: {self.cost_weights['rot']}")
        # print(f"seg_dice: {self.cost_weights['seg_dice']}")
        # print(f"seg_focal: {self.cost_weights['seg_focal']}")
        # print("--------------------------------")

        # Pre-compute and cache query type/object indices (constant tensors used every forward)
        tgt_types = torch.arange(TOKENS_PER_OBJECT, dtype=torch.int32).reshape(1, -1).expand(self.max_scene_objects, -1).reshape(-1)
        tgt_object_ids = torch.arange(self.max_scene_objects, dtype=torch.int32).reshape(-1, 1).expand(-1, TOKENS_PER_OBJECT).reshape(-1)
        self.register_buffer('_cached_tgt_types', tgt_types, persistent=False)
        self.register_buffer('_cached_tgt_object_ids', tgt_object_ids, persistent=False)

        # Embeddings
        # # Positional embedding for sequence position
        self.position_embedding = nn.Embedding(self.max_num_tokens, d_model)

        # Token type embedding (crucial for model to know attribute context)
        self.type_embedding = nn.Embedding(TOKENS_PER_OBJECT, d_model)
        self.object_embedding = nn.Embedding(self.max_scene_objects, d_model)

        # Transformer encoder and decoder
        self.transformer = DetrWSegWReconTransformer(
            d_model=d_model,
            d_seg_model=d_seg_model,
            nhead=num_attn_heads,
            nhead_seg=nhead_seg,
            num_encoder_layers=num_encoder_layers,
            num_seg_encoder_layers=num_seg_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dim_seg_feedforward=dim_seg_feedforward,
            dropout=dropout,
            return_intermediate_dec=aux,
            seg_pos_input_first=seg_pos_input_first
        )

        # Output Heads
        # Discrete head: Predicts bin indices for position/angle/scale
        # we want to create separate heads for each token of position, angle, and scale
        self.pos_bin_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_bins),
        )
        self.angle_bin_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_bins),
        )
        self.scale_bin_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_bins),
        )

        # # Continuous head: Predicts the segmentation feature vector (enlarged capacity)
        if not self.seg_feat_use_pose_transformer:
            if self.seg_feat_with_spatial:
                d_seg_model_in = 8 * d_model
                # print("enable seg_feat_with_spatial")
            else:
                d_seg_model_in = d_model
                # print("disable seg_feat_with_spatial")

            self.seg_feat_head = nn.Sequential(
                nn.Linear(d_seg_model_in, 2 * d_model),
                # nn.LayerNorm(2 * d_model),       # <--- 新增：稳定中间层分布
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                # nn.LayerNorm(d_model),           # <--- 新增
                nn.GELU(),
                nn.Linear(d_model, self.seg_feat_dim),
                # 注意：最后一层 Linear 后不要加 LayerNorm，
                # 归一化应该在 forward 里用 F.normalize (L2) 做，或者依靠最后一层的 Linear 自由学习
            )

            if self.seg_feat_with_spatial:
                self.pos_proj = nn.Sequential(
                    nn.LayerNorm(num_bins),
                    nn.Linear(num_bins, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                self.angle_proj = nn.Sequential(
                    nn.LayerNorm(num_bins),
                    nn.Linear(num_bins, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                self.scale_proj = nn.Sequential(
                    nn.LayerNorm(num_bins),
                    nn.Linear(num_bins, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
        else:
            print("enable seg_feat_use_pose_transformer")
            self.pos_proj = nn.Sequential(
                nn.LayerNorm(num_bins),
                nn.Linear(num_bins, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.angle_proj = nn.Sequential(
                nn.LayerNorm(num_bins),
                nn.Linear(num_bins, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.scale_proj = nn.Sequential(
                nn.LayerNorm(num_bins),
                nn.Linear(num_bins, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.pose_feat_head = nn.Sequential(
                nn.Linear(7*d_model, d_model), # 3*d_model for pos, angle; 1*d_model for scale
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            self.seg_feat_head_first = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )
            self.seg_feat_head = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.seg_feat_dim),
            )

            self.pos_transformer_encoder = PosTransformerEncoder(
                d_model=d_model,
                nhead=num_attn_heads,
                num_encoder_layers=num_seg_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

        self.context_feat_head = nn.Sequential(
            nn.Linear(d_seg_model, 2 * d_seg_model),
            # nn.LayerNorm(2 * d_model),       # <--- 新增：稳定中间层分布
            nn.GELU(),
            nn.Linear(2 * d_seg_model, d_seg_model),
            # nn.LayerNorm(d_model),           # <--- 新增
            nn.GELU(),
            nn.Linear(d_seg_model, seg_feat_dim),
        )

        # In foreground-only mode the background has no detection query, pose,
        # validity logit, or Hungarian assignment. Its mask classifier is this
        # single scene-independent learned embedding, applied to the
        # scene-dependent encoded point features.
        if self.learned_background_segmentation:
            self.background_seg_feat = nn.Parameter(torch.empty(seg_feat_dim))
        else:
            # A None parameter does not enter state_dict, keeping checkpoints
            # and execution byte-compatible with the historical disabled path.
            self.register_parameter("background_seg_feat", None)

        # Valid head: Predicts the valid token
        self.valid_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )




        # Initialize all heads with appropriate strategies
        self._init_weights()
        if self.background_seg_feat is not None:
            nn.init.normal_(self.background_seg_feat, mean=0.0, std=0.01)

        if dtype is not None:
            _dtype = str_to_dtype(dtype) if isinstance(dtype, str) else dtype
            self.apply(partial(convert_module_to, dtype=_dtype))
            # background_seg_feat is a direct Parameter rather than a child
            # Linear/Conv parameter, so convert_module_to does not visit it.
            if self.background_seg_feat is not None:
                self.background_seg_feat.data = (
                    self.background_seg_feat.data.to(dtype=_dtype)
                )

        # self.pos_bin_head = torch.compile(self.pos_bin_head)
        # self.angle_bin_head = torch.compile(self.angle_bin_head)
        # self.scale_bin_head = torch.compile(self.scale_bin_head)
        # self.valid_head = torch.compile(self.valid_head)
        # self.context_feat_head = torch.compile(self.context_feat_head)

        # if not self.seg_feat_use_pose_transformer:
        #     self.seg_feat_head = torch.compile(self.seg_feat_head)
        #     if self.seg_feat_with_spatial:
        #         self.pos_proj = torch.compile(self.pos_proj)
        #         self.angle_proj = torch.compile(self.angle_proj)
        #         self.scale_proj = torch.compile(self.scale_proj)
        # else:
        #     self.pos_proj = torch.compile(self.pos_proj)
        #     self.angle_proj = torch.compile(self.angle_proj)
        #     self.scale_proj = torch.compile(self.scale_proj)
        #     self.pose_feat_head = torch.compile(self.pose_feat_head)
        #     self.seg_feat_head_first = torch.compile(self.seg_feat_head_first)
        #     self.seg_feat_head = torch.compile(self.seg_feat_head)
        #     self.pos_transformer_encoder = torch.compile(self.pos_transformer_encoder)

        # for hungarian matching
        # Pre-compute and cache bin centers as buffers (avoid recomputing every forward)
        scale_centers, angle_centers, pos_centers = get_bin_centers(device='cpu', dtype=torch.float32)
        self.register_buffer('scale_centers', scale_centers, persistent=False)
        self.register_buffer('angle_centers', angle_centers, persistent=False)
        self.register_buffer('pos_centers', pos_centers, persistent=False)

        self.logit_scale = logit_scale
        # print(f"logit_scale: {self.logit_scale}")

    def _init_weights(self):
        """
        Initialize weights for all prediction heads.

        Strategy:
        """
        def _init_classification_head(module):
            """Xavier init for classification heads with zero bias."""
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        def _init_regression_head(module, small_final=True):
            """Xavier init for regression heads, optionally small final layer."""
            layers = [l for l in module if isinstance(l, nn.Linear)]
            for i, layer in enumerate(layers):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                # Make final layer output small values for stable training
                if small_final and i == len(layers) - 1:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)

        # Classification heads: uniform initial predictions over bins
        _init_classification_head(self.pos_bin_head)
        _init_classification_head(self.angle_bin_head)
        _init_classification_head(self.scale_bin_head)

        # Regression heads: stable output initialization
        _init_regression_head(self.context_feat_head, small_final=True)

        if not self.seg_feat_use_pose_transformer:
            _init_regression_head(self.seg_feat_head, small_final=True)
            if self.seg_feat_with_spatial:
                _init_regression_head(self.pos_proj, small_final=True)
                _init_regression_head(self.angle_proj, small_final=True)
                _init_regression_head(self.scale_proj, small_final=True)
        else:
            _init_regression_head(self.pos_proj, small_final=True)
            _init_regression_head(self.angle_proj, small_final=True)
            _init_regression_head(self.scale_proj, small_final=True)
            _init_regression_head(self.pose_feat_head, small_final=True)
            _init_regression_head(self.seg_feat_head_first, small_final=True)
            _init_regression_head(self.seg_feat_head, small_final=True)

        # Valid head: special initialization for imbalanced binary classification
        # With MAX_SCENE_OBJECTS=192 and ~10 objects per scene:
        # - ~5% positive (valid objects), ~95% negative (empty slots)
        # - Prior-based bias: log(p_pos / p_neg) = log(0.05 / 0.95) ≈ -2.94
        # - This makes the model initially predict "invalid" for all slots,
        #   then learn which slots should be valid
        for layer in self.valid_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Set negative bias on final layer of valid_head
        # bias = -3.0 corresponds to initial probability ~4.7% (sigmoid(-3) ≈ 0.047)
        final_layer = self.valid_head[-1]
        if hasattr(final_layer, 'bias') and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, -3.0)

    def embed_position(self, batch_size, seq_len, device):
        """Apply positional embedding.

        Args:
            batch_size: int. Batch size.
            seq_len: int. Sequence length.
            device: torch.device.

        Returns:
            pos_emb: [B, T, d_model] torch.FloatTensor.
        """
        t = torch.arange(seq_len, device=device)
        pos_emb = repeat(self.position_embedding(t), "t d -> b t d", b=batch_size)
        return pos_emb

    def forward(
        self,
        context,
        context_mask,
        coords,
        recon_context_feats,
        recon_context_mask,
        recon_coords,
        return_intermediate_feats=False,
        include_combined_outputs=True,
    ):
        """
        Memory-efficient forward pass. Only valid token values are passed, not full [B, T, dim] tensors.

        Args:
            context: [B, context_length, d_model] torch.FloatTensor. Encoder memory.
            context_mask: [B, context_length] torch.BoolTensor. True means ignore (padding).
            coords: [B, context_length, d_model] torch.FloatTensor. Fourier encoded coordinates.

            recon_context_feats: [B, full_context_length, d_model] torch.FloatTensor. Reconstructed context features.
            recon_context_mask: [B, full_context_length] torch.BoolTensor. Reconstructed context mask.
            recon_coords: [B, full_context_length, d_model] torch.FloatTensor. Reconstructed context coordinates.
        Returns:
            A list containing num_decoder_layers elements:
            each element is a dictionary containing the following keys:
            - pos_bin_logits: [B, MAX_SCENE_OBJECTS, 3, num_bins] torch.FloatTensor. Logits for position predictions.
            - angle_bin_logits: [B, MAX_SCENE_OBJECTS, 3, num_bins] torch.FloatTensor. Logits for angle predictions.
            - scale_bin_logits: [B, MAX_SCENE_OBJECTS, 1, num_bins] torch.FloatTensor. Logits for scale predictions.
            - valid_logits: [B, MAX_SCENE_OBJECTS] torch.FloatTensor. Logits for valid predictions.
            - seg_feat_logits: [B, MAX_SCENE_OBJECTS, seg_feat_dim] torch.FloatTensor. Logits for segmentation feature predictions.
            - background_pred_masks_logits: optional [B, 1, full_context_length]
              logits produced from the global learned background embedding.
        """

        B = context.shape[0]
        device = context.device
        dtype = context.dtype
        num_query_objects = self.max_scene_objects
        num_query_tokens = num_query_objects * TOKENS_PER_OBJECT

        # Use cached type/object indices (already on correct device via buffer mechanism)
        # Expand to batch dimension
        tgt_types = self._cached_tgt_types.unsqueeze(0).expand(B, -1)
        tgt_object_ids = self._cached_tgt_object_ids.unsqueeze(0).expand(B, -1)

        decoder_input = (
            self.embed_position(B, num_query_tokens, device)
            + self.type_embedding(tgt_types)
            + self.object_embedding(tgt_object_ids)
        )

        # Transformer decoder
        decoder_out_all_layers, context_memory, encoded_seg_context_feats = self.transformer(
            src=context,
            mask=context_mask,
            query_embed=decoder_input,
            pos_embed=coords,
            recon_src=recon_context_feats,
            recon_mask=recon_context_mask,
            recon_pos_embed=recon_coords,
        )  # decoder_out_all_layers: [num_decoder_layers, B, num_queries, d_model], encoded_context_feats: [B, full_context_length, d_model]

        encoded_seg_context_feats = self.context_feat_head(encoded_seg_context_feats)

        if self.logit_scale is not None:
            encoded_seg_context_feats = F.normalize(encoded_seg_context_feats, dim=-1, eps=1e-4)

        background_seg_feats = None
        background_pred_masks_logits = None
        if self.learned_background_segmentation:
            background_seg_feats = self.background_seg_feat.reshape(
                1, 1, -1
            ).expand(B, -1, -1)
            if self.logit_scale is not None:
                background_seg_feats = F.normalize(
                    background_seg_feats, dim=-1, eps=1e-4
                )
            background_pred_masks_logits = torch.matmul(
                background_seg_feats,
                encoded_seg_context_feats.transpose(-1, -2),
            )
            if self.logit_scale is not None:
                background_pred_masks_logits = (
                    background_pred_masks_logits * self.logit_scale
                )

        # Gather outputs at relevant positions and apply heads
        decoded_results = []

        num_decoder_layers = decoder_out_all_layers.shape[0]

        for layer_i in range(num_decoder_layers):
            decoder_out = decoder_out_all_layers[layer_i]
            # decoder_out: [B, num_queries, d_model]
            decoder_out = decoder_out.view(
                B, num_query_objects, TOKENS_PER_OBJECT, self.d_model
            )

            # Discrete predictions at shifted positions (predict next token)
            pos_bin_out = decoder_out[:, :, POS_TOKEN_START:POS_TOKEN_END+1]
            pos_bin_logits = self.pos_bin_head(pos_bin_out)  # [B, MAX_SCENE_OBJECTS, 3, num_bins]

            angle_bin_out = decoder_out[:, :, ANGLE_TOKEN_START:ANGLE_TOKEN_END+1, :]
            angle_bin_logits = self.angle_bin_head(angle_bin_out)  # [B, MAX_SCENE_OBJECTS, 3, num_bins]

            scale_bin_out = decoder_out[:, :, SCALE_TOKEN:SCALE_TOKEN+1, :]
            scale_bin_logits = self.scale_bin_head(scale_bin_out)  # [B, MAX_SCENE_OBJECTS, 1, num_bins]

            # Valid predictions at shifted positions (predict next token)
            valid_out = decoder_out[:, :, VALID_TOKEN:VALID_TOKEN+1, :].squeeze(2)
            valid_logits = self.valid_head(valid_out).squeeze(-1)  # [B, MAX_SCENE_OBJECTS]

            seg_feat_out = decoder_out[:, :, SEG_TOKEN:SEG_TOKEN+1, :].squeeze(2)  # [B, MAX_SCENE_OBJECTS, d_model]

            if not self.seg_feat_use_pose_transformer:
                if self.seg_feat_with_spatial:
                    pos_bin_logits_clamped = torch.clamp(pos_bin_logits.detach(), min=-20.0, max=20.0)
                    angle_bin_logits_clamped = torch.clamp(angle_bin_logits.detach(), min=-20.0, max=20.0)
                    scale_bin_logits_clamped = torch.clamp(scale_bin_logits.detach(), min=-20.0, max=20.0)

                    pos_bin_logits_proj_feats = self.pos_proj(
                        pos_bin_logits_clamped
                    ).reshape(B, num_query_objects, 3 * self.d_model)
                    angle_bin_logits_proj_feats = self.angle_proj(
                        angle_bin_logits_clamped
                    ).reshape(B, num_query_objects, 3 * self.d_model)
                    scale_bin_logits_proj_feats = self.scale_proj(
                        scale_bin_logits_clamped
                    ).reshape(B, num_query_objects, self.d_model)

                    seg_feat_out = torch.cat([
                        pos_bin_logits_proj_feats,
                        angle_bin_logits_proj_feats,
                        scale_bin_logits_proj_feats,
                        seg_feat_out
                    ], dim=-1)  # [B, MAX_SCENE_OBJECTS, 8*d_model]

                seg_feats = self.seg_feat_head(seg_feat_out)  # [B, MAX_SCENE_OBJECTS, seg_feat_dim]

            else:
                pos_bin_logits_clamped = torch.clamp(pos_bin_logits.detach(), min=-20.0, max=20.0)
                angle_bin_logits_clamped = torch.clamp(angle_bin_logits.detach(), min=-20.0, max=20.0)
                scale_bin_logits_clamped = torch.clamp(scale_bin_logits.detach(), min=-20.0, max=20.0)

                pos_bin_logits_proj_feats = self.pos_proj(
                    pos_bin_logits_clamped
                ).reshape(B, num_query_objects, 3 * self.d_model)
                angle_bin_logits_proj_feats = self.angle_proj(
                    angle_bin_logits_clamped
                ).reshape(B, num_query_objects, 3 * self.d_model)
                scale_bin_logits_proj_feats = self.scale_proj(
                    scale_bin_logits_clamped
                ).reshape(B, num_query_objects, self.d_model)

                pose_feats = torch.cat([
                    pos_bin_logits_proj_feats,
                    angle_bin_logits_proj_feats,
                    scale_bin_logits_proj_feats,
                ], dim=-1)  # [B, MAX_SCENE_OBJECTS, 7*d_model]
                pose_feats = self.pose_feat_head(pose_feats)  # [B, MAX_SCENE_OBJECTS, d_model]

                seg_feats_first = self.seg_feat_head_first(seg_feat_out)  # [B, MAX_SCENE_OBJECTS, d_model]

                seg_feats = self.pos_transformer_encoder(
                    src=seg_feats_first,
                    mask=None,
                    pos_embed=pose_feats,
                )  # [B, MAX_SCENE_OBJECTS, d_model]

                seg_feats = self.seg_feat_head(seg_feats)  # [B, MAX_SCENE_OBJECTS, seg_feat_dim]



            if self.logit_scale is not None:
                seg_feats = F.normalize(seg_feats, dim=-1, eps=1e-4)

            # if layer_i == num_decoder_layers - 1:
            #     context_target = encoded_seg_context_feats
            # else:
            #     context_target = encoded_seg_context_feats.detach()
            context_target = encoded_seg_context_feats

            pred_masks_logits = torch.matmul(seg_feats, context_target.transpose(-1, -2))  # [B, MAX_SCENE_OBJECTS, full_context_length]

            if self.logit_scale is not None:
                pred_masks_logits = pred_masks_logits * self.logit_scale

            layer_result = {
                "pos_bin_logits": pos_bin_logits,
                "angle_bin_logits": angle_bin_logits,
                "scale_bin_logits": scale_bin_logits,
                "valid_logits": valid_logits,
                "pred_masks_logits": pred_masks_logits,
                "seg_feats": seg_feats,
            }
            if self.learned_background_segmentation:
                layer_result.update({
                    "background_pred_masks_logits":
                        background_pred_masks_logits,
                    "background_seg_feats": background_seg_feats,
                })
                # Legacy inference consumers request the concatenated tensors.
                # The V2 trainer consumes the separate background/foreground
                # tensors and disables these copies, which are otherwise made
                # at every auxiliary decoder layer without affecting loss.
                if include_combined_outputs:
                    layer_result.update({
                        "all_pred_masks_logits": torch.cat(
                            [background_pred_masks_logits, pred_masks_logits],
                            dim=1,
                        ),
                        "all_seg_feats": torch.cat(
                            [background_seg_feats, seg_feats], dim=1
                        ),
                    })
            decoded_results.append(layer_result)

        if return_intermediate_feats:
            return decoded_results, context_memory, encoded_seg_context_feats

        return decoded_results

    # @torch.compile
    def get_cls_cost(
        self,
        valid_logits,
        max_num_objects,
    ):
        # Classification cost: high confidence -> low cost (negative probability)
        # We want the negative log-probability for the matching cost
        # Using log_sigmoid is numerically more stable than log(sigmoid(x))
        log_p = F.logsigmoid(valid_logits)  # [B, M]

        # Focal Loss version of the cost (Optional but highly recommended)
        # This mimics the focal loss gradient behavior in the matcher
        gamma = 2.0
        pred_valid_prob = valid_logits.sigmoid()  # [B, M]

        # The cost of matching query i to a target object
        # Cost = -(1-p)^gamma * log(p)
        focal_cost = -(1 - pred_valid_prob)**gamma * log_p

        # Expand to [B, M, num_objs]
        cls_cost = focal_cost.unsqueeze(-1).expand(-1, -1, max_num_objects)

        return cls_cost
    # @torch.compile
    # def get_cls_cost(
    #     self,
    #     valid_logits,
    #     max_num_objects,
    # ):
    #     # 1. Convert logits to probabilities
    #     pred_valid_prob = valid_logits.sigmoid()  # [B, M]

    #     # 2. Use pure negative probability for perfectly bounded matching cost [-1, 0]
    #     # High confidence (1.0) -> Cost = -1.0 (Highly preferred)
    #     # Low confidence (0.0) -> Cost = 0.0 (Least preferred)
    #     cls_cost = -pred_valid_prob

    #     # Expand to [B, M, num_objs]
    #     cls_cost = cls_cost.unsqueeze(-1).expand(-1, -1, max_num_objects)

    #     return cls_cost

    # @torch.compile
    def get_pose_cost(
        self,
        pos_bin_logits,
        angle_bin_logits,
        scale_bin_logits,
        tgt_pos_bins,
        tgt_angle_bins,
        tgt_scale_bins,
    ):

        # Use cached bin centers (already on correct device via buffer mechanism)
        scale_centers = self.scale_centers.float()
        angle_centers = self.angle_centers.float()
        pos_centers = self.pos_centers.float()

        # ------ start of pred processing ------
        pos_probs = torch.softmax(pos_bin_logits, dim=-1)  # [B, MAX_SCENE_OBJECTS, 3, num_bins]
        angle_probs = torch.softmax(angle_bin_logits, dim=-1)  # [B, MAX_SCENE_OBJECTS, 3, num_bins]
        scale_probs = torch.softmax(scale_bin_logits, dim=-1)  # [B, MAX_SCENE_OBJECTS, 1, num_bins]

        # Position: all 3 dimensions share the same bin centers (pos_centers[:, 0])
        pred_pos = (pos_probs * pos_centers[:, 0]).sum(dim=-1)  # [B, MAX_SCENE_OBJECTS, 3]

        # Angle: each dimension has different bin centers, transpose for broadcasting
        # angle_centers: [num_bins, 3] -> [3, num_bins] for [B, M, 3, num_bins] broadcasting
        pred_angle = (angle_probs * angle_centers.T).sum(dim=-1)  # [B, MAX_SCENE_OBJECTS, 3]

        # Scale: scale_centers is [num_bins]
        pred_scale = (scale_probs * scale_centers).sum(dim=-1)  # [B, MAX_SCENE_OBJECTS, 1]
        pred_rot = euler_angles_to_matrix(pred_angle)

        # ------ end of pred processing ------

        # ------ start of tgt processing ------
        tgt_scale, tgt_angle, tgt_pos = continue_transform_torch(tgt_scale_bins.clone(), tgt_angle_bins.clone(), tgt_pos_bins.clone())
        tgt_rot = euler_angles_to_matrix(tgt_angle)
        # ------ end of tgt processing ------

        # ------ pos and scale normalization ------
        tgt_scene_scale = POS_MAX
        tgt_pos_normalized = tgt_pos / tgt_scene_scale
        pred_pos_normalized = pred_pos / tgt_scene_scale
        tgt_scale_normalized = tgt_scale / tgt_scene_scale
        pred_scale_normalized = pred_scale / tgt_scene_scale

        # ------ calculate cost matrix ------

        # Position L2 distance
        pos_cost = torch.cdist(pred_pos_normalized, tgt_pos_normalized, p=2)  # [B, MAX_SCENE_OBJECTS, num_objects]

        # Scale L1 distance
        scale_cost = torch.cdist(pred_scale_normalized, tgt_scale_normalized, p=1)  # [B, MAX_SCENE_OBJECTS, num_objects]

        # Rotation geodesic distance. The legacy symmetry uses world-Z
        # quarter turns. The point-normalized model uses object-local-up
        # quarter turns, matching the current scene-pose perception branch
        # without changing the [0,24] position-token domain.
        if self.rotation_local_up_quarter_turns:
            equivalent_tgt_rot = local_up_quarter_turn_equivalent_rotations(tgt_rot)
            rot_cost = rotation_matrix_geodesic_distance(
                pred_rot[:, :, None, None, :, :],
                equivalent_tgt_rot[:, None, :, :, :, :],
            ).min(dim=-1).values
        elif self.rotation_z_quarter_turns:
            equivalent_tgt_rot = z_quarter_turn_equivalent_rotations(tgt_rot)
            rot_cost = rotation_matrix_geodesic_distance(
                pred_rot[:, :, None, None, :, :],
                equivalent_tgt_rot[:, None, :, :, :, :],
            ).min(dim=-1).values
        else:
            pred_rot_exp = pred_rot.unsqueeze(2)
            tgt_rot_exp = tgt_rot.unsqueeze(1)
            rot_cost = rotation_matrix_geodesic_distance(pred_rot_exp, tgt_rot_exp)
        rot_cost = rot_cost / torch.pi

        return pos_cost, scale_cost, rot_cost

    # @torch.compile
    def get_seg_cost(
        self,
        pred_masks_logits, # [B, MAX_SCENE_OBJECTS(K), full_context_length(N)], logits (pre-sigmoid)
        pred_masks, # [B, MAX_SCENE_OBJECTS(K), full_context_length(N)], logits (post-sigmoid)
        gt_masks, # [B, num_objects(M), full_context_length(N)], 0/1
    ):
        dice_cost = dice_cost_matrix(pred_masks, gt_masks) # [B, MAX_SCENE_OBJECTS, num_objects]
        focal_cost = sigmoid_focal_cost_matrix(pred_masks_logits, gt_masks) # [B, MAX_SCENE_OBJECTS, num_objects]
        return dice_cost, focal_cost

    # @torch.compile
    def get_cost_matrix(
        self,
        pos_bin_logits,
        angle_bin_logits,
        scale_bin_logits,
        tgt_pos_bins,
        tgt_angle_bins,
        tgt_scale_bins,
        valid_logits,
        pred_masks_logits, # [B, MAX_SCENE_OBJECTS, full_context_length]
        gt_masks, # [B, num_objects, full_context_length], 0/1
    ):
        # ==========================================
        # 🛡️ 强制进入 FP32 上下文，覆盖外层的 BF16 AMP
        # ==========================================
        with torch.autocast(device_type=pos_bin_logits.device.type, dtype=torch.float32):

            # 1. 在此上下文中，首先将所有浮点输入强制转为 FP32
            pos_bin_logits = pos_bin_logits.float()
            angle_bin_logits = angle_bin_logits.float()
            scale_bin_logits = scale_bin_logits.float()
            valid_logits = valid_logits.float()
            # pred_masks_logits = pred_masks_logits.float()
            # only calculate pred_masks in float32 context
            # pred_masks = pred_masks_logits.sigmoid()

            # 如果你的 tgt_bins 也是浮点数（或者即使是整数，为了后续统一计算），也转一下
            tgt_pos_bins = tgt_pos_bins.float()
            tgt_angle_bins = tgt_angle_bins.float()
            tgt_scale_bins = tgt_scale_bins.float()
            gt_masks = gt_masks.float()

            max_num_objects = tgt_pos_bins.shape[1]

            # Cost weights
            W_CLS = self.cost_weights["cls"]
            W_POS = self.cost_weights["pos"]
            W_SCL = self.cost_weights["scl"]
            W_ROT = self.cost_weights["rot"]
            # W_SEG_DICE = self.cost_weights["seg_dice"]
            # W_SEG_FOCAL = self.cost_weights["seg_focal"]

            # 2. 调用内部方法 (此时所有输入已经是 FP32，计算也会被强制在 FP32 下执行)
            cls_cost = self.get_cls_cost(valid_logits, max_num_objects) # [B, MAX_SCENE_OBJECTS, max_num_objects]
            pos_cost, scale_cost, rot_cost = self.get_pose_cost(
                pos_bin_logits=pos_bin_logits,
                angle_bin_logits=angle_bin_logits,
                scale_bin_logits=scale_bin_logits,
                tgt_pos_bins=tgt_pos_bins,
                tgt_angle_bins=tgt_angle_bins,
                tgt_scale_bins=tgt_scale_bins,
            ) # [B, MAX_SCENE_OBJECTS, max_num_objects] * 3
            # dice_cost, focal_cost = self.get_seg_cost(
            #     pred_masks_logits=pred_masks_logits,
            #     pred_masks=pred_masks,
            #     gt_masks=gt_masks,
            # ) # [B, MAX_SCENE_OBJECTS, max_num_objects]

            # Total cost matrix with weights without invalid objects
            cost_matrix = (W_CLS * cls_cost) + \
                (W_POS * pos_cost) + (W_SCL * scale_cost) + (W_ROT * rot_cost)
                #  + \
                # (W_SEG_DICE * dice_cost) + (W_SEG_FOCAL * focal_cost)

            return cost_matrix

    def get_mapping_by_hungarian_cpu(self, cost_matrix, B, device, num_valid_per_batch):

        # Apply Hungarian algorithm per batch
        # mapping[b, i] = j means pred[i] matches GT[j], -1 means no match
        num_queries = cost_matrix.shape[1]
        mapping = torch.full((B, num_queries), -1, dtype=torch.int32, device=device)

        # Move entire cost matrix to CPU once with non-blocking transfer
        # Use float32 for better numerical stability in matching
        cost_matrix_cpu = cost_matrix.float().cpu().numpy()

        # Prepare mapping results on CPU first to minimize GPU transfers
        mapping_cpu = [None] * B

        for b in range(B):
            num_valid = num_valid_per_batch[b]

            if num_valid == 0:
                # No objects to match
                continue

            # Hungarian matching on all objects
            # Only consider valid GT objects: gt[:num_valid]
            cost_b = cost_matrix_cpu[b, :, :num_valid] # [MAX_SCENE_OBJECTS, num_valid]

            cost_b = np.nan_to_num(cost_b, nan=1e6, posinf=1e6, neginf=-1e6)

            row_ind, col_ind = linear_sum_assignment(cost_b)
            mapping_cpu[b] = (row_ind.tolist(), col_ind.tolist())

        # Transfer all mappings to GPU in one batch
        for b in range(B):
            if mapping_cpu[b] is not None:
                row_ind, col_ind = mapping_cpu[b]
                if len(row_ind) > 0:
                    mapping[b, row_ind] = torch.tensor(col_ind, dtype=torch.int32, device=device)

        return mapping

    @torch.no_grad()
    def mapping_preds_gts_layers(
        self,
        decoded_results,
        tgt_pos_bins,
        tgt_angle_bins,
        tgt_scale_bins,
        gt_masks,
        num_objects,
    ):
        """Match all auxiliary decoder layers with one device synchronization.

        Each layer keeps its own cost matrix and its own SciPy Hungarian solve,
        exactly as in :meth:`mapping_preds_gts`. Stacking changes only the
        transfer schedule: six small GPU->CPU and CPU->GPU synchronizations
        become one transfer in each direction.
        """

        if not decoded_results:
            return []
        cost_matrices = []
        for decoded in decoded_results:
            cost_matrices.append(
                self.get_cost_matrix(
                    pos_bin_logits=decoded["pos_bin_logits"],
                    angle_bin_logits=decoded["angle_bin_logits"],
                    scale_bin_logits=decoded["scale_bin_logits"],
                    tgt_pos_bins=tgt_pos_bins,
                    tgt_angle_bins=tgt_angle_bins,
                    tgt_scale_bins=tgt_scale_bins,
                    valid_logits=decoded["valid_logits"],
                    pred_masks_logits=decoded["pred_masks_logits"],
                    gt_masks=gt_masks,
                )
            )

        device = cost_matrices[0].device
        layers = len(cost_matrices)
        batch_size, num_queries, _ = cost_matrices[0].shape
        # This is the sole cost synchronization for all layers.
        costs_cpu = torch.stack(cost_matrices, dim=0).float().cpu().numpy()
        valid_counts = num_objects.detach().cpu().long().tolist()
        mappings_cpu = np.full(
            (layers, batch_size, num_queries), -1, dtype=np.int32
        )
        for layer_index in range(layers):
            for batch_index, valid_count in enumerate(valid_counts):
                if valid_count <= 0:
                    continue
                cost = np.nan_to_num(
                    costs_cpu[
                        layer_index, batch_index, :, : int(valid_count)
                    ],
                    nan=1e6,
                    posinf=1e6,
                    neginf=-1e6,
                )
                row_indices, column_indices = linear_sum_assignment(cost)
                mappings_cpu[
                    layer_index, batch_index, row_indices
                ] = column_indices.astype(np.int32, copy=False)

        mappings = torch.from_numpy(mappings_cpu).to(device=device)
        return list(mappings.unbind(dim=0))

    @torch.no_grad()
    def mapping_preds_gts(
        self,
        pos_bin_logits,
        angle_bin_logits,
        scale_bin_logits,
        tgt_pos_bins,
        tgt_angle_bins,
        tgt_scale_bins,
        valid_logits,
        pred_masks_logits, # [B, MAX_SCENE_OBJECTS, full_context_length]
        gt_masks, # [B, num_objects, full_context_length], 0/1
        num_objects, # [B]
    ):
        # Soft argmax: convert logits to continuous values using probability-weighted bin centers
        # This uses the full probability distribution instead of just the argmax
        cost_matrix = self.get_cost_matrix(
            pos_bin_logits=pos_bin_logits,
            angle_bin_logits=angle_bin_logits,
            scale_bin_logits=scale_bin_logits,
            tgt_pos_bins=tgt_pos_bins,
            tgt_angle_bins=tgt_angle_bins,
            tgt_scale_bins=tgt_scale_bins,
            valid_logits=valid_logits,
            pred_masks_logits=pred_masks_logits,
            gt_masks=gt_masks,
        )

        B = pos_bin_logits.shape[0]
        device = pos_bin_logits.device

        mapping = self.get_mapping_by_hungarian_cpu(cost_matrix, B, device, num_objects)

        return mapping

    def resort_preds_by_mapping(
        self,
        pos_bin_logits,
        angle_bin_logits,
        scale_bin_logits,
        valid_logits,
        pred_masks_logits,
        mapping,
    ):
        """
        Resort predictions according to the Hungarian matching to align with GT ordering.
        Fully vectorized implementation - no Python loops over batches.

        After resorting: resorted_preds[j] = original_preds[i] where mapping[i] = j
        Invalid predictions (mapping == -1) are placed at the end.

        Args:
            pos_bin_logits: [B, MAX_SCENE_OBJECTS, 3, num_bins] float tensor.
            angle_bin_logits: [B, MAX_SCENE_OBJECTS, 3, num_bins] float tensor.
            scale_bin_logits: [B, MAX_SCENE_OBJECTS, 1, num_bins] float tensor.
            valid_logits: [B, MAX_SCENE_OBJECTS] float tensor.
            pred_masks_logits: [B, MAX_SCENE_OBJECTS, full_context_length] float tensor.
            mapping: [B, MAX_SCENE_OBJECTS] int32 tensor. mapping[b, i] = j means pred i matches GT j. -1 means unmatched.

        Returns:
            Resorted tensors in the same shapes, ordered to match GT sequence.
        """
        B = pos_bin_logits.shape[0]
        device = pos_bin_logits.device
        M = mapping.shape[1]
        if valid_logits.shape[1] != M:
            raise ValueError(
                "valid logits and mapping disagree on query count: "
                f"{valid_logits.shape[1]} vs {M}"
            )

        # Vectorized inverse mapping computation
        # We need to build resorted_indices[b, j] = i where mapping[b, i] = j

        # Create prediction indices for each batch
        pred_indices = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)  # [B, M]

        # Mask for matched predictions
        matched_mask = mapping >= 0  # [B, M]

        # For matched predictions, we want to place them at their GT positions
        # For unmatched, we place them at remaining positions

        # Create a large value for unmatched to push them to the end when sorted
        # mapping value for unmatched: use M + pred_index to maintain order
        sort_keys = torch.where(matched_mask, mapping.long(), M + pred_indices)  # [B, M]

        # Sort by the keys to get the reordering
        # After sorting: position j contains the pred index that should go to position j
        _, resorted_indices = sort_keys.sort(dim=1, stable=True)  # [B, M]

        # Apply the resorting to all prediction tensors using gather
        # This is more efficient than advanced indexing for this use case

        # Expand indices for gathering: [B, M, 1, 1] for broadcasting
        gather_idx_3d = resorted_indices.unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1]

        # Gather for 4D tensors: [B, M, D1, D2]
        resorted_pos_bin_logits = torch.gather(
            pos_bin_logits, 1,
            gather_idx_3d.expand(-1, -1, pos_bin_logits.shape[2], pos_bin_logits.shape[3])
        )
        resorted_angle_bin_logits = torch.gather(
            angle_bin_logits, 1,
            gather_idx_3d.expand(-1, -1, angle_bin_logits.shape[2], angle_bin_logits.shape[3])
        )
        resorted_scale_bin_logits = torch.gather(
            scale_bin_logits, 1,
            gather_idx_3d.expand(-1, -1, scale_bin_logits.shape[2], scale_bin_logits.shape[3])
        )
        resorted_valid_logits = torch.gather(
            valid_logits, 1,
            gather_idx_3d.squeeze(-1).squeeze(-1)
        )
        resorted_pred_masks_logits = torch.gather(
            pred_masks_logits, 1,
            gather_idx_3d.squeeze(-1).expand(-1, -1, pred_masks_logits.shape[2])
        )


        return (
            resorted_pos_bin_logits,
            resorted_angle_bin_logits,
            resorted_scale_bin_logits,
            resorted_valid_logits,
            resorted_pred_masks_logits,
        )
