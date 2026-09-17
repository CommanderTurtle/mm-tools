import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from utils.voxelize import InstanceVoxelize, Voxelize
from utils.constants import POS_MAX
import numpy as np
from utils.pos_embedding import fourier_encode_vector
from models.scene_decoder_w_seg_voxelize import SceneDecoder
from models.point_unet import PointCloudUNet
from torchvision.transforms import v2
from datetime import datetime


def split_background_supervision(
    object_translations,
    object_angles,
    object_scales,
    object_valid_masks,
    gt_masks,
    learned_background_segmentation=False,
):
    """Separate the fixed background mask target from foreground detections.

    Dataset/token order is `[background, foreground...]`. When the option is
    disabled, return the original tensors unchanged. When enabled, pose,
    validity, and Hungarian targets contain foreground objects only, while the
    background retains an explicit mask target.
    """
    if not learned_background_segmentation:
        return {
            "object_translations": object_translations,
            "object_angles": object_angles,
            "object_scales": object_scales,
            "object_valid_masks": object_valid_masks,
            "gt_masks": gt_masks,
            "background_gt_mask": None,
        }

    if object_translations.shape[1] < 1 or gt_masks.shape[1] < 1:
        raise ValueError(
            "learned background segmentation requires target slot 0"
        )
    if object_valid_masks.shape[1] < 1:
        raise ValueError(
            "learned background segmentation requires a background validity "
            "target in slot 0"
        )

    return {
        "object_translations": object_translations[:, 1:],
        "object_angles": object_angles[:, 1:],
        "object_scales": object_scales[:, 1:],
        "object_valid_masks": object_valid_masks[:, 1:],
        "gt_masks": gt_masks[:, 1:],
        "background_gt_mask": gt_masks[:, :1],
    }


def make_transform():
    """
    Create a transform for numpy RGB arrays with values in (0, 1).

    Returns:
        A transform function that takes numpy array and returns normalized tensor
    """
    def transform_fn(tensor):
        """
        Transform numpy array with values in (0, 1) to normalized tensor.

        Args:
            rgb_array: numpy array of shape [n, h, w] or [n, h, w, 3] with values in (0, 1)
                      If [n, h, w], assumes grayscale and will be converted to RGB by repeating channels

        Returns:
            Tensor of shape [n, 3, resize_size, resize_size] normalized for ImageNet
        """
        # Convert numpy to tensor
        # tensor = torch.from_numpy(rgb_array).float()

        # Handle shape [n, h, w] - add channel dimension and repeat for RGB
        if tensor.ndim == 3:
            # [n, h, w] -> [n, h, w, 3] by repeating the channel
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif tensor.ndim == 4:
            # Already [n, h, w, 3]
            pass
        else:
            raise ValueError(f"Expected input shape [n, h, w] or [n, h, w, 3], got {tensor.shape}")

        # Convert from [n, h, w, 3] to [n, 3, h, w] (channels first)
        tensor = tensor.permute(0, 3, 1, 2)

        # Normalize with ImageNet stats
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        tensor = normalize(tensor)

        return tensor

    return transform_fn


def visualize_points(points, rgbs, feats, masks, save_dir, save_name):
    import open3d as o3d
    import os
    from sklearn.decomposition import PCA
    if rgbs is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgbs)
        # downsample the points with voxel_down_sample
        # print(f"Downsampling points with voxel_size=0.01")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(os.path.join(save_dir, f"{save_name}_colors_pcd.ply"), pcd)

    if masks is not None:
        num_instances = np.max(masks) + 1
        instance_colors = np.random.randint(0, 255, size=(num_instances, 3), dtype=np.uint8)
        instance_colors[0] = [255, 255, 255] # background is white
        instance_colors  = instance_colors / 255.0
        instance_colors = instance_colors[masks]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        # print(f"Downsampling points with voxel_size=0.01")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(os.path.join(save_dir, f"{save_name}_instance_colors_pcd.ply"), pcd)

    if feats is not None:
        # Ensure numpy and same length as points
        if torch.is_tensor(feats):
            feats = feats.cpu().numpy()
        feats = np.asarray(feats, dtype=np.float64)
        assert feats.shape[0] == points.shape[0], (
            f"points and feats length mismatch: {points.shape[0]} vs {feats.shape[0]}"
        )

        n_components = min(3, feats.shape[1])
        pca = PCA(n_components=n_components)
        feats_pca = pca.fit_transform(feats)

        # Pad to 3 channels if we had fewer PCA components (e.g. 1 or 2 dims)
        if n_components < 3:
            pad = np.zeros((feats_pca.shape[0], 3 - n_components), dtype=feats_pca.dtype)
            feats_pca = np.hstack([feats_pca, pad])

        f_min = feats_pca.min(axis=0)
        f_max = feats_pca.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1
        colors = np.clip((feats_pca - f_min) / denom, 0.0, 1.0).astype(np.float64)
        colors = np.ascontiguousarray(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        output_path = os.path.join(save_dir, f"{save_name}_dino_pca_features.ply")
        o3d.io.write_point_cloud(output_path, pcd)


class ObjectPoseWSegVoxelize(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scene_scale = POS_MAX
        self.resolution = config.get('resolution', 1024)
        self.voxelize = InstanceVoxelize(
            voxel_size=self.scene_scale / self.resolution,
            resolution=self.resolution,
            apply_standardization=False
        )

        self.dino_model, self.dino_transform = self.get_dino_model_and_transform()
        self.dino_model.cuda().eval()
        for p in self.dino_model.parameters():
            p.requires_grad = False

        self.point_unet = PointCloudUNet(resolution=self.resolution, **config.get('point_unet', {}))

        scene_decoder_config = dict(config.get('scene_decoder', {}))
        scene_decoder_config['rotation_z_quarter_turns'] = bool(
            config.get('rotation_symmetry', {}).get('z_quarter_turns', False)
        )
        scene_decoder_config['rotation_local_up_quarter_turns'] = bool(
            config.get('rotation_symmetry', {}).get(
                'local_up_quarter_turns', False
            )
        )
        self.scene_decoder = SceneDecoder(**scene_decoder_config)
        self.learned_background_segmentation = bool(
            self.scene_decoder.learned_background_segmentation
        )

    @torch.no_grad()
    def get_dino_model_and_transform(self):

        model_path = self.config['dino']['model_path']
        REPO_DIR = self.config['dino']['repo_dir']

        model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=model_path)
        transform_fn = make_transform()
        return model, transform_fn

    @torch.no_grad()
    def get_dino_feats(self, rgbs):
        dino_downsample = 16
        N, H, W, _ = rgbs.shape
        output_H, output_W = H // dino_downsample, W // dino_downsample

        rgbs_ds = rgbs.reshape(N, output_H, dino_downsample, output_W, dino_downsample, 3)
        rgbs_ds = rgbs_ds[:, :, 0, :, 0][:, :output_H, :output_W]

        rgbs = self.dino_transform(rgbs)
        outputs = self.dino_model(rgbs, is_training=True)
        feats = outputs["x_norm_patchtokens"]


        feats = feats.reshape(N*output_H*output_W, -1)
        rgbs_ds = rgbs_ds.reshape(-1, 3)


        return feats, rgbs_ds

    def get_batched_dino_point_feats(
        self,
        rgbs,
        points,
        valid_point_masks=None,
        point_source_indices=None,
        point_valid_masks=None,
    ):
        """Build per-point DINO features for padded scene batches."""

        batch_size, max_num_points = points.shape[:2]
        padded_feats = []
        padded_rgbs = []
        for batch_index in range(batch_size):
            source_feats, source_rgbs = self.get_dino_feats(rgbs[batch_index])
            if valid_point_masks is not None:
                source_mask = valid_point_masks[batch_index].bool()
                if source_mask.numel() != source_feats.shape[0]:
                    raise ValueError(
                        "valid_point_masks/DINO rows mismatch for batch "
                        f"{batch_index}: {source_mask.numel()} vs "
                        f"{source_feats.shape[0]}"
                    )
                source_feats = source_feats[source_mask]
                source_rgbs = source_rgbs[source_mask]

            if point_valid_masks is None:
                target_mask = torch.ones(
                    max_num_points, dtype=torch.bool, device=points.device
                )
            else:
                target_mask = point_valid_masks[batch_index].bool()
            num_valid_points = int(target_mask.sum().item())

            if point_source_indices is None:
                if source_feats.shape[0] != num_valid_points:
                    raise ValueError(
                        "cannot infer source rows for batch "
                        f"{batch_index}: DINO rows={source_feats.shape[0]}, "
                        f"points={num_valid_points}"
                    )
                point_feats = source_feats
                point_rgbs = source_rgbs
            else:
                source_indices = point_source_indices[batch_index][target_mask].long()
                if source_indices.numel() != num_valid_points:
                    raise ValueError(
                        "point_source_indices/point_valid_masks mismatch for "
                        f"batch {batch_index}: {source_indices.numel()} vs "
                        f"{num_valid_points}"
                    )
                if source_indices.numel() > 0 and (
                    source_indices.min() < 0
                    or source_indices.max() >= source_feats.shape[0]
                ):
                    raise IndexError(
                        "point_source_indices outside post-mask DINO rows for "
                        f"batch {batch_index}: "
                        f"range=[{int(source_indices.min())}, "
                        f"{int(source_indices.max())}], "
                        f"rows={source_feats.shape[0]}"
                    )
                point_feats = source_feats.index_select(0, source_indices)
                point_rgbs = source_rgbs.index_select(0, source_indices)

            feat_pad = source_feats.new_zeros(
                (max_num_points, source_feats.shape[-1])
            )
            rgb_pad = source_rgbs.new_zeros((max_num_points, source_rgbs.shape[-1]))
            feat_pad[target_mask] = point_feats
            rgb_pad[target_mask] = point_rgbs
            padded_feats.append(feat_pad)
            padded_rgbs.append(rgb_pad)

        return torch.stack(padded_feats, dim=0), torch.stack(padded_rgbs, dim=0)

    def gather_sequence_instance_ids(
        self,
        voxel_coords,
        voxel_instance_ids,
        recon_voxel_coords,
        recon_context_mask,
    ):
        """Map flat voxel instance IDs to padded reconstructed voxel order."""

        device = voxel_coords.device
        batch_size, seq_len = recon_voxel_coords.shape[:2]
        resolution = self.resolution
        res_sq = resolution * resolution
        max_linear_idx = resolution ** 3

        voxel_coords_long = voxel_coords.long()
        voxel_keys = (
            voxel_coords_long[:, 0] * max_linear_idx
            + voxel_coords_long[:, 1] * res_sq
            + voxel_coords_long[:, 2] * resolution
            + voxel_coords_long[:, 3]
        )
        batch_ids = torch.arange(
            batch_size, dtype=torch.long, device=device
        )[:, None].expand(batch_size, seq_len)
        recon_coords_long = recon_voxel_coords.long()
        recon_keys = (
            batch_ids * max_linear_idx
            + recon_coords_long[..., 0] * res_sq
            + recon_coords_long[..., 1] * resolution
            + recon_coords_long[..., 2]
        ).reshape(-1)

        lookup = torch.searchsorted(voxel_keys, recon_keys)
        in_range = lookup < voxel_keys.numel()
        safe_lookup = lookup.clamp(max=max(0, voxel_keys.numel() - 1))
        matched = in_range & (voxel_keys[safe_lookup] == recon_keys)
        recon_padding = recon_context_mask.reshape(-1).bool()
        missing_valid = (~matched) & (~recon_padding)
        if bool(missing_valid.any().item()):
            missing_index = int(torch.nonzero(missing_valid, as_tuple=False)[0])
            raise RuntimeError(
                "reconstructed voxel coordinate is absent from input "
                f"voxelization at flat sequence index {missing_index}"
            )

        sequence_instance_ids = torch.zeros(
            batch_size * seq_len, dtype=voxel_instance_ids.dtype, device=device
        )
        sequence_instance_ids[matched] = voxel_instance_ids[safe_lookup[matched]]
        return sequence_instance_ids.view(batch_size, seq_len)


    def forward(
        self,
        points,
        rgbs,
        instance_ids,
        object_translations,
        object_angles,
        object_scales,
        object_valid_masks,
        max_num_objects,
        valid_point_masks=None,
        point_source_indices=None,
        point_valid_masks=None,
        return_feats=False,
    ):
        """
        Args:
            points: (B, N, 3)
            rgbs: (B, N_im, H, W, 3)
            instance_ids: (B, N), long, 0 to max_num_objects-1
            valid_point_masks: optional (B, source_rows), bool DINO source-row mask.
            point_source_indices: optional (B, N_aug) indices into DINO/RGB rows
                after applying valid_point_masks. Used when instance augmentation
                deletes or duplicates 3D point rows.
            point_valid_masks: optional (B, N), bool point-row mask for padded
                multi-scene batches.
            object_translations: (B, max_num_objects, 3), long
            object_angles: (B, max_num_objects, 3), long
            object_scales: (B, max_num_objects, 1), long
            object_valid_masks: (B, max_num_objects), bool, True for valid objects
            max_num_objects: int
        """

        # Get batch size from input points (avoids CPU sync from voxel_coords.max().item())
        B = points.shape[0]
        device = points.device

        dino_feats, rgbs_ds = self.get_batched_dino_point_feats(
            rgbs=rgbs,
            points=points,
            valid_point_masks=valid_point_masks,
            point_source_indices=point_source_indices,
            point_valid_masks=point_valid_masks,
        )

        # debug:
        # create a timestamp str
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # visualize_points(
        #     points=points[0].detach().clone().cpu().numpy(),
        #     rgbs=rgbs_ds.detach().clone().cpu().numpy(),
        #     feats=dino_feats[0].detach().clone().cpu().numpy(),
        #     masks=instance_ids[0].detach().clone().cpu().numpy(),
        #     save_dir="vis/render_points_dino",
        #     save_name=f"voxelized_points_in_training_{timestamp}",
        # )

        voxel_coords, voxel_feats, voxel_instance_ids, voxel_inverse_indices = self.voxelize(
            points=points.detach(),
            colors=dino_feats.detach(),
            instance_ids=instance_ids,
            valid_masks=point_valid_masks,
            return_inverse_indices=True,
        ) # [N_total, 4], [N_total, dino_feat_dim], [N_total], int, float, long

        outputs_dict = self.point_unet(
            voxel_coords=voxel_coords.detach(),
            voxel_feats=voxel_feats.detach(),
            B=B,
        )

        context = outputs_dict["context"]
        context_mask = outputs_dict["context_mask"]
        context_coords = outputs_dict["context_coords"]

        recon_context = outputs_dict["recon_context"]
        recon_context_mask = outputs_dict["recon_context_mask"]
        recon_context_coords = outputs_dict["recon_context_coords"]
        recon_voxel_coords = outputs_dict["recon_voxel_coords"]

        # print(f"context shape: {context.shape}, {context.dtype}, {context.device}, {context.min()}, {context.max()}")
        # print(f"context_mask shape: {context_mask.shape}, {context_mask.dtype}, {context_mask.device}, {context_mask.min()}, {context_mask.max()}")
        # print(f"context_coords shape: {context_coords.shape}, {context_coords.dtype}, {context_coords.device}, {context_coords.min()}, {context_coords.max()}")
        # print(f"recon_context shape: {recon_context.shape}, {recon_context.dtype}, {recon_context.device}, {recon_context.min()}, {recon_context.max()}")
        # print(f"recon_context_mask shape: {recon_context_mask.shape}, {recon_context_mask.dtype}, {recon_context_mask.device}, {recon_context_mask.min()}, {recon_context_mask.max()}")
        # print(f"recon_context_coords shape: {recon_context_coords.shape}, {recon_context_coords.dtype}, {recon_context_coords.device}, {recon_context_coords.min()}, {recon_context_coords.max()}")

        decoded_results = self.scene_decoder(
            context=context,
            context_mask=context_mask,
            coords=context_coords,
            recon_context_feats=recon_context,
            recon_context_mask=recon_context_mask,
            recon_coords=recon_context_coords,
        )

        sequence_instance_ids = self.gather_sequence_instance_ids(
            voxel_coords=voxel_coords,
            voxel_instance_ids=voxel_instance_ids,
            recon_voxel_coords=recon_voxel_coords,
            recon_context_mask=recon_context_mask,
        )
        sequence_instance_ids = sequence_instance_ids.clamp(
            min=0, max=max_num_objects - 1
        )

        # Get gt_masks: [B, M, N], with padded recon positions ignored.
        one_hot_voxel_instance_ids = F.one_hot(
            sequence_instance_ids, num_classes=max_num_objects
        )

        # 2. Transpose the last two dimensions to match [B, M, N].
        # Permute is a view operation (very fast, no data copy).
        gt_masks = one_hot_voxel_instance_ids.permute(0, 2, 1).bool()
        gt_masks = gt_masks & (~recon_context_mask[:, None, :])

        supervision = split_background_supervision(
            object_translations=object_translations,
            object_angles=object_angles,
            object_scales=object_scales,
            object_valid_masks=object_valid_masks,
            gt_masks=gt_masks,
            learned_background_segmentation=
                self.learned_background_segmentation,
        )
        matched_object_translations = supervision["object_translations"]
        matched_object_angles = supervision["object_angles"]
        matched_object_scales = supervision["object_scales"]
        matched_object_valid_masks = supervision["object_valid_masks"]
        matched_gt_masks = supervision["gt_masks"]
        background_gt_mask = supervision["background_gt_mask"]
        num_objects_batch = torch.sum(
            matched_object_valid_masks, dim=-1
        )

        results = []

        for decoded_result in decoded_results:
            layer_result = {}
            pos_bin_logits = decoded_result["pos_bin_logits"]
            angle_bin_logits = decoded_result["angle_bin_logits"]
            scale_bin_logits = decoded_result["scale_bin_logits"]
            valid_logits = decoded_result["valid_logits"]
            pred_masks_logits = decoded_result["pred_masks_logits"]
            pred_masks_logits = pred_masks_logits.masked_fill(
                recon_context_mask[:, None, :], -20.0
            )
            background_pred_masks_logits = decoded_result.get(
                "background_pred_masks_logits"
            )
            if background_pred_masks_logits is not None:
                background_pred_masks_logits = (
                    background_pred_masks_logits.masked_fill(
                        recon_context_mask[:, None, :], -20.0
                    )
                )

            mapping = self.scene_decoder.mapping_preds_gts(
                pos_bin_logits=pos_bin_logits,
                angle_bin_logits=angle_bin_logits,
                scale_bin_logits=scale_bin_logits,
                tgt_pos_bins=matched_object_translations,
                tgt_angle_bins=matched_object_angles,
                tgt_scale_bins=matched_object_scales,
                valid_logits=valid_logits,
                pred_masks_logits=pred_masks_logits,
                gt_masks=matched_gt_masks,
                num_objects=num_objects_batch,
            )

            pos_bin_logits, angle_bin_logits, scale_bin_logits, valid_logits, pred_masks_logits = \
                self.scene_decoder.resort_preds_by_mapping(
                pos_bin_logits=pos_bin_logits,
                angle_bin_logits=angle_bin_logits,
                scale_bin_logits=scale_bin_logits,
                valid_logits=valid_logits,
                pred_masks_logits=pred_masks_logits,
                mapping=mapping,
            )

            layer_result["pos_bin_logits"] = pos_bin_logits
            layer_result["angle_bin_logits"] = angle_bin_logits
            layer_result["scale_bin_logits"] = scale_bin_logits
            layer_result["valid_logits"] = valid_logits
            layer_result["pred_masks_logits"] = pred_masks_logits
            if self.learned_background_segmentation:
                if background_pred_masks_logits is None:
                    raise RuntimeError(
                        "scene decoder did not return a background mask"
                    )
                layer_result["background_pred_masks_logits"] = (
                    background_pred_masks_logits
                )
                layer_result["all_pred_masks_logits"] = torch.cat(
                    [background_pred_masks_logits, pred_masks_logits], dim=1
                )

            results.append(layer_result)

        gt_dict = {
            "gt_pos_bins": matched_object_translations,
            "gt_angle_bins": matched_object_angles,
            "gt_scale_bins": matched_object_scales,
            "gt_masks": matched_gt_masks,
            "object_valid_masks": matched_object_valid_masks,
            "max_num_objects": matched_object_translations.shape[1],
            "voxel_inverse_indices": voxel_inverse_indices,
        }
        if self.learned_background_segmentation:
            gt_dict["background_gt_mask"] = background_gt_mask
            gt_dict["all_gt_masks"] = torch.cat(
                [background_gt_mask, matched_gt_masks], dim=1
            )

        if return_feats:
            if point_valid_masks is None:
                gt_dict["input_points"] = points[0].detach().clone()
                gt_dict["input_feats"] = dino_feats[0].detach().clone()
            else:
                first_valid = point_valid_masks[0].bool()
                gt_dict["input_points"] = points[0, first_valid].detach().clone()
                gt_dict["input_feats"] = dino_feats[0, first_valid].detach().clone()

        return results, gt_dict

    @torch.no_grad()
    def get_voxelizer_inference(self):
        self.voxelize_inference = Voxelize(
            voxel_size=self.scene_scale / self.resolution,
            resolution=self.resolution,
            apply_standardization=False
        )

    @torch.no_grad()
    def forward_inference(
        self,
        points,
        colors,
    ):
        """
        Args:
            points: (B, N, 3)
            colors: (B, N, C)
        """
        B = points.shape[0]
        device = points.device

        assert B == 1, "Batch size must be 1 for ObjectPoseWSegVoxelize"

        voxel_coords, voxel_feats, voxel_inverse_indices = self.voxelize_inference(
            points=points,
            colors=colors,
            return_inverse_indices=True,
        ) # [N_total, 4], [N_total, dino_feat_dim], int

        outputs_dict = self.point_unet(
            voxel_coords=voxel_coords,
            voxel_feats=voxel_feats,
            B=B,
        )

        context = outputs_dict["context"]
        context_mask = outputs_dict["context_mask"]
        context_coords = outputs_dict["context_coords"]
        context_coords_normalised = outputs_dict["context_coords_normalised"]

        recon_context = outputs_dict["recon_context"]
        recon_context_mask = outputs_dict["recon_context_mask"]
        recon_context_coords = outputs_dict["recon_context_coords"]
        recon_context_coords_normalised = outputs_dict["recon_context_coords_normalised"]

        decoded_results, context_memory, encoded_seg_context_feats = self.scene_decoder(
            context=context,
            context_mask=context_mask,
            coords=context_coords,
            recon_context_feats=recon_context,
            recon_context_mask=recon_context_mask,
            recon_coords=recon_context_coords,
            return_intermediate_feats=True,
        )

        last_layer_decoded_result = decoded_results[-1]

        pos_bin_logits = last_layer_decoded_result["pos_bin_logits"]
        angle_bin_logits = last_layer_decoded_result["angle_bin_logits"]
        scale_bin_logits = last_layer_decoded_result["scale_bin_logits"]
        valid_logits = last_layer_decoded_result["valid_logits"]
        pred_masks_logits = last_layer_decoded_result["pred_masks_logits"]
        seg_feats = last_layer_decoded_result["seg_feats"]

        context_coords_denormalised = context_coords_normalised * self.scene_scale

        results_dict = {
            "pos_bin_logits": pos_bin_logits,
            "angle_bin_logits": angle_bin_logits,
            "scale_bin_logits": scale_bin_logits,
            "valid_logits": valid_logits,
            "pred_masks_logits": pred_masks_logits,
            "voxel_inverse_indices": voxel_inverse_indices,
            "encoded_seg_context_feats": encoded_seg_context_feats,
            "context_memory": context_memory,
            "context_coords": context_coords_denormalised,
            "seg_feats": seg_feats,
        }
        if self.learned_background_segmentation:
            background_pred_masks_logits = last_layer_decoded_result[
                "background_pred_masks_logits"
            ]
            background_seg_feats = last_layer_decoded_result[
                "background_seg_feats"
            ]
            results_dict.update({
                "background_pred_masks_logits":
                    background_pred_masks_logits,
                "background_seg_feats": background_seg_feats,
                "all_pred_masks_logits": torch.cat(
                    [background_pred_masks_logits, pred_masks_logits], dim=1
                ),
                "all_seg_feats": torch.cat(
                    [background_seg_feats, seg_feats], dim=1
                ),
            })

        return results_dict
