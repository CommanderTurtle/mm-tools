import torch
import yaml
from models.object_pose_w_seg_voxelize_dino import ObjectPoseWSegVoxelize

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_obj_pose_w_seg_voxelize(model_path, config):
    model = ObjectPoseWSegVoxelize(config['model'])
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    model_state_dict = ckpt["model"]
    incompatible = model.load_state_dict(model_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Perception checkpoint compatibility: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model
