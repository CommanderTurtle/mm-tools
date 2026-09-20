import importlib
from omegaconf import OmegaConf
from .base_inference_pipeline import BaseInferencePipeline

def build_object_from_dict(config):
    class_path = config.pop("type")
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    obj = cls(**config)
    
    return obj


def build_object_from_config_file(config_path):
    config = OmegaConf.load(config_path)
    class_path = config.pop("type")
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    obj = cls(config)
    return obj
