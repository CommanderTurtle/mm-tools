import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
    "Trellis2TexturingPipeline": "trellis2_texturing",
    "Pixal3DImageTo3DPipeline": "pixal3d_image_to_3d",
}


__submodules = ['samplers', 'rembg']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):
    """
    Load a pipeline from a project-local model folder.

    Args:
        path: The project-local path to the model.
    """
    import os
    import json
    config_file = os.path.join(path, "pipeline.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Missing project-local pipeline config: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)


# For PyLance
if __name__ == '__main__':
    from . import samplers, rembg
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_texturing import Trellis2TexturingPipeline
    from .pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
