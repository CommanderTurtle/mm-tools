dependencies = ["torch", "natten"]

from src.model.naf import NAF


def naf(pretrained: bool = False, device="cpu"):
    """
    NAF (Neighborhood Attention Filtering) model for feature upsampling.
    VFM-agnostic upsampler that works with any Vision Foundation Model without retraining.

    Dependencies:
        - torch: PyTorch framework
        - natten: Neighborhood Attention Extension (required for cross-scale attention)

    """
    model = NAF().to(device)
    if pretrained:
        raise RuntimeError("Load the project-local NAF checkpoint after constructing the model")
    return model
