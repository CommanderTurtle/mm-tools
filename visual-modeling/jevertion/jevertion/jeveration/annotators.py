"""Optional (--from-photo): the structure maps of the hed and depth tasks, computed from a photo.

    hed     soft edges of ControlNet's HED annotator, white on black
    depth   relative inverse depth of Depth Anything V2 Small, bright = near

Both networks are downloaded from the Hugging Face Hub under their own licenses on first use.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class _HEDBlock(torch.nn.Module):
    def __init__(self, cin, cout, n):
        super().__init__()
        self.convs = torch.nn.Sequential(*([torch.nn.Conv2d(cin, cout, 3, 1, 1)] + [torch.nn.Conv2d(cout, cout, 3, 1, 1) for _ in range(n - 1)]))
        self.projection = torch.nn.Conv2d(cout, 1, 1, 1, 0)

    def forward(self, x, down=False):
        h = F.max_pool2d(x, 2, 2) if down else x
        for c in self.convs:
            h = F.relu(c(h))
        return h, self.projection(h)


class _ControlNetHED(torch.nn.Module):
    """the HED network of ControlNet 1.1's annotator: five conv stages, each with a 1 x 1 side output"""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1, self.block2, self.block3 = _HEDBlock(3, 64, 2), _HEDBlock(64, 128, 2), _HEDBlock(128, 256, 3)
        self.block4, self.block5 = _HEDBlock(256, 512, 3), _HEDBlock(512, 512, 3)

    def forward(self, x):
        h, ps = x - self.norm, []
        for k, b in enumerate((self.block1, self.block2, self.block3, self.block4, self.block5)):
            h, p = b(h, down=k > 0)
            ps.append(p)
        return ps


@torch.no_grad()
def hed_map(img, device="cuda"):
    """uint8 [H, W, 3] photo -> uint8 [H, W, 3] soft edge map (mean of the five side outputs, sigmoid)"""
    from huggingface_hub import hf_hub_download
    net = _ControlNetHED()
    net.load_state_dict(torch.load(hf_hub_download("lllyasviel/Annotators", "ControlNetHED.pth"), map_location="cpu"), strict=True)
    net = net.to(device).eval()
    h, w = img.shape[:2]
    ps = net(torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(device))
    m = torch.stack([F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0] for p in ps]).mean(0)
    e = torch.sigmoid(m).cpu().numpy()
    return np.repeat((e * 255).round().astype(np.uint8)[..., None], 3, axis=2)


@torch.no_grad()
def depth_map(img, device="cuda"):
    """uint8 [H, W, 3] photo -> uint8 [H, W, 3] depth map, bright = near"""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Small-hf"
    proc = AutoImageProcessor.from_pretrained(name)
    net = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
    o = net(**{k: v.to(device) for k, v in proc(images=Image.fromarray(img), return_tensors="pt").items()}).predicted_depth
    d = F.interpolate(o[None], size=img.shape[:2], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
    return np.repeat((d * 255).round().astype(np.uint8)[..., None], 3, axis=2)


def structure_map(img, task, device="cuda"):
    return {"hed": hed_map, "depth": depth_map}[task](img, device)
