"""Small student encoders that replace the DINOv3-H+ teacher backbone.

A student maps a 224 (DINOv2/14) or 256 (DINOv3/16) right-hand crop (same region as the teacher's crop) to a (B, 1280, 16, 16)
feature map that the (frozen or fine-tuned) SAM 3D Body hand decoder consumes in place of the
teacher's 512x512 features avg-pooled to 16x16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STUDENTS = {
    # timm name, input size, patch size, embed dim
    "dinov3_vits16": ("vit_small_patch16_dinov3.lvd1689m", 256, 16, 384),
    "dinov3_vits16plus": ("vit_small_plus_patch16_dinov3.lvd1689m", 256, 16, 384),
    "dinov3_vitb16": ("vit_base_patch16_dinov3.lvd1689m", 256, 16, 768),
    "dinov3_vitl16": ("vit_large_patch16_dinov3.lvd1689m", 256, 16, 1024),
    "dinov3_vith16plus": ("vit_huge_plus_patch16_dinov3.lvd1689m", 256, 16, 1280),  # = the teacher's architecture
    "dinov2_vits14_reg": ("vit_small_patch14_reg4_dinov2.lvd142m", 224, 14, 384),
    "dinov2_vitb14_reg": ("vit_base_patch14_reg4_dinov2.lvd142m", 224, 14, 768),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StudentEncoder(nn.Module):
    def __init__(self, name="dinov3_vits16", out_dim=1280, pretrained=True, grid=16, size=None):
        super().__init__()
        import timm

        timm_name, self.input_size, self.patch, dim = STUDENTS[name]
        if size:  # e.g. 320 for small / egocentric hands: grid follows (20 x 20 tokens for patch 16)
            self.input_size, grid = size, size // self.patch
        self.name, self.grid, self.out_dim = name, grid, out_dim
        # fixed input size: position embeddings / RoPE are precomputed (no per-call CPU work, CUDA-graph safe)
        self.vit = timm.create_model(timm_name, pretrained=pretrained, img_size=self.input_size, num_classes=0,
                                     dynamic_img_size=False)
        self.num_prefix = self.vit.num_prefix_tokens
        self.adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), False)

    def forward(self, x):
        """x: (B, 3, S, S) RGB in [0, 1] at self.input_size -> (B, out_dim, grid, grid)."""
        x = (x - self.mean) / self.std
        tokens = self.vit.forward_features(x)[:, self.num_prefix:]
        B, N, C = tokens.shape
        g = int(round(N ** 0.5))
        feat = self.adapter(tokens).transpose(1, 2).reshape(B, self.out_dim, g, g)
        if g != self.grid:
            feat = F.interpolate(feat, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return feat


class TRTEncoder:
    """TensorRT engine of a StudentEncoder (scripts/build_pose_engine.py): x (B, 3, S, S) fp32 -> (B, out_dim, grid, grid)
    fp32 on the current torch stream (so it can sit inside a captured CUDA graph); one persistent output buffer per batch
    size (consume it before the next call with that size). Engines with a "hidden<N>" output: also last_hidden."""

    def __init__(self, path):
        import tensorrt as trt

        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.bufs = {}
        # engines built for a contact branch (build_pose_engine.py --hidden_block N) also give the encoder's hidden state
        # at block N (every token, before that block): output "hidden<N>"
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.hidden_name = next((n for n in names if n.startswith("hidden")), None)
        self.hidden_block = int(self.hidden_name[6:]) if self.hidden_name else None
        self.hbufs, self.last_hidden = {}, None

    def __call__(self, x):
        x = x.float().contiguous()
        B = x.shape[0]
        self.ctx.set_input_shape("x", tuple(x.shape))
        if B not in self.bufs:
            self.bufs[B] = torch.empty(tuple(self.ctx.get_tensor_shape("feat")), dtype=torch.float32, device=x.device)
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("feat", self.bufs[B].data_ptr())
        if self.hidden_name:
            if B not in self.hbufs:
                self.hbufs[B] = torch.empty(tuple(self.ctx.get_tensor_shape(self.hidden_name)), dtype=torch.float32, device=x.device)
            self.ctx.set_tensor_address(self.hidden_name, self.hbufs[B].data_ptr())
            self.last_hidden = self.hbufs[B]
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.bufs[B]


def teacher_target(teacher, img, grid=16):
    """Teacher features for (B, 3, S, S) RGB [0, 1] crops, area-pooled to grid x grid. The frozen
    decoder gives the same FreiHAND error on 512 -> 16 and 384 -> 16 pooled features (5.07 mm)."""
    x = (img - teacher.image_mean) / teacher.image_std
    f = teacher.backbone(x.type(teacher.backbone_dtype))
    if isinstance(f, tuple):
        f = f[-1]
    return F.adaptive_avg_pool2d(f.float(), grid)
