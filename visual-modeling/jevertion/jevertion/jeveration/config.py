"""Fixed settings of the released Jeveration checkpoint."""

# ---- public models the checkpoint builds on: (Hugging Face repository, revision) --------------------------------------------
BACKBONE = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
TATITOK = ("turkeyju/tokenizer_tatitok_bl128_vq", "7650d466a955c754265d20e57513518a4e049f00")
CLIP_TEXT = ("timm/vit_large_patch14_clip_336.openai", "81e38efc4637de5023b10e75a7f9bd1c6fa6b010")
CLIP_ARCH = "ViT-L-14-336"

# ---- image decisions ---------------------------------------------------------------------------------------------------------
NUM_CODES = 128            # decision tokens = codes per image
CODE_DIM = 64              # width of a codebook entry
CODEBOOK_SIZE = 8192
MASK_PROB = 0.6            # share of the random state's positions that are masked
STATE_NOISE = 0.1          # Gaussian noise on the random state
IMAGE_SIZE = 512           # side of the output image and of an image-to-image source

# ---- adapters and heads ------------------------------------------------------------------------------------------------------
LORA_RANK = 256
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",          # the full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",          # the linear-attention layers
    "gate_proj", "up_proj", "down_proj",             # the MLP of every layer
)
HEAD_WIDTH, HEAD_DEPTH, HEAD_HEADS = 2048, 8, 16     # transformer image head
COND_DIM = 512                                       # width of the head's condition vector
TIME_DIM = 256                                       # sinusoidal features of the masked share

# ---- linear-attention kernels ----------------------------------------------------------------------------------------------
# One Triton configuration per flash-linear-attention kernel (kernels.py): a seed then gives the same result in every
# process.
LINEAR_ATTENTION_KERNELS = {
    "fla.modules.fused_norm_gate.layer_norm_gated_fwd_kernel": ({"BT": 16}, 8, 3),
    "fla.modules.l2norm.l2norm_fwd_kernel": ({"BT": 16}, 8, 3),
    "fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ({"BV": 32}, 4, 3),
    "fla.ops.common.chunk_o.chunk_fwd_kernel_o": ({"BK": 32, "BV": 32}, 2, 3),
    "fla.ops.gated_delta_rule.chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel": ({"BK": 64}, 2, 3),
    "fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd_kernel": ({}, 4, 3),
    "fla.ops.utils.cumsum.chunk_local_cumsum_scalar_kernel": ({}, 2, 3),
}

# ---- prompts -----------------------------------------------------------------------------------------------------------------
MAX_PROMPT_TOKENS = 128                              # a prompt / instruction is cut to this many tokens
T2I_TEMPLATE = "Generate an image: {prompt}"         # user turn of a text-to-image request

# Image-to-image: three wordings per task, the first is the default. {cap} is a caption of the wanted image.
I2I_INSTRUCTIONS = {
    "hed": ("Generate an image that follows this soft edge map. {cap}",
            "Render a picture with the structure of these soft edges. {cap}",
            "Create an image whose contours match this HED edge map. {cap}"),
    "depth": ("Generate an image with this depth map. {cap}",
              "Render a picture whose depth layout matches this depth map. {cap}",
              "Create an image that follows this depth map (bright is near). {cap}"),
    "colorize": ("Colorize this black-and-white photo.",
                 "Add natural colors to this grayscale image.",
                 "Turn this grayscale picture into a color image."),
}
# Style transfer: one instruction per style; the generator reads it behind STYLE_PREFIX.
STYLE_PREFIX = "Edit the image: "
STYLE_INSTRUCTIONS = {
    "anime": "Convert this image into a clean modern anime illustration, keep the same composition.",
    "ghibli": "Transform this photo into a Studio Ghibli style anime illustration, keep the same composition and subjects.",
    "watercolor": "Repaint this entire image as a loose watercolor painting on paper: soft washes, bleeding edges, visible paper "
                  "texture, light pencil lines. Keep the same composition, people and objects in the same places.",
}
I2I_TASKS = tuple(I2I_INSTRUCTIONS) + tuple(STYLE_INSTRUCTIONS)
# Task ids of the source decoder.
DECODER_TASK_ID = {"hed": 1, "depth": 2, "colorize": 3, "anime": 4, "ghibli": 4, "watercolor": 4}

# ---- dense prediction --------------------------------------------------------------------------------------------------------
DENSE_INPUT_HW = (576, 768)                          # the picture is resized to this for the vision tower
DENSE_CELL = 32                                      # pixels per spatial query: 18 x 24 queries
DENSE_SUB = 8                                        # output samples per query and axis
DENSE_HEAD_WIDTH = 1024
NYU13 = ("bed", "books", "ceiling", "chair", "floor", "furniture", "objects", "picture", "sofa", "table", "tv", "wall", "window")
DENSE_INSTRUCTIONS = {
    "seg": "Segment the image: give every pixel one of these classes: " + ", ".join(NYU13) + ".",
    "depth": "Estimate the depth of every pixel of the image, in meters.",
    "normal": "Estimate the surface normal of every pixel of the image.",
}
DENSE_TASKS = tuple(DENSE_INSTRUCTIONS)
DENSE_CHANNELS = {"seg": len(NYU13), "depth": 1, "normal": 3}
