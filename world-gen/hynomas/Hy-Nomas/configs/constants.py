__all__ = [
    "PROMPT_TEMPLATE",
]

# =================== Prompt Templates =====================
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE = {
    "li-dit-encode": {"template": PROMPT_TEMPLATE_ENCODE, "crop_start": 36},
    "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO, "crop_start": 95},
    "li-dit-encode-image-json": {
        "template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON,
        "crop_start": -1,
    },
    "li-dit-encode-video-json": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON,
        "crop_start": -1,
    },
}
