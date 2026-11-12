from hymm.utils.inference_metrics import profiled
from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    AutoModel,
    LlavaForConditionalGeneration,
    LlamaTokenizerFast,
    CLIPImageProcessor,
)
from transformers.utils import ModelOutput

from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.model_cpu_offload import cpu_offloaded


def use_default(value, default):
    return value if value is not None else default


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if not text_encoder_path:
        raise ValueError("Set model.text_encoder.model_name to a model ID or local directory")
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llava-llama-3-8b-all":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True
        )
    else:
        import transformers as _tf

        _prev_verbosity = _tf.logging.get_verbosity()
        _tf.logging.set_verbosity_error()
        try:
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path, low_cpu_mem_usage=True
            )
        finally:
            _tf.logging.set_verbosity(_prev_verbosity)

        if hasattr(text_encoder, "language_model"):
            text_encoder = text_encoder.language_model
        text_encoder.final_layer_norm = text_encoder.norm

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    processor = None
    if not tokenizer_path:
        raise ValueError("Set model.text_encoder.model_name to a model ID or local directory")
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llava-llama-3-8b-all":
        tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
        processor = CLIPImageProcessor.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None
    image_features: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert use_attention_mask is True, (
                "Attention mask is True required when training videos."
            )
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), (
                f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            )
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), (
                    f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
                )
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif (
            "llm" in text_encoder_type
            or "glm" in text_encoder_type
            or "llava" in text_encoder_type
            or "qwen" in text_encoder_type
            or "gemma" in text_encoder_type
        ):
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

        # Pre-calculate crop_start for image and video
        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
            self.logger.info(
                f"crop_start for image: {self.prompt_template['crop_start']}"
            )
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")
            self.logger.info(
                f"crop_start for video: {self.prompt_template_video['crop_start']}"
            )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If True, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(
                        text if text else (" " if prevent_empty_text else "")
                    )
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input):
        """
        Automatically calculate the crop_start position based on identifying user tokens.
        """
        input_ids = tokenized_input["input_ids"][0].tolist()

        user_markers = {
            "llava": "<|start_header_id|>user<|end_header_id|>\n\n",
            "qwen": "<|im_start|>user\n",
        }

        marker = None
        for model_type, token_marker in user_markers.items():
            if model_type in self.text_encoder_type.lower():
                marker = token_marker
                break

        if marker is None:
            return 0

        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]

        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens)

        if hasattr(self.tokenizer, "special_tokens_map"):
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if "user" in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1

        return 0

    def text2tokens(self, text, data_type="image", max_length=300):
        """
        Tokenize the input text.
        """
        tokenize_input_type = "str"
        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template)
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

            if crop_start == -1:
                temp_kwargs = dict(
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt",
                )

                if tokenize_input_type == "str":
                    temp_tokenized = self.tokenizer(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = self.tokenizer.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )

                crop_start = self.calculate_crop_start(temp_tokenized)

                if data_type == "image":
                    self.prompt_template["crop_start"] = crop_start
                else:
                    self.prompt_template_video["crop_start"] = crop_start
        else:
            crop_start = 0

        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )

        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

        return tokenized_output

    @cpu_offloaded
    @profiled("text_encode")
    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        device=None,
        semantic_images=None,
        is_uncond=False,
    ):
        if device is None:
            device = getattr(self, "_fsdp_cpu_offload_device", self.model.device)
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)

        if self.text_encoder_type != "llava-llama-3-8b-all":
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", 0)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", 0)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = (
                        attention_mask[:, crop_start:] if use_attention_mask else None
                    )

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state, attention_mask, outputs.hidden_states
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)
        else:
            if semantic_images is not None:
                image_outputs = self.processor(semantic_images, return_tensors="pt")[
                    "pixel_values"
                ].to(self.model.device)
            else:
                image_outputs = None

            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
                pixel_values=image_outputs,
            )
            image_features = outputs.image_hidden_states
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", -1)
                    crop_end = self.prompt_template_video.get("assistant_emb_start", -1)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = (
                        crop_start
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    image_crop_start = self.prompt_template_video.get(
                        "image_emb_start", 5
                    )
                    image_crop_end = self.prompt_template_video.get(
                        "image_emb_end", 581
                    )
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"]
                        == self.prompt_template_video.get("double_return_token_id", 271)
                    )
                    last_double_return_token_indices = (
                        last_double_return_token_indices.reshape(
                            batch_encoding["input_ids"].shape[0], -1
                        )[:, -1]
                    )
                    batch_indices = batch_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1
                    )[:, -1]
                    assistant_crop_start = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                        - 4
                    )
                    assistant_crop_end = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )

                    attention_mask_assistant_crop_start = (
                        last_double_return_token_indices - 4
                    )
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                text_last_hidden_state = []
                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(
                        torch.cat(
                            [
                                last_hidden_state[
                                    i, text_crop_start : assistant_crop_start[i].item()
                                ],
                                last_hidden_state[i, assistant_crop_end[i].item() :],
                            ]
                        )
                    )
                    text_attention_mask.append(
                        torch.cat(
                            [
                                attention_mask[
                                    i,
                                    crop_start : attention_mask_assistant_crop_start[
                                        i
                                    ].item(),
                                ],
                                attention_mask[
                                    i, attention_mask_assistant_crop_end[i].item() :
                                ],
                            ]
                        )
                        if use_attention_mask
                        else None
                    )
                    image_last_hidden_state.append(
                        last_hidden_state[i, image_crop_start:image_crop_end]
                    )
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0])
                        .to(last_hidden_state.device)
                        .to(attention_mask.dtype)
                        if use_attention_mask
                        else None
                    )

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                assert (
                    text_last_hidden_state.shape[0] == text_attention_mask.shape[0]
                    and image_last_hidden_state.shape[0]
                    == image_attention_mask.shape[0]
                )

                last_hidden_state = torch.cat(
                    [image_last_hidden_state, text_last_hidden_state], dim=1
                )
                attention_mask = torch.cat(
                    [image_attention_mask, text_attention_mask], dim=1
                )

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state,
                    attention_mask,
                    image_features=image_features,
                    hidden_states_list=outputs.hidden_states,
                )
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, image_features=image_features
            )

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
