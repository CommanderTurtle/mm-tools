"""Template utilities for Jinja template processing.

This module provides utilities for analyzing and processing Jinja chat templates,
including content format detection and message processing.
"""

import logging

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils

from sglang.srt.utils import ImageData

logger = logging.getLogger(__name__)

# ============================================================================
# JINJA TEMPLATE CONTENT FORMAT DETECTION
# ============================================================================
#
# This adapts vLLM's approach for detecting chat template content format:
# https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/entrypoints/chat_utils.py#L296-L313
# - Analyzes Jinja template AST to detect content iteration patterns
# - 'openai' format: templates with {%- for content in message['content'] -%} loops
# - 'string' format: templates that expect simple string content
# - Processes content accordingly to match template expectations


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    """Check if node is a variable access like {{ varname }}"""
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    """Check if node is an attribute access like {{ varname['key'] }} or {{ varname.key }}"""
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str = None,
) -> bool:
    """Check if node accesses varname or varname[key] with filters/tests"""
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _try_extract_ast(chat_template: str):
    """Try to parse the Jinja template into an AST"""
    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception as e:
        logger.debug(f"Error when compiling Jinja template: {e}")
        return None


def detect_jinja_template_content_format(chat_template: str) -> str:
    """
    Detect whether a chat template expects 'string' or 'openai' content format.

    - 'string': content is a simple string (like DeepSeek templates)
    - 'openai': content is a list of structured dicts (like Llama4 templates)

    Detection logic:
    - If template has loops like {%- for content in message['content'] -%} → 'openai'
    - Otherwise → 'string'
    """
    # Shortcut for multimodal templates
    if any(
        keyword in chat_template for keyword in ["image", "audio", "video", "vision"]
    ):
        return "openai"

    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return "string"

    try:
        # Look for patterns like: {%- for content in message['content'] -%}
        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):
            loop_iter = loop_ast.iter

            # Check if iterating over message['content'] or similar
            if _is_var_or_elems_access(loop_iter, "message", "content"):
                return "openai"  # Found content iteration → openai format

            # Also check for patterns like: {%- for item in msg.content -%} or {%- for item in m.content -%}
            if _is_var_or_elems_access(
                loop_iter, "msg", "content"
            ) or _is_var_or_elems_access(loop_iter, "m", "content"):
                return "openai"  # Found content iteration → openai format (glm4v)

        return "string"  # No content loops found → string format
    except Exception as e:
        logger.debug(f"Error when parsing AST of Jinja template: {e}")
        return "string"


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
    use_dpsk_v32_encoding: bool = False,
    audio_placeholder: str | None = None,
) -> dict:
    """
    Process message content based on detected template format.

    Args:
        msg_dict: Message dictionary with content
        content_format: 'string' or 'openai' (detected via AST analysis)
        image_data: List to append extracted image URLs
        video_data: List to append extracted video URLs
        audio_data: List to append extracted audio URLs
        modalities: List to append modalities
        use_dpsk_v32_encoding: If True, extract multimodal data and convert content to string (for DeepSeek-V3.2 encoding)

    Returns:
        Processed message dictionary
    """
    if not isinstance(msg_dict.get("content"), list):
        # Already a string or None, no processing needed
        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format == "openai" or use_dpsk_v32_encoding:
        # OpenAI format: preserve structured content list, normalize types
        # V32 encoding: extract multimodal data but convert content to string
        processed_content_parts = []
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type in ("image_url", "input_image"):
                    image_obj = chunk.get("image_url") or {}
                    if isinstance(image_obj, str):
                        image_obj = {"url": image_obj, "detail": chunk.get("detail")}
                    mdp = image_obj.get("max_dynamic_patch", None)
                    # Also allow flat style: chunk["max_dynamic_patch"]
                    image_data.append(
                        ImageData(
                            url=image_obj["url"],
                            detail=image_obj.get("detail") or "auto",
                            max_dynamic_patch=mdp,
                        )
                    )

                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'image' type for template compatibility
                    processed_content_parts.append({"type": "image"})
                elif chunk_type in ("video_url", "video_audio_url"):
                    video_obj = chunk.get("video_url") or {}
                    if isinstance(video_obj, str):
                        video_obj = {"url": video_obj}
                    paired_audio = chunk.get("audio_url")
                    if isinstance(paired_audio, dict):
                        paired_audio = paired_audio.get("url")
                    if not video_obj.get("url"):
                        if chunk_type != "video_audio_url" or not paired_audio:
                            raise ValueError("Video part requires a video URL")
                        audio_data.append(paired_audio)
                        processed_content_parts.append(
                            {"type": "text", "text": audio_placeholder}
                            if audio_placeholder is not None else {"type": "audio"})
                        continue
                    video_options = {key: video_obj[key] for key in (
                        "max_dynamic_patch", "min_dynamic_patch", "fps", "num_frames",
                        "nframes",
                        "min_frames", "max_frames", "total_pixels", "min_pixels",
                        "max_pixels", "use_audio", "audio",
                    ) if video_obj.get(key) is not None}
                    if paired_audio:
                        # Same-part audio rides the video modality, preserving
                        # the interleaved WeMM video/audio token layout.
                        video_options["audio"] = paired_audio
                    if chunk.get("uuid") is not None:
                        video_options["uuid"] = str(chunk["uuid"])
                    video_data.append({"url": video_obj["url"], **video_options}
                                      if video_options else video_obj["url"])
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'video' type for template compatibility
                    processed_content_parts.append({"type": "video"})
                elif chunk_type == "audio_url":
                    audio_options = chunk["audio_url"]
                    options = {}
                    if audio_placeholder is not None:
                        value = chunk.get("timestamp_interval_sec")
                        if value is None:
                            value = audio_options.get("timestamp_interval_sec")
                        if value is not None:
                            options["timestamp_interval_sec"] = value
                    audio_data.append({"url": audio_options["url"], **options}
                                      if options else audio_options["url"])
                    # Normalize to simple 'audio' type
                    processed_content_parts.append(
                        {"type": "text", "text": audio_placeholder}
                        if audio_placeholder is not None else {"type": "audio"}
                    )
                elif chunk_type in ("text", "input_text"):
                    # For v32 encoding, collect text parts separately
                    if use_dpsk_v32_encoding:
                        text_parts.append(chunk["text"])
                    else:
                        # Keep text content as-is for openai format
                        processed_content_parts.append(
                            {"type": "text", "text": chunk["text"]}
                        )
                elif chunk_type == "tool_reference":
                    # GLM-specific extension: pass through so the chat template
                    # can match tool_reference.name against tools[*].function.name
                    # and render the referenced tool schemas inline.
                    processed_content_parts.append(chunk)

        new_msg = {
            k: v for k, v in msg_dict.items() if v is not None and k != "content"
        }
        if use_dpsk_v32_encoding:
            new_msg["content"] = " ".join(text_parts) if text_parts else ""
        else:
            new_msg["content"] = processed_content_parts
        return new_msg

    elif content_format == "string":
        # String format: flatten to text only (for templates like DeepSeek)
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") in ("text", "input_text"):
                text_parts.append(chunk["text"])
            # Note: For string format, we ignore images/audio since the template
            # doesn't expect structured content - multimodal placeholders would
            # need to be inserted differently

        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg

    else:
        raise ValueError(f"Invalid content format: {content_format}")
