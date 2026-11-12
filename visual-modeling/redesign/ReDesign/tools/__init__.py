# ReDesign/tools/__init__.py
"""Lazy exports for ReDesign's independent local tools."""

from importlib import import_module


_TOOLS = {
    "run_ocr": ("ocr_tool", "run_ocr"),
    "run_dino_batch_all": ("dino_tool", "run_dino_batch_all"),
    "run_hisam_union": ("hisam_tool", "run_hisam_union"),
    "run_sam2_union": ("sam2_tool", "run_sam2_union"),
    "run_lama": ("lama_tool", "run_lama"),
    "run_objectclear": ("objectclear_tool", "run_objectclear"),
    "run_nanobanana": ("nanobanana_tool", "run_nanobanana"),
    "run_vtracer": ("vtracer_tool", "run_vtracer"),
    "run_split_cca": ("cca_tool", "run_split_cca"),
    "analyze_separability": ("cca_tool", "analyze_separability"),
    "run_qwen_layered": ("qwen_layered_tool", "run_qwen_layered"),
}

__all__ = list(_TOOLS)


def __getattr__(name):
    try:
        module_name, attribute = _TOOLS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
