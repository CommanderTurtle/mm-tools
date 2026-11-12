# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""EasyMagpieTTS model and pipeline definitions for vLLM-Omni."""

from easymagpie_vllm_omni.config import EASYMAGPIE_SMALLMAMBA, EasyMagpieOmniArch

__all__ = ["EASYMAGPIE_SMALLMAMBA", "EasyMagpieOmniArch"]


def __getattr__(name: str):
    # Lazily expose pipeline configs without importing vllm_omni (a heavy
    # dependency) at package import time.
    if name == "EASYMAGPIE_PIPELINE":
        from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE

        return EASYMAGPIE_PIPELINE
    if name == "EASYMAGPIE_LM_PIPELINE":
        from easymagpie_vllm_omni.pipeline import EASYMAGPIE_LM_PIPELINE

        return EASYMAGPIE_LM_PIPELINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
