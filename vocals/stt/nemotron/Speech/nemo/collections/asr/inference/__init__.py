# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# SimulStream integration is an optional dependency (`pip install simulstream`); the adapter module
# guards its own `import simulstream` internally, so importing it here always succeeds.
from nemo.collections.asr.inference.utils.simulstream_manifest_utils import (
    load_manifest_audio_paths,
    manifest_to_audio_definition,
)
from nemo.collections.asr.inference.utils.simulstream_pipeline_adapter import NeMoStreamingPipelineAdapter

__all__ = [
    'NeMoStreamingPipelineAdapter',
    'load_manifest_audio_paths',
    'manifest_to_audio_definition',
]
