# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inworld Voice Isolator: the public API of inworld.voice_isolation.

    from inworld import voice_isolation
    isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator")
    voice = isolator.isolate(wav, sample_rate)
"""

from inworld.voice_isolation.audio import load_audio
from inworld.voice_isolation.audio import resample
from inworld.voice_isolation.audio import save_audio
from inworld.voice_isolation.audio import SR
from inworld.voice_isolation.isolator import load_model
from inworld.voice_isolation.isolator import VoiceIsolator
from inworld.voice_isolation.model import VoiceIsolatorConfig
from inworld.voice_isolation.model import VoiceIsolatorNetwork
from inworld.voice_isolation.registry import MODELS
from inworld.voice_isolation.streaming import Stream

__version__ = "1.0.0"

__all__ = [
    "SR",
    "MODELS",
    "VoiceIsolator",
    "VoiceIsolatorConfig",
    "VoiceIsolatorNetwork",
    "Stream",
    "load_audio",
    "load_model",
    "resample",
    "save_audio",
]
