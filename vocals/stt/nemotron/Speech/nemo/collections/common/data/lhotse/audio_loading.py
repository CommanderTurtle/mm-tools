# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Shared audio-loading policy helpers for Lhotse-backed NeMo datasets."""

from copy import copy


class LhotseAudioLoadingDatasetMixin:
    """Make a dataset's ``AudioSamples`` failure policy loader-configurable.

    Lhotse returns the surviving cuts as a third value only in fault-tolerant
    mode. ``load_audio_with_cuts`` preserves that stable three-value contract
    for NeMo datasets while allowing strict mode to propagate the original
    audio-loading exception.
    """

    def with_fault_tolerant_audio_loading(self, enabled: bool):
        """Return a shallow dataset copy with an independent audio loader."""
        enabled = bool(enabled)
        dataset = copy(self)
        dataset.load_audio = copy(self.load_audio)
        dataset.load_audio.fault_tolerant = enabled
        if dataset.load_audio.ais_batch_loader is not None:
            dataset.load_audio.ais_batch_loader = copy(dataset.load_audio.ais_batch_loader)
            dataset.load_audio.ais_batch_loader.skip_failed_fetches = enabled
        return dataset

    def load_audio_with_cuts(self, cuts):
        """Load audio and always return ``(audio, lengths, surviving_cuts)``."""
        loaded = self.load_audio(cuts)
        if self.load_audio.fault_tolerant:
            return loaded
        audio, audio_lens = loaded
        return audio, audio_lens, cuts


def configure_dataset_audio_loading(dataset, enabled: bool):
    """Apply a loader-level audio policy when the dataset exposes the protocol."""
    configure = getattr(dataset, "with_fault_tolerant_audio_loading", None)
    return configure(enabled) if callable(configure) else dataset
