# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class NeMoBaseException(Exception):
    """NeMo Base Exception. All exceptions created in NeMo should inherit from this class"""


class LightningNotInstalledException(NeMoBaseException):
    """Raised when optional Lightning dependencies are required but unavailable."""

    def __init__(self, obj):
        message = (
            f" You are trying to use {obj} without installing all of pytorch_lightning, hydra, and "
            f"omegaconf. Please install those packages before trying to access {obj}."
        )
        super().__init__(message)


class CheckInstall:
    """Placeholder that raises when optional Lightning dependencies are missing."""

    def __init__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __call__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __getattr__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)
