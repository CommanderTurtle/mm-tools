# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def read_config(config_path: str, config_name: str) -> DictConfig:
    """
    Read configuration file
    Args:
        config_path: (str) Absolute path to the configuration file
        config_name: (str) Name of the configuration file
    Returns:
        (DictConfig) Configuration object
    """

    rel_to = os.path.dirname(os.path.realpath(__file__))

    # Reset the global Hydra instance if already initialized to prevent duplicate initialization errors
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=os.path.relpath(config_path, rel_to)):
        cfg = compose(config_name=config_name)
    return cfg
