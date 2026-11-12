# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MLFlowParams:
    """ML Flow Configuration Dataclass."""

    # name of experiment, if none, defaults to the globally set experiment name
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    # if no run_name is set, it's set by version
    # local or remote tracking seerver. If tracking_uri is not set, it defaults to save_dir
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    save_dir: Optional[str] = "./mlruns"
    prefix: str = ""
    artifact_location: Optional[str] = None
    # provide run_id if resuming a previously started run
    run_id: Optional[str] = None
    # Log checkpoints created by ModelCheckpoint as MLFlow artifacts.
    log_model: bool = False
