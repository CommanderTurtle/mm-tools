# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.audio.data.data_simulation import RIRMixGenerator
from nemo.core.config import hydra_runner


"""
This script creates a corpus of signals using RIRs, speech and noise.

Usage:
    python rir_mix_generator.py --config-path PATH_TO_CONFIG_DIR --config-name CONFIG_NAME output_dir=OUTPUT_DIR

Details of the configuration can be found in the example configuration files in conf/*
"""


@hydra_runner(config_path="conf", config_name="rir_mix.yaml")
def main(cfg):
    rir_mix = RIRMixGenerator(cfg=cfg)
    rir_mix.generate()


if __name__ == "__main__":
    main()
