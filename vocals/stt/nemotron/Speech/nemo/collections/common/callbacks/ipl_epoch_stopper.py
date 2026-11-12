# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule


class IPLEpochStopper(Callback):
    """
    Callback to gracefully terminate training at the *end* of an epoch,
    typically used in Iterative Pseudo-Labeling (IPL) pipelines.

    IPL is a semi-supervised learning approach where models are trained
    iteratively, alternating between generating pseudo-labels and fine-tuning
    on them. For more details, see our paper:
    "TopIPL: Unified Semi-Supervised Pipeline for Automatic Speech Recognition"
    https://arxiv.org/abs/2506.07659

    This callback is used to signal the Trainer to stop training after a given number
    of epochs, allowing pseudo-label generation and model reinitialization to occur.

    Args:
        enable_stop (bool): If True, the trainer will be requested to stop during
            `on_train_epoch_end`. If False, the callback is inert.
        stop_every_n_epochs (int): Number of epochs to run before each stop. If set to 1,
            training will stop after every epoch.
    """

    def __init__(self, enable_stop: bool = False, stop_every_n_epochs: int = 1) -> None:
        super().__init__()
        self.enable_stop = bool(enable_stop)
        self.stop_every_n_epochs = stop_every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Sets `should_stop` stop flag to terminate the training.
        """
        super().__init__()

        if self.stop_every_n_epochs != 0:
            self.stop_every_n_epochs -= 1
        if self.stop_every_n_epochs == 0:
            trainer.should_stop = True
