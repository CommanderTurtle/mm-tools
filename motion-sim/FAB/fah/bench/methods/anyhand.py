"""Adapter: WiLoR / HaMeR checkpoints co-trained with AnyHand synthetic data (arXiv 2603.25726,
https://huggingface.co/chen-si-02/AnyHand-Models, CC-BY-NC-ND 4.0) -> benchmark prediction format.

The architectures are unchanged (AnyHandPredictor loads them with the official load_wilor / HAMER.load_from_checkpoint
and runs WiLoR's YOLO detector + ViTDetDataset crops, rescale_factor 2.0), so these are the WiLoR / HaMeR adapters with
the AnyHand weights (docs/evaluation.md, baselines). Same detector conf / NMS IoU as the wilor / hamer runs (AnyHand's
predictor uses NMS IoU 0.3), so the comparison isolates the weights. AnyHand-WiLoR's config uses EXTRA.FOCAL_LENGTH
1000 (WiLoR: 5000); the crop -> full-image camera uses the true intrinsics either way.
"""
import os

from bench.methods.hamer import HaMeRMethod
from bench.methods.wilor import WiLoRMethod

ANYHAND = os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "AnyHand")


class WiLoRAnyHandMethod(WiLoRMethod):
    name = "wilor-anyhand"

    def __init__(self, ckpt=None, cfg_path=None, **kw):
        super().__init__(ckpt=ckpt or os.path.join(ANYHAND, "anyhand_wilor.ckpt"),
                         cfg_path=cfg_path or os.path.join(ANYHAND, "model_config_wilor.yaml"), **kw)
        self.name = "wilor-anyhand-fast" if self.fast else "wilor-anyhand"


class HaMeRAnyHandMethod(HaMeRMethod):
    name = "hamer-anyhand"

    def __init__(self, ckpt=None, **kw):
        # load_hamer reads <ckpt>/../../model_config.yaml (the config embedded in the checkpoint, prepare_anyhand.sh)
        super().__init__(ckpt=ckpt or os.path.join(ANYHAND, "hamer", "checkpoints", "anyhand_hamer.ckpt"), **kw)
        self.name = "hamer-anyhand"
