from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import PeftAdapterMixin

from wanxiang.models.wan_animate_2_model import  Transformer


class WanxiangAnimate2Transformer(Transformer, PeftAdapterMixin, ModelMixin, ConfigMixin):
    _tied_weights_keys = None

    def load_from_official_state_dict(self, state_dict):
        state_dict_new = {}

        for k, v in state_dict.items():
            if k.startswith("blocks"):
                parts = k.split(".")
                parts_block = parts[:]
                parts_block.insert(2, "block")

                new_key_block = ".".join(parts_block)

                state_dict_new[new_key_block] = v
                continue

            state_dict_new[k] = v

        self.load_state_dict(state_dict_new, strict=True)