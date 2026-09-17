import copy
import torch

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        # Create a shadow copy of the model
        self.shadow = copy.deepcopy(model)
        # Ensure shadow model is in eval mode and detach params
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    def to(self, device):
        self.shadow.to(device)
        return self

    def update(self, model):
        """Update EMA parameters with the current model's parameters."""
        with torch.no_grad():
            msd = model.state_dict()

            # --- Optimization: Device Check ---
            # 1. Identify the device of the incoming model (using the first parameter)
            #    We check 'msd' because 'model' might be a wrapper, but state_dict is raw tensors.
            if len(msd) > 0:
                model_device = next(iter(msd.values())).device
            else:
                model_device = torch.device('cpu') # Fallback

            # 2. Check the device of the shadow model
            #    We check the first parameter of the shadow model
            shadow_params = list(self.shadow.parameters())
            if len(shadow_params) > 0:
                shadow_device = shadow_params[0].device
            else:
                # Fallback for models with only buffers or empty
                shadow_device = next(iter(self.shadow.state_dict().values())).device if len(self.shadow.state_dict()) > 0 else torch.device('cpu')

            # 3. If devices differ, move the shadow model ONCE
            if model_device != shadow_device:
                self.shadow.to(model_device)
            # ----------------------------------

            # 4. Get the shadow state_dict AFTER the potential move
            #    (This ensures ssd references the tensors on the correct device)
            ssd = self.shadow.state_dict()

            for k in ssd.keys():
                if k in msd:
                    # Since we ensured devices match above, .to() here is effectively a no-op
                    # but kept for safety.
                    model_param = msd[k] #.to(ssd[k].device)

                    # Optimization: In-place operations to save memory
                    # ssd[k] = ssd[k] * decay + new * (1 - decay)
                    ssd[k].mul_(self.decay).add_(model_param, alpha=(1 - self.decay))

    def state_dict(self):
        return self.shadow.state_dict()