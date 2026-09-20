import os

class BaseInferencePipeline:
    def __init__(self, cfg):
        self.hparams = cfg
        self.device = "cuda"
        self.global_rank = 0
        self.global_step = 0
        self.world_size = 1
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log(self, key, value):
        return