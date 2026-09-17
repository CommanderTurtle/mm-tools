import numpy as np
# Quantization constants
NUM_BINS = 1024
POS_MIN, POS_MAX = 0, 24.0
SCALE_MIN, SCALE_MAX = 0, 24.0
EULER_X_MIN, EULER_X_MAX = -np.pi, np.pi        # from atan2
EULER_Y_MIN, EULER_Y_MAX = -np.pi / 2, np.pi / 2  # from arcsin
EULER_Z_MIN, EULER_Z_MAX = -np.pi, np.pi        # from atan2
MAX_SCENE_OBJECTS = 384
LATENT_RESOLUTION = 8
LATENT_DIM = 16
SS_LATENT_RESOLUTION = 2
SS_LATENT_DIM = 8
