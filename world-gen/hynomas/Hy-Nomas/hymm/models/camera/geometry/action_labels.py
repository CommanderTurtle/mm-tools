"""Convert camera movement labels into model conditioning vectors."""

import numpy as np

from hymm.models.camera.geometry.discretize_utils import LABEL_TO_ONEHOT_4T4R


def action_labels_to_onehot(action_labels, mode="4t4r"):
    """
    Convert action labels to concatenated onehot vectors.

    For 4t4r mode:
        - action_label = trans_label * 9 + rotate_label
        - trans_label: 0-8 (maps to 4-dim onehot via LABEL_TO_ONEHOT_4T4R)
        - rotate_label: 0-8 (maps to 4-dim onehot via LABEL_TO_ONEHOT_4T4R)
        - Output: [T, 8] concatenated onehot (4-dim trans + 4-dim rotate)

    Args:
        action_labels: np.ndarray [T] - discretized action labels
        mode: str - "4t4r" or "6t6r"

    Returns:
        action_onehot: np.ndarray [T, 8] for 4t4r, [T, 12] for 6t6r
    """
    num_frames = len(action_labels)

    if mode == "4t4r":
        num_rot_labels = 9
        onehot_dim = 4
        label_to_onehot = LABEL_TO_ONEHOT_4T4R
    else:
        raise NotImplementedError(f"Mode {mode} not supported yet")

    # Decode action labels
    trans_labels = action_labels // num_rot_labels
    rotate_labels = action_labels % num_rot_labels

    # Convert to onehot
    trans_onehot = np.zeros((num_frames, onehot_dim), dtype=np.float32)
    rotate_onehot = np.zeros((num_frames, onehot_dim), dtype=np.float32)

    for i in range(num_frames):
        trans_label = trans_labels[i]
        rotate_label = rotate_labels[i]

        # Get onehot tuple from mapping
        if trans_label in label_to_onehot:
            trans_onehot[i] = np.array(label_to_onehot[trans_label], dtype=np.float32)
        if rotate_label in label_to_onehot:
            rotate_onehot[i] = np.array(label_to_onehot[rotate_label], dtype=np.float32)

    # Concatenate: [T, 4] + [T, 4] -> [T, 8]
    action_onehot = np.concatenate([trans_onehot, rotate_onehot], axis=1)

    return action_onehot
