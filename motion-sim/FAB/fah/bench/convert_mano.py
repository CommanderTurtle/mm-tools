"""Convert the official MANO .pkl files (which embed chumpy objects) into plain-numpy pickles
that smplx can load without chumpy.

    python bench/convert_mano.py checkpoints/mano/mano_v1_2/models checkpoints/mano/clean
"""
import os
import pickle
import sys
import types

import numpy as np


class _ChumpyStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})


def _install_chumpy_stub():
    for name in ["chumpy", "chumpy.ch", "chumpy.reordering", "chumpy.ch_ops", "chumpy.logic"]:
        mod = types.ModuleType(name)
        mod.__getattr__ = lambda attr: _ChumpyStub
        sys.modules[name] = mod


def to_numpy(v):
    if isinstance(v, _ChumpyStub):
        d = v.__dict__
        if "idxs" in d and "a" in d:  # chumpy Select / reordering
            arr = np.asarray(to_numpy(d["a"])).ravel()[d["idxs"]]
            return arr.reshape(d["preferred_shape"]) if d.get("preferred_shape") else arr
        for key in ["x", "a", "_x"]:
            if key in v.__dict__:
                return np.asarray(to_numpy(v.__dict__[key]))
        raise ValueError(f"cannot unwrap chumpy object with keys {list(v.__dict__)}")
    return v


def main(src, dst):
    _install_chumpy_stub()
    os.makedirs(dst, exist_ok=True)
    for name in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        with open(os.path.join(src, name), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        clean = {k: to_numpy(v) for k, v in data.items()}
        for k, v in clean.items():
            if hasattr(v, "shape"):
                print(f"  {name} {k}: {type(v).__name__} {v.shape}")
        with open(os.path.join(dst, name), "wb") as f:
            pickle.dump(clean, f)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
