"""The Triton kernels of the backbone's linear-attention layers, each held at one configuration.

flash-linear-attention picks each kernel's configuration by timing at its first call, and for two kernels the choice changes
the last bits of the result. With the configurations fixed, a seed gives the same image in every process.
"""
import sys

from . import config


def autotuned_kernels():
    """"module.function" -> Triton autotuner, for every flash-linear-attention kernel imported so far (empty without the
    package)"""
    if "fla" not in sys.modules:
        return {}
    from triton.runtime.autotuner import Autotuner, Heuristics
    found = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "fla" or name.startswith("fla.")):
            continue
        for obj in list(vars(module).values()):
            while isinstance(obj, Heuristics):
                obj = obj.fn
            if isinstance(obj, Autotuner):
                found[id(obj)] = obj
    return {t.base_fn.__module__ + "." + t.base_fn.__name__: t for t in found.values()}


def describe(cfg):
    """a Triton configuration -> (block sizes, warps, stages), the form of config.LINEAR_ATTENTION_KERNELS"""
    return dict(cfg.kwargs), cfg.num_warps, cfg.num_stages


def fix_kernel_configs(table=None):
    """Narrow each kernel of `table` (default: config.LINEAR_ATTENTION_KERNELS) to its listed configuration. Call it after the
    backbone is loaded and before the first forward pass. -> the kernels of the table that could not be fixed."""
    table = config.LINEAR_ATTENTION_KERNELS if table is None else table
    kernels, left = autotuned_kernels(), []
    if not kernels:
        return left
    for name, (blocks, warps, stages) in table.items():
        tuner = kernels.get(name)
        if tuner is None:
            left.append(name)
            continue
        if not hasattr(tuner, "jev_all_configs"):        # the kernel's full list, so that a later call can choose again
            tuner.jev_all_configs = list(tuner.configs)
        match = [c for c in tuner.jev_all_configs if describe(c) == (dict(blocks), warps, stages)]
        if match:
            tuner.configs = match[:1]
        else:
            left.append(name)
    return left


def ran_autotuned():
    """the kernels that have run in this process with a configuration chosen by timing (for checks)"""
    return sorted(name for name, tuner in autotuned_kernels().items() if tuner.cache)
