"""Minimal mmcv.utils stand-in for Metric3D inference."""
from mmengine import Config, DictAction  # noqa: F401
from mmengine.utils.dl_utils import collect_env  # noqa: F401


def get_git_hash(fallback="unknown", digits=None):
    return fallback
