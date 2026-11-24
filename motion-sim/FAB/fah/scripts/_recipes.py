"""Shared configuration and subprocess handling for the user-facing recipe scripts."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"fah-s": ("s", "dinov3_vits16"), "fah-b": ("b", "dinov3_vitb16"),
          "fah": ("l", "dinov3_vitl16"), "fah-h": ("h", "dinov3_vith16plus")}
POSE_POOLS = [("hot3d_lab_mesh", 1.3), ("coco_wb_kin", 1), ("cmu_lab_kin", .3), ("hint_lab_kin", .6),
              ("freihand_lab_mesh_kin", .5), ("egodex_test_mano", .6), ("visor_lab", .8), ("neg", .5),
              ("halpe_lab_kin", .8), ("egodex_part1_mano", .6), ("dexycb_lab", .5), ("interhand_lab", .5),
              ("oakink2_lab", .4), ("ho3d_lab", .3), ("hanco_lab", .2), ("oakink2_ego_lab", .5)]
CONTACT_POOLS = [("hot3d_lab_mesh", 1), ("dexycb_lab", .6), ("ho3d_lab", .3), ("oakink2_ego_lab", .8),
                 ("oakink2_lab", .5), ("interhand_lab", .3), ("mow_train_c35", .05), ("arctic_ego", .5), ("hoi4d", .5)]


def config(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise ValueError(f"Missing config: {p}. Copy configs/local.example.json to configs/local.json and edit it.")
    c = json.loads(p.read_text())
    if not isinstance(c, dict):
        raise ValueError("Config must be a JSON object.")
    return c


def required(c, key):
    value = c.get(key)
    if value is None or value == "":
        raise ValueError(f"Set {key} in your local config.")
    return value


def positive(c, key):
    value = required(c, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer chosen for your setup.")
    return value


def path(c, key, default=None):
    value = c.get(key) or default
    if not value:
        raise ValueError(f"Set {key} in your local config.")
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else ROOT / p


def args(**kw):
    return [str(x) for key, value in kw.items() if value is not None for x in ("--" + key, value)]


def module(c, name, *values):
    return [str(c.get("python") or sys.executable), "-m", name, *map(str, values)]


def distributed(c, name, *values):
    return module(c, "torch.distributed.run", "--nproc_per_node", positive(c, "train_processes"), "-m", name, *values)


def pools(c, key, default):
    value = c.get(key) or default
    if not isinstance(value, list) or not value or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in value):
        raise ValueError(f"{key} must contain [pool_name, sampling_weight] pairs.")
    if any(not isinstance(p[0], str) or not isinstance(p[1], (int, float)) or p[1] <= 0 for p in value):
        raise ValueError(f"Invalid pool or sampling weight in {key}.")
    return args(sources=",".join(p[0] for p in value), weights=",".join(str(p[1]) for p in value))


def execute(commands, c, dry_run=False):
    env = dict(os.environ)
    if c.get("mano_dir"):
        env["MANO_DIR"] = str(path(c, "mano_dir"))
    for command in commands:
        print(shlex.join(list(map(str, command))), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
