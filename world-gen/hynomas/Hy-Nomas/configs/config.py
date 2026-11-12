"""
Hydra-based configuration for Hy-Nomas.

Uses the Hydra Compose API so that torchrun / distributed launchers remain
in control of ``__main__``.  The returned ``DictConfig`` supports attribute
access (``cfg.field_name``) — a drop-in replacement for the old
``argparse.Namespace``.

Usage
-----
From a training script::

    from configs.config import parse_args
    cfg = parse_args()          # reads sys.argv for Hydra overrides
    print(cfg.learning_rate)

CLI examples::

    # Select an experiment preset
    torchrun ... hymm/train.py +experiment=sft

    # Override individual values
    torchrun ... hymm/train.py +experiment=sft training.optimizer.learning_rate=1e-4

    # Multiple overrides
    torchrun ... hymm/inference.py +experiment=inference dit_weight=/path/to/model.safetensors

Backward compatibility
----------------------
The old argparse-based ``parse_args`` is preserved as
``configs/config_argparse_backup.py`` for reference.
"""

import logging
import os
import sys
from typing import List, Optional

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


# Nested groups in config.yaml that should be flattened into the root namespace.
# Order matters: later groups can overwrite keys from earlier ones.
# Tuple format: (dotted_path_in_cfg, optional_key_rename_map)
_FLATTEN_GROUPS = [
    ("logging", {}),
    ("model.camera", {}),
    ("model.dit", {}),
    ("model.vae", {"name": "vae_type"}),
    ("model.text_encoder", {"name": "text_encoder_type"}),
    ("dataset", {}),
    ("training", {}),
    ("training.ema", {}),
    ("training.segment_self_forcing", {}),
    ("training.lora", {}),
    ("training.scheduler", {}),
    ("training.optimizer", {"name": "optimizer_type"}),
    ("profiling", {}),
    ("parallelism", {}),
    ("inference", {}),
]


def _flatten_cfg(cfg: DictConfig) -> DictConfig:
    """Flatten nested config groups into the root namespace.

    After this call every leaf value is accessible as ``cfg.field_name``
    (e.g. ``cfg.learning_rate`` instead of ``cfg.training.optimizer.learning_rate``).

    Section dicts are also re-attached as sub-namespaces for disambiguation:
      - ``cfg.inference.seed``, ``cfg.dit.model_type``, ``cfg.scheduler.flow_reverse``
      - ``cfg.vae.vae_precision``, ``cfg.optimizer.learning_rate``,
        ``cfg.text_encoder.text_len``.

    Scalar ``name`` keys from vae/text_encoder/optimizer sections are renamed
    to ``vae_type``, ``text_encoder_type``, ``optimizer_type`` to avoid
    colliding with the section dict name.

    Nested sub-dicts that are themselves groups (e.g. ``model``, ``training``)
    are removed from the root after their children have been promoted.
    """
    flat = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(flat, dict)

    # Collect the set of top-level group keys that will be removed later,
    # so we know which root-level keys are "real" overrides from experiment yamls
    # vs defaults from the nested config.
    group_top_keys = {
        "logging",
        "model",
        "dataset",
        "training",
        "profiling",
        "parallelism",
        "inference",
    }

    # section_dicts keeps a snapshot of leaves and selected runtime groups so that
    # colliding keys can still be accessed via cfg.<section>.<key>.
    # e.g. cfg.inference.seed even when cfg.seed is taken by root Common.
    section_dicts: dict = {}

    for dotted_path, rename_map in _FLATTEN_GROUPS:
        parts = dotted_path.split(".")
        section_name = parts[-1]  # e.g. "inference", "optimizer", "fsdp"
        node = flat
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is None or not isinstance(node, dict):
            continue

        # Runtime option groups stay intact, including in cfg.inference.
        # Their generic leaves (e.g. "enabled") must not become root flags.
        runtime_groups = (
            {"sliding_window", "sparse_attention", "sparse_optim", "qvg_kv_cache"} if dotted_path == "inference" else set()
        )
        section_leaf = {}
        for k, v in node.items():
            if isinstance(v, dict) and k not in runtime_groups:
                continue
            section_leaf[k] = v

        for k, v in node.items():
            # Skip sub-dicts that are themselves flatten targets (avoid double-promote)
            if isinstance(v, dict) and k not in runtime_groups:
                continue
            dest_key = rename_map.get(k, k)
            # Promote to root if no collision (first-write-wins), OR if the
            # existing value is None and the new value is not (so that a real
            # value from a later group can override a null default).
            if dest_key not in flat or dest_key in group_top_keys:
                flat[dest_key] = v
            elif flat[dest_key] is None and v is not None:
                flat[dest_key] = v
            else:
                # Key collision: dest_key already has a non-None value from
                # a previous section or root Common.  The value is kept as-is
                # (first-write-wins).  Use cfg.<section>.<key> to disambiguate.
                _logger = logging.getLogger(__name__)
                _logger.debug(
                    "_flatten_cfg: key '%s' from section '%s' skipped "
                    "(already set to %r). Use cfg.%s.%s to access this "
                    "section's value (%r).",
                    dest_key,
                    dotted_path,
                    flat[dest_key],
                    section_name,
                    k,
                    v,
                )

        # Store the section dict for nested access (uses the *last* segment
        # of the dotted path, e.g. "inference", "scheduler", "dit").
        section_dicts[section_name] = section_leaf

        # Also merge leaves into the *parent* section dict so that nested
        # paths like cfg.parallelism.fsdp_sharding_strategy work when
        # fsdp_sharding_strategy originates from parallelism.fsdp.
        if len(parts) > 1:
            parent_name = parts[-2]  # e.g. "parallelism" for "parallelism.fsdp"
            if parent_name in section_dicts:
                for k, v in section_leaf.items():
                    if k not in section_dicts[parent_name]:
                        section_dicts[parent_name][k] = v

    # Remove the original nested group dicts, then re-attach section
    # snapshots so that colliding keys are still reachable via
    # cfg.<section>.<key>  (e.g. cfg.inference.seed).
    for key in (
        "logging",
        "model",
        "dataset",
        "training",
        "profiling",
        "parallelism",
        "inference",
    ):
        flat.pop(key, None)

    # Re-attach section dicts as sub-namespaces for disambiguation.
    for section_name, leaf_dict in section_dicts.items():
        if section_name not in flat:
            flat[section_name] = leaf_dict
        elif isinstance(flat[section_name], dict):
            # Merge into existing dict (e.g. if "fsdp" was already created
            # by a prior group under the same top-level key).
            flat[section_name].update(leaf_dict)
        # If a scalar key already took the section name, skip to avoid
        # overwriting.  With the current rename_map this should not happen.

    # Codec defaults remain available to Hydra overrides, but only belong in
    # runtime/serialized configs when QVG storage is selected. Inactive defaults
    # otherwise create unrelated differences against pre-QVG experiment records.
    if flat.get("inference", {}).get("kv_cache_quantization", "none") != "qvg":
        flat.pop("qvg_kv_cache", None)
        flat.get("inference", {}).pop("qvg_kv_cache", None)

    return OmegaConf.create(flat)


def _extract_hydra_overrides() -> List[str]:
    """Extract Hydra-style overrides from ``sys.argv``.

    ``torchrun`` injects its own ``--local_rank=...`` (or ``--local-rank=...``)
    arguments.  We filter those out and keep only Hydra override tokens, which
    are positional args (no ``--`` prefix) **or** ``+key=value`` / ``~key``
    forms.

    We also support the legacy ``--config-file`` / ``--config-name`` flags for
    backward-compatible invocation from shell scripts.
    """
    overrides: List[str] = []
    skip_next = False

    for i, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue

        # Skip torchrun / distributed-launcher injected args
        if arg.startswith("--local_rank") or arg.startswith("--local-rank"):
            if "=" not in arg:
                skip_next = True
            continue

        # Legacy compat: --config-file / --config-name  (skip them)
        if arg in ("--config-file", "--config-name"):
            skip_next = True
            continue
        if arg.startswith("--config-file=") or arg.startswith("--config-name="):
            continue

        # Skip the script path (argv[0] is already excluded by [1:])
        if arg.endswith(".py"):
            continue

        # Skip any other --flag style args that aren't Hydra overrides
        if arg.startswith("--"):
            # Convert --key=value or --key value to Hydra key=value
            if "=" in arg:
                key_val = arg.lstrip("-")
                key, val = key_val.split("=", 1)
                key = key.replace("-", "_")
                overrides.append(f"{key}={val}")
            else:
                # --key value  (next arg is the value)
                key = arg.lstrip("-").replace("-", "_")
                if i < len(sys.argv) - 1:
                    val = sys.argv[i + 1]
                    if not val.startswith("-"):
                        overrides.append(f"{key}={val}")
                        skip_next = True
            continue

        # Hydra native overrides: key=value, +key=value, ~key, ++key=value
        overrides.append(arg)

    return overrides


def _validate_defaults_exist(
    configs_root: str,
    overrides: List[str],
) -> None:
    """Recursively validate that all files referenced in ``defaults`` lists exist.

    Starting from the experiment YAML (extracted from *overrides*), walk the
    ``defaults`` inheritance chain and raise ``FileNotFoundError`` with a
    clear message if any referenced config file is missing.

    This runs **before** ``hydra.compose()`` so the user sees a human-readable
    error instead of Hydra's cryptic traceback.
    """
    # Find the experiment file from overrides
    experiment_path: Optional[str] = None
    for ov in overrides:
        for prefix in ("+experiment=", "experiment="):
            if ov.startswith(prefix):
                experiment_path = ov[len(prefix) :]
                break
        if experiment_path is not None:
            break

    if experiment_path is None:
        return  # no experiment override → nothing to validate

    experiment_file = os.path.join(
        configs_root, "experiment", experiment_path + ".yaml"
    )
    if not os.path.isfile(experiment_file):
        raise FileNotFoundError(
            f"Experiment config file not found: {experiment_file}\n"
            f"  (resolved from override '+experiment={experiment_path}')\n"
            f"  Please check that the file exists under configs/experiment/."
        )

    # Recursively walk the defaults chain
    _check_defaults_recursive(experiment_file, configs_root, visited=set())


def _check_defaults_recursive(
    yaml_path: str,
    configs_root: str,
    visited: set,
) -> None:
    """Recursively check that all ``defaults`` references in *yaml_path* exist.

    Raises ``FileNotFoundError`` with a clear message including the source
    file and the missing reference.
    """
    yaml_path = os.path.abspath(yaml_path)
    if yaml_path in visited:
        return
    visited.add(yaml_path)

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return  # file itself is unreadable; Hydra will report this later

    defaults = data.get("defaults", [])
    if not isinstance(defaults, list):
        return

    for entry in defaults:
        # Each entry is either a string or a single-key dict
        if isinstance(entry, str):
            ref = entry
        elif isinstance(entry, dict):
            # e.g. {override hydra/hydra_logging: disabled} or {experiment: ...}
            values = list(entry.values())
            if not values:
                continue
            ref = values[0]
            # Skip Hydra internal overrides (e.g. "override hydra/...")
            keys = list(entry.keys())
            if keys and str(keys[0]).startswith("override "):
                continue
        else:
            continue

        # Skip _self_ (Hydra built-in)
        if ref == "_self_":
            continue

        # Resolve Hydra path: '/share/solver/euler' -> 'configs/share/solver/euler.yaml'
        ref_path = ref.lstrip("/")
        resolved_file = os.path.join(configs_root, ref_path + ".yaml")

        if not os.path.isfile(resolved_file):
            raise FileNotFoundError(
                f"Default config file not found!\n"
                f"  Source file : {yaml_path}\n"
                f"  Missing ref : '{ref}'\n"
                f"  Expected at : {resolved_file}\n"
                f"  Please check that the referenced config file exists."
            )

        # Recurse into the referenced file
        _check_defaults_recursive(resolved_file, configs_root, visited)


def parse_args(mode: str = "train", namespace=None) -> DictConfig:
    """Load configuration via Hydra Compose API.

    Parameters
    ----------
    mode : str
        Retained for call-site compatibility. Configuration composition does
        not validate inference optimization combinations.
    namespace :
        Unused — kept for API compatibility.

    Returns
    -------
    DictConfig
        A flat OmegaConf config with attribute access (``cfg.field_name``).
    """
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # Clear any previous Hydra state (e.g. when called multiple times)
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        overrides = _extract_hydra_overrides()

        # ---- Normalize experiment path ----
        # Users (or shell scripts) may pass full filesystem paths like
        #   +experiment=configs/experiment/sft.yaml
        # but Hydra expects a path relative to configs/experiment/, e.g.
        #   +experiment=sft
        # Strip the prefix and suffix so both forms work.
        normalized = []
        for ov in overrides:
            if ov.startswith("+experiment=") or ov.startswith("experiment="):
                key, _, val = ov.partition("=")
                val = val.removesuffix(".yaml")
                val = val.removeprefix("configs/experiment/")
                ov = f"{key}={val}"
            normalized.append(ov)
        overrides = normalized

        # ---- Validate defaults references before Hydra compose ----
        # Recursively check that all files referenced in 'defaults' lists
        # actually exist.  This gives a clear, early error message instead
        # of Hydra's cryptic "Could not load" traceback.
        _validate_defaults_exist(config_dir, overrides)

        cfg = compose(config_name="config", overrides=overrides)

    # Resolve interpolations eagerly so downstream code sees plain values
    OmegaConf.resolve(cfg)

    # Flatten nested groups (model.dit.*, training.optimizer.*, etc.)
    # into the root namespace so that downstream code can access them as
    # cfg.field_name — matching the old argparse.Namespace interface.
    cfg = _flatten_cfg(cfg)

    # Enable struct mode so that accessing / setting undeclared keys raises
    # an error.  Downstream code that *must* add runtime-computed keys should
    # wrap the mutation with ``from omegaconf import open_dict; with open_dict(cfg): ...``.
    OmegaConf.set_struct(cfg, True)

    return cfg
