"""Helpers for tool-side SOMA rig asset loading."""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from soma.io import (
    SOMA_NEUTRAL_RIG_KEYS,
    SOMA_TEMPLATE_RIG_FILENAME,
    load_lod_rig_from_usd,
    missing_soma_neutral_rig_keys,
)
from soma.procedural_transforms import (
    SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME,
    derive_soma_rig_without_procedural_joints,
    load_soma_procedural_transform_definition,
)


def _public_joint_names_from_assets(
    rig_data: Mapping[str, Any],
    *,
    core_asset: Path,
    definition_path: Path,
) -> np.ndarray:
    if "joint_names" in rig_data:
        return np.array(rig_data["joint_names"]).copy()
    if definition_path.exists():
        definition = load_soma_procedural_transform_definition(definition_path)
        return np.array(definition.public_joint_names)
    raise FileNotFoundError(
        f"Core asset '{core_asset}' does not contain joint_names. "
        f"Install '{definition_path.name}' next to it so the public SOMA joint contract "
        "can be derived from the procedural definition."
    )


def _raise_missing_template_for_slim_npz(
    missing_keys: tuple[str, ...],
    *,
    core_asset: Path,
    template_rig_path: Path,
) -> None:
    if missing_keys:
        raise FileNotFoundError(
            f"Template rig asset not found: {template_rig_path}. "
            f"Core asset '{core_asset}' is a slim SOMA_neutral.npz and no longer contains "
            f"rig fields: {', '.join(missing_keys)}. Install "
            f"'{SOMA_TEMPLATE_RIG_FILENAME}' next to the core asset."
        )


def load_public_mid_soma_rig(
    data_root: str | Path,
    *,
    template_rig_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load SOMA core arrays plus public mid-LOD rig arrays for tools.

    ``SOMA_neutral.npz`` is now a slim core asset in the checked-in release.
    Tooling that needs bind poses, joint parents, bind shape, or skinning
    weights should use the canonical template rig when it is present, while
    retaining non-rig arrays such as PCA, mirror indices, and topology from
    the core NPZ.
    """
    data_root = Path(data_root)
    if template_rig_path is None:
        template_rig_path = data_root / SOMA_TEMPLATE_RIG_FILENAME
    else:
        template_rig_path = Path(template_rig_path)

    return dict(
        _load_public_mid_soma_rig_cached(
            str(data_root.resolve()),
            str(template_rig_path.resolve()),
        )
    )


@lru_cache(maxsize=4)
def _load_public_mid_soma_rig_cached(
    data_root_str: str,
    template_rig_path_str: str,
) -> dict[str, Any]:
    data_root = Path(data_root_str)
    template_rig_path = Path(template_rig_path_str)
    core_asset = data_root / "SOMA_neutral.npz"
    if not core_asset.exists():
        raise FileNotFoundError(
            f"Core asset not found: {core_asset}\nRun 'git lfs pull' to fetch LFS-tracked files."
        )

    rig_data = dict(np.load(core_asset, allow_pickle=False))
    if template_rig_path.exists():
        public_joint_names = _public_joint_names_from_assets(
            rig_data,
            core_asset=core_asset,
            definition_path=data_root / SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME,
        )
        public_rig = derive_soma_rig_without_procedural_joints(
            load_lod_rig_from_usd(template_rig_path, "mid"),
            public_joint_names,
        )
        rig_data.update({key: public_rig[key] for key in SOMA_NEUTRAL_RIG_KEYS})
    else:
        _raise_missing_template_for_slim_npz(
            missing_soma_neutral_rig_keys(rig_data),
            core_asset=core_asset,
            template_rig_path=template_rig_path,
        )

    return rig_data
