#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""YAML configs with ``a.b=value`` command-line overrides."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

# YAML 1.1 reads exponents without a decimal point (``1e-6``) as strings.
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


class Config(dict):
    """A dict with attribute access for nested sections."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return Config(value) if isinstance(value, dict) else value


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, overrides=()):
    """Load ``path`` (following an optional ``base:`` chain) and apply overrides."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    config = _merge(load_config(path.parent / base), data) if base else data
    for item in overrides:
        key, _, raw = item.partition("=")
        # ``+a.b=value`` adds a key; without ``+`` the key must exist, so that typos fail loudly.
        add = key.startswith("+")
        node = config
        *parents, leaf = key.lstrip("+").split(".")
        for name in parents:
            if not isinstance(node.get(name, {} if add else None), dict):
                raise KeyError(f"override {item!r}: {path} has no section {name!r} (use +{key.lstrip('+')}=...)")
            node = node.setdefault(name, {})
        if not add and leaf not in node:
            raise KeyError(f"override {item!r}: {path} has no key {key!r} (use +{key}=... to add it)")
        value = yaml.safe_load(raw)
        node[leaf] = float(value) if isinstance(value, str) and _NUMBER.match(value) else value
    return Config(config)
