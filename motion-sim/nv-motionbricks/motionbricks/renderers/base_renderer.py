# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Track which Newton viewer primitives a renderer shows or hides."""

from collections.abc import Callable, MutableMapping
from typing import Protocol


class _ViewerPrimitive(Protocol):
    def destroy(self) -> None: ...


class BaseRenderer:
    """Track which named Newton viewer primitives are shown or hidden.

    Each renderer supplies the geometry its Newton log function expects. This
    class logs shown and hidden states the same way for every renderer. Hiding a
    primitive that is already hidden logs nothing.
    """

    def __init__(self) -> None:
        self._registered_names: set[str] = set()
        self._hidden_names: set[str] = set()

    def _log_visible(
        self,
        log: Callable[..., object],
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Log a visible primitive and remember its current state."""
        log(name, *args, hidden=False, **kwargs)
        self._registered_names.add(name)
        self._hidden_names.discard(name)

    def _log_hidden_once(
        self,
        log: Callable[..., object],
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Hide a registered primitive, logging it only once until it is shown again."""
        if name not in self._registered_names or name in self._hidden_names:
            return

        log(name, *args, hidden=True, **kwargs)
        self._hidden_names.add(name)

    def _clear(self, primitives: MutableMapping[str, _ViewerPrimitive]) -> None:
        """Destroy and remove every viewer primitive owned by this renderer."""
        for name in self._registered_names:
            primitive = primitives.pop(name, None)
            if primitive is None:
                continue

            primitive.destroy()

        self._registered_names.clear()
        self._hidden_names.clear()
