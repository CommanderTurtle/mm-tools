"""Shared local-first studio runtime for mm-tools.

The package deliberately contains no model code.  Project adapters own their
native inference APIs while this module supplies the durable queue, artifact
library, and browser transport shared by every new mm-tools studio.
"""

from .runtime import StudioAdapter, StudioContext, StudioOutput

__all__ = ["StudioAdapter", "StudioContext", "StudioOutput"]
