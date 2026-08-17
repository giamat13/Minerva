"""The Minerva model family.

Each model is a :class:`~minerva.models.base.ModelSpec` in its own module,
listed in :mod:`minerva.models.registry`. The family currently has one member:

* **Swift** (:mod:`minerva.models.swift`) - small, fast, tool-capable.

See ``docs/ADDING_A_MODEL.md`` to add another.
"""

from __future__ import annotations

from .base import MinervaModel, ModelSpec
from .registry import get_spec, list_specs, load_model, model_names, register_spec
from .swift import SWIFT

__all__ = [
    "SWIFT",
    "MinervaModel",
    "ModelSpec",
    "get_spec",
    "list_specs",
    "load_model",
    "model_names",
    "register_spec",
]
