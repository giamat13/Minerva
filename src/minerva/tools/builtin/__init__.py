"""Minerva's built-in tools.

These ship with the platform and are loaded into
:func:`minerva.tools.registry.default_registry`.

ADDING A BUILT-IN TOOL
----------------------
1. Write it in a new module in this package (or extend an existing one). Use the
   ``@tool`` decorator from :mod:`minerva.tools.base` - the JSON Schema the
   model sees is derived from your type hints and docstring.
2. Import it below and add it to :data:`_BUILTIN_TOOLS`.
3. Add tests in ``tests/test_tools_builtin.py`` that exercise the real
   behaviour, including the failure paths.

That is the whole procedure - no registration boilerplate anywhere else.
See ``docs/ADDING_A_TOOL.md`` for the longer version.
"""

from __future__ import annotations

from ..base import Tool
from .calculator import calculate
from .clock import current_time, days_between
from .web_search import web_search

__all__ = ["builtin_tools", "calculate", "current_time", "days_between", "web_search"]

# The canonical list of built-in tools. Order is irrelevant; the registry
# sorts by name when it hands specs to an engine.
_BUILTIN_TOOLS: tuple[Tool, ...] = (
    calculate,
    current_time,
    days_between,
    web_search,
)


def builtin_tools() -> tuple[Tool, ...]:
    """Every tool that ships with Minerva."""
    return _BUILTIN_TOOLS
