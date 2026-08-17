"""Tool calling (a.k.a. function calling) for Minerva.

The public surface is small on purpose::

    from minerva.tools import ToolRegistry, tool

    @tool
    def celsius_to_fahrenheit(celsius: float) -> float:
        \"\"\"Convert a temperature from Celsius to Fahrenheit.

        Args:
            celsius: The temperature in degrees Celsius.
        \"\"\"
        return celsius * 9 / 5 + 32

    registry = ToolRegistry([celsius_to_fahrenheit])

Pass that registry to a model or an :class:`~minerva.runtime.agent.Agent` and
the model can call it. See ``docs/ADDING_A_TOOL.md``.
"""

from __future__ import annotations

from .base import FunctionTool, Tool, ToolSpec, python_type_to_json_schema, tool
from .registry import ToolRegistry, default_registry

__all__ = [
    "FunctionTool",
    "Tool",
    "ToolRegistry",
    "ToolSpec",
    "default_registry",
    "python_type_to_json_schema",
    "tool",
]
