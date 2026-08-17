"""The tool registry.

A :class:`ToolRegistry` is the set of tools a given conversation may use.
Registries are cheap and independent, so different agents can expose different
tool surfaces from the same process - which is exactly what you want when one
agent may read the filesystem and another may not.

``default_registry()`` returns the process-wide registry pre-loaded with
Minerva's built-in tools.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..errors import ToolExecutionError, ToolNotFoundError, ToolValidationError
from ..messages import Message, ToolCall, ToolResult
from .base import FunctionTool, Tool, ToolSpec

__all__ = ["ToolRegistry", "default_registry"]


class ToolRegistry:
    """A named collection of :class:`~minerva.tools.base.Tool` objects."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools or ():
            self.register(item)

    # -- registration ----------------------------------------------------

    def register(self, tool: Tool | Callable[..., Any], *, replace: bool = False) -> Tool:
        """Add a tool (or a plain function, which is wrapped automatically).

        :param replace: allow overwriting an existing name. Off by default so a
            duplicate registration is a loud error rather than a silent swap.
        """
        if not isinstance(tool, Tool):
            if not callable(tool):
                raise TypeError(f"cannot register {tool!r}: not a Tool and not callable")
            tool = FunctionTool(tool)

        if tool.name in self._tools and not replace:
            raise ValueError(
                f"a tool named {tool.name!r} is already registered; "
                f"pass replace=True to override it"
            )
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> None:
        """Remove a tool. Raises :class:`ToolNotFoundError` if it is not there."""
        if name not in self._tools:
            raise ToolNotFoundError(f"no tool named {name!r}")
        del self._tools[name]

    # -- lookup ----------------------------------------------------------

    def get(self, name: str) -> Tool:
        """Return the tool called ``name``.

        :raises ToolNotFoundError: with the list of names that *are* available,
            which is what the model needs in order to correct itself.
        """
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise ToolNotFoundError(
                f"no tool named {name!r}; available tools: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """A new registry containing only the named tools."""
        return ToolRegistry(self.get(name) for name in names)

    def specs(self) -> list[ToolSpec]:
        """Wire-format specs for every tool, ready to hand to an engine."""
        return [self._tools[name].spec() for name in sorted(self._tools)]

    # -- execution -------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call and capture the outcome.

        Errors are returned as :class:`~minerva.messages.ToolResult` objects
        with ``is_error=True`` rather than raised: the model gets to see what
        went wrong and try again, which is how a tool-using agent recovers.
        Bugs in Minerva itself still propagate.
        """
        try:
            tool = self.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult(call_id=call.id, name=call.name, content=str(exc), is_error=True)

        try:
            value = tool.invoke(call.arguments)
        except (ToolValidationError, ToolExecutionError) as exc:
            return ToolResult(call_id=call.id, name=call.name, content=str(exc), is_error=True)

        return ToolResult(
            call_id=call.id, name=call.name, content=_stringify(value), is_error=False
        )

    def execute_all(self, calls: Iterable[ToolCall]) -> list[Message]:
        """Run several calls in order and return the ``tool`` messages to append."""
        return [self.execute(call).to_message() for call in calls]

    # -- container protocol ----------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools[name] for name in sorted(self._tools))

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry {len(self._tools)} tool(s): {', '.join(self.names()) or '-'}>"


def _stringify(value: Any) -> str:
    """Render a tool's return value as the text the model will read."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    try:
        return json.dumps(value, ensure_ascii=False, default=str, indent=None)
    except (TypeError, ValueError):
        return str(value)


_DEFAULT_REGISTRY: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """The process-wide registry, populated with Minerva's built-in tools.

    Built lazily so importing :mod:`minerva.tools` never triggers the import of
    every built-in tool's dependencies.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from .builtin import builtin_tools

        _DEFAULT_REGISTRY = ToolRegistry(builtin_tools())
    return _DEFAULT_REGISTRY
