"""Tool definitions and JSON-Schema generation.

A *tool* is a Python callable the model may invoke mid-conversation. Minerva
supports two ways to define one, and both produce the same
:class:`Tool` interface:

1. The :func:`tool` decorator, for the common case. The JSON Schema the model
   sees is derived from the function's real type hints and docstring - there is
   no second, hand-written copy of the signature to drift out of sync.

2. Subclassing :class:`Tool` directly, when a tool needs state (a database
   handle, an HTTP session, a cache) or a schema too intricate to infer.

See ``docs/ADDING_A_TOOL.md`` for a walkthrough.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from ..errors import ToolExecutionError, ToolValidationError

__all__ = ["FunctionTool", "Tool", "ToolSpec", "python_type_to_json_schema", "tool"]


class ToolSpec(dict):
    """The OpenAI/Ollama-style function description sent to the engine.

    It is a plain ``dict`` so it can be serialised straight onto the wire::

        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": {...}}}
    """

    @classmethod
    def build(cls, name: str, description: str, parameters: dict[str, Any]) -> ToolSpec:
        return cls(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )


class Tool(ABC):
    """Base class for everything the model can call.

    Subclasses provide :attr:`name`, :attr:`description`, :attr:`parameters`
    (a JSON Schema object) and :meth:`run`.
    """

    #: Engines and models identify a tool by this name; keep it stable.
    name: str
    #: Shown to the model. This is prompt text - write it for the model, and be
    #: specific about when the tool should and should not be used.
    description: str

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema (an ``object`` schema) describing the tool's arguments."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Return anything JSON-serialisable or a string."""

    # -- plumbing --------------------------------------------------------

    def spec(self) -> ToolSpec:
        """The wire-format description handed to the engine."""
        return ToolSpec.build(self.name, self.description, self.parameters)

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check ``arguments`` against :attr:`parameters` before running.

        This is intentionally a light structural check, not a full JSON Schema
        validator: it catches the mistakes small models actually make (missing
        required argument, hallucinated argument name) without pulling in a
        dependency. Subclasses may override for stricter validation.

        :raises ToolValidationError: when the arguments cannot be accepted.
        """
        schema = self.parameters
        properties: dict[str, Any] = schema.get("properties", {})
        required: Sequence[str] = schema.get("required", [])

        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolValidationError(
                f"tool {self.name!r} is missing required argument(s): {', '.join(missing)}"
            )

        if schema.get("additionalProperties") is False:
            unknown = [key for key in arguments if key not in properties]
            if unknown:
                raise ToolValidationError(
                    f"tool {self.name!r} received unknown argument(s): {', '.join(unknown)}; "
                    f"expected only: {', '.join(properties) or '(none)'}"
                )

        return arguments

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """Validate then run, wrapping any failure as a :class:`ToolExecutionError`."""
        validated = self.validate_arguments(dict(arguments))
        try:
            return self.run(**validated)
        except ToolValidationError:
            raise
        except Exception as exc:
            raise ToolExecutionError(self.name, exc) from exc

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class FunctionTool(Tool):
    """A :class:`Tool` backed by a plain Python function.

    Built by the :func:`tool` decorator; you rarely construct it by hand.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        summary, param_docs = _parse_docstring(doc)
        self.description = description or summary or f"Call the {self.name} function."
        self._parameters = parameters or build_parameters_schema(func, param_docs)

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def run(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)

    @property
    def func(self) -> Callable[..., Any]:
        """The wrapped function, so the tool stays directly callable in tests."""
        return self._func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


def tool(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Turn a function into a :class:`Tool`.

    Usage::

        @tool
        def celsius_to_fahrenheit(celsius: float) -> float:
            \"\"\"Convert a temperature from Celsius to Fahrenheit.

            Args:
                celsius: The temperature in degrees Celsius.
            \"\"\"
            return celsius * 9 / 5 + 32

    The schema the model sees comes from the annotations and the ``Args:``
    section, so the documentation and the contract cannot diverge.
    """

    def wrap(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, parameters=parameters)

    if _func is not None:
        return wrap(_func)
    return wrap


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON Schema fragment.

    Handles the shapes that show up in real tool signatures: primitives,
    ``list[T]``, ``dict[str, T]``, ``Literal[...]``, ``Enum`` subclasses,
    ``Optional[T]`` / unions, and ``Any``. Anything else degrades to an
    unconstrained schema rather than raising - a slightly loose schema is far
    better than a tool that refuses to be defined.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    if annotation is None or annotation is type(None):
        return {"type": "null"}

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]
        schema: dict[str, Any] = {"enum": values}
        value_types = {type(v) for v in values}
        if len(value_types) == 1:
            only = next(iter(value_types))
            if only in _PRIMITIVES:
                schema["type"] = _PRIMITIVES[only]
        return schema

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        schema = {"enum": list(args)}
        value_types = {type(v) for v in args}
        if len(value_types) == 1 and next(iter(value_types)) in _PRIMITIVES:
            schema["type"] = _PRIMITIVES[next(iter(value_types))]
        return schema

    # Optional[T] and unions. `X | None` produces types.UnionType on 3.10+.
    if origin is Union or isinstance(annotation, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        if len(non_null) == 1:
            # Optional[T] - the schema is just T's; optionality is expressed by
            # leaving the field out of `required`.
            return python_type_to_json_schema(non_null[0])
        return {"anyOf": [python_type_to_json_schema(a) for a in non_null]}

    if origin in (list, set, frozenset, tuple) or annotation in (list, set, tuple):
        item_schema = python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema} if item_schema else {"type": "array"}

    if origin is dict or annotation is dict:
        value_schema = python_type_to_json_schema(args[1]) if len(args) == 2 else {}
        out: dict[str, Any] = {"type": "object"}
        if value_schema:
            out["additionalProperties"] = value_schema
        return out

    # Unknown type: leave it unconstrained.
    return {}


def build_parameters_schema(
    func: Callable[..., Any], param_docs: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build the ``parameters`` JSON Schema object for ``func``.

    Parameters without a default become ``required``. ``*args``/``**kwargs``
    are skipped: a tool call is a mapping of named arguments, so variadics have
    no meaningful schema.
    """
    param_docs = param_docs or {}
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = getattr(func, "__annotations__", {})

    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        schema = python_type_to_json_schema(hints.get(name, param.annotation))
        if name in param_docs:
            schema["description"] = param_docs[name]
        if param.default is not inspect.Parameter.empty:
            # Only advertise defaults that survive a JSON round-trip.
            if isinstance(param.default, (str, int, float, bool, type(None))):
                schema["default"] = param.default
        else:
            required.append(name)

        properties[name] = schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return parameters


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Params|Parameters)\s*:\s*$", re.IGNORECASE)
_OTHER_HEADER = re.compile(
    r"^\s*(Returns|Raises|Yields|Examples?|Notes?|Attributes)\s*:\s*$", re.IGNORECASE
)
_ARG_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:[\w\[\], ]+\s+)?(\w+)\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param_name: description}).

    Understands Google style (``Args:`` block) and Sphinx style (``:param x:``),
    which between them cover essentially every docstring in the wild.
    """
    if not doc:
        return "", {}

    lines = doc.splitlines()
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}

    in_args = False
    seen_header = False  # everything before the first section header is the summary
    current: str | None = None

    for raw in lines:
        line = raw.rstrip()

        sphinx = _SPHINX_PARAM.match(line)
        if sphinx:
            in_args = False
            seen_header = True
            current = sphinx.group(1)
            param_docs[current] = sphinx.group(2).strip()
            continue

        if _ARGS_HEADER.match(line):
            in_args = True
            seen_header = True
            current = None
            continue

        if _OTHER_HEADER.match(line):
            in_args = False
            seen_header = True
            current = None
            continue

        if in_args:
            if not line.strip():
                current = None
                continue
            match = _ARG_LINE.match(line)
            if match:
                current = match.group(1).lstrip("*")
                param_docs[current] = match.group(2).strip()
            elif current:
                # A continuation line of the previous parameter's description.
                param_docs[current] = f"{param_docs[current]} {line.strip()}".strip()
            continue

        if current and line.startswith((" ", "\t")) and line.strip():
            param_docs[current] = f"{param_docs[current]} {line.strip()}".strip()
            continue

        current = None
        if not seen_header:
            summary_lines.append(line)

    summary = " ".join(part.strip() for part in summary_lines if part.strip())
    return summary.strip(), param_docs
