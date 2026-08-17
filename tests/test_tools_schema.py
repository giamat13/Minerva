"""Tests for tool definition and JSON Schema inference.

The schema a model sees is derived from real type hints and real docstrings, so
these tests assert on the actual generated schema - if inference regresses, a
model starts calling tools with the wrong arguments.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import pytest

from minerva.errors import ToolExecutionError, ToolValidationError
from minerva.tools.base import (
    FunctionTool,
    Tool,
    python_type_to_json_schema,
    tool,
)


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


class TestTypeMapping:
    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (str, {"type": "string"}),
            (int, {"type": "integer"}),
            (float, {"type": "number"}),
            (bool, {"type": "boolean"}),
            (list[str], {"type": "array", "items": {"type": "string"}}),
            (dict[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
        ],
    )
    def test_primitives_and_containers(self, annotation: object, expected: dict) -> None:
        assert python_type_to_json_schema(annotation) == expected

    def test_optional_unwraps_to_the_inner_type(self) -> None:
        assert python_type_to_json_schema(str | None) == {"type": "string"}

    def test_union_becomes_any_of(self) -> None:
        schema = python_type_to_json_schema(int | str)
        assert schema == {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    def test_literal_becomes_an_enum(self) -> None:
        schema = python_type_to_json_schema(Literal["celsius", "fahrenheit"])
        assert schema == {"enum": ["celsius", "fahrenheit"], "type": "string"}

    def test_enum_class_becomes_an_enum_of_values(self) -> None:
        assert python_type_to_json_schema(Colour) == {"enum": ["red", "blue"], "type": "string"}

    def test_unknown_types_degrade_to_an_open_schema(self) -> None:
        class Whatever:
            pass

        assert python_type_to_json_schema(Whatever) == {}


class TestDecorator:
    def test_builds_a_tool_from_hints_and_docstring(self) -> None:
        @tool
        def convert_temperature(value: float, unit: str = "celsius") -> str:
            """Convert a temperature between units.

            Args:
                value: The numeric temperature to convert.
                unit: The unit to convert into.
            """
            return f"{value} {unit}"

        assert isinstance(convert_temperature, Tool)
        assert convert_temperature.name == "convert_temperature"
        assert convert_temperature.description == "Convert a temperature between units."

        params = convert_temperature.parameters
        assert params["type"] == "object"
        assert params["required"] == ["value"], "only the argument without a default is required"
        assert params["properties"]["value"] == {
            "type": "number",
            "description": "The numeric temperature to convert.",
        }
        assert params["properties"]["unit"]["default"] == "celsius"
        assert params["additionalProperties"] is False

    def test_supports_sphinx_style_docstrings(self) -> None:
        @tool
        def lookup(term: str) -> str:
            """Look something up.

            :param term: The term to look up.
            """
            return term

        assert lookup.parameters["properties"]["term"]["description"] == "The term to look up."

    def test_name_and_description_can_be_overridden(self) -> None:
        @tool(name="add_numbers", description="Add two integers together.")
        def _add(a: int, b: int) -> int:
            """Internal docstring nobody should see."""
            return a + b

        assert _add.name == "add_numbers"
        assert _add.description == "Add two integers together."

    def test_multiline_summary_is_joined(self) -> None:
        @tool
        def wordy(x: int) -> int:
            """A summary that happens
            to span two lines.

            Args:
                x: A number.
            """
            return x

        assert wordy.description == "A summary that happens to span two lines."

    def test_the_wrapped_function_is_still_callable(self) -> None:
        @tool
        def double(x: int) -> int:
            """Double a number.

            Args:
                x: The number.
            """
            return x * 2

        assert double(21) == 42
        assert double.run(x=21) == 42

    def test_variadics_are_excluded_from_the_schema(self) -> None:
        @tool
        def flexible(first: str, *rest: str, **options: str) -> str:
            """Do something flexible.

            Args:
                first: The first argument.
            """
            return first

        assert list(flexible.parameters["properties"]) == ["first"]

    def test_spec_is_the_openai_style_wire_format(self) -> None:
        @tool
        def ping() -> str:
            """Return pong."""
            return "pong"

        spec = ping.spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "ping"
        assert spec["function"]["parameters"]["properties"] == {}
        assert "required" not in spec["function"]["parameters"]


class TestValidationAndInvocation:
    @pytest.fixture
    def greet(self) -> FunctionTool:
        @tool
        def greet(name: str, excited: bool = False) -> str:
            """Greet somebody.

            Args:
                name: Who to greet.
                excited: Whether to shout.
            """
            if name == "boom":
                raise RuntimeError("kaboom")
            return f"Hello {name}{'!' if excited else '.'}"

        return greet

    def test_invoke_runs_the_real_function(self, greet: FunctionTool) -> None:
        assert greet.invoke({"name": "Dana", "excited": True}) == "Hello Dana!"

    def test_missing_required_argument_is_rejected(self, greet: FunctionTool) -> None:
        with pytest.raises(ToolValidationError, match="missing required argument"):
            greet.invoke({"excited": True})

    def test_unknown_argument_is_rejected(self, greet: FunctionTool) -> None:
        with pytest.raises(ToolValidationError, match="unknown argument"):
            greet.invoke({"name": "Dana", "volume": 11})

    def test_an_exception_inside_the_tool_is_wrapped(self, greet: FunctionTool) -> None:
        with pytest.raises(ToolExecutionError) as exc_info:
            greet.invoke({"name": "boom"})
        assert isinstance(exc_info.value.original, RuntimeError)
        assert exc_info.value.tool_name == "greet"


class TestSubclassing:
    def test_a_stateful_tool_can_subclass_tool_directly(self) -> None:
        class CounterTool(Tool):
            name = "counter"
            description = "Increment and return a counter."

            def __init__(self) -> None:
                self.count = 0

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                    "additionalProperties": False,
                }

            def run(self, step: int = 1) -> int:
                self.count += step
                return self.count

        counter = CounterTool()
        assert counter.invoke({"step": 5}) == 5
        assert counter.invoke({"step": 2}) == 7
        assert counter.spec()["function"]["name"] == "counter"
