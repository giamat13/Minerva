"""Tests for the tool registry and real tool execution."""

from __future__ import annotations

import json

import pytest

from minerva.errors import ToolNotFoundError
from minerva.messages import Role, ToolCall
from minerva.tools.base import tool
from minerva.tools.registry import ToolRegistry, default_registry


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def explode() -> str:
    """Always fail, on purpose."""
    raise ValueError("as designed")


class TestRegistration:
    def test_register_and_get(self, empty_registry: ToolRegistry) -> None:
        empty_registry.register(add)
        assert empty_registry.get("add") is add
        assert empty_registry.names() == ["add"]
        assert "add" in empty_registry
        assert len(empty_registry) == 1

    def test_a_plain_function_is_wrapped_automatically(self, empty_registry: ToolRegistry) -> None:
        def subtract(a: int, b: int) -> int:
            """Subtract b from a.

            Args:
                a: The first number.
                b: The number to subtract.
            """
            return a - b

        registered = empty_registry.register(subtract)
        assert registered.name == "subtract"
        call = ToolCall(name="subtract", arguments={"a": 5, "b": 3})
        assert empty_registry.execute(call).content == "2"

    def test_duplicate_registration_is_an_error_by_default(
        self, empty_registry: ToolRegistry
    ) -> None:
        empty_registry.register(add)
        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register(add)

    def test_replace_allows_overriding(self, empty_registry: ToolRegistry) -> None:
        empty_registry.register(add)
        empty_registry.register(add, replace=True)
        assert len(empty_registry) == 1

    def test_unregister(self, empty_registry: ToolRegistry) -> None:
        empty_registry.register(add)
        empty_registry.unregister("add")
        assert "add" not in empty_registry

    def test_unregister_unknown_is_an_error(self, empty_registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            empty_registry.unregister("nope")

    def test_non_callables_are_refused(self, empty_registry: ToolRegistry) -> None:
        with pytest.raises(TypeError):
            empty_registry.register(42)  # type: ignore[arg-type]


class TestLookup:
    def test_missing_tool_error_lists_what_is_available(self) -> None:
        registry = ToolRegistry([add])
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get("multiply")
        assert "add" in str(exc_info.value)

    def test_subset_builds_a_narrower_registry(self) -> None:
        registry = ToolRegistry([add, explode])
        narrow = registry.subset(["add"])
        assert narrow.names() == ["add"]
        assert len(registry) == 2, "the original registry is untouched"

    def test_specs_are_sorted_and_well_formed(self) -> None:
        specs = ToolRegistry([explode, add]).specs()
        assert [spec["function"]["name"] for spec in specs] == ["add", "explode"]
        assert json.dumps(specs), "specs must be JSON-serialisable for the wire"


class TestExecution:
    def test_a_successful_call_returns_the_real_result(self) -> None:
        registry = ToolRegistry([add])
        result = registry.execute(ToolCall(name="add", arguments={"a": 17, "b": 43}))
        assert result.content == "60"
        assert result.is_error is False

    def test_a_failing_tool_becomes_an_error_result_not_an_exception(self) -> None:
        # The model must get to see the failure and recover from it.
        registry = ToolRegistry([explode])
        result = registry.execute(ToolCall(name="explode", arguments={}))
        assert result.is_error is True
        assert "as designed" in result.content

    def test_an_unknown_tool_becomes_an_error_result(self) -> None:
        registry = ToolRegistry([add])
        result = registry.execute(ToolCall(name="teleport", arguments={}))
        assert result.is_error is True
        assert "add" in result.content, "the model needs to see what it may call instead"

    def test_bad_arguments_become_an_error_result(self) -> None:
        registry = ToolRegistry([add])
        result = registry.execute(ToolCall(name="add", arguments={"a": 1}))
        assert result.is_error is True
        assert "missing required argument" in result.content

    def test_results_are_rendered_as_tool_messages(self) -> None:
        registry = ToolRegistry([add])
        call = ToolCall(name="add", arguments={"a": 1, "b": 2})
        messages = registry.execute_all([call])
        assert len(messages) == 1
        assert messages[0].role is Role.TOOL
        assert messages[0].tool_call_id == call.id
        assert messages[0].name == "add"
        assert messages[0].content == "3"

    def test_structured_return_values_are_json_encoded(self, empty_registry: ToolRegistry) -> None:
        @tool
        def profile() -> dict:
            """Return a small record."""
            return {"name": "swift", "wingspan_cm": 42}

        empty_registry.register(profile)
        content = empty_registry.execute(ToolCall(name="profile", arguments={})).content
        assert json.loads(content) == {"name": "swift", "wingspan_cm": 42}

    def test_unicode_survives_the_round_trip(self, empty_registry: ToolRegistry) -> None:
        @tool
        def greet_hebrew() -> dict:
            """Return a Hebrew greeting."""
            return {"greeting": "שלום"}

        empty_registry.register(greet_hebrew)
        content = empty_registry.execute(ToolCall(name="greet_hebrew", arguments={})).content
        assert "שלום" in content


class TestDefaultRegistry:
    def test_it_is_a_shared_singleton(self) -> None:
        assert default_registry() is default_registry()

    def test_it_contains_the_builtins(self) -> None:
        names = default_registry().names()
        assert {"calculate", "current_time", "days_between"} <= set(names)
