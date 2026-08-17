"""Tests for the agent loop's tool-execution machinery.

The full loop needs a live model, so it is exercised end to end in
``tests/integration/test_live_ollama.py``. What is tested here is everything in
the loop that does *not* require inference: real tool execution, the callbacks,
the iteration ceiling, and how the loop reports a tool that fails.
"""

from __future__ import annotations

import pytest

from conftest import CAPABLE_SPEC
from minerva.engines.ollama import OllamaEngine
from minerva.messages import Role, ToolCall, ToolResult
from minerva.models.base import MinervaModel
from minerva.models.swift import SWIFT
from minerva.runtime.agent import Agent
from minerva.thinking import ThinkingLevel
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
    raise RuntimeError("as designed")


@pytest.fixture
def agent() -> Agent:
    """An agent bound to a real engine object; no inference happens in these tests."""
    model = MinervaModel(CAPABLE_SPEC, OllamaEngine(host="http://127.0.0.1:11434"))
    return Agent(model, tools=ToolRegistry([add, explode]))


class TestToolExecution:
    def test_calls_run_for_real_and_produce_tool_messages(self, agent: Agent) -> None:
        call = ToolCall(name="add", arguments={"a": 17, "b": 43})
        results, messages = agent._run_tools([call])

        assert [r.content for r in results] == ["60"]
        assert messages[0].role is Role.TOOL
        assert messages[0].tool_call_id == call.id

    def test_several_calls_run_in_order(self, agent: Agent) -> None:
        calls = [
            ToolCall(name="add", arguments={"a": 1, "b": 1}),
            ToolCall(name="add", arguments={"a": 10, "b": 10}),
        ]
        results, messages = agent._run_tools(calls)
        assert [r.content for r in results] == ["2", "20"]
        assert len(messages) == 2

    def test_a_failing_tool_is_reported_back_not_raised(self, agent: Agent) -> None:
        results, messages = agent._run_tools([ToolCall(name="explode", arguments={})])
        assert results[0].is_error is True
        assert "as designed" in results[0].content
        assert messages[0].is_error is True

    def test_an_unknown_tool_is_reported_back(self, agent: Agent) -> None:
        results, _ = agent._run_tools([ToolCall(name="teleport", arguments={})])
        assert results[0].is_error is True
        assert "add" in results[0].content

    def test_a_call_with_no_tools_available_says_so_clearly(self) -> None:
        # Silently dropping the call sends the model into a loop.
        model = MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))
        bare = Agent(model, tools=None)
        results, _ = bare._run_tools([ToolCall(name="add", arguments={})])
        assert results[0].is_error is True
        assert "no tools are available" in results[0].content


class TestCallbacks:
    def test_on_tool_call_fires_before_execution(self) -> None:
        seen: list[ToolCall] = []
        model = MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))
        agent = Agent(model, tools=ToolRegistry([add]), on_tool_call=seen.append)

        agent._run_tools([ToolCall(name="add", arguments={"a": 1, "b": 2})])
        assert [call.name for call in seen] == ["add"]

    def test_on_tool_result_fires_after_execution(self) -> None:
        seen: list[ToolResult] = []
        model = MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))
        agent = Agent(model, tools=ToolRegistry([add]), on_tool_result=seen.append)

        agent._run_tools([ToolCall(name="add", arguments={"a": 1, "b": 2})])
        assert [result.content for result in seen] == ["3"]


class TestIterationCeiling:
    def test_it_defaults_to_the_thinking_levels_ceiling(self, agent: Agent) -> None:
        assert agent._iteration_ceiling(ThinkingLevel.DO, None) == (
            ThinkingLevel.DO.profile.max_tool_iterations
        )
        assert agent._iteration_ceiling(ThinkingLevel.SI, None) == (
            ThinkingLevel.SI.profile.max_tool_iterations
        )

    def test_deeper_thinking_gets_a_longer_leash(self, agent: Agent) -> None:
        assert agent._iteration_ceiling(ThinkingLevel.SI, None) > agent._iteration_ceiling(
            ThinkingLevel.DO, None
        )

    def test_an_agent_level_override_wins_over_the_level(self) -> None:
        model = MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))
        agent = Agent(model, max_iterations=3)
        assert agent._iteration_ceiling(ThinkingLevel.SI, None) == 3

    def test_a_per_run_override_wins_over_everything(self, agent: Agent) -> None:
        assert agent._iteration_ceiling(ThinkingLevel.SI, 1) == 1


class TestThinkingResolution:
    def test_the_models_ceiling_is_honoured(self, agent: Agent) -> None:
        # CAPABLE_SPEC tops out at SOL, so asking for SI must clamp.
        assert agent._effective_level("si") is ThinkingLevel.SOL

    def test_the_agent_default_is_used_when_no_level_is_given(self) -> None:
        model = MinervaModel(CAPABLE_SPEC, OllamaEngine(host="http://127.0.0.1:11434"))
        agent = Agent(model, thinking="mi")
        assert agent._effective_level(None) is ThinkingLevel.MI

    def test_a_per_run_level_wins(self) -> None:
        model = MinervaModel(CAPABLE_SPEC, OllamaEngine(host="http://127.0.0.1:11434"))
        agent = Agent(model, thinking="mi")
        assert agent._effective_level("do") is ThinkingLevel.DO

    def test_a_base_model_collapses_every_level_to_silence(self) -> None:
        # Swift was never trained to reason, so the scale degrades to DO.
        swift = MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))
        assert Agent(swift, thinking="si")._effective_level(None) is ThinkingLevel.DO


class TestPromptPreparation:
    def test_a_string_becomes_a_user_message(self, agent: Agent) -> None:
        messages = agent._prepare("hello")
        assert len(messages) == 1
        assert messages[0].role is Role.USER
        assert messages[0].content == "hello"

    def test_a_message_list_is_copied_not_aliased(self, agent: Agent) -> None:
        from minerva.messages import user

        original = [user("hi")]
        prepared = agent._prepare(original)
        prepared.append(user("more"))
        assert len(original) == 1


class TestDefaults:
    def test_an_agent_inherits_the_models_tools(self) -> None:
        model = MinervaModel(
            CAPABLE_SPEC, OllamaEngine(host="http://127.0.0.1:11434"), tools=default_registry()
        )
        assert Agent(model).tools is default_registry()
