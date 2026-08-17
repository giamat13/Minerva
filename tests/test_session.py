"""Tests for session state: history, trimming, the thinking policy, persistence.

Sending a message needs a live model, so that path lives in the integration
suite. Everything here is state management, which does not.
"""

from __future__ import annotations

from pathlib import Path

from minerva.engines.ollama import OllamaEngine
from minerva.messages import Message, Role, ToolCall, assistant, user
from minerva.models.base import MinervaModel
from minerva.models.swift import SWIFT
from minerva.runtime.session import Session
from minerva.thinking import ThinkingLevel


def _model() -> MinervaModel:
    return MinervaModel(SWIFT, OllamaEngine(host="http://127.0.0.1:11434"))


class TestInitialState:
    def test_the_models_system_prompt_seeds_the_transcript(self) -> None:
        session = Session(_model())
        assert len(session) == 1
        assert session.messages[0].role is Role.SYSTEM
        assert session.messages[0].content == SWIFT.system_prompt

    def test_an_explicit_system_prompt_wins(self) -> None:
        session = Session(_model(), system_prompt="be terse")
        assert session.messages[0].content == "be terse"

    def test_an_empty_system_prompt_seeds_nothing(self) -> None:
        assert len(Session(_model(), system_prompt="")) == 0

    def test_the_session_inherits_the_models_tools(self) -> None:
        from minerva.tools.registry import default_registry

        model = MinervaModel(SWIFT, OllamaEngine(host="http://x"), tools=default_registry())
        assert Session(model).tools is default_registry()


class TestThinkingPolicy:
    def test_the_level_is_clamped_to_the_models_ceiling(self) -> None:
        assert Session(_model(), thinking="si").level is ThinkingLevel.SOL

    def test_extended_levels_preserve_reasoning(self) -> None:
        # Use a model whose ceiling actually reaches LA.
        spec = SWIFT.with_overrides(name="deep", max_thinking=ThinkingLevel.SI)
        model = MinervaModel(spec, OllamaEngine(host="http://x"))
        assert Session(model, thinking="la").preserves_thinking is True

    def test_lower_levels_do_not_preserve_reasoning(self) -> None:
        assert Session(_model(), thinking="mi").preserves_thinking is False

    def test_keep_thinking_overrides_the_policy_either_way(self) -> None:
        assert Session(_model(), thinking="mi", keep_thinking=True).preserves_thinking is True
        assert Session(_model(), thinking="si", keep_thinking=False).preserves_thinking is False

    def test_reasoning_is_stripped_when_the_policy_says_so(self) -> None:
        session = Session(_model(), thinking="mi")
        messages = [assistant("answer", thinking="scratch work")]
        assert session._apply_thinking_policy(messages)[0].thinking is None

    def test_reasoning_survives_when_extended_thinking_is_on(self) -> None:
        session = Session(_model(), thinking="mi", keep_thinking=True)
        messages = [assistant("answer", thinking="scratch work")]
        assert session._apply_thinking_policy(messages)[0].thinking == "scratch work"


class TestTrimming:
    def test_history_is_unbounded_by_default(self) -> None:
        session = Session(_model(), system_prompt="")
        session.messages = [user(f"message {i}") for i in range(50)]
        session._trim()
        assert len(session) == 50

    def test_the_oldest_turns_are_dropped_past_the_limit(self) -> None:
        session = Session(_model(), system_prompt="", max_history=4)
        session.messages = [user(f"message {i}") for i in range(10)]
        session._trim()
        assert len(session) == 4
        assert session.messages[0].content == "message 6"

    def test_the_system_prompt_is_pinned(self) -> None:
        session = Session(_model(), max_history=2)
        session.messages += [user(f"message {i}") for i in range(10)]
        session._trim()
        assert session.messages[0].role is Role.SYSTEM
        assert len(session) == 3  # system + 2

    def test_an_orphaned_tool_result_is_not_left_at_the_front(self) -> None:
        # A tool result with no visible call confuses the model.
        session = Session(_model(), system_prompt="", max_history=2)
        session.messages = [
            assistant("", tool_calls=[ToolCall(name="add", id="call_1")]),
            Message(role=Role.TOOL, content="3", tool_call_id="call_1", name="add"),
            assistant("the answer is 3"),
        ]
        session._trim()
        assert session.messages[0].role is not Role.TOOL


class TestReset:
    def test_reset_keeps_the_system_prompt_by_default(self) -> None:
        session = Session(_model())
        session.messages.append(user("hello"))
        session.reset()
        assert len(session) == 1
        assert session.messages[0].role is Role.SYSTEM

    def test_reset_can_drop_everything(self) -> None:
        session = Session(_model())
        session.messages.append(user("hello"))
        session.reset(keep_system=False)
        assert len(session) == 0


class TestPersistence:
    def test_round_trip_through_disk(self, tmp_path: Path) -> None:
        session = Session(_model())
        session.messages += [
            user("what is 17 * 43?"),
            assistant("731", thinking="17*40=680, +51"),
        ]
        path = session.save(tmp_path / "transcript.json")
        assert path.exists()

        restored = Session(_model(), system_prompt="")
        restored.load(path)
        assert [m.content for m in restored] == [m.content for m in session]
        assert restored.messages[-1].thinking == "17*40=680, +51"

    def test_json_records_the_model_and_level(self) -> None:
        import json

        data = json.loads(Session(_model(), thinking="mi").to_json())
        assert data["model"] == "swift"
        assert data["thinking_level"] == "mi"

    def test_unicode_is_not_escaped(self) -> None:
        session = Session(_model(), system_prompt="")
        session.messages.append(user("שלום"))
        assert "שלום" in session.to_json()


class TestAgentConstruction:
    def test_the_agent_does_not_re_add_the_system_prompt(self) -> None:
        # The prompt is already message zero of the transcript.
        session = Session(_model())
        assert session.agent().system_prompt == ""

    def test_overrides_reach_the_agent(self) -> None:
        session = Session(_model(), thinking="mi")
        assert session.agent(thinking="do").thinking == "do"
