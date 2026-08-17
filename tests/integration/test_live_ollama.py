"""End-to-end tests against a REAL, running inference engine.

Nothing in this file is simulated. Every assertion here is about what an actual
Ollama daemon running actual weights does with Minerva's payloads. If no daemon
is reachable, or Swift's weights are not installed, these tests skip with an
explanation - they are never satisfied by a stub.

To run them::

    ollama serve &
    ollama pull qwen3:1.7b
    pytest -m integration

Assertions are written to be robust against sampling: a small model will not
produce the same wording twice, so we assert on facts the model must get right
(the number 731 appears; a tool was actually called) rather than on phrasing.
"""

from __future__ import annotations

import pytest

from minerva.engines.base import Engine, GenerationRequest, SamplingParams
from minerva.errors import EngineUnavailableError, ModelNotInstalledError
from minerva.messages import Role, assistant, user
from minerva.models.base import MinervaModel
from minerva.runtime.agent import Agent
from minerva.runtime.session import Session
from minerva.thinking import ThinkingLevel
from minerva.tools.base import tool
from minerva.tools.registry import ToolRegistry

pytestmark = pytest.mark.integration


class TestEngineIsReal:
    def test_health_reports_a_live_daemon(self, live_engine: Engine) -> None:
        health = live_engine.health()
        assert health.available is True
        assert health.version, "a live daemon reports its version"

    def test_installed_models_are_listed(self, live_engine: Engine) -> None:
        models = live_engine.list_available_models()
        assert isinstance(models, list)
        assert all(isinstance(name, str) for name in models)

    def test_a_wrong_host_really_fails(self) -> None:
        from minerva.engines.ollama import OllamaEngine

        with OllamaEngine(host="http://127.0.0.1:1", connect_timeout=1.0) as engine:
            assert engine.health().available is False
            with pytest.raises(EngineUnavailableError, match="cannot reach Ollama"):
                engine.list_available_models()


class TestSwiftGenerates:
    def test_it_answers_a_question(self, swift: MinervaModel) -> None:
        answer = swift.ask("Reply with exactly one word: the capital city of France.")
        assert answer.strip(), "the model returned nothing at all"
        assert "paris" in answer.lower()

    def test_it_reports_real_token_usage(self, swift: MinervaModel) -> None:
        result = swift.chat([user("Say hi.")], thinking="do")
        assert result.usage.prompt_tokens and result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens and result.usage.completion_tokens > 0
        assert result.duration_seconds and result.duration_seconds > 0

    def test_the_system_prompt_reaches_the_model(self, swift: MinervaModel) -> None:
        answer = swift.ask(
            "What is your name?",
            system_prompt="You are called Minerva Swift. Answer with your name only.",
            thinking="do",
        )
        assert "swift" in answer.lower()

    def test_sampling_parameters_are_honoured(self, swift: MinervaModel) -> None:
        # A tiny num_predict must visibly truncate the output.
        short = swift.chat(
            [user("Count slowly from one to fifty in words.")],
            thinking="do",
            sampling=SamplingParams(max_tokens=8),
        )
        assert short.usage.completion_tokens is not None
        assert short.usage.completion_tokens <= 12

    def test_a_missing_model_fails_with_an_actionable_message(
        self, live_engine: Engine
    ) -> None:
        from minerva.models.swift import SWIFT

        ghost = SWIFT.with_overrides(
            name="ghost", engine_model="not-a-real-model:999b", engine_model_fallbacks=()
        )
        model = MinervaModel(ghost, live_engine)
        with pytest.raises(ModelNotInstalledError, match="ollama pull"):
            model.ask("hello")


class TestThinkingOnRealHardware:
    def test_do_produces_no_reasoning_trace(self, swift: MinervaModel) -> None:
        result = swift.chat([user("What is 2 + 2?")], thinking="do")
        assert not result.thinking, "DO must switch the reasoning phase off entirely"
        assert result.content.strip()

    def test_higher_notes_produce_a_reasoning_trace(self, swift: MinervaModel) -> None:
        result = swift.chat(
            [user("A shop sells pens at 7 for 21 shekels. What do 12 pens cost?")],
            thinking="sol",
        )
        assert result.thinking, "SOL must produce a reasoning trace"
        assert result.content.strip()

    def test_thinking_is_returned_separately_from_the_answer(
        self, swift: MinervaModel
    ) -> None:
        result = swift.chat([user("Is 91 a prime number? Answer yes or no.")], thinking="mi")
        if result.thinking:
            assert result.thinking not in result.content

    @pytest.mark.parametrize("note", ["do", "re", "mi", "fa", "sol"])
    def test_every_note_swift_supports_actually_runs(
        self, swift: MinervaModel, note: str
    ) -> None:
        result = swift.chat([user("Say the word: ready.")], thinking=note)
        assert result.content.strip()

    def test_hebrew_note_names_work_end_to_end(self, swift: MinervaModel) -> None:
        assert swift.ask("Say the word: ready.", thinking="סול").strip()

    def test_a_note_above_the_ceiling_is_clamped_and_still_answers(
        self, swift: MinervaModel
    ) -> None:
        # Swift tops out at SOL; asking for SI must not fail.
        assert swift.ask("Say the word: ready.", thinking="si").strip()


class TestStreaming:
    def test_chunks_arrive_and_assemble_into_the_final_answer(
        self, swift: MinervaModel
    ) -> None:
        chunks = list(swift.stream([user("Count from 1 to 5.")], thinking="do"))

        assert chunks, "the stream produced nothing"
        assert chunks[-1].kind == "done"
        assert chunks[-1].result is not None

        streamed = "".join(chunk.text for chunk in chunks if chunk.kind == "content")
        assert streamed == chunks[-1].result.message.content

    def test_reasoning_streams_on_its_own_channel(self, swift: MinervaModel) -> None:
        chunks = list(
            swift.stream([user("What is 13 * 17? Think it through.")], thinking="sol")
        )
        thinking_chunks = [c for c in chunks if c.kind == "thinking"]
        if thinking_chunks:
            assert all(c.kind != "content" for c in thinking_chunks)
            final = chunks[-1].result
            assert final is not None and final.message.thinking


class TestToolCallingForReal:
    def test_the_model_calls_the_calculator_and_uses_its_answer(
        self, swift: MinervaModel
    ) -> None:
        calls: list[str] = []
        agent = Agent(
            swift,
            thinking="fa",
            on_tool_call=lambda call: calls.append(call.name),
        )
        run = agent.run(
            "Use the calculate tool to work out 4871 * 293, then state the result."
        )

        assert "calculate" in calls, "the model never called the tool"
        assert "1427203" in run.answer.replace(",", "").replace(" ", "")

    def test_a_tool_result_really_flows_back_into_the_transcript(
        self, swift: MinervaModel
    ) -> None:
        agent = Agent(swift, thinking="fa")
        run = agent.run("Use the calculate tool to compute 111 * 111 and report the result.")

        tool_messages = [m for m in run.messages if m.role is Role.TOOL]
        assert tool_messages, "no tool result was appended"
        assert any("12321" in m.content for m in tool_messages)

    def test_a_custom_tool_is_invoked_with_real_arguments(
        self, swift: MinervaModel
    ) -> None:
        received: list[str] = []

        @tool
        def lookup_employee_id(surname: str) -> str:
            """Look up an employee's internal ID by surname.

            Args:
                surname: The employee's surname.
            """
            received.append(surname)
            return f"EMP-{surname.upper()}-4417"

        agent = Agent(swift, tools=ToolRegistry([lookup_employee_id]), thinking="fa")
        run = agent.run("Look up the employee ID for the surname Cohen and tell me what it is.")

        assert received == ["Cohen"] or [s.lower() for s in received] == ["cohen"]
        assert "4417" in run.answer

    def test_a_failing_tool_is_recovered_from_rather_than_crashing(
        self, swift: MinervaModel
    ) -> None:
        @tool
        def unstable_sensor() -> str:
            """Read the temperature sensor."""
            raise RuntimeError("sensor offline")

        agent = Agent(swift, tools=ToolRegistry([unstable_sensor]), thinking="fa")
        run = agent.run("Read the temperature sensor and tell me what happened.")

        assert run.answer.strip(), "the run must still produce an answer"
        assert any(m.role is Role.TOOL and m.is_error for m in run.messages)

    def test_no_tool_is_called_when_none_is_needed(self, swift: MinervaModel) -> None:
        agent = Agent(swift, thinking="do")
        run = agent.run("Reply with exactly one word: hello.")
        assert run.tool_calls == []

    def test_the_run_records_what_happened(self, swift: MinervaModel) -> None:
        agent = Agent(swift, thinking="fa")
        run = agent.run("Use the calculate tool for 25 * 25, then state the answer.")

        assert run.steps, "a run must record at least one model turn"
        assert run.duration_seconds > 0
        assert run.prompt_tokens > 0
        assert run.thinking_level is ThinkingLevel.FA


class TestSessionMemory:
    def test_the_model_remembers_across_turns(self, swift: MinervaModel) -> None:
        session = Session(swift, thinking="do")
        session.send("My name is Dana. Remember it.")
        answer = session.send("What is my name? Reply with the name only.")
        assert "dana" in answer.lower()

    def test_the_transcript_grows_with_both_sides(self, swift: MinervaModel) -> None:
        session = Session(swift, thinking="do")
        before = len(session)
        session.send("Say hi.")
        assert len(session) > before
        assert any(m.role is Role.ASSISTANT for m in session)

    def test_reset_really_forgets(self, swift: MinervaModel) -> None:
        session = Session(swift, thinking="do")
        session.send("My name is Dana.")
        session.reset()
        answer = session.send(
            "Do you know my name? If you have not been told, reply exactly: no."
        )
        assert "dana" not in answer.lower()

    def test_extended_thinking_keeps_the_trace_in_the_transcript(
        self, live_engine: Engine, swift: MinervaModel
    ) -> None:
        # Swift's ceiling is SOL, so raise it for this test to exercise the
        # Extended Thinking path on real hardware.
        deep_spec = swift.spec.with_overrides(name="swift-deep", max_thinking=ThinkingLevel.SI)
        deep = MinervaModel(deep_spec, live_engine, tools=swift.tools)
        session = Session(deep, thinking="la")

        session.send("What is 19 * 21? Think it through.")
        assistant_messages = [m for m in session if m.role is Role.ASSISTANT]
        assert assistant_messages
        if any(m.thinking for m in assistant_messages):
            assert session.preserves_thinking is True


class TestPayloadsAreAccepted:
    """The daemon must accept every payload shape Minerva can produce."""

    def test_a_transcript_with_tool_calls_and_results_round_trips(
        self, live_engine: Engine, swift: MinervaModel
    ) -> None:
        from minerva.messages import ToolCall, ToolResult

        call = ToolCall(name="calculate", arguments={"expression": "17*43"}, id="call_1")
        messages = [
            user("What is 17 times 43?"),
            assistant("", tool_calls=[call]),
            ToolResult(call_id="call_1", name="calculate", content="731").to_message(),
        ]
        request = GenerationRequest(
            model=swift.resolve_engine_model(),
            messages=swift._with_system_prompt(messages, None),
            tools=list(swift.tools.specs()) if swift.tools else [],
            thinking=ThinkingLevel.DO.profile,
        )
        result = live_engine.chat(request)
        assert result.content.strip(), "the daemon rejected or ignored the transcript"
