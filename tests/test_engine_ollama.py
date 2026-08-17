"""Tests for the Ollama engine's request/response codec.

Scope note: these are unit tests of pure translation functions - Minerva types
in, Ollama's documented wire shapes out, and back. No HTTP happens here and no
daemon is simulated; the payload shapes are the ones Ollama's API actually
produces (https://github.com/ollama/ollama/blob/main/docs/api.md).

The engine's *behaviour over the wire* - that a real daemon accepts these
payloads and answers with these shapes - is covered by
``tests/integration/test_live_ollama.py``, which runs against a live daemon or
skips.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from minerva.engines.base import GenerationRequest, SamplingParams
from minerva.engines.ollama import (
    OllamaEngine,
    _downgrade_think,
    _is_think_value_rejection,
)
from minerva.errors import EngineRequestError, EngineResponseError
from minerva.messages import Message, Role, ToolCall, ToolResult, assistant, system, user
from minerva.thinking import ThinkingLevel


@pytest.fixture
def engine() -> OllamaEngine:
    # Constructing an engine opens no socket, so this needs no daemon.
    return OllamaEngine(host="http://127.0.0.1:11434")


class TestMessageEncoding:
    def test_plain_messages(self, engine: OllamaEngine) -> None:
        assert engine._encode_message(user("hello")) == {"role": "user", "content": "hello"}
        assert engine._encode_message(system("rules")) == {"role": "system", "content": "rules"}

    def test_assistant_reasoning_is_sent_back_for_extended_thinking(
        self, engine: OllamaEngine
    ) -> None:
        encoded = engine._encode_message(assistant("answer", thinking="my reasoning"))
        assert encoded["thinking"] == "my reasoning"

    def test_reasoning_on_a_user_message_is_not_sent(self, engine: OllamaEngine) -> None:
        message = Message(role=Role.USER, content="hi", thinking="not mine")
        assert "thinking" not in engine._encode_message(message)

    def test_tool_calls_use_ollamas_function_envelope(self, engine: OllamaEngine) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "17*43"})
        encoded = engine._encode_message(assistant("", tool_calls=[call]))
        assert encoded["tool_calls"] == [
            {"function": {"name": "calculate", "arguments": {"expression": "17*43"}}}
        ]

    def test_tool_results_carry_the_tool_name(self, engine: OllamaEngine) -> None:
        message = ToolResult(call_id="call_1", name="calculate", content="731").to_message()
        encoded = engine._encode_message(message)
        assert encoded == {
            "role": "tool",
            "content": "731",
            "tool_name": "calculate",
            "tool_call_id": "call_1",
        }


class TestThinkingEncoding:
    def test_no_profile_omits_the_field(self, engine: OllamaEngine) -> None:
        assert engine._encode_thinking(None) is None

    def test_do_switches_thinking_off(self, engine: OllamaEngine) -> None:
        assert engine._encode_thinking(ThinkingLevel.DO.profile) is False

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (ThinkingLevel.RE, "low"),
            (ThinkingLevel.MI, "low"),
            (ThinkingLevel.FA, "medium"),
            (ThinkingLevel.SOL, "medium"),
            (ThinkingLevel.LA, "high"),
            (ThinkingLevel.SI, "high"),
        ],
    )
    def test_each_note_maps_to_an_effort(
        self, engine: OllamaEngine, level: ThinkingLevel, expected: str
    ) -> None:
        assert engine._encode_thinking(level.profile) == expected


class TestOptionEncoding:
    def test_unset_parameters_are_not_sent(self, engine: OllamaEngine) -> None:
        # Sending a default would silently override whatever is baked into the
        # model file, so an unset field must produce no key at all.
        assert engine._encode_options(SamplingParams()) == {}

    def test_names_are_translated_to_ollamas(self, engine: OllamaEngine) -> None:
        options = engine._encode_options(
            SamplingParams(temperature=0.7, max_tokens=256, context_length=8192, seed=11)
        )
        assert options == {
            "temperature": 0.7,
            "num_predict": 256,
            "num_ctx": 8192,
            "seed": 11,
        }

    def test_stop_sequences_become_a_list(self, engine: OllamaEngine) -> None:
        assert engine._encode_options(SamplingParams(stop=("</s>", "\n\n")))["stop"] == [
            "</s>",
            "\n\n",
        ]


class TestPayloadConstruction:
    def test_a_minimal_payload(self, engine: OllamaEngine) -> None:
        payload = engine._build_payload(
            GenerationRequest(model="qwen3:1.7b", messages=[user("hi")]), stream=False
        )
        assert payload["model"] == "qwen3:1.7b"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert "tools" not in payload
        assert "think" not in payload

    def test_tools_and_thinking_are_included_when_present(self, engine: OllamaEngine) -> None:
        spec = {"type": "function", "function": {"name": "calculate", "parameters": {}}}
        payload = engine._build_payload(
            GenerationRequest(
                model="qwen3:1.7b",
                messages=[user("hi")],
                tools=[spec],  # type: ignore[list-item]
                thinking=ThinkingLevel.SOL.profile,
                keep_alive="5m",
            ),
            stream=True,
        )
        assert payload["stream"] is True
        assert payload["tools"] == [spec]
        assert payload["think"] == "medium"
        assert payload["keep_alive"] == "5m"

    def test_extra_overrides_everything_else(self, engine: OllamaEngine) -> None:
        payload = engine._build_payload(
            GenerationRequest(
                model="qwen3:1.7b", messages=[user("hi")], extra={"format": "json"}
            ),
            stream=False,
        )
        assert payload["format"] == "json"


class TestResponseDecoding:
    # A real, complete /api/chat response.
    RESPONSE: ClassVar[dict[str, Any]] = {
        "model": "qwen3:1.7b",
        "created_at": "2026-08-17T09:12:33.1Z",
        "message": {
            "role": "assistant",
            "content": "17 * 43 = 731.",
            "thinking": "Let me multiply: 17 * 40 = 680, plus 17 * 3 = 51, so 731.",
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1_234_567_890,
        "prompt_eval_count": 31,
        "eval_count": 42,
    }

    def test_decodes_content_reasoning_and_metadata(self, engine: OllamaEngine) -> None:
        result = engine._decode_result(self.RESPONSE, fallback_model="qwen3:1.7b")
        assert result.message.role is Role.ASSISTANT
        assert result.content == "17 * 43 = 731."
        assert result.thinking is not None and "731" in result.thinking
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 31
        assert result.usage.completion_tokens == 42
        assert result.usage.total_tokens == 73
        assert result.duration_seconds == pytest.approx(1.23456789)

    def test_missing_reasoning_becomes_none(self, engine: OllamaEngine) -> None:
        payload = {"model": "m", "message": {"role": "assistant", "content": "hi"}}
        assert engine._decode_result(payload, fallback_model="m").thinking is None

    def test_a_response_without_a_message_is_an_error(self, engine: OllamaEngine) -> None:
        with pytest.raises(EngineResponseError, match="no 'message' object"):
            engine._decode_result({"model": "m", "done": True}, fallback_model="m")

    def test_the_raw_payload_is_kept(self, engine: OllamaEngine) -> None:
        result = engine._decode_result(self.RESPONSE, fallback_model="m")
        assert result.raw["created_at"] == "2026-08-17T09:12:33.1Z"


class TestToolCallDecoding:
    def test_decodes_an_object_argument_payload(self, engine: OllamaEngine) -> None:
        calls = engine._decode_tool_calls(
            [{"function": {"name": "calculate", "arguments": {"expression": "17*43"}}}]
        )
        assert len(calls) == 1
        assert calls[0].name == "calculate"
        assert calls[0].arguments == {"expression": "17*43"}

    def test_decodes_arguments_that_arrive_as_a_json_string(self, engine: OllamaEngine) -> None:
        # Some model templates emit a JSON string rather than an object.
        calls = engine._decode_tool_calls(
            [{"function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}]
        )
        assert calls[0].arguments == {"expression": "1+1"}

    def test_an_empty_argument_string_means_no_arguments(self, engine: OllamaEngine) -> None:
        calls = engine._decode_tool_calls([{"function": {"name": "ping", "arguments": ""}}])
        assert calls[0].arguments == {}

    def test_an_engine_supplied_id_is_preserved(self, engine: OllamaEngine) -> None:
        calls = engine._decode_tool_calls(
            [{"id": "call_xyz", "function": {"name": "ping", "arguments": {}}}]
        )
        assert calls[0].id == "call_xyz"

    def test_missing_ids_are_generated(self, engine: OllamaEngine) -> None:
        calls = engine._decode_tool_calls([{"function": {"name": "ping", "arguments": {}}}])
        assert calls[0].id.startswith("call_")

    def test_no_tool_calls_decodes_to_an_empty_list(self, engine: OllamaEngine) -> None:
        assert engine._decode_tool_calls(None) == []
        assert engine._decode_tool_calls([]) == []

    def test_unparseable_arguments_raise_rather_than_silently_dropping(
        self, engine: OllamaEngine
    ) -> None:
        with pytest.raises(EngineResponseError, match="neither an object nor valid JSON"):
            engine._decode_tool_calls(
                [{"function": {"name": "calculate", "arguments": "{not json"}}]
            )

    def test_entries_without_a_name_are_skipped(self, engine: OllamaEngine) -> None:
        assert engine._decode_tool_calls([{"function": {"arguments": {}}}]) == []


class TestThinkValueCompatibility:
    """The effort string degrades to a boolean only when the daemon rejects it."""

    @staticmethod
    def _payload() -> dict[str, Any]:
        return {"model": "qwen3:1.7b", "messages": [], "think": "medium"}

    def test_a_rejection_of_the_think_field_is_recognised(self) -> None:
        exc = EngineRequestError("ollama returned HTTP 400: invalid think value", status_code=400)
        assert _is_think_value_rejection(exc, self._payload()) is True

    def test_an_unrelated_400_is_not_treated_as_a_think_rejection(self) -> None:
        exc = EngineRequestError("ollama returned HTTP 400: model not found", status_code=400)
        assert _is_think_value_rejection(exc, self._payload()) is False

    def test_a_server_error_is_never_retried(self) -> None:
        exc = EngineRequestError("ollama returned HTTP 500: think exploded", status_code=500)
        assert _is_think_value_rejection(exc, self._payload()) is False

    def test_a_boolean_think_value_is_never_downgraded_again(self) -> None:
        # There is nothing left to fall back to, so this must not loop.
        payload = {"model": "m", "messages": [], "think": True}
        exc = EngineRequestError("ollama returned HTTP 400: invalid think value", status_code=400)
        assert _is_think_value_rejection(exc, payload) is False

    def test_a_request_without_thinking_is_never_retried(self) -> None:
        exc = EngineRequestError("ollama returned HTTP 400: think", status_code=400)
        assert _is_think_value_rejection(exc, {"model": "m", "messages": []}) is False

    def test_downgrading_keeps_everything_else_intact(self) -> None:
        downgraded = _downgrade_think({**self._payload(), "tools": ["x"]})
        assert downgraded["think"] is True
        assert downgraded["tools"] == ["x"]
        assert downgraded["model"] == "qwen3:1.7b"


class TestCapabilities:
    def test_the_engine_declares_what_it_really_supports(self, engine: OllamaEngine) -> None:
        capabilities = engine.capabilities
        assert capabilities.streaming and capabilities.tools and capabilities.thinking
        assert capabilities.thinking_trace, "Ollama returns reasoning as its own field"
        assert not capabilities.thinking_budget, "Ollama takes an effort knob, not a budget"


class TestUnavailableEngine:
    def test_health_reports_failure_instead_of_raising(self) -> None:
        # Port 1 is reserved and nothing listens there, so this is a genuine
        # connection failure rather than a simulated one.
        engine = OllamaEngine(host="http://127.0.0.1:1", timeout=2.0, connect_timeout=1.0)
        health = engine.health()
        assert health.available is False
        assert health.detail
        assert "127.0.0.1:1" in str(health)
        engine.close()
