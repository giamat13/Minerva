"""Tests for the conversation primitives."""

from __future__ import annotations

from minerva.messages import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    assistant,
    system,
    tool_result,
    user,
)


class TestConstructors:
    def test_helpers_set_the_right_roles(self) -> None:
        assert system("rules").role is Role.SYSTEM
        assert user("hello").role is Role.USER
        assert assistant("hi").role is Role.ASSISTANT

    def test_a_string_role_is_normalised_to_the_enum(self) -> None:
        assert Message(role="user", content="hi").role is Role.USER  # type: ignore[arg-type]

    def test_assistant_can_carry_reasoning_and_tool_calls(self) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "1+1"})
        message = assistant("", thinking="let me compute", tool_calls=[call])
        assert message.thinking == "let me compute"
        assert message.has_tool_calls
        assert message.tool_calls[0] is call


class TestToolCall:
    def test_ids_are_generated_and_unique(self) -> None:
        first = ToolCall(name="x")
        second = ToolCall(name="x")
        assert first.id != second.id
        assert first.id.startswith("call_")

    def test_an_explicit_id_is_kept(self) -> None:
        assert ToolCall(name="x", id="call_fixed").id == "call_fixed"

    def test_str_renders_a_readable_invocation(self) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "17*43"})
        assert str(call) == 'calculate({"expression": "17*43"})'

    def test_str_keeps_unicode_readable(self) -> None:
        call = ToolCall(name="echo", arguments={"text": "שלום"})
        assert "שלום" in str(call)


class TestToolResult:
    def test_to_message_links_back_to_the_call(self) -> None:
        result = ToolResult(call_id="call_1", name="calculate", content="731")
        message = result.to_message()
        assert message.role is Role.TOOL
        assert message.tool_call_id == "call_1"
        assert message.name == "calculate"
        assert message.content == "731"
        assert message.is_error is False

    def test_the_helper_wires_a_call_to_its_result(self) -> None:
        call = ToolCall(name="calculate", arguments={})
        message = tool_result(call, "731")
        assert message.tool_call_id == call.id
        assert message.name == "calculate"

    def test_errors_are_marked(self) -> None:
        message = tool_result(ToolCall(name="boom"), "it failed", is_error=True)
        assert message.is_error is True


class TestSerialisation:
    def test_round_trip_preserves_everything(self) -> None:
        original = Message(
            role=Role.ASSISTANT,
            content="the answer is 731",
            thinking="17 * 43 = 731",
            tool_calls=[
                ToolCall(name="calculate", arguments={"expression": "17*43"}, id="call_abc")
            ],
            metadata={"latency_ms": 12},
        )
        restored = Message.from_dict(original.to_dict())

        assert restored.role is original.role
        assert restored.content == original.content
        assert restored.thinking == original.thinking
        assert restored.metadata == original.metadata
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "call_abc"
        assert restored.tool_calls[0].arguments == {"expression": "17*43"}

    def test_empty_optional_fields_are_omitted(self) -> None:
        data = user("hello").to_dict()
        assert data == {"role": "user", "content": "hello"}

    def test_tool_messages_round_trip(self) -> None:
        message = ToolResult(
            call_id="call_1", name="calculate", content="boom", is_error=True
        ).to_message()
        restored = Message.from_dict(message.to_dict())
        assert restored.role is Role.TOOL
        assert restored.is_error is True
        assert restored.tool_call_id == "call_1"
