"""Tests for the chat format.

This format is a contract between three places: the fine-tuning data, the
engine's prompt, and the parser that reads the reply. If they drift apart the
model is trained on one thing and asked for another, and the symptom is a
model that "just doesn't work" with no error anywhere. These tests pin the
contract down.
"""

from __future__ import annotations

import pytest

from minerva.messages import Message, Role, ToolCall, assistant, system, user
from minerva.training.chat import (
    ASSISTANT,
    CALL_CLOSE,
    CALL_OPEN,
    CHAT_SPECIAL_TOKENS,
    END,
    THINK_OPEN,
    USER,
    format_conversation,
    format_tool_call,
    parse_response,
    supervised_segments,
)


class TestMarkers:
    def test_every_marker_is_unique(self) -> None:
        assert len(set(CHAT_SPECIAL_TOKENS)) == len(CHAT_SPECIAL_TOKENS)

    def test_markers_cannot_occur_in_ordinary_text(self) -> None:
        # They are added to the tokenizer as single tokens, so they must be
        # shapes no normal prose produces.
        for marker in CHAT_SPECIAL_TOKENS:
            assert marker.startswith("<|") and marker.endswith("|>")


class TestFormatting:
    def test_a_simple_exchange(self) -> None:
        text = format_conversation(
            [user("hello"), assistant("hi")], add_generation_prompt=False
        )
        assert text == f"{USER}hello{ASSISTANT}hi{END}"

    def test_a_generation_prompt_ends_where_the_model_writes(self) -> None:
        assert format_conversation([user("hello")]).endswith(ASSISTANT)

    def test_thinking_opens_the_reasoning_block(self) -> None:
        # This is what makes the thinking level real: the reasoning phase is
        # forced by the prompt, not requested politely.
        assert format_conversation([user("hi")], thinking=True).endswith(THINK_OPEN)

    def test_the_system_prompt_is_dropped(self) -> None:
        # The model was never fine-tuned with one; including it would put it
        # off the distribution it was trained on.
        text = format_conversation([system("be nice"), user("hi")])
        assert "be nice" not in text

    def test_a_tool_call_renders_as_name_then_json(self) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "1+1"})
        assert format_tool_call(call) == f'{CALL_OPEN}calculate {{"expression": "1+1"}}{CALL_CLOSE}'

    def test_argument_order_is_stable(self) -> None:
        # An unstable rendering asks the model to learn noise.
        first = format_tool_call(ToolCall(name="t", arguments={"b": 2, "a": 1}))
        second = format_tool_call(ToolCall(name="t", arguments={"a": 1, "b": 2}))
        assert first == second

    def test_a_full_tool_turn(self) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "2*2"})
        text = format_conversation(
            [
                user("what is 2*2?"),
                assistant("", tool_calls=[call]),
                Message(role=Role.TOOL, content="4", name="calculate"),
                assistant("It is 4."),
            ],
            add_generation_prompt=False,
        )
        assert "<|result|>4<|/result|>" in text
        assert text.endswith(f"It is 4.{END}")


class TestSupervisedSegments:
    def test_the_users_words_are_never_a_target(self) -> None:
        segments = supervised_segments([user("hello"), assistant("hi")])
        for text, is_target in segments:
            if "hello" in text:
                assert not is_target

    def test_the_answer_is_a_target(self) -> None:
        segments = supervised_segments([user("hello"), assistant("hi")])
        assert any(is_target and "hi" in text for text, is_target in segments)

    def test_the_assistant_marker_is_not_a_target(self) -> None:
        # The harness writes it to hand over the turn, so the model is never
        # asked to predict it.
        segments = supervised_segments([user("x"), assistant("y")])
        assert (ASSISTANT, False) in segments

    def test_tool_results_are_context_not_targets(self) -> None:
        """The single most important masking decision.

        A tool result comes from a real tool. Training the model to predict it
        teaches it to invent tool output, which is the exact failure this
        project refuses to ship.
        """
        call = ToolCall(name="calculate", arguments={"expression": "2*2"})
        segments = supervised_segments(
            [
                user("2*2?"),
                assistant("", tool_calls=[call]),
                Message(role=Role.TOOL, content="4", name="calculate"),
                assistant("It is 4."),
            ]
        )
        for text, is_target in segments:
            if "<|result|>" in text:
                assert not is_target, "the model must never be trained to write tool output"

    def test_the_tool_call_itself_is_a_target(self) -> None:
        call = ToolCall(name="calculate", arguments={"expression": "2*2"})
        segments = supervised_segments([user("2*2?"), assistant("", tool_calls=[call])])
        assert any(is_target and CALL_OPEN in text for text, is_target in segments)

    def test_reassembling_the_segments_gives_the_rendered_conversation(self) -> None:
        messages = [user("hello"), assistant("hi", thinking="be brief")]
        joined = "".join(text for text, _ in supervised_segments(messages))
        assert joined == format_conversation(messages, add_generation_prompt=False)


class TestParsing:
    def test_plain_content(self) -> None:
        assert parse_response("It is 4.").content == "It is 4."

    def test_the_end_marker_is_stripped(self) -> None:
        assert parse_response(f"done{END}").content == "done"

    def test_thinking_is_separated_from_the_answer(self) -> None:
        parsed = parse_response("<|think|>let me see<|/think|>The answer is 4.")
        assert parsed.thinking == "let me see"
        assert parsed.content == "The answer is 4."

    def test_a_prompt_opened_block_is_read_back(self) -> None:
        # The prompt supplied "<|think|>", so the text starts inside it.
        parsed = parse_response("weighing it up<|/think|>Yes.", thinking_started=True)
        assert parsed.thinking == "weighing it up"
        assert parsed.content == "Yes."

    def test_an_unclosed_reasoning_block_is_all_reasoning(self) -> None:
        parsed = parse_response("still thinking and never stopped", thinking_started=True)
        assert parsed.thinking == "still thinking and never stopped"
        assert parsed.content == ""

    def test_a_tool_call_is_extracted(self) -> None:
        parsed = parse_response('<|call|>calculate {"expression": "17 * 43"}<|/call|>')
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "calculate"
        assert parsed.tool_calls[0].arguments == {"expression": "17 * 43"}

    def test_broken_arguments_keep_the_tool_name(self) -> None:
        # Knowing WHICH tool was wanted lets the tool's own validation produce
        # an error the model can recover from; dropping the call teaches
        # nothing.
        parsed = parse_response("<|call|>calculate {not json<|/call|>")
        assert parsed.tool_calls[0].name == "calculate"
        assert parsed.tool_calls[0].arguments == {}

    def test_a_nameless_call_is_dropped(self) -> None:
        assert parse_response(f"{CALL_OPEN}   {CALL_CLOSE}").tool_calls == ()

    def test_orphaned_markers_never_reach_the_user(self) -> None:
        # Regression: a stray close tag was printed as if it were text.
        for text in (f"answer{CALL_CLOSE}", f"{ASSISTANT}answer", f"answer{USER}"):
            assert "<|" not in parse_response(text).content

    def test_a_truncated_call_fragment_is_removed(self) -> None:
        # Regression: the model restarted a call, was cut off by the stop
        # token, and left `100 * 000"}` behind as the "answer".
        assert parse_response('100 * 000"}<|/call|>').content == ""

    def test_ordinary_prose_is_not_eaten_by_the_fragment_rule(self) -> None:
        assert parse_response("The total is 42.").content == "The total is 42."
        assert parse_response("Yes, that is right.").content == "Yes, that is right."

    def test_parsing_survives_empty_input(self) -> None:
        parsed = parse_response("")
        assert parsed.content == ""
        assert parsed.tool_calls == ()


class TestRoundTrip:
    @pytest.mark.parametrize(
        "message",
        [
            assistant("a plain answer"),
            assistant("", thinking="some reasoning"),
            assistant("", tool_calls=[ToolCall(name="calculate", arguments={"expression": "1+1"})]),
            assistant(
                "final",
                thinking="reasoned",
                tool_calls=[ToolCall(name="clock", arguments={})],
            ),
        ],
    )
    def test_format_then_parse_recovers_the_turn(self, message: Message) -> None:
        """Rendering and parsing must be inverses, or training and inference disagree."""
        rendered = format_conversation([user("q"), message], add_generation_prompt=False)
        generated = rendered.split(ASSISTANT, 1)[1]
        parsed = parse_response(generated)

        assert parsed.content == message.content
        assert (parsed.thinking or None) == (message.thinking or None)
        assert [c.name for c in parsed.tool_calls] == [c.name for c in message.tool_calls]
        assert [c.arguments for c in parsed.tool_calls] == [
            c.arguments for c in message.tool_calls
        ]
