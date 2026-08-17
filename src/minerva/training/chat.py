"""The chat format Swift-Instruct is fine-tuned on.

A base model has never seen a conversation, so instruction tuning has to teach
it one. This module is the single definition of that format: the fine-tuning
data is rendered with :func:`format_conversation`, the engine prompts with the
same function, and :func:`parse_response` reads back what the model produced.
One definition means training and inference cannot drift apart - which is the
most common and most invisible way a fine-tune goes wrong.

The format
----------
::

    <|user|>What is 17 times 43?<|assistant|><|think|>Arithmetic - I should
    use the calculator rather than guess.<|/think|><|call|>calculate
    {"expression": "17 * 43"}<|/call|><|result|>731<|/result|>17 times 43 is
    731.<|end|>

Every marker is **one token**. A byte-level BPE would otherwise spend five to
seven tokens on ``<|assistant|>``, which is unaffordable in a 512-token context
and much harder for a small model to learn to emit exactly.

Thinking is optional in the rendering, which is what makes the solfege scale
real for this model: at ``DO`` the prompt ends at ``<|assistant|>`` and the
model answers directly, and above ``DO`` the prompt ends at ``<|think|>``, so
the reasoning phase is *forced* by the prompt rather than hoped for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..messages import Message, Role, ToolCall

__all__ = [
    "CHAT_SPECIAL_TOKENS",
    "ParsedResponse",
    "format_conversation",
    "format_tool_call",
    "parse_response",
]

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
THINK_OPEN = "<|think|>"
THINK_CLOSE = "<|/think|>"
CALL_OPEN = "<|call|>"
CALL_CLOSE = "<|/call|>"
RESULT_OPEN = "<|result|>"
RESULT_CLOSE = "<|/result|>"
END = "<|end|>"

#: Added to the tokenizer before fine-tuning, in this order.
CHAT_SPECIAL_TOKENS: tuple[str, ...] = (
    USER,
    ASSISTANT,
    THINK_OPEN,
    THINK_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    RESULT_OPEN,
    RESULT_CLOSE,
    END,
)


def format_tool_call(call: ToolCall) -> str:
    """Render one tool call as the model is trained to emit it.

    ``name`` then a compact JSON object. Keys are sorted so the same call is
    always rendered identically - an inconsistent rendering would ask the model
    to learn noise.
    """
    arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    return f"{CALL_OPEN}{call.name} {arguments}{CALL_CLOSE}"


def format_conversation(
    messages: list[Message],
    *,
    add_generation_prompt: bool = True,
    thinking: bool = False,
) -> str:
    """Render a conversation into the training/prompting format.

    :param add_generation_prompt: end the string where the model should start
        writing. Used at inference; off when rendering a training example that
        already contains the assistant's turn.
    :param thinking: when a generation prompt is added, open the reasoning
        block so the model must think before answering.
    """
    parts: list[str] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            # Swift-Instruct was not trained with system prompts; folding one in
            # silently would put the model off-distribution.
            continue

        if message.role is Role.USER:
            parts.append(f"{USER}{message.content}")

        elif message.role is Role.ASSISTANT:
            parts.append(ASSISTANT)
            if message.thinking:
                parts.append(f"{THINK_OPEN}{message.thinking}{THINK_CLOSE}")
            for call in message.tool_calls:
                parts.append(format_tool_call(call))
            if message.content:
                parts.append(message.content)
            if not message.tool_calls:
                parts.append(END)

        elif message.role is Role.TOOL:
            parts.append(f"{RESULT_OPEN}{message.content}{RESULT_CLOSE}")

    if add_generation_prompt:
        parts.append(ASSISTANT)
        if thinking:
            parts.append(THINK_OPEN)

    return "".join(parts)


def supervised_segments(messages: list[Message]) -> list[tuple[str, bool]]:
    """Split a training conversation into ``(text, is_target)`` segments.

    Only the parts the model must *produce* are targets. Everything it is
    merely *given* - the user's words, the assistant marker that the harness
    writes to start a turn, and tool results, which come from real tools rather
    than from the model - is context, and its loss is masked out.

    Training on context tokens is the classic instruction-tuning mistake: the
    model spends capacity learning to predict the user's questions, which is a
    skill nobody wants and which competes directly with learning to answer.
    """
    segments: list[tuple[str, bool]] = []

    def add(text: str, is_target: bool) -> None:
        if text:
            segments.append((text, is_target))

    for message in messages:
        if message.role is Role.SYSTEM:
            continue

        if message.role is Role.USER:
            add(f"{USER}{message.content}", False)

        elif message.role is Role.ASSISTANT:
            # The harness emits this marker to hand the turn over, so the model
            # is never asked to predict it.
            add(ASSISTANT, False)
            if message.thinking:
                add(f"{THINK_OPEN}{message.thinking}{THINK_CLOSE}", True)
            for call in message.tool_calls:
                add(format_tool_call(call), True)
            if message.content:
                add(message.content, True)
            if not message.tool_calls:
                add(END, True)

        elif message.role is Role.TOOL:
            add(f"{RESULT_OPEN}{message.content}{RESULT_CLOSE}", False)

    return segments


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """What the model produced, split into its three channels."""

    content: str
    thinking: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> Message:
        return Message(
            role=Role.ASSISTANT,
            content=self.content,
            thinking=self.thinking,
            tool_calls=list(self.tool_calls),
        )


# The opening marker is optional as a WHOLE: when the prompt already opened the
# block, the model's output starts inside it and only the close tag appears.
# `re.escape(THINK_OPEN) + "?"` would make just the final ">" optional, which
# silently sends every prompt-opened turn down the "unclosed block" path and
# throws the answer away.
_THINK_RE = re.compile(
    f"(?:{re.escape(THINK_OPEN)})?(.*?)" + re.escape(THINK_CLOSE), re.DOTALL
)
_CALL_RE = re.compile(re.escape(CALL_OPEN) + r"(.*?)" + re.escape(CALL_CLOSE), re.DOTALL)
_RESULT_RE = re.compile(
    re.escape(RESULT_OPEN) + r".*?" + re.escape(RESULT_CLOSE), re.DOTALL
)
# Trailing debris from a tool call the model began and never finished, e.g.
# `100 * 000"}`. Matched only at the end, so real prose is untouched.
_CALL_FRAGMENT = re.compile(r"[^.!?\n]*[{}\"]+\s*$")


def parse_response(text: str, *, thinking_started: bool = False) -> ParsedResponse:
    """Read a generated turn back into thinking, tool calls and content.

    Tolerant by design. A 9.9M model does not close every tag, so an unclosed
    reasoning block or a malformed argument object must degrade to *something
    usable* rather than raise - the alternative is that one bad character
    discards an otherwise fine answer.

    :param thinking_started: the prompt already opened the reasoning block, so
        the text begins inside it even though no opening marker is present.
    """
    remaining = text

    thinking: str | None = None
    match = _THINK_RE.search(remaining)
    if match:
        thinking = _strip_markers(match.group(1)) or None
        remaining = remaining[: match.start()] + remaining[match.end() :]
    elif thinking_started:
        # The block was opened by the prompt and never closed: everything the
        # model wrote is reasoning, and there is no answer.
        return ParsedResponse(content="", thinking=_strip_markers(remaining) or None)

    calls: list[ToolCall] = []
    for call_match in _CALL_RE.finditer(remaining):
        parsed = _parse_call(call_match.group(1))
        if parsed is not None:
            calls.append(parsed)
    remaining = _CALL_RE.sub("", remaining)
    remaining = _RESULT_RE.sub("", remaining)

    remaining = _strip_markers(remaining)

    # An unpaired call close usually means the model restarted a call and was
    # cut off by the stop token, leaving a fragment of JSON behind it.
    remaining = _CALL_FRAGMENT.sub("", remaining)

    return ParsedResponse(
        content=remaining.strip(), thinking=thinking, tool_calls=tuple(calls)
    )


def _strip_markers(text: str) -> str:
    """Remove any chat marker from user-visible text.

    A small model does not always pair its tags, and a stray ``<|/call|>`` or
    ``<|end|>`` inside an otherwise fine answer - or inside a reasoning trace -
    must never be shown as if it were words.
    """
    for marker in CHAT_SPECIAL_TOKENS:
        text = text.replace(marker, "")
    return text.strip()


def _parse_call(payload: str) -> ToolCall | None:
    """Parse ``name {"arg": ...}``, or return None if it is unusable.

    A tool call with a broken argument object still tells us *which* tool the
    model wanted, so it is kept with empty arguments; the tool's own validation
    then produces an error message the model can recover from. Only a call with
    no name at all is dropped.
    """
    text = payload.strip()
    if not text:
        return None

    name, _, arguments_text = text.partition(" ")
    name = name.strip()
    if not name:
        return None

    arguments: dict[str, object] = {}
    arguments_text = arguments_text.strip()
    if arguments_text:
        try:
            decoded = json.loads(arguments_text)
            if isinstance(decoded, dict):
                arguments = decoded
        except json.JSONDecodeError:
            arguments = {}

    return ToolCall(name=name, arguments=arguments)
