"""Conversation primitives: roles, messages, tool calls and tool results.

These types are the shared vocabulary between the model layer, the engine
layer and the agent loop. They are engine-agnostic on purpose: an engine
translates them into its own wire format (see
:meth:`minerva.engines.ollama.OllamaEngine._encode_message`) and nothing
outside that translation layer knows what the wire looks like.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "assistant",
    "system",
    "tool_result",
    "user",
]


class Role(StrEnum):
    """Who produced a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to invoke one tool.

    ``arguments`` is always a decoded mapping. Engines that hand us a JSON
    string (some do) must decode it before constructing this object, so the
    rest of the system never has to care which form the engine used.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return f"{self.name}({json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)})"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of executing a :class:`ToolCall`.

    A failed tool is *not* an exception at this layer: the failure is fed back
    to the model as content so it can recover, retry with different arguments
    or explain the problem to the user. :attr:`is_error` marks which happened.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_message(self) -> Message:
        """Render this result as the ``tool`` message that goes back to the model."""
        return Message(
            role=Role.TOOL,
            content=self.content,
            tool_call_id=self.call_id,
            name=self.name,
            is_error=self.is_error,
        )


@dataclass(slots=True)
class Message:
    """A single turn in a conversation.

    Not every field is meaningful for every role:

    * ``thinking``      - assistant only; the reasoning trace, when the engine
      returned one (see :mod:`minerva.thinking`).
    * ``tool_calls``    - assistant only; tools the model wants run.
    * ``tool_call_id`` / ``name`` - tool messages only; which call this answers.
    * ``metadata``      - free-form, never sent to the engine. Use it for
      timings, token counts, provenance, anything local.
    """

    role: Role
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept plain strings for convenience: Message("user", "hi") works.
        if not isinstance(self.role, Role):
            self.role = Role(str(self.role))

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """A plain-dict view, useful for logging and persistence."""
        data: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.thinking:
            data["thinking"] = self.thinking
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.is_error:
            data["is_error"] = True
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Rebuild a message from :meth:`to_dict` output."""
        return cls(
            role=Role(data["role"]),
            content=data.get("content", ""),
            thinking=data.get("thinking"),
            tool_calls=[
                ToolCall(name=c["name"], arguments=c.get("arguments", {}), id=c["id"])
                for c in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            is_error=bool(data.get("is_error", False)),
            metadata=dict(data.get("metadata", {})),
        )


# A transcript is just an ordered list of messages. The alias exists so
# signatures read clearly and so we can swap in a richer type later without
# touching every call site.
Transcript = list[Message]


# --------------------------------------------------------------------------
# Constructors - short, readable helpers used all over the codebase.
# --------------------------------------------------------------------------


def system(content: str) -> Message:
    """Build a system message."""
    return Message(role=Role.SYSTEM, content=content)


def user(content: str) -> Message:
    """Build a user message."""
    return Message(role=Role.USER, content=content)


def assistant(
    content: str = "",
    *,
    thinking: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    """Build an assistant message, optionally carrying reasoning and tool calls."""
    return Message(
        role=Role.ASSISTANT,
        content=content,
        thinking=thinking,
        tool_calls=list(tool_calls or []),
    )


def tool_result(call: ToolCall, content: str, *, is_error: bool = False) -> Message:
    """Build the ``tool`` message that answers ``call``."""
    return ToolResult(
        call_id=call.id, name=call.name, content=content, is_error=is_error
    ).to_message()
