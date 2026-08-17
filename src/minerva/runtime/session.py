"""Multi-turn conversation state.

:class:`~minerva.runtime.agent.Agent` is stateless by design - you hand it the
messages, it hands you an answer. A :class:`Session` is the thin layer that
remembers the transcript between turns, applies the Extended Thinking policy,
and keeps the history from growing without bound.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..messages import Message, Role, user
from ..models.base import MinervaModel
from ..thinking import ThinkingLevel
from ..tools.registry import ToolRegistry
from .agent import Agent, AgentRun

__all__ = ["Session"]


@dataclass(slots=True)
class Session:
    """A running conversation with one model.

    Example::

        session = Session(load_model("swift"), thinking="sol")
        print(session.send("My name is Dana."))
        print(session.send("What is my name?"))   # remembers

    Extended Thinking
    -----------------
    At levels ``LA`` and ``SI`` the reasoning trace of each assistant turn is
    kept in the transcript and sent back to the model, so it can build on its
    earlier reasoning. Below those levels the trace is stripped once the turn
    is over: it was scratch work, and carrying it forward would just burn
    context. :attr:`keep_thinking` overrides the policy either way.
    """

    model: MinervaModel
    thinking: ThinkingLevel | str | int | None = None
    system_prompt: str | None = None
    tools: ToolRegistry | None = None
    max_iterations: int | None = None

    #: Cap on stored messages, excluding the system prompt. ``None`` = unlimited.
    max_history: int | None = None

    #: ``None`` follows the thinking level's Extended Thinking policy.
    keep_thinking: bool | None = None

    messages: list[Message] = field(default_factory=list)
    runs: list[AgentRun] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = self.model.tools
        prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else self.model.spec.system_prompt
        )
        if prompt and not self.messages:
            self.messages.append(Message(role=Role.SYSTEM, content=prompt))

    # -- properties ------------------------------------------------------

    @property
    def level(self) -> ThinkingLevel:
        """The effective thinking level, clamped to the model's ceiling."""
        return self.model.spec.resolve_thinking(self.thinking)

    @property
    def preserves_thinking(self) -> bool:
        """Whether reasoning traces stay in the transcript across turns."""
        if self.keep_thinking is not None:
            return self.keep_thinking
        return self.level.profile.preserve_thinking

    # -- conversation ----------------------------------------------------

    def agent(self, **overrides: Any) -> Agent:
        """Build an :class:`Agent` configured from this session."""
        params: dict[str, Any] = {
            "tools": self.tools,
            "thinking": self.thinking,
            # The system prompt is already the first message, so don't add it twice.
            "system_prompt": "",
            "max_iterations": self.max_iterations,
        }
        params.update(overrides)
        return Agent(self.model, **params)

    def send(self, prompt: str, **kwargs: Any) -> str:
        """Send a user message and return the model's answer text."""
        return self.send_run(prompt, **kwargs).answer

    def send_run(self, prompt: str, **kwargs: Any) -> AgentRun:
        """Send a user message and return the full :class:`AgentRun`."""
        self.messages.append(user(prompt))

        agent = self.agent(**{k: v for k, v in kwargs.items() if k in _AGENT_KWARGS})
        run_kwargs = {k: v for k, v in kwargs.items() if k not in _AGENT_KWARGS}
        run = agent.run(list(self.messages), **run_kwargs)

        # agent.run() returns the transcript it worked with; adopt it wholesale
        # so tool calls and tool results are preserved in order.
        self.messages = self._apply_thinking_policy(run.messages)
        self._trim()
        self.runs.append(run)
        return run

    def _apply_thinking_policy(self, messages: list[Message]) -> list[Message]:
        if self.preserves_thinking:
            return messages
        for message in messages:
            if message.role is Role.ASSISTANT and message.thinking:
                message.thinking = None
        return messages

    def _trim(self) -> None:
        """Drop the oldest turns once the history exceeds :attr:`max_history`.

        The system prompt is pinned - it defines the model's behaviour and must
        survive trimming.
        """
        if self.max_history is None:
            return
        system: list[Message] = []
        rest = self.messages
        if rest and rest[0].role is Role.SYSTEM:
            system, rest = [rest[0]], rest[1:]
        if len(rest) <= self.max_history:
            return

        trimmed = rest[-self.max_history :]
        # Never start the surviving history with an orphaned tool result: the
        # model would see an answer to a question it cannot see.
        while trimmed and trimmed[0].role is Role.TOOL:
            trimmed = trimmed[1:]
        self.messages = system + trimmed

    def reset(self, *, keep_system: bool = True) -> None:
        """Clear the conversation, optionally keeping the system prompt."""
        system = [m for m in self.messages[:1] if m.role is Role.SYSTEM] if keep_system else []
        self.messages = system
        self.runs = []

    # -- persistence -----------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the transcript (not the model binding) to JSON."""
        return json.dumps(
            {
                "model": self.model.spec.name,
                "thinking_level": self.level.latin_name,
                "messages": [m.to_dict() for m in self.messages],
            },
            ensure_ascii=False,
            indent=indent,
        )

    def save(self, path: str | Path) -> Path:
        """Write the transcript to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> Session:
        """Replace the transcript with one previously saved by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        self.runs = []
        return self

    # -- container protocol ----------------------------------------------

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)


# Keyword arguments that configure the Agent rather than a single run.
_AGENT_KWARGS = frozenset(
    {"tools", "thinking", "system_prompt", "sampling", "on_tool_call", "on_tool_result"}
)
