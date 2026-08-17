"""The engine abstraction.

An *engine* is the thing that actually runs inference. Minerva models are
descriptions; engines are execution. Keeping them apart is what lets the same
Minerva model run on a laptop through Ollama today and on a served backend
tomorrow without a single change to the model definition, the tools or the
agent loop.

ADDING AN ENGINE
----------------
1. Subclass :class:`Engine` in a new module in this package.
2. Implement :meth:`Engine.chat`, :meth:`Engine.stream`, :meth:`Engine.health`
   and :meth:`Engine.list_available_models`.
3. Declare what you support in :attr:`Engine.capabilities`.
4. Register the class in :mod:`minerva.engines.registry`.

The contract is deliberately narrow. Everything above the engine - thinking
levels, tool schemas, the agent loop - is engine-agnostic, and the engine's
only job is to translate Minerva's vocabulary into its own wire format and
back. See ``docs/ADDING_AN_ENGINE.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..messages import Message
from ..thinking import ThinkingProfile
from ..tools.base import ToolSpec

__all__ = [
    "Engine",
    "EngineCapabilities",
    "EngineHealth",
    "GenerationRequest",
    "GenerationResult",
    "SamplingParams",
    "StreamChunk",
    "Usage",
]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """What an engine can do.

    Callers check these before asking for something the backend cannot deliver,
    so a missing feature produces a clear
    :class:`~minerva.errors.CapabilityError` instead of a confusing wire error.
    """

    streaming: bool = False
    tools: bool = False
    thinking: bool = False
    #: True when the engine returns the reasoning trace as a separate field
    #: rather than inline in the content (required for Extended Thinking).
    thinking_trace: bool = False
    #: True when the engine accepts a numeric reasoning-token budget.
    thinking_budget: bool = False
    vision: bool = False


@dataclass(frozen=True, slots=True)
class EngineHealth:
    """The result of a real health probe against the engine."""

    available: bool
    endpoint: str
    version: str | None = None
    detail: str | None = None
    models: tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.available:
            version = f" v{self.version}" if self.version else ""
            return f"reachable at {self.endpoint}{version} ({len(self.models)} model(s) installed)"
        return f"unreachable at {self.endpoint}: {self.detail or 'unknown reason'}"


@dataclass(slots=True)
class SamplingParams:
    """Decoding parameters, in engine-neutral terms.

    ``None`` means "do not send it" - the engine's own default wins. That
    matters: sending an explicit default silently overrides whatever the model
    author baked into the model file.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None
    context_length: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def merged_with(self, override: SamplingParams | None) -> SamplingParams:
        """Return a copy where every set field of ``override`` wins."""
        if override is None:
            return self
        merged = SamplingParams(**{f: getattr(self, f) for f in self.__slots__})
        for name in self.__slots__:
            value = getattr(override, name)
            if value is None or value == ():
                continue
            setattr(merged, name, value)
        return merged

    def as_dict(self) -> dict[str, Any]:
        """Only the fields that were actually set."""
        out: dict[str, Any] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if value is None or value == ():
                continue
            out[name] = value
        return out


@dataclass(slots=True)
class GenerationRequest:
    """Everything an engine needs for one turn of generation."""

    model: str
    """The engine-side model identifier (e.g. ``"qwen3:1.7b"``), not the
    Minerva model name. The model layer performs that translation."""

    messages: Sequence[Message]
    tools: Sequence[ToolSpec] = ()
    thinking: ThinkingProfile | None = None
    sampling: SamplingParams = field(default_factory=SamplingParams)
    #: How long the engine should keep the model resident after the call.
    #: Engine-specific string (Ollama accepts e.g. "5m", "0", "-1").
    keep_alive: str | None = None
    #: Escape hatch for engine-specific knobs Minerva does not model yet.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one generation, when the engine reports it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A completed, non-streaming generation."""

    message: Message
    """The assistant message, including any reasoning trace and tool calls."""

    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    duration_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    """The untouched engine payload, for debugging and for features Minerva
    has not surfaced yet."""

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def thinking(self) -> str | None:
        return self.message.thinking

    @property
    def tool_calls(self) -> list[Any]:
        return self.message.tool_calls


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One incremental piece of a streamed generation.

    ``kind`` tells the consumer which channel the delta belongs to:

    * ``"thinking"`` - reasoning tokens (render them differently, or hide them)
    * ``"content"``  - the answer itself
    * ``"tool_call"``- a completed tool call arrived (``tool_call`` is set)
    * ``"done"``     - terminal chunk; ``result`` carries the assembled result
    """

    kind: str
    text: str = ""
    tool_call: Any = None
    result: GenerationResult | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Engine(ABC):
    """Base class for inference backends."""

    #: Short, stable identifier used in config and on the CLI (e.g. "ollama").
    name: str
    #: What this engine supports. Override in the subclass.
    capabilities: EngineCapabilities = EngineCapabilities()

    # -- required interface ----------------------------------------------

    @abstractmethod
    def chat(self, request: GenerationRequest) -> GenerationResult:
        """Run one non-streaming generation."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Run one streaming generation, yielding :class:`StreamChunk` objects.

        The final chunk must have ``kind == "done"`` and carry the assembled
        :class:`GenerationResult`, so callers that only care about the end
        state can ignore everything before it.
        """

    @abstractmethod
    def health(self) -> EngineHealth:
        """Probe the backend for real. Must never raise - report, don't throw."""

    @abstractmethod
    def list_available_models(self) -> list[str]:
        """Model identifiers currently installed/served by this backend."""

    # -- convenience -----------------------------------------------------

    def is_available(self) -> bool:
        """True when the backend answered a health probe."""
        return self.health().available

    def has_model(self, model: str) -> bool:
        """True when ``model`` is installed on this backend.

        Matches Ollama-style tags: a bare name matches ``name:latest``.
        """
        installed = self.list_available_models()
        if model in installed:
            return True
        if ":" not in model:
            return f"{model}:latest" in installed
        return False

    def close(self) -> None:  # noqa: B027 - a no-op default is correct here
        """Release any held resources. Safe to call more than once.

        Deliberately concrete and empty: an engine that holds nothing (a local
        in-process backend, say) should not be forced to implement it.
        """

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
