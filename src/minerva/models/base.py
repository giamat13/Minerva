"""What a Minerva model *is*.

A Minerva model is a named, versioned specification: which weights to run,
with what decoding parameters, under what system prompt, at what default
thinking level, with which capabilities. It is deliberately not an engine and
not a network client - a :class:`ModelSpec` is inert data that can be printed,
diffed and reviewed.

:class:`MinervaModel` binds a spec to an engine and gives you something you can
actually talk to.

ADDING A MODEL
--------------
See ``src/minerva/models/swift.py`` for the reference implementation and
``docs/ADDING_A_MODEL.md`` for the walkthrough. In short: write a new module
with a :class:`ModelSpec`, then register it in
:mod:`minerva.models.registry`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..engines.base import (
    Engine,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
)
from ..errors import CapabilityError, ModelNotInstalledError
from ..messages import Message, Role, user
from ..thinking import ThinkingLevel, parse_thinking_level
from ..tools.registry import ToolRegistry

__all__ = ["MinervaModel", "ModelSpec"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """The declarative definition of one Minerva model."""

    # -- identity --------------------------------------------------------
    name: str
    """Minerva's name for the model, lowercase and stable (e.g. ``"swift"``).
    This is what users type; it never changes when the weights do."""

    display_name: str
    """Human-facing name (e.g. ``"Minerva Swift"``)."""

    version: str
    """Version of *this specification*, not of the weights."""

    description: str
    """One or two sentences: what this model is for and when to pick it."""

    tier: str
    """Where it sits in the family - ``"small"``, ``"medium"``, ``"large"``.
    Used for sorting and for picking a sensible default."""

    # -- execution -------------------------------------------------------
    engine: str = "ollama"
    """Which engine runs it. Must be a key in :mod:`minerva.engines.registry`."""

    engine_model: str = ""
    """The engine-side identifier - for Ollama, the model tag to run."""

    engine_model_fallbacks: tuple[str, ...] = ()
    """Alternative tags to accept if :attr:`engine_model` is not installed,
    tried in order. Lets a spec survive a retagged or quantised build."""

    system_prompt: str | None = None
    """Prepended to every conversation unless the caller supplies their own."""

    sampling: SamplingParams = field(default_factory=SamplingParams)
    """The model's tuned decoding defaults. Callers may override per call."""

    # -- behaviour -------------------------------------------------------
    default_thinking: ThinkingLevel = ThinkingLevel.FA
    """Where on the solfege scale this model sits when nothing is specified."""

    max_thinking: ThinkingLevel = ThinkingLevel.SI
    """The highest note this model can usefully reach. Requests above it are
    clamped rather than rejected - a small model asked to think at SI should
    still answer, just at its own ceiling."""

    supports_tools: bool = True
    supports_thinking: bool = True
    supports_vision: bool = False

    context_length: int = 4_096
    """Advertised context window, in tokens."""

    parameter_count: str = ""
    """Human-readable size, e.g. ``"1.7B"``. Documentation only."""

    tags: tuple[str, ...] = ()
    """Free-form labels for search and filtering (e.g. ``("fast", "local")``)."""

    def resolve_thinking(self, requested: ThinkingLevel | str | int | None) -> ThinkingLevel:
        """Pick the effective thinking level, honouring this model's ceiling."""
        level = self.default_thinking if requested is None else parse_thinking_level(requested)
        if not self.supports_thinking:
            return ThinkingLevel.DO
        return min(level, self.max_thinking)

    def candidate_engine_models(self) -> tuple[str, ...]:
        """Engine-side identifiers to try, most preferred first."""
        return (self.engine_model, *self.engine_model_fallbacks)

    def with_overrides(self, **changes: Any) -> ModelSpec:
        """A copy of this spec with fields replaced (specs are immutable)."""
        return replace(self, **changes)


class MinervaModel:
    """A :class:`ModelSpec` bound to a live :class:`~minerva.engines.base.Engine`.

    This is the object you talk to::

        model = load_model("swift")
        print(model.ask("Why is the sky blue?", thinking="sol"))

    It is stateless: every call takes the full message list. Multi-turn state
    lives in :class:`~minerva.runtime.session.Session`, which is what you want
    when one process serves many conversations.
    """

    def __init__(
        self,
        spec: ModelSpec,
        engine: Engine,
        *,
        tools: ToolRegistry | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.spec = spec
        self.engine = engine
        self.tools = tools
        self.keep_alive = keep_alive
        # Which engine-side tag we actually run. Resolved lazily on first use so
        # constructing a model never requires the engine to be up.
        self._resolved_engine_model: str | None = None

    # -- identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    def __repr__(self) -> str:
        return (
            f"<MinervaModel {self.spec.name!r} v{self.spec.version} "
            f"engine={self.engine.name!r}>"
        )

    # -- engine binding --------------------------------------------------

    def resolve_engine_model(self, *, verify: bool = True) -> str:
        """The engine-side tag this model will actually run.

        With ``verify=True`` the engine is asked which models are installed and
        the first candidate that is present wins. With ``verify=False`` the
        preferred tag is returned without touching the network.

        :raises ModelNotInstalledError: when none of the candidates is installed.
        """
        if self._resolved_engine_model is not None:
            return self._resolved_engine_model

        candidates = self.spec.candidate_engine_models()
        if not verify:
            return candidates[0]

        for candidate in candidates:
            if self.engine.has_model(candidate):
                self._resolved_engine_model = candidate
                return candidate

        raise ModelNotInstalledError(
            f"Minerva model {self.spec.name!r} needs one of "
            f"{', '.join(repr(c) for c in candidates)} on engine {self.engine.name!r}, "
            f"but none is installed. Install it with: "
            f"ollama pull {candidates[0]}   (or: minerva pull {self.spec.name})"
        )

    def is_installed(self) -> bool:
        """True when the engine has weights we can run."""
        return any(self.engine.has_model(c) for c in self.spec.candidate_engine_models())

    # -- request construction --------------------------------------------

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        thinking: ThinkingLevel | str | int | None = None,
        tools: ToolRegistry | None = None,
        sampling: SamplingParams | None = None,
        system_prompt: str | None = None,
        verify_installed: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> GenerationRequest:
        """Turn Minerva-level arguments into an engine-level request.

        Kept public because it is the natural place to inspect what will be
        sent, and because engine-level code (and tests) can then build a
        request without duplicating this logic.
        """
        level = self.spec.resolve_thinking(thinking)
        profile = level.profile

        if profile.enabled and not self.engine.capabilities.thinking:
            raise CapabilityError(
                f"thinking level {level.latin_name} ({level.hebrew_name}) was requested but "
                f"engine {self.engine.name!r} does not support thinking; use 'do' to disable it"
            )

        registry = tools if tools is not None else self.tools
        tool_specs: list[Any] = []
        if registry is not None and len(registry) > 0:
            if not self.spec.supports_tools:
                raise CapabilityError(
                    f"model {self.spec.name!r} does not support tool calling"
                )
            if not self.engine.capabilities.tools:
                raise CapabilityError(
                    f"engine {self.engine.name!r} does not support tool calling"
                )
            tool_specs = list(registry.specs())

        return GenerationRequest(
            model=self.resolve_engine_model(verify=verify_installed),
            messages=self._with_system_prompt(messages, system_prompt),
            tools=tool_specs,
            thinking=profile,
            sampling=self.spec.sampling.merged_with(sampling),
            keep_alive=self.keep_alive,
            extra=dict(extra or {}),
        )

    def _with_system_prompt(
        self, messages: Sequence[Message], override: str | None
    ) -> list[Message]:
        """Prepend the system prompt, unless the caller already supplied one."""
        prompt = override if override is not None else self.spec.system_prompt
        result = list(messages)
        if not prompt:
            return result
        if result and result[0].role is Role.SYSTEM:
            return result
        return [Message(role=Role.SYSTEM, content=prompt), *result]

    # -- generation ------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        thinking: ThinkingLevel | str | int | None = None,
        tools: ToolRegistry | None = None,
        sampling: SamplingParams | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """One non-streaming turn. Tool calls are returned, not executed.

        To have tool calls executed automatically, use
        :class:`minerva.runtime.agent.Agent`.
        """
        request = self.build_request(
            messages,
            thinking=thinking,
            tools=tools,
            sampling=sampling,
            system_prompt=system_prompt,
            **kwargs,
        )
        return self.engine.chat(request)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        thinking: ThinkingLevel | str | int | None = None,
        tools: ToolRegistry | None = None,
        sampling: SamplingParams | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """One streaming turn."""
        if not self.engine.capabilities.streaming:
            raise CapabilityError(f"engine {self.engine.name!r} does not support streaming")
        request = self.build_request(
            messages,
            thinking=thinking,
            tools=tools,
            sampling=sampling,
            system_prompt=system_prompt,
            **kwargs,
        )
        return self.engine.stream(request)

    def ask(
        self,
        prompt: str,
        *,
        thinking: ThinkingLevel | str | int | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Ask one question, get the answer text. The shortest useful path in."""
        result = self.chat(
            [user(prompt)], thinking=thinking, system_prompt=system_prompt, **kwargs
        )
        return result.content
