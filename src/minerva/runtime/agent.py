"""The agent loop: generation plus real tool execution.

:class:`~minerva.models.base.MinervaModel` gives you one turn and hands back
any tool calls the model made. The :class:`Agent` closes that loop: it runs the
tools for real, feeds the results back, and repeats until the model produces an
answer instead of another tool call.

    user -> model -> tool calls? -> run tools -> model -> ... -> answer

Nothing here is simulated. A tool call executes the actual Python function; a
failure is reported back to the model as a tool result so it can recover.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..engines.base import GenerationResult, SamplingParams, StreamChunk
from ..errors import MaxIterationsExceeded
from ..messages import Message, ToolCall, ToolResult, user
from ..models.base import MinervaModel
from ..thinking import ThinkingLevel
from ..tools.registry import ToolRegistry

__all__ = ["Agent", "AgentRun", "AgentStep"]


@dataclass(slots=True)
class AgentStep:
    """One model turn inside a run, plus whatever tools it triggered."""

    index: int
    result: GenerationResult
    tool_results: list[ToolResult] = field(default_factory=list)

    @property
    def called_tools(self) -> bool:
        return bool(self.tool_results)


@dataclass(slots=True)
class AgentRun:
    """The complete outcome of :meth:`Agent.run`."""

    answer: str
    """The model's final text."""

    messages: list[Message]
    """The full transcript, including tool calls and tool results."""

    steps: list[AgentStep]
    """One entry per model turn, oldest first."""

    thinking_level: ThinkingLevel
    duration_seconds: float

    @property
    def final_message(self) -> Message:
        return self.steps[-1].result.message

    @property
    def thinking(self) -> str | None:
        """The reasoning trace of the final turn, if the engine returned one."""
        return self.final_message.thinking

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call made during the run, in order."""
        return [call for step in self.steps for call in step.result.message.tool_calls]

    @property
    def prompt_tokens(self) -> int:
        return sum(step.result.usage.prompt_tokens or 0 for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.result.usage.completion_tokens or 0 for step in self.steps)


class Agent:
    """Runs a model in a loop, executing whatever tools it asks for.

    Example::

        agent = Agent(load_model("swift"))
        run = agent.run("What is 17 * 43, and what day is it today?", thinking="sol")
        print(run.answer)

    :param model: the bound model to drive.
    :param tools: tools to expose. Defaults to the model's own registry.
    :param thinking: default thinking level for runs (any solfege name/index).
    :param max_iterations: ceiling on model turns per run. ``None`` derives it
        from the thinking level - deeper thinking gets a longer leash.
    :param on_tool_call: optional callback fired just before each tool runs;
        the natural hook for logging, tracing or a human-in-the-loop gate.
    """

    def __init__(
        self,
        model: MinervaModel,
        *,
        tools: ToolRegistry | None = None,
        thinking: ThinkingLevel | str | int | None = None,
        system_prompt: str | None = None,
        sampling: SamplingParams | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools if tools is not None else model.tools
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.sampling = sampling
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

    # -- helpers ---------------------------------------------------------

    def _effective_level(self, thinking: ThinkingLevel | str | int | None) -> ThinkingLevel:
        requested = thinking if thinking is not None else self.thinking
        return self.model.spec.resolve_thinking(requested)

    def _iteration_ceiling(
        self, level: ThinkingLevel, override: int | None
    ) -> int:
        if override is not None:
            return override
        if self.max_iterations is not None:
            return self.max_iterations
        # Deeper thinking implies a longer plan, so the ceiling rides the scale.
        return level.profile.max_tool_iterations

    def _prepare(self, prompt: str | Sequence[Message]) -> list[Message]:
        if isinstance(prompt, str):
            return [user(prompt)]
        return list(prompt)

    def _run_tools(self, calls: Sequence[ToolCall]) -> tuple[list[ToolResult], list[Message]]:
        """Execute tool calls for real and build the messages that answer them."""
        results: list[ToolResult] = []
        messages: list[Message] = []

        for call in calls:
            if self.on_tool_call is not None:
                self.on_tool_call(call)

            if self.tools is None:
                # The model invented a tool call although none were offered.
                # Say so plainly rather than dropping it - a silent no-op here
                # sends the model into a loop.
                result = ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=f"no tools are available in this conversation, so {call.name!r} "
                    f"cannot be called; answer using your own knowledge instead",
                    is_error=True,
                )
            else:
                result = self.tools.execute(call)

            if self.on_tool_result is not None:
                self.on_tool_result(result)

            results.append(result)
            messages.append(result.to_message())

        return results, messages

    # -- the loop --------------------------------------------------------

    def run(
        self,
        prompt: str | Sequence[Message],
        *,
        thinking: ThinkingLevel | str | int | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> AgentRun:
        """Drive the conversation to a final answer.

        :raises MaxIterationsExceeded: if the model keeps calling tools past
            the ceiling. That is a real failure - returning a half-finished
            answer would hide it.
        """
        started = time.monotonic()
        level = self._effective_level(thinking)
        ceiling = self._iteration_ceiling(level, max_iterations)

        messages = self._prepare(prompt)
        steps: list[AgentStep] = []

        for index in range(ceiling):
            result = self.model.chat(
                messages,
                thinking=level,
                tools=self.tools,
                sampling=self.sampling,
                system_prompt=self.system_prompt,
                **kwargs,
            )
            # The assistant turn, reasoning trace included, goes into the
            # transcript before the tool results so ordering stays honest.
            messages.append(result.message)
            step = AgentStep(index=index, result=result)
            steps.append(step)

            if not result.message.tool_calls:
                return AgentRun(
                    answer=result.content,
                    messages=messages,
                    steps=steps,
                    thinking_level=level,
                    duration_seconds=time.monotonic() - started,
                )

            tool_results, tool_messages = self._run_tools(result.message.tool_calls)
            step.tool_results = tool_results
            messages.extend(tool_messages)

        raise MaxIterationsExceeded(ceiling)

    def stream(
        self,
        prompt: str | Sequence[Message],
        *,
        thinking: ThinkingLevel | str | int | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Same loop, but yielding chunks as they arrive.

        Chunk kinds are the engine's (``"thinking"``, ``"content"``,
        ``"tool_call"``, ``"done"``) plus one the agent adds:
        ``"tool_result"``, carrying the real output of a tool that just ran.
        Only the last ``"done"`` chunk represents the end of the run; the
        intermediate ones close a turn that was followed by tool calls.
        """
        level = self._effective_level(thinking)
        ceiling = self._iteration_ceiling(level, max_iterations)
        messages = self._prepare(prompt)

        for _ in range(ceiling):
            result: GenerationResult | None = None

            for chunk in self.model.stream(
                messages,
                thinking=level,
                tools=self.tools,
                sampling=self.sampling,
                system_prompt=self.system_prompt,
                **kwargs,
            ):
                if chunk.kind == "done":
                    result = chunk.result
                yield chunk

            if result is None:
                raise RuntimeError(
                    "engine stream ended without a terminal 'done' chunk; "
                    "this is an engine bug"
                )

            messages.append(result.message)
            if not result.message.tool_calls:
                return

            tool_results, tool_messages = self._run_tools(result.message.tool_calls)
            messages.extend(tool_messages)
            for tool_res in tool_results:
                yield StreamChunk(
                    kind="tool_result",
                    text=tool_res.content,
                    tool_call=tool_res,
                )

        raise MaxIterationsExceeded(ceiling)
