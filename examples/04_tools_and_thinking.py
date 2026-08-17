#!/usr/bin/env python3
"""Example 4: tools and the thinking scale, on Minerva's own weights.

Swift-Instruct really does call tools. This runs the full loop - the model
decides to call, the *real* tool executes, the result is fed back, the model
answers - against a checkpoint trained in this repository.

It also shows where the model fails, because a demo that only shows the good
cases is advertising rather than documentation.

Run it::

    minerva prepare-data && minerva train && minerva finetune
    python examples/04_tools_and_thinking.py
"""

from __future__ import annotations

import sys

from minerva import Agent, ThinkingLevel, ToolRegistry, load_model, tool
from minerva.engines.base import SamplingParams
from minerva.errors import CapabilityError, MinervaError
from minerva.messages import ToolCall, user
from minerva.models.registry import get_spec
from minerva.tools import default_registry


@tool
def wingspan_cm(bird: str) -> dict:
    """Look up the wingspan of a bird in centimetres.

    Args:
        bird: The common name of the bird, e.g. "swift".
    """
    table = {"swift": 42, "kestrel": 75, "owl": 105}
    key = bird.strip().lower()
    if key not in table:
        return {"bird": bird, "known": False, "known_birds": sorted(table)}
    return {"bird": key, "wingspan_cm": table[key]}


def show_registry() -> None:
    print("=" * 72)
    print("TOOLS - real functions, really executed")
    print("=" * 72)

    registry = ToolRegistry([*default_registry(), wingspan_cm])
    print(f"\n  registered: {', '.join(registry.names())}")

    # The schema comes from the type hints and the docstring, so there is no
    # second copy of the signature to drift out of sync.
    schema = wingspan_cm.spec()["function"]
    print("\n  schema inferred for wingspan_cm:")
    print(f"    parameters: {schema['parameters']['properties']}")
    print(f"    required:   {schema['parameters'].get('required')}")

    print("\n  executing calls directly:")
    for call in (
        ToolCall(name="calculate", arguments={"expression": "(17 * 43) / 6"}),
        ToolCall(name="wingspan_cm", arguments={"bird": "swift"}),
        ToolCall(name="calculate", arguments={"expression": "1/0"}),
    ):
        result = registry.execute(call)
        marker = "ERROR" if result.is_error else "ok   "
        print(f"    [{marker}] {call.name} -> {result.content[:64]}")
    print(
        "\n  A failing tool becomes a result the model can read and recover\n"
        "  from, not an exception that kills the conversation."
    )


def show_agent_loop() -> None:
    print("\n" + "=" * 72)
    print("THE AGENT LOOP - Minerva's own model calling Minerva's own tools")
    print("=" * 72)

    model = load_model("swift-instruct")
    print(f"\n  {model.spec.display_name}  ({model.spec.parameter_count}, "
          f"trained in this repository)")

    agent = Agent(
        model,
        tools=default_registry(),
        sampling=SamplingParams(temperature=0.0),
        on_tool_call=lambda call: print(f"      -> model asked for: {call}"),
        on_tool_result=lambda res: print(f"      <- real tool returned: {res.content!r}"),
    )

    for question in (
        "What is 17 times 43?",          # routes to the calculator
        "Hello.",                         # needs no tool
        "Who won the World Cup in 2022?", # should decline rather than invent
        "What time is it in London?",     # routes to the clock
    ):
        print(f"\n  Q: {question}")
        run = agent.run(question)
        print(f"  A: {run.answer.strip()!r}")


def show_the_limits() -> None:
    print("\n" + "=" * 72)
    print("WHERE IT FAILS - measured, not hidden")
    print("=" * 72)

    model = load_model("swift-instruct")
    agent = Agent(
        model, tools=default_registry(), sampling=SamplingParams(temperature=0.0)
    )

    print(
        "\n  It decides WHETHER and WHICH tool well (94% / 89% on held-out\n"
        "  prompts) but copies ARGUMENTS badly (28%). Watch the operands:"
    )
    for question in ("What is 23 times 19?", "Work out 48 times 12."):
        run = agent.run(question)
        calls = [str(c) for c in run.tool_calls]
        print(f"\n    Q: {question}")
        print(f"       called: {calls}")
        print(f"       answer: {run.answer.strip()[:70]!r}")

    print(
        "\n  Routing is one token and learnable from 185 examples. Copying an\n"
        "  arbitrary second operand is a general skill that needs far more\n"
        "  data. Run `minerva evaluate-instruct` for the full numbers."
    )


def show_thinking_scale() -> None:
    print("\n" + "=" * 72)
    print("THE THINKING SCALE - seven notes, engine-agnostic")
    print("=" * 72)
    print()
    for level in ThinkingLevel:
        profile = level.profile
        budget = f"{profile.budget_tokens:,}" if profile.budget_tokens else "-"
        extended = "Extended Thinking" if profile.extended else ""
        print(
            f"  {int(level)}  {level.latin_name:<4} {level.hebrew_name:<4} "
            f"budget {budget:>7}  effort {(profile.effort or 'off'):<7} {extended}"
        )

    print("\n  accepted by Latin name, Hebrew name, index or alias:")
    for spelling in ("sol", "סול", 4, "high"):
        print(f"    {spelling!r:<8} -> {ThinkingLevel.parse(spelling)}")

    print("\n  Each model declares its own ceiling, and requests above it are")
    print("  CLAMPED rather than rejected:")
    for name in ("swift", "swift-instruct"):
        spec = get_spec(name)
        resolved = {str(lvl): str(spec.resolve_thinking(lvl)) for lvl in ("do", "fa", "si")}
        print(f"    {name:<15} ceiling {spec.max_thinking}   asked->got {resolved}")

    print(
        "\n  Both ceilings are DO, and that is a MEASUREMENT. Swift-Instruct\n"
        "  really can emit reasoning traces - the machinery is wired and\n"
        "  tested - but forcing it to reason first drops routing accuracy\n"
        "  from 94% to 62%. CLAUDE.md forbids advertising a capability the\n"
        "  weights do not have, so the scale stays clamped until a model is\n"
        "  trained that benefits from it. See docs/TRAINING.md."
    )


def show_base_model_refuses_tools() -> None:
    print("\n" + "=" * 72)
    print("AND THE BASE MODEL STILL SAYS NO")
    print("=" * 72)

    base = load_model("swift")
    print(f"\n  {base.spec.display_name}: supports_tools={base.spec.supports_tools}")
    try:
        base.chat([user("what is 2+2?")], tools=default_registry())
    except CapabilityError as exc:
        print(f"    CapabilityError: {exc}")


def main() -> int:
    show_registry()
    show_agent_loop()
    show_the_limits()
    show_thinking_scale()
    show_base_model_refuses_tools()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
