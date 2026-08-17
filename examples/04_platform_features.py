#!/usr/bin/env python3
"""Example 4: the platform around the model.

Tools and the thinking scale are platform machinery. Swift cannot use either -
it is a base model - so this example does two things:

1. Exercises both for real, independently of any model.
2. Shows Minerva **refusing** to pretend Swift can use them.

The second half is the point. A platform that silently accepted tools it could
not run would be lying to you.

Run it (no trained checkpoint needed for most of it)::

    python examples/04_platform_features.py
"""

from __future__ import annotations

from minerva import ThinkingLevel, ToolRegistry, tool
from minerva.engines.native import NativeEngine
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


def show_tools() -> None:
    print("=" * 70)
    print("TOOLS - real functions, really executed")
    print("=" * 70)

    registry = ToolRegistry([*default_registry(), wingspan_cm])
    print(f"\n  registered: {', '.join(registry.names())}")

    # The schema the model would see is derived from the type hints and the
    # docstring - there is no second copy of the signature to drift.
    schema = wingspan_cm.spec()["function"]
    print("\n  schema inferred for wingspan_cm:")
    print(f"    description: {schema['description'][:60]}...")
    print(f"    parameters:  {schema['parameters']['properties']}")
    print(f"    required:    {schema['parameters'].get('required')}")

    print("\n  executing calls for real:")
    for call in (
        ToolCall(name="calculate", arguments={"expression": "(17 * 43) / 6"}),
        ToolCall(name="current_time", arguments={"timezone": "Asia/Jerusalem"}),
        ToolCall(name="wingspan_cm", arguments={"bird": "swift"}),
        ToolCall(name="wingspan_cm", arguments={"bird": "penguin"}),
        ToolCall(name="calculate", arguments={"expression": "1/0"}),
    ):
        result = registry.execute(call)
        marker = "ERROR" if result.is_error else "ok   "
        print(f"    [{marker}] {call.name}({call.arguments}) -> {result.content[:70]}")

    print(
        "\n  A failing tool becomes a result the model can read and recover from,\n"
        "  not an exception that kills the conversation."
    )


def show_thinking_scale() -> None:
    print("\n" + "=" * 70)
    print("THE THINKING SCALE - seven notes, engine-agnostic")
    print("=" * 70)
    print()
    for level in ThinkingLevel:
        profile = level.profile
        budget = f"{profile.budget_tokens:,}" if profile.budget_tokens else "-"
        extended = "Extended Thinking" if profile.extended else ""
        print(
            f"  {int(level)}  {level.latin_name:<4} {level.hebrew_name:<4} "
            f"budget {budget:>7}  effort {(profile.effort or 'off'):<7} {extended}"
        )

    print("\n  every level is accepted by name, Hebrew name, index or alias:")
    for spelling in ("sol", "סול", 4, "high"):
        print(f"    {spelling!r:<8} -> {ThinkingLevel.parse(spelling)}")


def show_honest_refusals() -> None:
    print("\n" + "=" * 70)
    print("WHAT SWIFT CANNOT DO - and how Minerva says so")
    print("=" * 70)

    spec = get_spec("swift")
    print(f"\n  spec.supports_tools    = {spec.supports_tools}")
    print(f"  spec.supports_thinking = {spec.supports_thinking}")

    print("\n  so every thinking level collapses to silence:")
    for requested in ("do", "fa", "sol", "si"):
        print(f"    asked {requested:<4} -> {spec.resolve_thinking(requested)}")
    print("    (clamped, never rejected - the same code works on a bigger model)")

    engine = NativeEngine()
    print(f"\n  engine capabilities: tools={engine.capabilities.tools}, "
          f"thinking={engine.capabilities.thinking}, "
          f"streaming={engine.capabilities.streaming}")

    print("\n  and asking for tools anyway raises instead of silently doing nothing:")
    from minerva.engines.base import GenerationRequest

    request = GenerationRequest(
        model="swift",
        messages=[user("what is 2+2?")],
        tools=list(default_registry().specs()),
    )
    try:
        list(engine.stream(request))
    except CapabilityError as exc:
        print(f"    CapabilityError: {exc}")
    except MinervaError as exc:
        print(f"    {type(exc).__name__}: {exc}")


def main() -> int:
    show_tools()
    show_thinking_scale()
    show_honest_refusals()
    print(
        "\n  Making Swift use any of this is a training problem, not an engine\n"
        "  problem. See docs/TRAINING.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
