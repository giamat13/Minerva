#!/usr/bin/env python3
"""Example 2: the solfege thinking scale.

Runs the same problem at every note of the scale and prints what changed:
whether a reasoning trace appeared, how long it was, how many tokens it cost
and how long it took.

Run it::

    python examples/02_thinking_levels.py
"""

from __future__ import annotations

import sys

from minerva import ThinkingLevel, load_model
from minerva.errors import MinervaError
from minerva.messages import user

PROBLEM = (
    "A train leaves at 14:20 and the journey takes 95 minutes. "
    "There is then a 25 minute wait and a second journey of 40 minutes. "
    "What time does the traveller arrive? Give the time only."
)


def main() -> int:
    model = load_model("swift")

    print("The scale, quietest to loudest:")
    for level in ThinkingLevel:
        print(
            f"  {int(level)}  {level.latin_name:<4} {level.hebrew_name:<4} "
            f"{level.description}"
        )

    print(f"\nModel: {model.spec.display_name}")
    print(
        f"Its ceiling is {model.spec.max_thinking.latin_name} "
        f"({model.spec.max_thinking.hebrew_name}) - higher requests are clamped, "
        f"never rejected.\n"
    )

    print(
        f"{'NOTE':<6} {'ASKED':<6} {'ACTUAL':<7} {'REASONING':>9} "
        f"{'TOKENS':>7} {'TIME':>7}  ANSWER"
    )
    for level in ThinkingLevel:
        # `resolve_thinking` shows what the model will really do with the
        # request, after its own ceiling is applied.
        effective = model.spec.resolve_thinking(level)
        result = model.chat([user(PROBLEM)], thinking=level)

        trace = len(result.thinking or "")
        answer = " ".join(result.content.split())[:40]
        print(
            f"{level.hebrew_name:<6} {level.latin_name:<6} {effective.latin_name:<7} "
            f"{trace:>9} {result.usage.completion_tokens or 0:>7} "
            f"{result.duration_seconds or 0:>6.1f}s  {answer}"
        )

    # Hebrew note names work everywhere a level is accepted.
    print("\nThe same request using a Hebrew note name:")
    print("  ", model.ask("What is 6 * 7?", thinking="סול").strip()[:80])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
