#!/usr/bin/env python3
"""Example 1: running Swift.

Swift is a **base** language model: it continues text. It does not answer
questions, and this example is written to show that honestly rather than to
flatter the model.

Run it::

    minerva prepare-data
    minerva train
    python examples/01_text_continuation.py
"""

from __future__ import annotations

import sys

from minerva import load_model
from minerva.engines.base import SamplingParams
from minerva.errors import MinervaError
from minerva.messages import user

# Openings drawn from the registers Swift was actually trained on. A base model
# continues the distribution it saw, so these are the prompts it can do
# something with.
PROMPTS = [
    "It is a truth universally acknowledged, that",          # literature
    "Showers continued throughout the week in the",           # newswire
    "Mr President, I should like to thank the",               # parliamentary
    "My fellow citizens, we gather today to",                 # oratory
]


def main() -> int:
    model = load_model("swift")
    spec = model.spec
    print(f"{spec.display_name}  {spec.parameter_count} parameters")
    print(f"engine: {model.engine.name} (in-process)   context: {spec.context_length}\n")

    print("=" * 70)
    print("CONTINUING TEXT - what a base model is actually for")
    print("=" * 70)
    for prompt in PROMPTS:
        result = model.chat(
            [user(prompt)], sampling=SamplingParams(max_tokens=60, seed=1729)
        )
        print(f"\n  prompt:     {prompt!r}")
        print(f"  continues:  {' '.join(result.content.split())}")

    print("\n" + "=" * 70)
    print("STREAMING - tokens as they are generated")
    print("=" * 70)
    print("\n  It is a truth universally acknowledged, that", end="", flush=True)
    for chunk in model.stream(
        [user("It is a truth universally acknowledged, that")],
        sampling=SamplingParams(max_tokens=50, seed=7),
    ):
        if chunk.kind == "content":
            print(chunk.text, end="", flush=True)
    print("\n")

    print("=" * 70)
    print("WHAT IT CANNOT DO - shown, not hidden")
    print("=" * 70)
    question = "What is the capital of France?"
    answer = model.chat([user(question)], sampling=SamplingParams(max_tokens=40, seed=3))
    print(f"\n  asked:    {question!r}")
    print(f"  replied:  {' '.join(answer.content.split())}")
    print(
        "\n  Swift had no instruction tuning, so it continues the text of a\n"
        "  question rather than answering it. That is a training stage it has\n"
        "  not had, not a bug. See docs/TRAINING.md."
    )

    print("\n  temperature controls how far it strays:")
    for temperature in (0.2, 0.8, 1.2):
        result = model.chat(
            [user("The ship sailed")],
            sampling=SamplingParams(max_tokens=25, temperature=temperature, seed=11),
        )
        print(f"    T={temperature}:  {' '.join(result.content.split())}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
