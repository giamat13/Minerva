#!/usr/bin/env python3
"""Example 1: ask Swift a question.

Run it::

    ollama serve &
    ollama pull qwen3:1.7b
    python examples/01_basic_chat.py

This talks to a real Ollama daemon. If none is running the script says so and
exits - it does not fall back to a canned answer.
"""

from __future__ import annotations

import sys

from minerva import load_model
from minerva.errors import MinervaError


def main() -> int:
    model = load_model("swift")
    print(f"model: {model.spec.display_name} ({model.spec.parameter_count})")

    # One question, one answer. `ask` is the shortest path in.
    print("\n--- a plain question ---")
    print(model.ask("Reply in one sentence: why is the sky blue?", thinking="do"))

    # Streaming, so the answer appears as it is generated.
    print("\n--- the same question, streamed ---")
    from minerva.messages import user

    for chunk in model.stream([user("Name three birds that migrate.")], thinking="mi"):
        if chunk.kind == "content":
            print(chunk.text, end="", flush=True)
    print()

    # The full result carries reasoning, token counts and timings.
    print("\n--- the full result object ---")
    result = model.chat([user("What is 12 * 12?")], thinking="fa")
    print(f"answer:     {result.content.strip()}")
    print(f"reasoning:  {(result.thinking or '(none)')[:120]}")
    print(f"tokens:     {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out")
    print(f"took:       {result.duration_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
