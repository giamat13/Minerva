#!/usr/bin/env python3
"""Example 4: multi-turn conversations and Extended Thinking.

Shows a :class:`~minerva.runtime.session.Session` remembering across turns,
changing thinking level mid-conversation, and saving the transcript to disk.

Run it::

    python examples/04_session_memory.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from minerva import Session, load_model
from minerva.errors import MinervaError


def main() -> int:
    model = load_model("swift")
    session = Session(model, thinking="do")

    print(f"model:    {model.spec.display_name}")
    print(f"thinking: {session.level.latin_name} ({session.level.hebrew_name})")
    print(f"extended: {session.preserves_thinking}\n")

    for message in (
        "My name is Dana and I keep bees.",
        "How many hives would you guess a hobbyist keeps?",
        "What is my name, and what is my hobby?",
    ):
        print(f"you>   {message}")
        print(f"swift> {session.send(message).strip()}\n")

    # Turn the thinking up mid-conversation. The transcript is unaffected.
    session.thinking = "sol"
    print(f"[thinking level -> {session.level.latin_name} ({session.level.hebrew_name})]")
    print("you>   If each hive yields 18 kg of honey, what do 7 hives yield?")
    run = session.send_run("If each hive yields 18 kg of honey, what do 7 hives yield?")
    print(f"swift> {run.answer.strip()}")
    if run.thinking:
        print(f"       [reasoning trace: {len(run.thinking)} characters]")

    # Extended Thinking (LA and SI) keeps reasoning in the transcript so the
    # model can build on it. Swift's ceiling is SOL, so this reports False -
    # a larger Minerva model would report True at the same setting.
    print(f"\nreasoning preserved across turns: {session.preserves_thinking}")

    print(f"\ntranscript: {len(session)} messages")
    with tempfile.TemporaryDirectory() as tmp:
        path = session.save(Path(tmp) / "conversation.json")
        print(f"saved to:   {path} ({path.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
