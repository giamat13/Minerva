"""Measuring Swift-Instruct on held-out conversations.

PROVENANCE
==========
Every prompt below was written by hand for this evaluation and **none appears
in the fine-tuning set**. That is the whole point: a set built by permuting the
training prompts would measure memorisation and report it as capability, which
`CLAUDE.md` calls out specifically as the way an eval comes to measure nothing.

The prompts are deliberately *near* the training distribution but not in it -
different numbers, different verbs, different phrasings. That is the honest
question for a model this size: not "can it recite" but "does the habit
generalise one step". The same holds for the ten Hebrew cases added alongside
instruct_data.py's Hebrew training section: different numbers, different
dates, different phrasings from every Hebrew training example.

What is measured
----------------
``format``    The turn parses cleanly and terminates. If this fails nothing
              else matters, because the agent loop cannot use the output.
``routing``   Did it make the right call/no-call decision? This is a single
              token and the capability the fine-tune was really for.
``tool``      Given that it called a tool, was it the right one?
``arguments`` Do the arguments produce the correct answer when the tool is
              actually run? This is where a 9.9M model is weakest, and it is
              reported separately for exactly that reason.
``answer``    End to end through the agent loop: does the final reply contain
              the correct value?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["EVAL_CASES", "EvalCase", "evaluate_instruct"]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One held-out prompt and what a correct response looks like."""

    prompt: str
    expects_tool: str | None = None
    """Tool the model should call, or None if it should answer directly."""
    expected_value: str | None = None
    """The value the final answer must contain, for questions with one."""
    expects_refusal: bool = False
    """True when the honest response is admitting it does not know."""


#: Hand-written, held out. None of these prompts is in `instruct_data.py`.
EVAL_CASES: tuple[EvalCase, ...] = (
    # -- arithmetic: different operands and verbs from any training example --
    EvalCase("What is 23 times 19?", "calculate", "437"),
    EvalCase("Add 314 and 159.", "calculate", "473"),
    EvalCase("Subtract 76 from 500.", "calculate", "424"),
    EvalCase("What is 630 divided by 9?", "calculate", "70"),
    EvalCase("Work out 48 times 12.", "calculate", "576"),
    EvalCase("How much is 7 plus 8?", "calculate", "15"),
    EvalCase("What is 11 squared?", "calculate", "121"),
    EvalCase("Compute the square root of 169.", "calculate", "13"),
    EvalCase("What is 5 factorial?", "calculate", "120"),
    EvalCase("Give me 25 percent of 800.", "calculate", "200"),
    EvalCase("Multiply 33 by 3.", "calculate", "99"),
    EvalCase("What is 1000 minus 1?", "calculate", "999"),
    EvalCase("Divide 144 by 12.", "calculate", "12"),
    EvalCase("What is 2 to the power of 10?", "calculate", "1024"),
    # -- clock ---------------------------------------------------------------
    EvalCase("What time is it in Paris?", "current_time"),
    EvalCase("Tell me today's date please.", "current_time"),
    EvalCase("How many days between 2026-02-01 and 2026-02-28?", "days_between", "27"),
    EvalCase("Count the days from 2026-05-01 to 2026-06-01.", "days_between", "31"),
    # -- no tool needed ------------------------------------------------------
    EvalCase("Good evening.", None),
    EvalCase("Hello there, how are you?", None),
    EvalCase("What should I call you?", None),
    EvalCase("Many thanks.", None),
    EvalCase("Say the word apple.", None),
    EvalCase("How many days are in a fortnight?", None),
    EvalCase("Name the first month of the year.", None),
    EvalCase("Make this upper case: quiet please", None),
    EvalCase("What comes after Saturday?", None),
    EvalCase("Spell the word cat backwards.", None),
    # -- should decline rather than invent -----------------------------------
    EvalCase("Who is the president of Brazil?", None, expects_refusal=True),
    EvalCase("What is the stock price of Apple today?", None, expects_refusal=True),
    EvalCase("What happened in the news yesterday?", None, expects_refusal=True),
    EvalCase("How many people live in Tokyo?", None, expects_refusal=True),
    EvalCase("Write me a complete database engine.", None, expects_refusal=True),
    EvalCase("Translate 'good night' into Korean.", None, expects_refusal=True),
    # -- Hebrew: different numbers, dates and phrasings from every Hebrew
    #    training example in instruct_data.py's section 8 - same "near but not
    #    in the training distribution" standard as the English cases above.
    EvalCase("כמה זה 23 כפול 19?", "calculate", "437"),
    EvalCase("תוסיף 314 ל-159.", "calculate", "473"),
    EvalCase("מה זה 630 חלקי 9?", "calculate", "70"),
    EvalCase("12 בריבוע.", "calculate", "144"),
    EvalCase("מה השעה בלונדון?", "current_time"),
    EvalCase("כמה ימים יש בין 2026-02-01 ל-2026-02-28?", "days_between", "27"),
    EvalCase("ערב טוב.", None),
    EvalCase("איך קוראים לך?", None),
    EvalCase("מי המנכ״ל של החברה?", None, expects_refusal=True),
    EvalCase("מה מחיר המניה של אפל היום?", None, expects_refusal=True),
)

_REFUSAL_MARKERS = (
    "do not know",
    "don't know",
    "cannot",
    "can not",
    "not able",
    "no access",
    "not confident",
    "would rather not",
    "do not remember",
    # Hebrew equivalents, for the Hebrew eval cases above. Hebrew has no case
    # distinction, so the .lower() call in evaluate_instruct is a harmless
    # no-op on these.
    "לא יודע",
    "אין לי",
    "לא יכול",
    "לא בטוח",
    "מעדיף לא",
)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _mentions(value: str, text: str) -> bool:
    """True when ``text`` states ``value`` as a number, ignoring separators."""
    target = float(value)
    for found in _NUMBER.findall(text):
        try:
            if abs(float(found.replace(",", "")) - target) < 1e-6:
                return True
        except ValueError:
            continue
    return False


def evaluate_instruct(
    model_name: str = "swift-instruct",
    *,
    checkpoint_dir: Path | None = None,
    cases: tuple[EvalCase, ...] = EVAL_CASES,
    thinking: str | int | None = None,
    seed: int = 1729,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the held-out evaluation through the real agent loop.

    Uses the same public path a user would - `Agent.run` with the real tool
    registry - so what is measured is the shipped behaviour, not a private
    shortcut through it.

    Decoding is **greedy** (temperature 0). Sampling makes the numbers move:
    the same checkpoint scored 94.4% and 83.3% tool accuracy on two different
    seeds, and a figure that swings eleven points with the random seed cannot
    support a claim about the model. Greedy decoding measures the model rather
    than the draw. The shipped defaults do sample, so live output will be
    slightly more varied than these numbers suggest.
    """
    from ..config import get_config
    from ..engines.base import SamplingParams
    from ..engines.native import NativeEngine
    from ..models.base import MinervaModel
    from ..models.registry import get_spec
    from ..runtime.agent import Agent
    from ..tools.registry import default_registry

    config = get_config()
    engine = NativeEngine(
        checkpoint_dir=checkpoint_dir or config.checkpoint_dir, device="cpu"
    )
    model = MinervaModel(get_spec(model_name), engine)
    registry = default_registry()

    totals = {
        "cases": 0,
        "format_ok": 0,
        "routing_ok": 0,
        "tool_expected": 0,
        "tool_called": 0,
        "tool_name_ok": 0,
        "arguments_ok": 0,
        "value_expected": 0,
        "answer_ok": 0,
        "refusal_expected": 0,
        "refusal_ok": 0,
    }
    rows: list[dict[str, Any]] = []

    for case in cases:
        agent = Agent(
            model,
            tools=registry,
            thinking=thinking,
            sampling=SamplingParams(seed=seed, temperature=0.0),
            max_iterations=3,
        )
        try:
            run = agent.run(case.prompt)
            answer, calls, failed = run.answer, run.tool_calls, None
        except Exception as exc:
            answer, calls, failed = "", [], f"{type(exc).__name__}: {exc}"

        totals["cases"] += 1
        called = calls[0].name if calls else None

        # Format: it produced something usable and left no markup behind.
        format_ok = failed is None and (bool(answer.strip()) or bool(calls))
        format_ok = format_ok and "<|" not in answer
        totals["format_ok"] += format_ok

        routing_ok = (called is not None) == (case.expects_tool is not None)
        totals["routing_ok"] += routing_ok

        arguments_ok = False
        if case.expects_tool:
            totals["tool_expected"] += 1
            totals["tool_called"] += called is not None
            totals["tool_name_ok"] += called == case.expects_tool
            if calls and case.expected_value is not None:
                outcome = registry.execute(calls[0])
                arguments_ok = not outcome.is_error and _mentions(
                    case.expected_value, outcome.content
                )
                totals["arguments_ok"] += arguments_ok

        if case.expected_value is not None:
            totals["value_expected"] += 1
            totals["answer_ok"] += _mentions(case.expected_value, answer)

        if case.expects_refusal:
            totals["refusal_expected"] += 1
            totals["refusal_ok"] += any(m in answer.lower() for m in _REFUSAL_MARKERS)

        rows.append(
            {
                "prompt": case.prompt,
                "expected_tool": case.expects_tool,
                "called": called,
                "routing_ok": routing_ok,
                "arguments_ok": arguments_ok,
                "answer": answer,
                "error": failed,
            }
        )
        if verbose:
            flag = "ok " if routing_ok else "ROUTE"
            print(f"  [{flag}] {case.prompt!r} -> {called or 'direct'} | {answer[:60]!r}")

    def pct(numerator: str, denominator: str) -> float:
        total = totals[denominator]
        return round(100 * totals[numerator] / total, 1) if total else 0.0

    summary = {
        "model": model_name,
        "seed": seed,
        "thinking": str(thinking) if thinking else "do",
        "cases": totals["cases"],
        "format_valid_pct": pct("format_ok", "cases"),
        "routing_accuracy_pct": pct("routing_ok", "cases"),
        "tool_name_accuracy_pct": pct("tool_name_ok", "tool_expected"),
        "argument_accuracy_pct": pct("arguments_ok", "tool_expected"),
        "final_answer_accuracy_pct": pct("answer_ok", "value_expected"),
        "honest_refusal_pct": pct("refusal_ok", "refusal_expected"),
        "counts": totals,
    }
    return {"summary": summary, "cases": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an instruction-tuned model.")
    parser.add_argument("--model", default="swift-instruct")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--thinking", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_instruct(
        args.model,
        checkpoint_dir=args.checkpoint_dir,
        thinking=args.thinking,
        seed=args.seed,
        verbose=not args.quiet,
    )
    summary = report["summary"]

    print(f"\nHeld-out evaluation - {summary['model']} ({summary['cases']} hand-written cases)")
    print(f"  format valid          {summary['format_valid_pct']:5.1f}%")
    print(f"  routing accuracy      {summary['routing_accuracy_pct']:5.1f}%")
    print(f"  tool name accuracy    {summary['tool_name_accuracy_pct']:5.1f}%")
    print(f"  argument accuracy     {summary['argument_accuracy_pct']:5.1f}%")
    print(f"  final answer correct  {summary['final_answer_accuracy_pct']:5.1f}%")
    print(f"  honest refusal        {summary['honest_refusal_pct']:5.1f}%")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
