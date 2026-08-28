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
``language``  Was the reply in the language the question was asked in? Added
              in v0.4.0. Every metric above can pass while the model answers
              a Hebrew question in English - v0.3.0's own eval log contains
              exactly that, scored as a pass. For a bilingual model that is
              the most visible failure there is, so it gets its own number.
``relevance`` Did the reply engage the question at all, right or wrong?
              Deliberately a weaker bar than ``answer``: "Thursday" for
              "what comes after Saturday" is relevant and wrong, while
              "Nice to meet you" for "name the first month" is not about
              months at all. The second is what a user means by "it says
              things unrelated to what I asked", and it is the one this
              number tracks.
``coherence`` Did the reply avoid collapsing into repetition? Also v0.4.0.
              A small model that has lost the thread tends to echo a phrase
              rather than stop, which reads as broken to a user even when
              routing was right.

The last two exist because a user reported that replies were "not related to
what I asked" while routing accuracy read 97.7%. Both numbers were true; the
eval was simply not measuring the thing that was wrong.
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
    relevant_if: tuple[str, ...] = ()
    """Substrings marking a reply that **engaged the question**, right or wrong.

    Relevance is deliberately a weaker bar than correctness, because they are
    different failures and a user feels them differently. "What comes after
    Saturday?" answered "Thursday." is relevant and wrong - it is at least
    about weekdays. "Name the first month of the year." answered "Nice to
    meet you." is not about months at all, and that is the failure a user
    describes as "it says things unrelated to what I asked".

    Hand-written per case, like every other expectation in this file. Left
    empty where a sensible default exists: a case with an `expected_value`
    is relevant if the reply contains any number, and a refusal case is
    relevant if the reply refuses.
    """


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
    EvalCase("Good evening.", None, relevant_if=("evening", "morning", "hello", "hi", "good")),
    EvalCase("Hello there, how are you?", None,
             relevant_if=("hello", "hi", "well", "fine", "good", "help", "how")),
    EvalCase("What should I call you?", None, relevant_if=("swift", "name", "call")),
    EvalCase("Many thanks.", None,
             relevant_if=("welcome", "glad", "pleasure", "happy", "any time", "sure")),
    EvalCase("Say the word apple.", None, relevant_if=("apple",)),
    EvalCase("How many days are in a fortnight?", None,
             relevant_if=("day", "fourteen", "14", "two week")),
    EvalCase("Name the first month of the year.", None,
             relevant_if=("january", "month")),
    EvalCase("Make this upper case: quiet please", None, relevant_if=("quiet", "please")),
    EvalCase("What comes after Saturday?", None,
             relevant_if=("sunday", "monday", "tuesday", "wednesday", "thursday",
                          "friday", "saturday", "day")),
    # NOT ("c", "a", "t") - single letters match almost any English sentence
    # and would score this case as relevant no matter what came back.
    EvalCase("Spell the word cat backwards.", None, relevant_if=("tac", "cat")),
    # -- should decline rather than invent: no tool fixes these ---------------
    EvalCase("Write me a complete database engine.", None, expects_refusal=True),
    EvalCase("Translate 'good night' into Korean.", None, expects_refusal=True),
    # -- should search rather than invent or refuse: web_search can resolve
    #    these, and CLAUDE.md's honesty rule says a tool call beats a guess
    #    or a flat refusal when one is available. Different facts and
    #    phrasings from every _WEB_SEARCH training example in instruct_data.py.
    EvalCase("Who is the president of Brazil?", "web_search"),
    EvalCase("What is the stock price of Apple today?", "web_search"),
    EvalCase("What happened in the news yesterday?", "web_search"),
    EvalCase("How many people live in Tokyo?", "web_search"),
    # -- Hebrew: different numbers, dates and phrasings from every Hebrew
    #    training example in instruct_data.py's section 8 - same "near but not
    #    in the training distribution" standard as the English cases above.
    EvalCase("כמה זה 23 כפול 19?", "calculate", "437"),
    EvalCase("תוסיף 314 ל-159.", "calculate", "473"),
    EvalCase("מה זה 630 חלקי 9?", "calculate", "70"),
    EvalCase("12 בריבוע.", "calculate", "144"),
    EvalCase("מה השעה בלונדון?", "current_time"),
    EvalCase("כמה ימים יש בין 2026-02-01 ל-2026-02-28?", "days_between", "27"),
    EvalCase("ערב טוב.", None, relevant_if=("ערב", "בוקר", "שלום", "טוב")),
    EvalCase("איך קוראים לך?", None, relevant_if=("swift", "שם", "קוראים", "אני")),
    EvalCase("מי המנכ״ל של החברה?", None, expects_refusal=True),
    EvalCase("מה מחיר המניה של אפל היום?", "web_search"),
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

# --- Answering in the language you were asked in ---------------------------
# Routing and argument accuracy say nothing about the most visible failure a
# bilingual model has: being asked in Hebrew and replying in English. v0.3.0's
# eval log contains exactly that ("מה מחיר המניה של אפל היום?" -> "I do not
# know."), scored as a pass by every metric there was. This measures it.
#
# Script is counted over letters only - digits, punctuation and whitespace are
# shared between the two languages and would just add noise. A proper noun in
# the other script ("Swift") is normal and should not fail the case, so the
# test is which script *dominates*, not whether the other appears at all.
_HEBREW_LETTER = re.compile(r"[֐-׿]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def _dominant_script(text: str) -> str | None:
    """``"he"``, ``"en"``, or ``None`` when there are too few letters to tell."""
    hebrew = len(_HEBREW_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if hebrew + latin < 3:
        return None
    return "he" if hebrew > latin else "en"


# A small model that has lost the thread often repeats itself rather than
# stopping - "I do not know. I do not know." or a word echoed a dozen times.
# That reads as broken to a user even when routing was correct, so it is
# measured rather than left to impressions.
_MIN_WORDS_FOR_REPETITION = 6


def _is_degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < _MIN_WORDS_FOR_REPETITION:
        return False
    # One token dominating the whole answer.
    counts: dict[str, int] = {}
    for word in words:
        key = word.strip(".,!?;:\"'").lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    if counts and max(counts.values()) / len(words) > 0.4:
        return True
    # A repeated multi-word phrase, e.g. an answer that says the same
    # sentence twice.
    for size in (3, 4, 5):
        if len(words) < size * 2:
            continue
        seen: dict[str, int] = {}
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size]).lower()
            seen[phrase] = seen.get(phrase, 0) + 1
            if seen[phrase] >= 3:
                return True
    return False


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
        "language_scorable": 0,
        "language_ok": 0,
        "answered": 0,
        "coherent": 0,
        "relevance_scorable": 0,
        "relevant": 0,
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

        # Answered in the language it was asked in? Only scorable when both
        # sides carry enough letters to tell - an answer of "42" has no script.
        want_script = _dominant_script(case.prompt)
        got_script = _dominant_script(answer)
        language_ok = None
        if want_script is not None and got_script is not None:
            totals["language_scorable"] += 1
            language_ok = want_script == got_script
            totals["language_ok"] += language_ok

        # Relevance: did the reply engage the question at all? Deliberately a
        # weaker bar than correctness - "Thursday" for "what comes after
        # Saturday" is relevant and wrong, while "Nice to meet you" for "name
        # the first month" is the failure a user calls "unrelated".
        low = answer.lower()
        if case.relevant_if:
            relevant = any(marker.lower() in low for marker in case.relevant_if)
        elif case.expects_refusal:
            relevant = any(m in low for m in _REFUSAL_MARKERS)
        elif case.expected_value is not None:
            # It was asked for a quantity; a reply with no number in it did
            # not engage the question, whatever else it did.
            relevant = bool(_NUMBER.search(answer))
        else:
            # Tool-routing cases with no expected value (clock, web_search):
            # engaging means actually reaching for a tool.
            relevant = bool(calls)
        if answer.strip() or calls:
            totals["relevance_scorable"] += 1
            totals["relevant"] += relevant

        # Coherence: produced something, and did not collapse into repetition.
        degenerate = _is_degenerate(answer)
        if answer.strip():
            totals["answered"] += 1
            totals["coherent"] += not degenerate

        rows.append(
            {
                "prompt": case.prompt,
                "expected_tool": case.expects_tool,
                "called": called,
                "routing_ok": routing_ok,
                "arguments_ok": arguments_ok,
                "relevant": relevant,
                "language_ok": language_ok,
                "degenerate": degenerate,
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
        "relevance_pct": pct("relevant", "relevance_scorable"),
        "language_match_pct": pct("language_ok", "language_scorable"),
        "coherence_pct": pct("coherent", "answered"),
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
    print(f"  related to question   {summary['relevance_pct']:5.1f}%")
    print(f"  answered in-language  {summary['language_match_pct']:5.1f}%")
    print(f"  coherent (no repeats) {summary['coherence_pct']:5.1f}%")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
