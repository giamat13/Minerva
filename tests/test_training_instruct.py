"""Tests for the instruction-tuning stack.

The data tests are the important ones. `CLAUDE.md` makes claims about this
dataset - hand-written, individually considered, tool results never fabricated,
balanced between tool and no-tool - and a claim nobody checks is a claim that
quietly stops being true. These check them.
"""

from __future__ import annotations

import re

import pytest

from minerva.messages import Role
from minerva.training.instruct_data import (
    INSTRUCT_EXAMPLES,
    build_examples,
)

torch = pytest.importorskip("torch", reason="needs the 'training' extra")

from minerva.training.chat import CHAT_SPECIAL_TOKENS  # noqa: E402
from minerva.training.finetune import (  # noqa: E402
    IGNORE,
    encode_supervised,
    routing_accuracy,
)
from minerva.training.instruct_eval import EVAL_CASES, _mentions  # noqa: E402
from minerva.training.model import SwiftConfig, SwiftLM  # noqa: E402
from minerva.training.tokenizer import BPETokenizer  # noqa: E402


class TestDatasetIntegrity:
    def test_prompts_are_not_duplicated(self) -> None:
        """Duplicates are the signature of copy-paste rather than authorship."""
        seen: dict[tuple[str, tuple], int] = {}
        for example in INSTRUCT_EXAMPLES:
            key = (example.user, example.history)
            seen[key] = seen.get(key, 0) + 1
        duplicates = [prompt for (prompt, _), count in seen.items() if count > 1]
        assert not duplicates, f"duplicated prompts: {duplicates}"

    def test_the_same_question_may_repeat_with_different_context(self) -> None:
        # "What is my name?" appears twice on purpose: once with no history
        # (answer: I don't know) and once after being told (answer: Dana).
        variants = [e for e in INSTRUCT_EXAMPLES if e.user == "What is my name?"]
        assert len(variants) == 2
        assert {bool(v.history) for v in variants} == {True, False}
        assert variants[0].answer != variants[1].answer

    def test_tool_and_no_tool_examples_are_balanced(self) -> None:
        """The balance is the whole defence against "always call the calculator".

        Measured: tipping the set to 91 tool against 66 without dropped honest
        refusal from 50% to 17% in one run.
        """
        with_tool = sum(1 for e in INSTRUCT_EXAMPLES if e.call)
        without = len(INSTRUCT_EXAMPLES) - with_tool
        ratio = with_tool / without
        assert 0.7 < ratio < 1.4, f"{with_tool} tool vs {without} no-tool is unbalanced"

    def test_it_teaches_refusal(self) -> None:
        # Honest ignorance is taught two ways since v0.3.0: a flat refusal for
        # what no tool can fix, and a web_search call for what a search
        # actually could - see CLAUDE.md's "must always be able to say I
        # don't know" rule. Both count: the point being measured is that
        # confabulation is trained against, not which of the two mechanisms
        # handles a given question.
        refusals = [
            e
            for e in INSTRUCT_EXAMPLES
            if any(
                marker in e.answer.lower()
                for marker in (
                    "do not know",
                    "cannot",
                    "not able",
                    "would rather not",
                    "לא יודע",
                    "אין לי",
                    "לא יכול",
                )
            )
        ]
        web_searches = [e for e in INSTRUCT_EXAMPLES if e.call and e.call[0] == "web_search"]
        assert len(web_searches) >= 8, "too few examples teaching search-instead-of-guessing"
        assert len(refusals) + len(web_searches) >= 20, "too few examples teaching honest ignorance"

    def test_every_answer_is_non_empty(self) -> None:
        for example in INSTRUCT_EXAMPLES:
            assert example.answer.strip(), f"{example.user!r} has no answer"

    def test_answers_are_short_enough_for_the_context(self) -> None:
        for example in INSTRUCT_EXAMPLES:
            assert len(example.answer) < 300, f"{example.user!r} has an over-long answer"

    def test_no_answer_contains_a_chat_marker(self) -> None:
        for example in INSTRUCT_EXAMPLES:
            for marker in CHAT_SPECIAL_TOKENS:
                assert marker not in example.answer


class TestToolResultsAreReal:
    def test_building_runs_every_declared_tool(self) -> None:
        """A declared call that does not run would train the model on a fake result."""
        conversations = build_examples()
        assert len(conversations) == len(INSTRUCT_EXAMPLES)

    def test_computed_results_come_from_the_real_tool(self) -> None:
        from minerva.messages import ToolCall
        from minerva.tools.registry import default_registry

        registry = default_registry()
        conversations = build_examples()
        checked = 0

        for example, messages in zip(INSTRUCT_EXAMPLES, conversations, strict=True):
            if example.call is None or example.pinned:
                continue
            tool_message = next(m for m in messages if m.role is Role.TOOL)
            name, arguments = example.call
            expected = registry.execute(ToolCall(name=name, arguments=arguments))
            # The point of this test is that a stored result is what the tool
            # really returns. That holds for the failure-path examples too, so
            # the check is `is_error` matching intent rather than never erroring.
            assert expected.is_error == example.expect_error
            assert tool_message.content == expected.content
            checked += 1

        assert checked > 40, "expected many tool examples to be verified"

    def test_pinned_examples_declare_their_result(self) -> None:
        # Only the clock and web_search may pin: the clock's output depends on
        # the wall clock, and web_search's depends on the live web - neither
        # is reproducible, unlike calculate/days_between which always return
        # the same result for the same arguments.
        for example in INSTRUCT_EXAMPLES:
            if example.pinned:
                assert example.result
                assert example.call and example.call[0] in ("current_time", "web_search")

    def test_stated_answers_agree_with_the_real_tool_output(self) -> None:
        """A wrong worked example teaches a wrong habit."""
        from minerva.messages import ToolCall
        from minerva.tools.registry import default_registry

        registry = default_registry()
        number = re.compile(r"-?\d+(?:\.\d+)?")
        mismatches: list[str] = []

        for example in INSTRUCT_EXAMPLES:
            if example.call is None or example.pinned or example.call[0] != "calculate":
                continue
            name, arguments = example.call
            output = registry.execute(ToolCall(name=name, arguments=arguments)).content
            if not number.fullmatch(output.strip()):
                continue
            if not _mentions(output.strip(), example.answer):
                # A rounded restatement is fine ("about 142.86" for 142.857...).
                rounded = f"{float(output):.2f}".rstrip("0").rstrip(".")
                if rounded not in example.answer:
                    mismatches.append(f"{example.user!r}: tool={output} answer={example.answer!r}")

        assert not mismatches, "answers disagree with the tool:\n" + "\n".join(mismatches)


class TestEvalSetIsHeldOut:
    def test_no_eval_prompt_appears_in_training(self) -> None:
        """Otherwise the eval measures memorisation and reports it as skill."""
        training = {e.user.strip().lower() for e in INSTRUCT_EXAMPLES}
        leaked = [c.prompt for c in EVAL_CASES if c.prompt.strip().lower() in training]
        assert not leaked, f"eval prompts present in training data: {leaked}"

    def test_it_covers_tools_and_direct_answers_and_refusals(self) -> None:
        assert any(c.expects_tool for c in EVAL_CASES)
        assert any(c.expects_tool is None and not c.expects_refusal for c in EVAL_CASES)
        # Since v0.3.0, most "should decline" cases became "should search"
        # instead - a flat refusal remains only for what no tool fixes.
        assert sum(1 for c in EVAL_CASES if c.expects_refusal) >= 2
        assert sum(1 for c in EVAL_CASES if c.expects_tool == "web_search") >= 4

    def test_expected_values_are_arithmetically_right(self) -> None:
        # The eval's own answers must be correct, or it scores the model
        # against a mistake.
        known = {
            "What is 23 times 19?": 23 * 19,
            "Add 314 and 159.": 314 + 159,
            "Subtract 76 from 500.": 500 - 76,
            "What is 630 divided by 9?": 630 // 9,
            "Work out 48 times 12.": 48 * 12,
            "What is 11 squared?": 11**2,
            "What is 5 factorial?": 120,
            "What is 2 to the power of 10?": 2**10,
            "כמה זה 23 כפול 19?": 23 * 19,
            "תוסיף 314 ל-159.": 314 + 159,
            "מה זה 630 חלקי 9?": 630 // 9,
            "12 בריבוע.": 12**2,
        }
        for case in EVAL_CASES:
            if case.prompt in known:
                assert float(case.expected_value or 0) == float(known[case.prompt])


class TestLanguageAndCoherenceMetrics:
    """v0.4.0 metrics, added because routing read 97.7% while a user was
    reporting that the answers had nothing to do with the questions."""

    def test_hebrew_and_english_are_told_apart(self) -> None:
        from minerva.training.instruct_eval import _dominant_script

        assert _dominant_script("What time is it in Paris?") == "en"
        assert _dominant_script("מה השעה בלונדון?") == "he"

    def test_a_proper_noun_in_the_other_script_does_not_flip_the_verdict(self) -> None:
        from minerva.training.instruct_eval import _dominant_script

        # Naming the model in Latin letters inside a Hebrew sentence is
        # normal, not a language failure.
        assert _dominant_script("השם שלי הוא Swift ואני מודל קטן.") == "he"

    def test_too_few_letters_is_unscorable_rather_than_guessed(self) -> None:
        from minerva.training.instruct_eval import _dominant_script

        assert _dominant_script("42") is None
        assert _dominant_script("") is None

    def test_the_real_v030_failure_is_now_caught(self) -> None:
        """The exact case v0.3.0 scored as a pass: Hebrew in, English out."""
        from minerva.training.instruct_eval import _dominant_script

        prompt, answer = "מה מחיר המניה של אפל היום?", "I do not know."
        assert _dominant_script(prompt) != _dominant_script(answer)

    def test_repetition_is_detected(self) -> None:
        from minerva.training.instruct_eval import _is_degenerate

        assert _is_degenerate("I do not know. I do not know. I do not know.")
        assert _is_degenerate("yes yes yes yes yes yes yes yes")

    def test_normal_answers_are_not_flagged(self) -> None:
        from minerva.training.instruct_eval import _is_degenerate

        assert not _is_degenerate("Argentina won the 2022 World Cup, beating France on penalties.")
        assert not _is_degenerate("It is 11:41 in Jerusalem.")
        # Short answers cannot be judged for repetition and must not be
        # flagged - "Twelve." is a perfectly good reply.
        assert not _is_degenerate("Twelve.")


class TestRelevanceExpectations:
    """The relevance metric is only as honest as its hand-written markers."""

    def test_no_marker_is_short_enough_to_match_anything(self) -> None:
        """A one-letter marker scores every English sentence as relevant.

        This is not hypothetical: "Spell the word cat backwards." was first
        written with markers ("tac", "cat", "c", "a", "t"), and "c"/"a"/"t"
        would have passed the case no matter what came back.
        """
        for case in EVAL_CASES:
            for marker in case.relevant_if:
                assert len(marker.strip()) >= 2, (
                    f"{case.prompt!r} has marker {marker!r}, too short to mean anything"
                )

    def test_no_marker_is_a_bare_pronoun_or_filler(self) -> None:
        """Found by hand-checking the scorer against real replies.

        "איך קוראים לך?" answered "אני מודה על זה." scored as relevant purely
        because "אני" ("I") was in the marker list - a word that appears in
        almost any first-person reply and says nothing about whether the
        question was engaged. The percentage looked fine; the verdict was
        wrong. Percentages get spot-checked by eye for this reason.
        """
        banned = {"אני", "am", "is", "the", "a", "i", "it", "של", "זה", "הוא"}
        for case in EVAL_CASES:
            for marker in case.relevant_if:
                assert marker.lower().strip() not in banned, (
                    f"{case.prompt!r} uses {marker!r}, which matches almost any reply"
                )

    def test_every_conversational_case_says_what_relevant_means(self) -> None:
        """Otherwise it silently falls through to the tool-call default.

        A direct-answer case has no tool and no expected value, so without
        explicit markers the scorer would fall back to "did it call a tool",
        which is False for every such case - marking all of them irrelevant
        and quietly understating the metric.
        """
        for case in EVAL_CASES:
            conversational = (
                case.expects_tool is None
                and case.expected_value is None
                and not case.expects_refusal
            )
            if conversational:
                assert case.relevant_if, f"{case.prompt!r} needs relevant_if markers"

    def test_relevance_is_a_weaker_bar_than_correctness(self) -> None:
        """"Thursday" for "what comes after Saturday" is relevant and wrong."""
        case = next(c for c in EVAL_CASES if c.prompt == "What comes after Saturday?")
        assert any(m in "thursday." for m in case.relevant_if), (
            "a wrong-but-on-topic weekday should still count as engaging the question"
        )
        assert not any(m in "nice to meet you." for m in case.relevant_if), (
            "an off-topic pleasantry must not count as engaging the question"
        )


class TestSupervisedEncoding:
    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls) -> BPETokenizer:
        text = "It is a truth universally acknowledged that a good tokenizer helps. " * 12
        tok = BPETokenizer.train(text, 400, verbose=False)
        tok.add_special_tokens(CHAT_SPECIAL_TOKENS)
        return tok

    def test_chat_markers_become_single_tokens(self, tokenizer: BPETokenizer) -> None:
        for marker in CHAT_SPECIAL_TOKENS:
            assert len(tokenizer.encode(marker)) == 1, f"{marker} is not one token"

    def test_adding_markers_does_not_renumber_existing_tokens(self) -> None:
        text = "hello world hello world hello"
        first = BPETokenizer.train(text, 300, verbose=False)
        before = first.encode("hello world")
        first.add_special_tokens(CHAT_SPECIAL_TOKENS)
        assert first.encode("hello world") == before

    def test_inputs_and_labels_are_the_same_length(self, tokenizer: BPETokenizer) -> None:
        for messages in build_examples()[:20]:
            ids, labels = encode_supervised(messages, tokenizer, 512)
            assert len(ids) == len(labels)

    def test_some_positions_are_supervised_and_some_are_masked(
        self, tokenizer: BPETokenizer
    ) -> None:
        _, labels = encode_supervised(build_examples()[0], tokenizer, 512)
        assert any(label != IGNORE for label in labels)
        assert any(label == IGNORE for label in labels)

    def test_the_users_question_is_never_supervised(
        self, tokenizer: BPETokenizer
    ) -> None:
        """Checked structurally, not by string search.

        An answer legitimately restates the question ("17 times 43 is 731"),
        so looking for the question's words in the supervised span finds a
        false positive. What must be true is that the user *segment* is masked.
        """
        from minerva.training.chat import USER, supervised_segments

        messages = build_examples()[0]
        for text, is_target in supervised_segments(messages):
            if text.startswith(USER):
                assert not is_target

        _, labels = encode_supervised(messages, tokenizer, 512)
        supervised = tokenizer.decode([lab for lab in labels if lab != IGNORE])
        assert USER not in supervised

    def test_the_tool_result_is_never_supervised(self, tokenizer: BPETokenizer) -> None:
        """The model must never be trained to write a tool's output."""
        messages = build_examples()[0]
        _, labels = encode_supervised(messages, tokenizer, 512)
        supervised = tokenizer.decode([lab for lab in labels if lab != IGNORE])
        assert "<|result|>" not in supervised


class TestEmbeddingResize:
    def test_learned_rows_survive_and_new_ones_are_added(self) -> None:
        torch.manual_seed(0)
        model = SwiftLM(SwiftConfig(vocab_size=300, n_layer=1, n_head=2, d_model=32))
        original = model.embed.weight[:300].clone()

        model.resize_token_embeddings(309)

        assert model.config.vocab_size == 309
        assert torch.equal(model.embed.weight[:300], original)
        assert model.lm_head.weight is model.embed.weight, "tying must be preserved"

    def test_shrinking_is_refused(self) -> None:
        model = SwiftLM(SwiftConfig(vocab_size=300, n_layer=1, n_head=2, d_model=32))
        with pytest.raises(ValueError, match="refusing to shrink"):
            model.resize_token_embeddings(200)

    def test_resizing_to_the_same_size_is_a_no_op(self) -> None:
        model = SwiftLM(SwiftConfig(vocab_size=300, n_layer=1, n_head=2, d_model=32))
        weight = model.embed.weight
        model.resize_token_embeddings(300)
        assert model.embed.weight is weight


class TestRoutingMetric:
    def test_it_returns_a_fraction(self) -> None:
        text = "hello world this is a small corpus for a small tokenizer. " * 12
        tokenizer = BPETokenizer.train(text, 400, verbose=False)
        tokenizer.add_special_tokens(CHAT_SPECIAL_TOKENS)

        torch.manual_seed(0)
        model = SwiftLM(
            SwiftConfig(vocab_size=tokenizer.vocab_size, n_layer=1, n_head=2, d_model=32)
        )
        score = routing_accuracy(model, tokenizer, build_examples()[:8])
        assert 0.0 <= score <= 1.0
