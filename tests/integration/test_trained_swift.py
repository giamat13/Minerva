"""End-to-end tests against the REAL trained Swift checkpoint.

These load the actual weights produced by `minerva train` and generate real
text with them. If Swift has not been trained on this machine they skip with
the command that would produce it — they are never satisfied by a stub.

    minerva prepare-data
    minerva train
    pytest -m integration

Assertions are written to hold for *any* competently trained checkpoint, not
for one particular run: a language model is stochastic, and asserting on exact
output would be asserting on a random seed. What is asserted is what training
must have achieved — real vocabulary, coherent bytes, sane perplexity — plus
the properties the serving path must have regardless of the weights.
"""

from __future__ import annotations

import math

import pytest

from minerva.engines.base import SamplingParams
from minerva.errors import CapabilityError
from minerva.messages import Role, user
from minerva.models.base import MinervaModel
from minerva.thinking import ThinkingLevel
from minerva.tools.registry import default_registry

pytestmark = pytest.mark.integration


class TestTheCheckpointIsReal:
    def test_it_loads_and_reports_its_own_architecture(
        self, trained_swift: MinervaModel
    ) -> None:
        network, tokenizer = trained_swift.engine.load("swift")  # type: ignore[attr-defined]
        assert network.config.vocab_size == tokenizer.vocab_size
        assert network.config.n_layer > 0
        assert network.num_parameters() > 1_000_000

    def test_it_is_the_size_the_documentation_claims(
        self, trained_swift: MinervaModel
    ) -> None:
        network, _ = trained_swift.engine.load("swift")  # type: ignore[attr-defined]
        assert 9.5e6 < network.num_parameters() < 10.5e6

    def test_the_weights_are_trained_not_random(
        self, trained_swift: MinervaModel
    ) -> None:
        """A randomly initialised model has loss ln(V); a trained one is well below.

        This is the assertion that distinguishes "the file loaded" from
        "training actually happened".
        """
        import torch

        network, tokenizer = trained_swift.engine.load("swift")  # type: ignore[attr-defined]
        text = (
            "It is a truth universally acknowledged, that a single man in "
            "possession of a good fortune, must be in want of a wife."
        )
        ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)

        with torch.no_grad():
            _, loss, _ = network(ids[:, :-1], targets=ids[:, 1:])

        assert loss is not None
        uniform = math.log(tokenizer.vocab_size)
        assert loss.item() < uniform * 0.75, (
            f"loss {loss.item():.3f} is too close to the untrained baseline "
            f"{uniform:.3f} - these weights do not look trained"
        )


class TestGeneration:
    def test_it_generates_real_text(self, trained_swift: MinervaModel) -> None:
        result = trained_swift.chat(
            [user("It is a truth universally acknowledged, that")],
            sampling=SamplingParams(max_tokens=60, seed=1729),
        )
        assert result.message.role is Role.ASSISTANT
        assert result.content.strip(), "the model produced nothing at all"
        assert "�" not in result.content, "output contains broken UTF-8"

    def test_the_output_is_mostly_real_english_words(
        self, trained_swift: MinervaModel
    ) -> None:
        """A trained model produces real words; an untrained one produces noise."""
        result = trained_swift.chat(
            [user("The company said that the")],
            sampling=SamplingParams(max_tokens=80, temperature=0.7, seed=11),
        )
        words = [w.strip(".,;:!?\"'()").lower() for w in result.content.split()]
        words = [w for w in words if w.isalpha()]
        assert len(words) >= 10, f"too little word-like output: {result.content!r}"

        # Common function words are the highest-frequency tokens in any English
        # corpus, so a model that learned anything at all emits them.
        common = {
            "the", "of", "and", "to", "a", "in", "is", "that", "for", "it",
            "was", "with", "as", "on", "be", "at", "by", "he", "this", "not",
            "have", "from", "but", "they", "we", "will", "would", "has", "an",
        }
        hits = sum(1 for w in words if w in common)
        assert hits >= 2, f"no common English words in: {result.content!r}"

    def test_generation_is_reproducible_from_a_seed(
        self, trained_swift: MinervaModel
    ) -> None:
        prompt = [user("Mr President, I should like to")]
        first = trained_swift.chat(prompt, sampling=SamplingParams(max_tokens=30, seed=5))
        second = trained_swift.chat(prompt, sampling=SamplingParams(max_tokens=30, seed=5))
        assert first.content == second.content

    def test_different_seeds_give_different_text(
        self, trained_swift: MinervaModel
    ) -> None:
        prompt = [user("The ship sailed")]
        first = trained_swift.chat(
            prompt, sampling=SamplingParams(max_tokens=40, temperature=1.0, seed=1)
        )
        second = trained_swift.chat(
            prompt, sampling=SamplingParams(max_tokens=40, temperature=1.0, seed=2)
        )
        assert first.content != second.content

    def test_greedy_decoding_is_deterministic(self, trained_swift: MinervaModel) -> None:
        prompt = [user("It is a truth")]
        first = trained_swift.chat(
            prompt, sampling=SamplingParams(max_tokens=20, temperature=0.0)
        )
        second = trained_swift.chat(
            prompt, sampling=SamplingParams(max_tokens=20, temperature=0.0)
        )
        assert first.content == second.content

    def test_it_honours_the_token_budget(self, trained_swift: MinervaModel) -> None:
        result = trained_swift.chat(
            [user("The")], sampling=SamplingParams(max_tokens=12, seed=0)
        )
        assert result.usage.completion_tokens is not None
        assert result.usage.completion_tokens <= 12

    def test_it_reports_real_usage_and_timing(self, trained_swift: MinervaModel) -> None:
        result = trained_swift.chat(
            [user("It is a truth universally")], sampling=SamplingParams(max_tokens=20, seed=0)
        )
        assert result.usage.prompt_tokens and result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens and result.usage.completion_tokens > 0
        assert result.duration_seconds is not None and result.duration_seconds > 0


class TestStreaming:
    def test_chunks_reassemble_into_the_final_text(
        self, trained_swift: MinervaModel
    ) -> None:
        chunks = list(
            trained_swift.stream(
                [user("Showers continued throughout the week")],
                sampling=SamplingParams(max_tokens=40, seed=3),
            )
        )
        assert chunks, "the stream produced nothing"
        assert chunks[-1].kind == "done"
        final = chunks[-1].result
        assert final is not None

        streamed = "".join(c.text for c in chunks if c.kind == "content")
        assert streamed == final.content

    def test_no_chunk_contains_broken_utf8(self, trained_swift: MinervaModel) -> None:
        """Multi-byte characters can straddle two tokens; the engine must buffer."""
        chunks = list(
            trained_swift.stream(
                [user("It is a truth")], sampling=SamplingParams(max_tokens=60, seed=9)
            )
        )
        for chunk in chunks:
            if chunk.kind == "content":
                assert "�" not in chunk.text


class TestPerplexity:
    def test_held_out_perplexity_is_far_below_chance(
        self, trained_swift: MinervaModel
    ) -> None:
        """The headline quality number, measured on real held-out text.

        Chance for an 8,192-token vocabulary is a perplexity of 8,192. Any
        competently trained model is orders of magnitude below that; the bound
        here is loose on purpose so it holds for a short run too.
        """
        import torch

        from minerva.training.dataset import TokenDataset

        network, tokenizer = trained_swift.engine.load("swift")  # type: ignore[attr-defined]
        try:
            val = TokenDataset("data/val.bin", network.config.max_seq_len)
        except (FileNotFoundError, ValueError) as exc:
            pytest.skip(f"no tokenized validation split: {exc}")

        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val.sequential_batches(4, limit=10):
                _, loss, _ = network(inputs, targets=targets)
                assert loss is not None
                total += loss.item()
                count += 1

        mean_loss = total / count
        perplexity = math.exp(mean_loss)
        print(f"\n  held-out loss {mean_loss:.4f}  perplexity {perplexity:.2f}")
        assert perplexity < tokenizer.vocab_size / 4, (
            f"perplexity {perplexity:.1f} is not meaningfully below chance "
            f"({tokenizer.vocab_size})"
        )


class TestBaseModelHonesty:
    """Swift is a base model, and the serving path must not pretend otherwise."""

    def test_every_thinking_level_resolves_to_silence(
        self, trained_swift: MinervaModel
    ) -> None:
        for level in ThinkingLevel:
            assert trained_swift.spec.resolve_thinking(level) is ThinkingLevel.DO

    def test_no_reasoning_trace_is_ever_returned(
        self, trained_swift: MinervaModel
    ) -> None:
        result = trained_swift.chat(
            [user("What is 2 + 2?")], sampling=SamplingParams(max_tokens=20, seed=0)
        )
        assert result.thinking is None

    def test_asking_for_tools_raises_rather_than_silently_ignoring_them(
        self, trained_swift: MinervaModel
    ) -> None:
        with pytest.raises(CapabilityError):
            trained_swift.chat([user("hello")], tools=default_registry())
