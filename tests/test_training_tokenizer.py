"""Tests for the from-scratch BPE tokenizer.

These train real tokenizers on real text - small ones, so the suite stays
fast, but the same code path the 8,192-token production vocabulary came from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minerva.training.tokenizer import (
    END_OF_TEXT,
    BPETokenizer,
    _pretokenize,
    train_bpe,
)

# Real English prose. Repetition is what BPE learns from, so a short natural
# passage is enough to exercise merging without being a generated pattern.
SAMPLE = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property of "
    "some one or other of their daughters."
) * 4


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(SAMPLE, 400, verbose=False)


class TestPretokenization:
    def test_a_leading_space_joins_the_following_word(self) -> None:
        assert _pretokenize("the cat") == ["the", " cat"]

    def test_contractions_split_off(self) -> None:
        assert " don" in _pretokenize("I don't")
        assert "'t" in _pretokenize("I don't")

    def test_digits_are_always_split_individually(self) -> None:
        # Grouped digits ruin place-value consistency, so 1987 must be 4 tokens.
        # The leading space is its own piece rather than riding on the first
        # digit: this test used to expect " 1", which made a number's tokens
        # depend on what preceded it. See the next test for why that mattered.
        assert _pretokenize("in 1987") == ["in", " ", "1", "9", "8", "7"]

    def test_a_number_tokenizes_identically_wherever_it_appears(self) -> None:
        # The property that makes operand copying learnable at all. When the
        # leading space fused onto the first digit, "314" was [' 3','1','4'] in
        # "What is 314 plus 159?" but ['3','1','4'] in the expression "314 +
        # 159" - so copying an operand into a tool call was a different token
        # edit for every number, and the finetuned model invented digits
        # instead ("Add 314 and 159" -> "314 times 15 is 471").
        in_question = _pretokenize("What is 314 plus 159?")
        in_expression = _pretokenize("314 + 159")
        for run in (in_question, in_expression):
            digits = [p for p in run if p.isdigit()]
            assert digits == ["3", "1", "4", "1", "5", "9"]

    def test_punctuation_runs_group_together(self) -> None:
        assert "..." in _pretokenize("wait... really")

    def test_whitespace_is_preserved(self) -> None:
        assert "".join(_pretokenize("a\n\n  b")) == "a\n\n  b"

    def test_non_ascii_survives(self) -> None:
        assert "".join(_pretokenize("café שלום")) == "café שלום"

    def test_underscores_are_not_swallowed(self) -> None:
        # Regression: `\w` includes "_", so it fell through every alternative
        # of the original pattern and was silently deleted from the corpus.
        assert "".join(_pretokenize("snake_case _italic_ __dunder__")) == (
            "snake_case _italic_ __dunder__"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "".join(chr(c) for c in range(32, 127)),
            "a_b|c~d`e^f",
            "\u00a0\u2014\u2019 em-dash and curly quotes",
            "mixed: ab12_!? \t\n x",
        ],
    )
    def test_pretokenization_never_loses_a_character(self, text: str) -> None:
        """Any character matching no alternative is deleted, silently."""
        assert "".join(_pretokenize(text)) == text


class TestTraining:
    def test_the_vocabulary_reaches_the_requested_size(self, tokenizer: BPETokenizer) -> None:
        assert tokenizer.vocab_size == 400

    def test_the_first_256_ids_are_the_raw_bytes(self, tokenizer: BPETokenizer) -> None:
        for value in range(256):
            assert tokenizer.vocab[value] == bytes([value])

    def test_merges_build_real_words(self, tokenizer: BPETokenizer) -> None:
        learned = {tok for tok in tokenizer.vocab.values() if len(tok) > 3}
        assert learned, "no multi-byte merges were learned at all"
        # The passage is about a "truth universally acknowledged"; frequent
        # substrings of it must appear in the vocabulary.
        text = b" ".join(sorted(learned))
        assert b"th" in text or b"the" in text

    def test_training_is_deterministic(self) -> None:
        first, _ = train_bpe(SAMPLE, 320, verbose=False)
        second, _ = train_bpe(SAMPLE, 320, verbose=False)
        assert first == second

    def test_a_vocab_smaller_than_the_byte_alphabet_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            train_bpe(SAMPLE, 100, verbose=False)

    def test_it_stops_early_rather_than_padding_the_vocabulary(self) -> None:
        # A tiny corpus cannot supply thousands of useful merges; the trainer
        # must stop, not invent tokens to hit the target.
        vocab, merges = train_bpe("ab ab ab", 5000, verbose=False)
        assert len(vocab) < 5000
        assert len(merges) < 5000


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "It is a truth universally acknowledged",
            "Numbers: 1987, 42, 3.14159",
            "Punctuation!? -- yes; and (parentheses).",
            "Unicode: café, naïve, שלום, 日本語",
            "   leading and trailing whitespace   ",
            "\n\nnewlines\n\tand tabs\n",
            "",
        ],
    )
    def test_encode_decode_is_lossless(self, tokenizer: BPETokenizer, text: str) -> None:
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_every_byte_value_round_trips(self, tokenizer: BPETokenizer) -> None:
        # Byte-level means there is no such thing as an unknown token.
        raw = bytes(range(256)).decode("latin-1")
        assert tokenizer.decode(tokenizer.encode(raw)) == raw

    def test_unseen_text_still_encodes(self, tokenizer: BPETokenizer) -> None:
        unseen = "quantum chromodynamics ⚛"
        assert tokenizer.decode(tokenizer.encode(unseen)) == unseen

    def test_ids_are_always_in_range(self, tokenizer: BPETokenizer) -> None:
        ids = tokenizer.encode(SAMPLE[:2000])
        assert all(0 <= i < tokenizer.vocab_size for i in ids)


class TestSpecialTokens:
    def test_the_document_separator_is_a_single_token(
        self, tokenizer: BPETokenizer
    ) -> None:
        assert tokenizer.encode(END_OF_TEXT) == [tokenizer.eot_id]

    def test_it_survives_inside_text(self, tokenizer: BPETokenizer) -> None:
        text = f"before{END_OF_TEXT}after"
        ids = tokenizer.encode(text)
        assert tokenizer.eot_id in ids
        assert tokenizer.decode(ids) == text

    def test_it_can_be_disabled(self, tokenizer: BPETokenizer) -> None:
        ids = tokenizer.encode(END_OF_TEXT, allow_special=False)
        assert tokenizer.eot_id not in ids
        assert tokenizer.decode(ids) == END_OF_TEXT


class TestCompression:
    def test_bpe_beats_raw_bytes(self, tokenizer: BPETokenizer) -> None:
        text = SAMPLE[:1000]
        assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))

    def test_a_bigger_vocabulary_compresses_better(self) -> None:
        small = BPETokenizer.train(SAMPLE, 300, verbose=False)
        large = BPETokenizer.train(SAMPLE, 800, verbose=False)
        assert len(large.encode(SAMPLE)) < len(small.encode(SAMPLE))


class TestPersistence:
    def test_save_and_load_round_trips(
        self, tokenizer: BPETokenizer, tmp_path: Path
    ) -> None:
        path = tokenizer.save(tmp_path / "tok.json")
        restored = BPETokenizer.load(path)

        assert restored.vocab_size == tokenizer.vocab_size
        assert restored.vocab == tokenizer.vocab
        assert restored.merges == tokenizer.merges
        text = "It is a truth universally acknowledged, 1987."
        assert restored.encode(text) == tokenizer.encode(text)

    def test_high_bytes_survive_the_json_round_trip(
        self, tokenizer: BPETokenizer, tmp_path: Path
    ) -> None:
        # Bytes are stored as latin-1; anything else would corrupt 0x80-0xFF.
        restored = BPETokenizer.load(tokenizer.save(tmp_path / "tok.json"))
        raw = bytes(range(128, 256)).decode("latin-1")
        assert restored.decode(restored.encode(raw)) == raw
