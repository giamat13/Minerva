"""Tests for the corpus pipeline and the batch loader.

The cleaners and the chunker are pure functions, so they are tested offline
against the real distribution quirks they exist to handle. Downloading the
corpus is covered by the integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minerva.training.data import (
    DOCUMENT_SEPARATOR,
    SOURCES,
    _chunk_document,
    _clean_benyehuda,
    _clean_brown,
    _clean_europarl,
    _clean_generic,
    _clean_gutenberg,
)


class TestSourceCatalogue:
    def test_every_source_records_its_provenance(self) -> None:
        # CLAUDE.md requires provenance for every corpus. Enforce it.
        for source in SOURCES:
            assert source.name
            assert source.url.startswith("https://")
            assert source.licence, f"{source.name} has no licence recorded"
            assert source.origin, f"{source.name} has no origin recorded"
            assert len(source.description) > 40, f"{source.name} needs a real description"

    def test_source_names_are_unique(self) -> None:
        names = [source.name for source in SOURCES]
        assert len(names) == len(set(names))

    def test_the_corpus_mixes_registers(self) -> None:
        # A single-register corpus produces a single-register model.
        names = {source.name for source in SOURCES}
        assert {"gutenberg", "benyehuda"} <= names, "literature is the backbone, in two languages"
        assert len(names) >= 4, "at least four distinct registers"

    def test_degraded_corpora_are_not_included(self) -> None:
        # Brown WAS rejected for its POS-tag formatting, then re-included once
        # _clean_brown made that a solved problem - see the module docstring's
        # "Added back in v0.3.0" section. Only the still-genuinely-rejected
        # sources belong in this test now.
        names = {source.name for source in SOURCES}
        assert "movie_reviews" not in names, "lowercased pre-tokenised text was rejected"

    def test_general_knowledge_and_event_sources_are_not_included(self) -> None:
        # v0.3.0: a 9.9M-parameter model cannot reliably hold facts, only
        # imitate their register while confabulating the content. web_search
        # is the honest substitute - see the module docstring in data.py.
        names = {source.name for source in SOURCES}
        assert "wikipedia_simple_en" not in names, "encyclopedic general knowledge was removed"
        assert "reuters" not in names, "newswire event reporting was removed"


class TestGenericCleaner:
    def test_windows_line_endings_are_normalised(self) -> None:
        assert _clean_generic("a\r\nb") == "a\nb"

    def test_runs_of_blank_lines_collapse_to_one(self) -> None:
        assert _clean_generic("a\n\n\n\n\nb") == "a\n\nb"

    def test_trailing_whitespace_is_stripped(self) -> None:
        assert _clean_generic("line   \nnext\t\n") == "line\nnext"

    def test_a_paragraph_break_is_preserved(self) -> None:
        assert _clean_generic("one\n\ntwo") == "one\n\ntwo"

    def test_unicode_is_normalised_but_not_destroyed(self) -> None:
        assert "café" in _clean_generic("café")


class TestGutenbergCleaner:
    def test_the_bracketed_title_line_is_removed(self) -> None:
        raw = "[Emma by Jane Austen 1816]\n\nEmma Woodhouse, handsome, clever."
        cleaned = _clean_gutenberg(raw)
        assert "[Emma by Jane Austen 1816]" not in cleaned
        assert cleaned.startswith("Emma Woodhouse")

    def test_bracketed_text_inside_prose_is_kept(self) -> None:
        # Only a bracket line on its own is metadata.
        raw = "He paused [and looked up] before speaking."
        assert "[and looked up]" in _clean_gutenberg(raw)


class TestEuroparlCleaner:
    def test_sgml_speaker_tags_are_removed(self) -> None:
        raw = "<SPEAKER ID=1 NAME=President>\nI declare the session resumed."
        cleaned = _clean_europarl(raw)
        assert "<SPEAKER" not in cleaned
        assert "I declare the session resumed." in cleaned


class TestBrownCleaner:
    def test_pos_tags_are_stripped(self) -> None:
        raw = "The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr ./."
        cleaned = _clean_brown(raw)
        assert "/at" not in cleaned
        assert "/vbd" not in cleaned
        assert cleaned == "The Fulton County Grand Jury said Friday."

    def test_no_space_creeps_in_before_punctuation(self) -> None:
        # This is exactly why Brown was rejected before v0.3.0's _clean_brown:
        # a naive "join words with spaces" detokenizer puts a space in front
        # of every comma and period.
        raw = "He/pps said/vbd hello/nn ,/, then/rb left/vbd ./."
        cleaned = _clean_brown(raw)
        assert " ," not in cleaned
        assert " ." not in cleaned
        assert cleaned == "He said hello, then left."

    def test_quote_tag_pair_becomes_a_real_paired_quote(self) -> None:
        raw = '``/`` no/at evidence/nn \'\'/\'\' that/cs anything/nn happened/vbd ./.'
        cleaned = _clean_brown(raw)
        assert cleaned.count('"') == 2
        assert cleaned == '"no evidence" that anything happened.'

    def test_a_quote_immediately_followed_by_a_comma_has_no_stray_space(self) -> None:
        raw = "It/pps was/bedz ``/`` received/vbn ''/'' ,/, he/pps said/vbd ./."
        cleaned = _clean_brown(raw)
        assert '" ,' not in cleaned
        assert cleaned == 'It was "received", he said.'

    def test_contractions_stay_a_single_token(self) -> None:
        # Brown distributes these already joined ("didn't/dod*", not two
        # tokens), so no reassembly is needed - just confirm de-tagging
        # doesn't split them.
        raw = "He/pps didn't/dod* want/vb it/ppo ./."
        assert "didn't" in _clean_brown(raw)


class TestBenyehudaCleaner:
    def test_the_volunteer_credit_paragraph_is_removed(self) -> None:
        raw = (
            "שָׁלוֹם רָב שׁוּבֵךְ, צִפֹּרָה נֶחְמֶדֶת\n\n"
            "את הטקסט[ים] לעיל הפיקו מתנדבי פרויקט בן־יהודה באינטרנט."
            "  הכל זמין תמיד בכתובת הבאה:https://benyehuda.org/read/20"
        )
        cleaned = _clean_benyehuda(raw)
        assert "הפיקו מתנדבי" not in cleaned
        assert "benyehuda.org" not in cleaned
        assert cleaned == "שָׁלוֹם רָב שׁוּבֵךְ, צִפֹּרָה נֶחְמֶדֶת"

    def test_the_word_text_in_ordinary_prose_is_not_a_false_match(self) -> None:
        # "את הטקסט" ("the text") is ordinary Hebrew, not just the credit
        # boilerplate's opening words - only the full, distinctive sentence
        # should trigger a cut.
        raw = "הוא דן בפרשנות של את הטקסט המקראי לעומק."
        assert _clean_benyehuda(raw) == raw.strip()


class TestChunking:
    def test_a_short_document_is_left_alone(self) -> None:
        text = "A short paragraph."
        assert _chunk_document(text) == [text]

    def test_a_long_document_is_split(self) -> None:
        text = "\n\n".join(["A paragraph of reasonable length." * 20] * 60)
        chunks = _chunk_document(text)
        assert len(chunks) > 1

    def test_chunks_break_only_at_paragraph_boundaries(self) -> None:
        paragraph = "Sentence one. Sentence two. " * 40
        text = "\n\n".join([paragraph] * 40)
        for chunk in _chunk_document(text):
            assert chunk.startswith("Sentence one."), "a chunk started mid-sentence"
            assert chunk.rstrip().endswith(("two.", "one.")), "a chunk ended mid-sentence"

    def test_no_text_is_lost(self) -> None:
        paragraphs = [f"Paragraph number {i}. " * 30 for i in range(50)]
        text = "\n\n".join(paragraphs)
        rejoined = "\n\n".join(_chunk_document(text))
        assert rejoined == text

    def test_an_oversized_single_paragraph_is_kept_whole(self) -> None:
        # Cutting mid-sentence is worse training signal than an oversized chunk.
        giant = "word " * 20_000
        assert len(_chunk_document(giant)) == 1


class TestDocumentSeparator:
    def test_it_is_the_end_of_text_token(self) -> None:
        assert "<|endoftext|>" in DOCUMENT_SEPARATOR

    def test_it_is_surrounded_by_blank_lines(self) -> None:
        assert DOCUMENT_SEPARATOR.startswith("\n\n")
        assert DOCUMENT_SEPARATOR.endswith("\n\n")


torch = pytest.importorskip("torch", reason="needs the 'training' extra")
np = pytest.importorskip("numpy")

from minerva.training.dataset import TokenDataset  # noqa: E402


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    """A real binary token file: ids 0..999 repeated."""
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint16).repeat(5).tofile(path)
    return path


@pytest.mark.torch
class TestTokenDataset:
    def test_it_reports_the_token_count(self, token_file: Path) -> None:
        assert len(TokenDataset(token_file, 16)) == 5000

    def test_a_batch_has_the_requested_shape(self, token_file: Path) -> None:
        dataset = TokenDataset(token_file, 16)
        inputs, targets = dataset.batch(4, np.random.default_rng(0))
        assert inputs.shape == (4, 16)
        assert targets.shape == (4, 16)
        assert inputs.dtype == torch.int64

    def test_targets_are_inputs_shifted_by_one(self, token_file: Path) -> None:
        """The next-token objective lives or dies on this alignment."""
        dataset = TokenDataset(token_file, 16)
        inputs, targets = dataset.batch(4, np.random.default_rng(0))
        assert torch.equal(inputs[:, 1:], targets[:, :-1])

    def test_sampling_is_reproducible_from_a_seed(self, token_file: Path) -> None:
        dataset = TokenDataset(token_file, 16)
        first, _ = dataset.batch(4, np.random.default_rng(7))
        second, _ = dataset.batch(4, np.random.default_rng(7))
        assert torch.equal(first, second)

    def test_different_seeds_give_different_windows(self, token_file: Path) -> None:
        dataset = TokenDataset(token_file, 16)
        first, _ = dataset.batch(8, np.random.default_rng(1))
        second, _ = dataset.batch(8, np.random.default_rng(2))
        assert not torch.equal(first, second)

    def test_sequential_batches_are_deterministic(self, token_file: Path) -> None:
        """Validation must not resample, or the number moves for no reason."""
        dataset = TokenDataset(token_file, 16)
        first = dataset.sequential_batches(4, limit=3)
        second = dataset.sequential_batches(4, limit=3)
        assert len(first) == len(second)
        for (xa, ya), (xb, yb) in zip(first, second, strict=True):
            assert torch.equal(xa, xb)
            assert torch.equal(ya, yb)

    def test_sequential_batches_do_not_overlap(self, token_file: Path) -> None:
        dataset = TokenDataset(token_file, 16)
        batch = dataset.sequential_batches(2, limit=1)[0][0]
        assert not torch.equal(batch[0], batch[1])

    def test_a_corpus_too_small_for_the_window_is_refused(self, tmp_path: Path) -> None:
        tiny = tmp_path / "tiny.bin"
        np.arange(4, dtype=np.uint16).tofile(tiny)
        with pytest.raises(ValueError, match="too few"):
            TokenDataset(tiny, 512)
