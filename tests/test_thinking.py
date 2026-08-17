"""Tests for the solfege thinking scale."""

from __future__ import annotations

import pytest

from minerva.errors import ThinkingLevelError
from minerva.thinking import ThinkingLevel, parse_thinking_level


class TestScaleOrder:
    def test_the_scale_is_do_re_mi_fa_sol_la_si_in_order(self) -> None:
        assert [level.latin_name for level in ThinkingLevel] == [
            "do",
            "re",
            "mi",
            "fa",
            "sol",
            "la",
            "si",
        ]

    def test_hebrew_names_follow_the_same_order(self) -> None:
        assert [level.hebrew_name for level in ThinkingLevel] == [
            "דו",
            "רה",
            "מי",
            "פה",
            "סול",
            "לה",
            "סי",
        ]

    def test_levels_compare_by_intensity(self) -> None:
        assert ThinkingLevel.DO < ThinkingLevel.RE < ThinkingLevel.SI
        assert max(ThinkingLevel) is ThinkingLevel.SI
        assert min(ThinkingLevel) is ThinkingLevel.DO

    def test_budgets_increase_monotonically(self) -> None:
        budgets = [level.profile.budget_tokens for level in ThinkingLevel]
        assert budgets == sorted(budgets)
        assert len(set(budgets)) == len(budgets), "each note must have its own budget"

    def test_tool_iteration_ceilings_never_decrease(self) -> None:
        ceilings = [level.profile.max_tool_iterations for level in ThinkingLevel]
        assert ceilings == sorted(ceilings)


class TestSemantics:
    def test_do_is_the_only_silent_level(self) -> None:
        silent = [level for level in ThinkingLevel if level.is_silent]
        assert silent == [ThinkingLevel.DO]
        assert ThinkingLevel.DO.profile.enabled is False
        assert ThinkingLevel.DO.profile.effort is None

    def test_extended_thinking_starts_at_la(self) -> None:
        assert [level for level in ThinkingLevel if level.is_extended] == [
            ThinkingLevel.LA,
            ThinkingLevel.SI,
        ]

    def test_extended_levels_preserve_reasoning(self) -> None:
        assert ThinkingLevel.SI.profile.preserve_thinking is True
        assert ThinkingLevel.SOL.profile.preserve_thinking is False

    def test_every_non_silent_level_has_an_effort(self) -> None:
        for level in ThinkingLevel:
            if level.is_silent:
                continue
            assert level.profile.effort in {"low", "medium", "high"}

    def test_profile_reports_its_own_level(self) -> None:
        for level in ThinkingLevel:
            assert level.profile.level is level

    def test_every_level_is_documented(self) -> None:
        for level in ThinkingLevel:
            assert level.description.strip()


class TestLaddering:
    def test_louder_climbs_and_clamps_at_si(self) -> None:
        assert ThinkingLevel.DO.louder() is ThinkingLevel.RE
        assert ThinkingLevel.FA.louder(2) is ThinkingLevel.LA
        assert ThinkingLevel.SI.louder() is ThinkingLevel.SI

    def test_quieter_descends_and_clamps_at_do(self) -> None:
        assert ThinkingLevel.SI.quieter() is ThinkingLevel.LA
        assert ThinkingLevel.MI.quieter(5) is ThinkingLevel.DO


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("do", ThinkingLevel.DO),
            ("SOL", ThinkingLevel.SOL),
            ("  si  ", ThinkingLevel.SI),
            ("דו", ThinkingLevel.DO),
            ("סול", ThinkingLevel.SOL),
            ("סי", ThinkingLevel.SI),
            ("ti", ThinkingLevel.SI),
            ("so", ThinkingLevel.SOL),
            ("off", ThinkingLevel.DO),
            ("medium", ThinkingLevel.FA),
            ("max", ThinkingLevel.SI),
            ("4", ThinkingLevel.SOL),
        ],
    )
    def test_accepted_spellings(self, text: str, expected: ThinkingLevel) -> None:
        assert parse_thinking_level(text) is expected

    def test_parses_indices(self) -> None:
        for index, level in enumerate(ThinkingLevel):
            assert parse_thinking_level(index) is level

    def test_is_idempotent(self) -> None:
        assert parse_thinking_level(ThinkingLevel.LA) is ThinkingLevel.LA

    def test_classmethod_matches_the_function(self) -> None:
        assert ThinkingLevel.parse("לה") is parse_thinking_level("לה")

    @pytest.mark.parametrize("bad", ["", "doh-re", "octave", "-1", "7", None, 3.5])
    def test_rejects_nonsense(self, bad: object) -> None:
        with pytest.raises(ThinkingLevelError):
            parse_thinking_level(bad)  # type: ignore[arg-type]

    def test_rejects_out_of_range_indices(self) -> None:
        with pytest.raises(ThinkingLevelError):
            parse_thinking_level(7)
        with pytest.raises(ThinkingLevelError):
            parse_thinking_level(-1)

    def test_rejects_booleans_even_though_bool_is_an_int(self) -> None:
        with pytest.raises(ThinkingLevelError):
            parse_thinking_level(True)  # type: ignore[arg-type]

    def test_error_message_lists_every_note(self) -> None:
        with pytest.raises(ThinkingLevelError) as exc_info:
            parse_thinking_level("octave")
        message = str(exc_info.value)
        for level in ThinkingLevel:
            assert level.latin_name in message
            assert level.hebrew_name in message

    def test_str_is_the_latin_name(self) -> None:
        assert str(ThinkingLevel.SOL) == "sol"
