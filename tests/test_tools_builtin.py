"""Tests for the built-in tools.

These run the actual implementations - the calculator really evaluates, the
clock really reads the system clock.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from minerva.errors import ToolExecutionError
from minerva.tools.builtin import builtin_tools, calculate, current_time, days_between


class TestCalculatorArithmetic:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 + 1", "2"),
            ("17 * 43", "731"),
            ("(17 * 43) / 6", "121.83333333333333"),
            ("2 ** 10", "1024"),
            ("7 // 2", "3"),
            ("7 % 2", "1"),
            ("-5 + 3", "-2"),
            ("10 / 4", "2.5"),
            ("12 / 4", "3"),
        ],
    )
    def test_basic_operators(self, expression: str, expected: str) -> None:
        assert calculate.invoke({"expression": expression}) == expected

    def test_operator_precedence_is_pythons(self) -> None:
        assert calculate.invoke({"expression": "2 + 3 * 4"}) == "14"
        assert calculate.invoke({"expression": "(2 + 3) * 4"}) == "20"

    def test_constants(self) -> None:
        assert calculate.invoke({"expression": "pi"}).startswith("3.14159")
        assert calculate.invoke({"expression": "e"}).startswith("2.718")

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("sqrt(144)", 12.0),
            ("log(100, 10)", 2.0),
            ("log10(1000)", 3.0),
            ("factorial(10)", 3628800),
            ("gcd(48, 18)", 6),
            ("hypot(3, 4)", 5.0),
            ("floor(3.7)", 3),
            ("ceil(3.2)", 4),
            ("round(3.14159, 2)", 3.14),
            ("max(3, 9, 2)", 9),
            ("abs(-4)", 4),
            ("degrees(pi)", 180.0),
        ],
    )
    def test_functions(self, expression: str, expected: float) -> None:
        result = float(calculate.invoke({"expression": expression}))
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_comparisons_return_booleans(self) -> None:
        assert calculate.invoke({"expression": "2 + 2 == 4"}) == "true"
        assert calculate.invoke({"expression": "1 > 2"}) == "false"
        assert calculate.invoke({"expression": "1 < 2 < 3"}) == "true"

    def test_integral_floats_print_as_integers(self) -> None:
        # A model reading "6.0" tends to repeat it back verbatim; "6" is the
        # honest rendering of an exact result.
        assert calculate.invoke({"expression": "sqrt(36)"}) == "6"

    def test_non_integral_results_keep_full_precision(self) -> None:
        assert calculate.invoke({"expression": "1/3"}) == repr(1 / 3)


class TestCalculatorSafety:
    """The evaluator walks an AST allow-list; nothing outside it executes."""

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",
            "().__class__",
            "open('/etc/passwd').read()",
            "eval('1+1')",
            "x := 5",
            "[i for i in range(3)]",
            "lambda: 1",
            "print(1)",
        ],
    )
    def test_dangerous_expressions_are_refused(self, expression: str) -> None:
        with pytest.raises(ToolExecutionError):
            calculate.invoke({"expression": expression})

    def test_unknown_names_are_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="unknown name"):
            calculate.invoke({"expression": "secret + 1"})

    def test_unknown_functions_are_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="unknown function"):
            calculate.invoke({"expression": "exec_me(1)"})

    def test_huge_exponents_are_refused_rather_than_hanging(self) -> None:
        with pytest.raises(ToolExecutionError, match="exponent exceeds"):
            calculate.invoke({"expression": "9 ** 9 ** 9"})

    def test_huge_factorials_are_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="above the"):
            calculate.invoke({"expression": "factorial(100000)"})

    def test_syntax_errors_report_the_expression(self) -> None:
        with pytest.raises(ToolExecutionError, match="could not parse"):
            calculate.invoke({"expression": "2 +"})

    def test_empty_expression_is_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="empty"):
            calculate.invoke({"expression": "   "})

    def test_division_by_zero_surfaces_as_a_tool_error(self) -> None:
        with pytest.raises(ToolExecutionError):
            calculate.invoke({"expression": "1/0"})


class TestClock:
    def test_reports_the_real_current_time(self) -> None:
        answer = current_time.invoke({"timezone": "UTC"})
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", answer)
        assert match, f"unexpected format: {answer}"

        reported = datetime.fromisoformat(match.group(1)).replace(tzinfo=ZoneInfo("UTC"))
        assert abs(reported - datetime.now(ZoneInfo("UTC"))) < timedelta(seconds=30)

    def test_defaults_to_utc(self) -> None:
        assert "UTC" in current_time.invoke({})

    def test_honours_a_real_timezone(self) -> None:
        answer = current_time.invoke({"timezone": "Asia/Jerusalem"})
        assert "Asia/Jerusalem" in answer
        assert re.search(r"UTC[+-]\d{2}:\d{2}", answer)

    def test_includes_the_weekday(self) -> None:
        answer = current_time.invoke({"timezone": "UTC"})
        today = datetime.now(ZoneInfo("UTC")).strftime("%A")
        assert today in answer

    def test_unknown_timezone_is_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="unknown time zone"):
            current_time.invoke({"timezone": "Middle/Earth"})


class TestDaysBetween:
    def test_counts_days_forward(self) -> None:
        assert days_between.invoke(
            {"start_date": "2026-01-01", "end_date": "2026-01-31"}
        ).startswith("30 day(s)")

    def test_counts_days_backward(self) -> None:
        answer = days_between.invoke({"start_date": "2026-03-01", "end_date": "2026-02-01"})
        assert answer.startswith("28 day(s)")
        assert "before" in answer

    def test_leap_year_is_handled_by_the_real_calendar(self) -> None:
        assert days_between.invoke(
            {"start_date": "2024-02-28", "end_date": "2024-03-01"}
        ).startswith("2 day(s)")

    def test_bad_dates_are_refused(self) -> None:
        with pytest.raises(ToolExecutionError, match="not a valid ISO date"):
            days_between.invoke({"start_date": "yesterday", "end_date": "2026-01-01"})


class TestBuiltinCatalogue:
    def test_every_builtin_has_a_name_and_a_description(self) -> None:
        for builtin in builtin_tools():
            assert builtin.name
            assert len(builtin.description) > 20, f"{builtin.name} needs a real description"

    def test_names_are_unique(self) -> None:
        names = [builtin.name for builtin in builtin_tools()]
        assert len(names) == len(set(names))

    def test_every_builtin_advertises_an_object_schema(self) -> None:
        for builtin in builtin_tools():
            assert builtin.parameters["type"] == "object"
