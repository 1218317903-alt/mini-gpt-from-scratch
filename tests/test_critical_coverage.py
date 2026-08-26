"""Tests for per-module coverage gate parsing and failures."""

from scripts.check_critical_coverage import check_thresholds


def test_critical_coverage_reports_missing_and_undercovered_modules() -> None:
    errors = check_thresholds(
        {"repo/src/covered.py": 79.0},
        {"src/covered.py": 80.0, "src/missing.py": 10.0},
    )

    assert errors == [
        "src/covered.py: 79.00% is below required 80.00%",
        "Missing critical module in coverage data: src/missing.py",
    ]


def test_critical_coverage_accepts_normalized_suffixes() -> None:
    assert (
        check_thresholds(
            {"workspace/src/covered.py": 80.0},
            {"src/covered.py": 80.0},
        )
        == []
    )
