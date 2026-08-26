"""Enforce per-module coverage floors that an aggregate percentage can hide."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CRITICAL_THRESHOLDS = {
    "src/agent/domain_adapter.py": 80.0,
    "src/agent/orchestrator.py": 90.0,
    "src/minigpt_service/app.py": 60.0,
    "src/minigpt_service/domain_runtime.py": 75.0,
    "src/minigpt_service/model_service.py": 35.0,
    "hf_lora/src/hf_lora/training.py": 18.0,
}


def load_module_coverage(path: Path) -> dict[str, float]:
    """Return normalized file names and their coverage percentages."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("Coverage JSON does not contain a files mapping.")
    result: dict[str, float] = {}
    for filename, details in files.items():
        if not isinstance(filename, str) or not isinstance(details, dict):
            continue
        summary = details.get("summary")
        if not isinstance(summary, dict):
            continue
        percent = summary.get("percent_covered")
        if isinstance(percent, int | float):
            result[filename.replace("\\", "/")] = float(percent)
    return result


def check_thresholds(
    coverage: dict[str, float],
    thresholds: dict[str, float] = CRITICAL_THRESHOLDS,
) -> list[str]:
    """Return actionable errors for missing or under-covered critical modules."""
    errors: list[str] = []
    for expected, minimum in thresholds.items():
        matches = [
            value for filename, value in coverage.items() if filename.endswith(expected)
        ]
        if not matches:
            errors.append(f"Missing critical module in coverage data: {expected}")
            continue
        actual = matches[0]
        if actual + 1e-9 < minimum:
            errors.append(f"{expected}: {actual:.2f}% is below required {minimum:.2f}%")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", type=Path)
    args = parser.parse_args(argv)
    errors = check_thresholds(load_module_coverage(args.coverage_json))
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Critical module coverage thresholds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
