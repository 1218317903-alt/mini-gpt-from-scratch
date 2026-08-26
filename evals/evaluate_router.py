"""Evaluate the deterministic route decision independently of model generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.routing import HeuristicAgentRouter  # noqa: E402
from evals.metrics import load_cases  # noqa: E402


def evaluate_router(cases_dir: Path) -> dict[str, Any]:
    router = HeuristicAgentRouter()
    available = ("calculator", "current_time", "search_docs")
    exact = 0
    expected_nonempty = 0
    expected_nonempty_hits = 0
    expected_empty = 0
    expected_empty_hits = 0
    rows: list[dict[str, Any]] = []
    for case in load_cases(cases_dir):
        decision = router.route(case.messages, available)
        expected = tuple(sorted(set(case.expected_tools)))
        actual = tuple(sorted(decision.allowed_tools))
        matched = expected == actual
        exact += int(matched)
        if expected:
            expected_nonempty += 1
            expected_nonempty_hits += int(set(expected).issubset(set(actual)))
        else:
            expected_empty += 1
            expected_empty_hits += int(not actual)
        rows.append(
            {
                "case_id": case.id,
                "expected": expected,
                "actual": actual,
                "route": decision.kind,
                "matched": matched,
                "reason": decision.reason,
            }
        )
    return {
        "case_count": len(rows),
        "exact_tool_set_accuracy": exact / len(rows) if rows else None,
        "required_tool_recall": (
            expected_nonempty_hits / expected_nonempty if expected_nonempty else None
        ),
        "no_tool_accuracy": (
            expected_empty_hits / expected_empty if expected_empty else None
        ),
        "failures": [row for row in rows if not row["matched"]],
        "predictions": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = evaluate_router(args.cases_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
