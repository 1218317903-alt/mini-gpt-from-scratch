"""Validate development/frozen inventories and cross-split contamination."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.inventory import compare_split_texts  # noqa: E402
from evals.run_agent_eval import validate_cases  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("evals/frozen_cases"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dev = validate_cases(args.dev_dir)
    frozen = validate_cases(args.frozen_dir)
    overlap = compare_split_texts(args.dev_dir, args.frozen_dir)
    dev_ids = {
        case_id for path in args.dev_dir.glob("*.jsonl") for case_id in _ids(path)
    }
    frozen_ids = {
        case_id for path in args.frozen_dir.glob("*.jsonl") for case_id in _ids(path)
    }
    duplicate_ids = sorted(dev_ids & frozen_ids)
    ok = bool(dev["ok"] and frozen["ok"] and not duplicate_ids and not overlap["exact"])
    report = {
        "ok": ok,
        "dev": dev,
        "frozen_test": frozen,
        "duplicate_case_ids": duplicate_ids,
        "prompt_overlap": overlap,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def _ids(path: Path) -> list[str]:
    return [
        str(json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
