"""Create or intentionally refresh a versioned evaluation manifest."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.inventory import build_manifest_payload, write_manifest  # noqa: E402
from evals.metrics import load_cases  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument(
        "--review-status",
        default="pending_owner_signoff",
    )
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--signed-at", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    counts = Counter(case.category for case in load_cases(args.cases_dir))
    payload = build_manifest_payload(
        args.cases_dir,
        split=args.split,
        version=args.version,
        purpose=args.purpose,
        frozen=args.frozen,
        expected_counts=dict(sorted(counts.items())),
        review_status=args.review_status,
        provenance=args.provenance,
        reviewer=args.reviewer,
        signed_at=args.signed_at,
    )
    path = write_manifest(args.cases_dir, payload)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
