"""Validate one P10 private-domain dataset before any training job can start."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agent.domain_adapter import DomainAdapterRegistry, validate_domain_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    registry = DomainAdapterRegistry(args.root)
    registry.load()
    manifest = registry.manifest(args.domain_id)
    report = validate_domain_dataset(manifest, registry.dataset_path(args.domain_id))
    payload = asdict(report)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
