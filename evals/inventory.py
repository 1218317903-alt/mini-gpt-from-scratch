"""Versioned evaluation inventory manifests and split-contamination checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class EvalInventory:
    """Version and integrity contract for one directory of evaluation cases."""

    split: str
    version: str
    purpose: str
    frozen: bool
    expected_counts: dict[str, int]
    minimum_cases: int
    maximum_cases: int
    case_files: dict[str, str]
    fingerprint: str
    review_status: str


def case_file_hashes(cases_dir: Path) -> dict[str, str]:
    """Return SHA-256 hashes for every JSONL case file in stable order."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(cases_dir.glob("*.jsonl"))
    }


def split_fingerprint(cases_dir: Path) -> str:
    """Fingerprint filenames and exact bytes for one evaluation split."""
    digest = hashlib.sha256()
    for path in sorted(cases_dir.glob("*.jsonl")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_inventory(cases_dir: Path) -> EvalInventory | None:
    """Load an optional manifest stored beside a case split."""
    path = cases_dir / MANIFEST_NAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    bounds = payload.get("case_count_range", [0, 2**31 - 1])
    review = payload.get("review", {})
    return EvalInventory(
        split=str(payload["split"]),
        version=str(payload["version"]),
        purpose=str(payload.get("purpose", "")),
        frozen=bool(payload.get("frozen", False)),
        expected_counts={
            str(name): int(count)
            for name, count in payload.get("expected_counts", {}).items()
        },
        minimum_cases=int(bounds[0]),
        maximum_cases=int(bounds[1]),
        case_files={
            str(name): str(checksum)
            for name, checksum in payload.get("case_files", {}).items()
        },
        fingerprint=str(payload.get("fingerprint", "")),
        review_status=str(review.get("status", "not_recorded")),
    )


def validate_inventory(cases_dir: Path, inventory: EvalInventory) -> list[str]:
    """Return integrity errors without treating human review as automated proof."""
    errors: list[str] = []
    actual_files = case_file_hashes(cases_dir)
    if actual_files != inventory.case_files:
        errors.append("case file SHA-256 hashes do not match manifest")
    actual_fingerprint = split_fingerprint(cases_dir)
    if actual_fingerprint != inventory.fingerprint:
        errors.append("split fingerprint does not match manifest")
    return errors


def build_manifest_payload(
    cases_dir: Path,
    *,
    split: str,
    version: str,
    purpose: str,
    frozen: bool,
    expected_counts: dict[str, int],
    review_status: str,
    provenance: str,
    reviewer: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Build a serializable manifest after cases have been written."""
    total = sum(expected_counts.values())
    return {
        "schema_version": 1,
        "split": split,
        "version": version,
        "purpose": purpose,
        "frozen": frozen,
        "expected_counts": expected_counts,
        "case_count_range": [total, total],
        "case_files": case_file_hashes(cases_dir),
        "fingerprint": split_fingerprint(cases_dir),
        "review": {
            "status": review_status,
            "reviewer": reviewer,
            "signed_at": signed_at,
        },
        "provenance": provenance,
    }


def write_manifest(cases_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic UTF-8 manifest beside the case files."""
    path = cases_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def normalized_user_texts(cases_dir: Path) -> dict[str, str]:
    """Return normalized first user messages without importing metric objects."""
    texts: dict[str, str] = {}
    for path in sorted(cases_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            user = next(
                (
                    str(message.get("content", ""))
                    for message in payload.get("messages", [])
                    if message.get("role") == "user"
                ),
                "",
            )
            texts[str(payload["id"])] = _normalize_text(user)
    return texts


def compare_split_texts(
    left_dir: Path,
    right_dir: Path,
    *,
    near_duplicate_threshold: float = 0.92,
) -> dict[str, list[dict[str, Any]]]:
    """Find exact and high-similarity prompts across two evaluation splits."""
    left = normalized_user_texts(left_dir)
    right = normalized_user_texts(right_dir)
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for left_id, left_text in left.items():
        for right_id, right_text in right.items():
            if not left_text or not right_text:
                continue
            if left_text == right_text:
                exact.append({"left": left_id, "right": right_id})
                continue
            if min(len(left_text), len(right_text)) < 12:
                continue
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= near_duplicate_threshold:
                near.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "similarity": round(ratio, 6),
                    }
                )
    return {"exact": exact, "near": near}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()
