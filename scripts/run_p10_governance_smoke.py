"""Create synthetic governed domains and verify P10 fail-closed controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from agent.domain_adapter import DomainAdapterRegistry, validate_domain_dataset

ROOT = Path(__file__).resolve().parents[1]
SMOKE_TOKEN = "p10-smoke-token-not-for-production"


def manifest(domain_id: str, *, adult: bool) -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "display_name": "合成治理验收域",
        "description": "只用于验证隔离、授权和数据门禁，不是正式领域模型。",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": f"adapters/{domain_id}",
        "dataset_path": f"datasets/{domain_id}.jsonl",
        "enabled": True,
        "access_mode": "restricted" if adult else "private",
        "access_token_sha256": hashlib.sha256(SMOKE_TOKEN.encode()).hexdigest(),
        "data_classification": "restricted" if adult else "confidential",
        "adult_content_enabled": adult,
        "minimum_age": 18 if adult else None,
        "require_consent_attestation": True,
        "allowed_capabilities": ["text"],
        "license_summary": "Synthetic test-only data with no real person.",
    }


def record(record_id: str, *, adult: bool, unsafe: bool = False) -> dict[str, object]:
    return {
        "record_id": record_id,
        "prompt": "涉及未成年人的露骨请求" if unsafe else "合成测试问题",
        "response": "必须拒绝" if unsafe else "合成测试回答",
        "source": "synthetic:p10-smoke",
        "license": "test-only",
        "consent_basis": "No real person; generated solely for governance tests.",
        "contains_personal_data": False,
        "personal_data_reviewed": False,
        "all_human_subjects_adults": True if adult else None,
        "consensual_context": True if adult else None,
    }


def main() -> None:
    root = ROOT / "output/p10_governance_smoke"
    (root / "adapters/standard_smoke").mkdir(parents=True, exist_ok=True)
    (root / "adapters/adult_smoke").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "standard.json").write_text(
        json.dumps(manifest("standard_smoke", adult=False), ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "adult.json").write_text(
        json.dumps(manifest("adult_smoke", adult=True), ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "datasets/standard_smoke.jsonl").write_text(
        json.dumps(record("standard-1", adult=False), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    adult_path = root / "datasets/adult_smoke.jsonl"
    adult_path.write_text(
        json.dumps(record("adult-safe-1", adult=True), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    registry = DomainAdapterRegistry(root)
    registry.load()
    standard_report = validate_domain_dataset(
        registry.manifest("standard_smoke"),
        registry.dataset_path("standard_smoke"),
    )
    adult_safe_report = validate_domain_dataset(
        registry.manifest("adult_smoke"), adult_path
    )
    adult_path.write_text(
        json.dumps(
            record("adult-unsafe-1", adult=True, unsafe=True), ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    adult_unsafe_report = validate_domain_dataset(
        registry.manifest("adult_smoke"), adult_path
    )
    unauthorized_denied = False
    try:
        registry.authorize("standard_smoke", "wrong-token")
    except PermissionError:
        unauthorized_denied = True
    selection = registry.authorize("standard_smoke", SMOKE_TOKEN)

    payload = {
        "date": "2026-08-25",
        "author": "李林知",
        "status": "governance_smoke_no_adapter_training",
        "standard_dataset": asdict(standard_report),
        "adult_safe_dataset": asdict(adult_safe_report),
        "adult_prohibited_dataset": asdict(adult_unsafe_report),
        "unauthorized_denied": unauthorized_denied,
        "authorized_domain": selection.domain_id,
        "catalog_hides_paths_and_tokens": all(
            "adapter_path" not in item and "access_token_sha256" not in item
            for item in registry.catalog()
        ),
        "default_public_model_modified": False,
        "limitations": [
            "本实验只验证 manifest、路径隔离、Token 授权与数据门禁，没有训练领域 Adapter。",
            "关键词门禁只是第一层静态检查，正式数据还需要人工、法律和隐私审核。",
            "成人模板默认关闭，不包含语料和权重，不能据此宣称已具备成人生成质量。",
            "Adapter 激活是进程级状态；生产环境应按租户隔离 worker，避免跨请求状态泄漏。",
        ],
    }
    output = ROOT / "docs/p10_domain_governance_smoke.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
