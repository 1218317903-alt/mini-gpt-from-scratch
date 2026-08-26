"""P10 private-domain isolation, authorization, and data-governance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from agent.domain_adapter import (
    DomainAdapterController,
    DomainAdapterManifest,
    DomainAdapterRegistry,
    validate_domain_dataset,
)
from minigpt_service.app import create_app
from minigpt_service.chat_model import ChatResult

TOKEN = "correct-private-token"


def manifest_payload(*, adult: bool = False, enabled: bool = True) -> dict[str, object]:
    return {
        "domain_id": "adult_domain" if adult else "legal_domain",
        "display_name": "受控领域",
        "description": "独立测试领域",
        "base_model": "base/model",
        "adapter_path": "adapters/current",
        "dataset_path": "datasets/current.jsonl",
        "enabled": enabled,
        "access_mode": "restricted" if adult else "private",
        "access_token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest(),
        "data_classification": "restricted" if adult else "confidential",
        "adult_content_enabled": adult,
        "minimum_age": 18 if adult else None,
        "require_consent_attestation": True,
        "allowed_capabilities": ["text"],
        "license_summary": "Test-only licensed synthetic data.",
    }


def record_payload(record_id: str, *, adult: bool = False) -> dict[str, object]:
    return {
        "record_id": record_id,
        "prompt": "测试问题",
        "response": "测试回答",
        "source": "synthetic:test",
        "license": "test-only",
        "consent_basis": "synthetic record with no real person",
        "contains_personal_data": False,
        "personal_data_reviewed": False,
        "all_human_subjects_adults": True if adult else None,
        "consensual_context": True if adult else None,
    }


def create_registry(tmp_path: Path, *, adult: bool = False) -> DomainAdapterRegistry:
    (tmp_path / "adapters/current").mkdir(parents=True)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "domain.json").write_text(
        json.dumps(manifest_payload(adult=adult)), encoding="utf-8"
    )
    (tmp_path / "datasets/current.jsonl").write_text(
        json.dumps(record_payload("one", adult=adult), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    registry = DomainAdapterRegistry(tmp_path)
    registry.load()
    return registry


def test_registry_catalog_hides_paths_and_tokens(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    catalog = registry.catalog()
    assert catalog[0]["domain_id"] == "legal_domain"
    assert "adapter_path" not in catalog[0]
    assert "access_token_sha256" not in catalog[0]


def test_registry_authorization_is_fail_closed(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    with pytest.raises(PermissionError):
        registry.authorize("legal_domain", "wrong")
    with pytest.raises(PermissionError):
        registry.authorize("unknown", TOKEN)
    selection = registry.authorize("legal_domain", TOKEN)
    assert selection.adapter_path == (tmp_path / "adapters/current").resolve()


def test_registry_rejects_paths_outside_domain_root(tmp_path: Path) -> None:
    payload = manifest_payload()
    payload["adapter_path"] = "../outside"
    (tmp_path / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
    registry = DomainAdapterRegistry(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        registry.load()


def test_standard_domain_dataset_validation(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    report = validate_domain_dataset(
        registry.manifest("legal_domain"), registry.dataset_path("legal_domain")
    )
    assert report.valid is True
    assert report.records == 1
    assert report.adult_policy_records == 0


def test_adult_domain_requires_adults_consent_and_rejects_prohibited_markers(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path, adult=True)
    manifest = registry.manifest("adult_domain")
    dataset = registry.dataset_path("adult_domain")
    invalid = record_payload("unsafe", adult=True)
    invalid["prompt"] = "涉及未成年人的露骨请求"
    dataset.write_text(json.dumps(invalid, ensure_ascii=False) + "\n", encoding="utf-8")
    report = validate_domain_dataset(manifest, dataset)
    assert report.valid is False
    assert report.prohibited_records == ("unsafe",)

    dataset.write_text(
        json.dumps(record_payload("safe", adult=True), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = validate_domain_dataset(manifest, dataset)
    assert report.valid is True
    assert report.adult_policy_records == 1


def test_adult_manifest_cannot_be_public_or_skip_consent() -> None:
    payload = manifest_payload(adult=True)
    payload["access_mode"] = "private"
    with pytest.raises(ValueError, match="restricted"):
        DomainAdapterManifest.model_validate(payload)
    payload = manifest_payload(adult=True)
    payload["require_consent_attestation"] = False
    with pytest.raises(ValueError, match="consent"):
        DomainAdapterManifest.model_validate(payload)


class FakeActivationBackend:
    def __init__(self) -> None:
        self.active: str | None = None

    def activate_adapter(self, domain_id: str, adapter_path: Path) -> None:
        assert adapter_path.is_dir()
        self.active = domain_id

    def deactivate_adapter(self) -> None:
        self.active = None


def test_controller_authorizes_before_activation(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    backend = FakeActivationBackend()
    controller = DomainAdapterController(registry, backend)
    with pytest.raises(PermissionError):
        controller.activate("legal_domain", "wrong")
    assert backend.active is None
    controller.activate("legal_domain", TOKEN)
    assert controller.active_domain == "legal_domain"
    controller.deactivate()
    assert backend.active is None


def test_request_scoped_domain_inference_serializes_default_work(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)
    backend = FakeActivationBackend()
    controller = DomainAdapterController(registry, backend)
    domain_entered = Event()
    release_domain = Event()
    default_completed = Event()

    def domain_operation() -> None:
        assert backend.active == "legal_domain"
        domain_entered.set()
        assert release_domain.wait(timeout=1)

    domain_thread = Thread(
        target=lambda: controller.run_authorized(
            "legal_domain", TOKEN, domain_operation
        )
    )
    default_thread = Thread(
        target=lambda: controller.run_default(default_completed.set)
    )
    domain_thread.start()
    assert domain_entered.wait(timeout=1)
    default_thread.start()
    assert not default_completed.wait(timeout=0.05)
    release_domain.set()
    domain_thread.join(timeout=1)
    default_thread.join(timeout=1)

    assert default_completed.is_set()
    assert backend.active is None
    assert controller.active_domain is None


class DomainAwareChatModel:
    is_ready = True

    def __init__(self) -> None:
        self.active: str | None = None

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def activate_adapter(self, domain_id: str, adapter_path: Path) -> None:
        assert adapter_path.is_dir()
        self.active = domain_id

    def deactivate_adapter(self) -> None:
        self.active = None

    def chat(self, _messages: object, **_kwargs: object) -> ChatResult:
        return ChatResult(
            text=self.active or "default",
            prompt_tokens=1,
            generated_tokens=1,
            finish_reason="stop",
            model="domain-aware-fake",
        )


def test_domain_http_inference_is_request_scoped(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    backend = DomainAwareChatModel()
    service_app = create_app(chat_model=backend, domain_registry=registry)  # type: ignore[arg-type]

    with TestClient(service_app) as client:
        scoped = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "受控问题"}],
                "domain_id": "legal_domain",
                "domain_access_token": TOKEN,
            },
        )
        default = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "普通问题"}]},
        )
        persistent = client.post(
            "/domains/legal_domain/activate",
            json={"access_token": TOKEN},
        )

    assert scoped.status_code == 200
    assert scoped.json()["generated_text"] == "legal_domain"
    assert default.status_code == 200
    assert default.json()["generated_text"] == "default"
    assert persistent.status_code == 409
    assert backend.active is None


def test_domain_http_catalog_hides_secrets_and_authorizes(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    client = TestClient(create_app(domain_registry=registry))
    catalog = client.get("/domains")
    assert catalog.status_code == 200
    domain = catalog.json()["domains"][0]
    assert domain["domain_id"] == "legal_domain"
    assert "adapter_path" not in domain
    assert "access_token_sha256" not in domain
    denied = client.post(
        "/domains/legal_domain/authorize", json={"access_token": "wrong"}
    )
    assert denied.status_code == 403
    allowed = client.post(
        "/domains/legal_domain/authorize", json={"access_token": TOKEN}
    )
    assert allowed.status_code == 200
    assert allowed.json()["authorized"] is True
