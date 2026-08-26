"""Isolated private-domain adapter manifests, authorization, and data governance."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DOMAIN_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_PROHIBITED_SEXUAL_MARKERS = (
    "未成年",
    "儿童色情",
    "强迫性行为",
    "非自愿性行为",
    "minor sexual",
    "child sexual",
    "non-consensual sexual",
)
_ResultT = TypeVar("_ResultT")


class DomainAdapterManifest(BaseModel):
    """Public-safe metadata plus confined paths for one optional Adapter."""

    model_config = ConfigDict(extra="forbid")

    domain_id: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    base_model: str = Field(min_length=1, max_length=300)
    adapter_path: str = Field(min_length=1, max_length=300)
    dataset_path: str = Field(min_length=1, max_length=300)
    enabled: bool = False
    access_mode: Literal["private", "restricted"] = "private"
    access_token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_classification: Literal["internal", "confidential", "restricted"]
    adult_content_enabled: bool = False
    minimum_age: int | None = Field(default=None, ge=18, le=100)
    require_consent_attestation: bool = True
    allowed_capabilities: list[str] = Field(default_factory=lambda: ["text"])
    license_summary: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_policy(self) -> DomainAdapterManifest:
        if not _DOMAIN_ID.fullmatch(self.domain_id):
            raise ValueError("domain_id must use lowercase letters, digits, _ or -.")
        if self.adult_content_enabled:
            if self.access_mode != "restricted":
                raise ValueError("Adult-domain adapters must use restricted access.")
            if self.minimum_age != 18:
                raise ValueError("Adult-domain adapters must set minimum_age=18.")
            if not self.require_consent_attestation:
                raise ValueError(
                    "Adult-domain adapters must require consent attestation."
                )
        return self


class DomainDatasetRecord(BaseModel):
    """One governed SFT record; provenance is mandatory rather than optional notes."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=16_000)
    response: str = Field(min_length=1, max_length=16_000)
    source: str = Field(min_length=1, max_length=1000)
    license: str = Field(min_length=1, max_length=300)
    consent_basis: str = Field(min_length=1, max_length=500)
    contains_personal_data: bool = False
    personal_data_reviewed: bool = False
    all_human_subjects_adults: bool | None = None
    consensual_context: bool | None = None

    @model_validator(mode="after")
    def validate_privacy(self) -> DomainDatasetRecord:
        if self.contains_personal_data and not self.personal_data_reviewed:
            raise ValueError("Personal data requires an explicit completed review.")
        return self


@dataclass(frozen=True)
class DomainDatasetReport:
    domain_id: str
    records: int
    duplicate_ids: tuple[str, ...]
    prohibited_records: tuple[str, ...]
    privacy_records: int
    adult_policy_records: int
    valid: bool


@dataclass(frozen=True)
class DomainSelection:
    domain_id: str
    adapter_path: Path
    base_model: str
    adult_content_enabled: bool


class AdapterActivationBackend(Protocol):
    def activate_adapter(self, domain_id: str, adapter_path: Path) -> None: ...

    def deactivate_adapter(self) -> None: ...


class DomainAdapterRegistry:
    """Load manifests from one root and fail closed on path or access violations."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, DomainAdapterManifest] = {}

    def load(self) -> None:
        loaded: dict[str, DomainAdapterManifest] = {}
        for path in sorted(self.root.glob("*.json")):
            manifest = DomainAdapterManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if manifest.domain_id in loaded:
                raise ValueError(f"Duplicate domain_id: {manifest.domain_id}")
            self._resolve_confined(manifest.adapter_path)
            self._resolve_confined(manifest.dataset_path)
            loaded[manifest.domain_id] = manifest
        self._manifests = loaded

    def _resolve_confined(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                "Domain paths must remain inside the registry root."
            ) from error
        return path

    def catalog(self, *, enabled_only: bool = False) -> list[dict[str, object]]:
        """Return metadata safe for an authenticated control-plane listing."""
        return [
            {
                "domain_id": item.domain_id,
                "display_name": item.display_name,
                "description": item.description,
                "enabled": item.enabled,
                "access_mode": item.access_mode,
                "data_classification": item.data_classification,
                "adult_content_enabled": item.adult_content_enabled,
                "allowed_capabilities": item.allowed_capabilities,
            }
            for item in sorted(
                self._manifests.values(), key=lambda value: value.domain_id
            )
            if not enabled_only or item.enabled
        ]

    def authorize(self, domain_id: str, access_token: str) -> DomainSelection:
        manifest = self._manifests.get(domain_id)
        if manifest is None or not manifest.enabled:
            raise PermissionError("Domain adapter is unavailable.")
        supplied = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(supplied, manifest.access_token_sha256):
            raise PermissionError("Domain adapter authorization failed.")
        adapter_path = self._resolve_confined(manifest.adapter_path)
        if not adapter_path.is_dir():
            raise FileNotFoundError("Authorized adapter directory is missing.")
        return DomainSelection(
            domain_id=manifest.domain_id,
            adapter_path=adapter_path,
            base_model=manifest.base_model,
            adult_content_enabled=manifest.adult_content_enabled,
        )

    def manifest(self, domain_id: str) -> DomainAdapterManifest:
        try:
            return self._manifests[domain_id]
        except KeyError as error:
            raise KeyError(f"Unknown domain: {domain_id}") from error

    def dataset_path(self, domain_id: str) -> Path:
        return self._resolve_confined(self.manifest(domain_id).dataset_path)


class DomainAdapterController:
    """Serialize Adapter selection so one request cannot leak into another."""

    def __init__(
        self, registry: DomainAdapterRegistry, backend: AdapterActivationBackend
    ) -> None:
        self.registry = registry
        self.backend = backend
        self._lock = RLock()
        self._active_domain: str | None = None

    @property
    def active_domain(self) -> str | None:
        with self._lock:
            return self._active_domain

    def activate(self, domain_id: str, access_token: str) -> DomainSelection:
        selection = self.registry.authorize(domain_id, access_token)
        with self._lock:
            self.backend.activate_adapter(selection.domain_id, selection.adapter_path)
            self._active_domain = selection.domain_id
        return selection

    def deactivate(self) -> None:
        with self._lock:
            self.backend.deactivate_adapter()
            self._active_domain = None

    def run_default(self, operation: Callable[[], _ResultT]) -> _ResultT:
        """Run base/default inference without overlapping a domain request."""
        with self._lock:
            if self._active_domain is not None:
                raise RuntimeError(
                    "Persistent domain activation cannot be mixed with request-scoped "
                    "inference."
                )
            return operation()

    def run_authorized(
        self,
        domain_id: str,
        access_token: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        """Activate, infer, and restore the default Adapter as one transaction."""
        selection = self.registry.authorize(domain_id, access_token)
        with self._lock:
            if self._active_domain is not None:
                raise RuntimeError("A persistent domain Adapter is already active.")
            self.backend.activate_adapter(selection.domain_id, selection.adapter_path)
            self._active_domain = selection.domain_id
            try:
                return operation()
            finally:
                self.backend.deactivate_adapter()
                self._active_domain = None


def validate_domain_dataset(
    manifest: DomainAdapterManifest, path: Path
) -> DomainDatasetReport:
    if not path.is_file():
        raise FileNotFoundError(f"Domain dataset is missing: {path.name}")
    seen: set[str] = set()
    duplicates: list[str] = []
    prohibited: list[str] = []
    privacy_records = 0
    adult_policy_records = 0
    record_count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = DomainDatasetRecord.model_validate_json(line)
        except Exception as error:
            raise ValueError(f"Invalid domain record at line {line_number}.") from error
        record_count += 1
        if record.record_id in seen:
            duplicates.append(record.record_id)
        seen.add(record.record_id)
        if record.contains_personal_data:
            privacy_records += 1
        combined = f"{record.prompt}\n{record.response}".casefold()
        if any(marker in combined for marker in _PROHIBITED_SEXUAL_MARKERS):
            prohibited.append(record.record_id)
        if manifest.adult_content_enabled:
            adult_policy_records += 1
            if record.all_human_subjects_adults is not True:
                prohibited.append(record.record_id)
            if record.consensual_context is not True:
                prohibited.append(record.record_id)
            if not record.consent_basis.strip():
                prohibited.append(record.record_id)
    prohibited = sorted(set(prohibited))
    return DomainDatasetReport(
        domain_id=manifest.domain_id,
        records=record_count,
        duplicate_ids=tuple(sorted(set(duplicates))),
        prohibited_records=tuple(prohibited),
        privacy_records=privacy_records,
        adult_policy_records=adult_policy_records,
        valid=record_count > 0 and not duplicates and not prohibited,
    )
