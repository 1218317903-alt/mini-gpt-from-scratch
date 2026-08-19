"""Tests for the MiniGPT service lifecycle and device selection."""

from pathlib import Path

import pytest

from minigpt_service.config import ServiceConfig
from minigpt_service.model_service import ModelService, select_device


def test_select_device_cpu_is_explicit() -> None:
    assert select_device("cpu").type == "cpu"


def test_select_device_rejects_unavailable_explicit_accelerator() -> None:
    if not __import__("torch").cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            select_device("cuda")


def test_service_config_rejects_invalid_runtime_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        ServiceConfig(tmp_path / "config.yaml", tmp_path / "best.pt", device="tpu")

    with pytest.raises(ValueError, match="compile_mode"):
        ServiceConfig(
            tmp_path / "config.yaml",
            tmp_path / "best.pt",
            compile_mode="fast",
        )


def test_service_starts_not_ready_and_close_is_safe(tmp_path: Path) -> None:
    service = ModelService(
        ServiceConfig(tmp_path / "config.yaml", tmp_path / "best.pt", device="cpu")
    )

    assert service.is_ready is False
    service.close()
    assert service.is_ready is False
