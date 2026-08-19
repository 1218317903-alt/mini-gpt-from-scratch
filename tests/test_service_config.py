"""Tests for container-facing service configuration."""

from pathlib import Path

import pytest

from minigpt_service.config import ServiceConfig


def test_environment_config_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("MINIGPT_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MINIGPT_CHECKPOINT_PATH", raising=False)

    assert ServiceConfig.from_environment() is None


def test_environment_config_requires_both_model_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIGPT_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("MINIGPT_CHECKPOINT_PATH", raising=False)

    with pytest.raises(ValueError, match="must be set together"):
        ServiceConfig.from_environment()


def test_environment_config_parses_runtime_switches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "best.pt"
    monkeypatch.setenv("MINIGPT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MINIGPT_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("MINIGPT_DEVICE", "cpu")
    monkeypatch.setenv("MINIGPT_COMPILE", "yes")
    monkeypatch.setenv("MINIGPT_COMPILE_MODE", "reduce-overhead")

    config = ServiceConfig.from_environment()

    assert config == ServiceConfig(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cpu",
        compile_model=True,
        compile_mode="reduce-overhead",
    )


def test_environment_config_rejects_invalid_compile_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIGPT_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("MINIGPT_CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    monkeypatch.setenv("MINIGPT_COMPILE", "sometimes")

    with pytest.raises(ValueError, match="MINIGPT_COMPILE"):
        ServiceConfig.from_environment()
