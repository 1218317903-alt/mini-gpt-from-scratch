"""Runtime configuration for the MiniGPT inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    """Paths and runtime switches required to load an inference model."""

    config_path: Path
    checkpoint_path: Path
    device: str = "auto"
    compile_model: bool = False
    compile_mode: str = "default"

    @classmethod
    def from_environment(cls) -> ServiceConfig | None:
        """Build a service config from container environment variables."""
        config_value = os.getenv("MINIGPT_CONFIG_PATH")
        checkpoint_value = os.getenv("MINIGPT_CHECKPOINT_PATH")
        if config_value is None and checkpoint_value is None:
            return None
        if not config_value or not checkpoint_value:
            raise ValueError(
                "MINIGPT_CONFIG_PATH and MINIGPT_CHECKPOINT_PATH must be set together."
            )

        compile_value = os.getenv("MINIGPT_COMPILE", "false").strip().lower()
        if compile_value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "MINIGPT_COMPILE must be one of: 0, 1, false, true, no, yes."
            )

        return cls(
            config_path=Path(config_value),
            checkpoint_path=Path(checkpoint_value),
            device=os.getenv("MINIGPT_DEVICE", "auto"),
            compile_model=compile_value in {"1", "true", "yes"},
            compile_mode=os.getenv("MINIGPT_COMPILE_MODE", "default"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, Path):
            raise TypeError("config_path must be a pathlib.Path.")
        if not isinstance(self.checkpoint_path, Path):
            raise TypeError("checkpoint_path must be a pathlib.Path.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps.")
        if not isinstance(self.compile_model, bool):
            raise TypeError("compile_model must be a bool.")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("compile_mode is not supported.")
