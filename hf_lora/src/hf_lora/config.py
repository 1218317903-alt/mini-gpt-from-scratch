"""阶段六实验配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str
    max_length: int


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    minimum_lr_ratio: float
    max_grad_norm: float
    eval_interval: int
    scaler_initial_scale: float


@dataclass(frozen=True)
class PathsConfig:
    train_data: Path
    validation_data: Path
    artifact_directory: Path


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    prompts: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    paths: PathsConfig
    generation: GenerationConfig


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} 必须是映射。")
    return payload


def load_experiment_config(path: Path) -> ExperimentConfig:
    """读取并校验 YAML，所有相对路径相对于项目根目录。"""
    payload = _require_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "配置文件",
    )
    model = _require_mapping(payload.get("model"), "model")
    lora = _require_mapping(payload.get("lora"), "lora")
    training = _require_mapping(payload.get("training"), "training")
    paths = _require_mapping(payload.get("paths"), "paths")
    generation = _require_mapping(payload.get("generation"), "generation")

    config = ExperimentConfig(
        model=ModelConfig(
            name=str(model["name"]),
            dtype=str(model.get("dtype", "float16")),
            max_length=int(model["max_length"]),
        ),
        lora=LoRAConfig(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=tuple(str(x) for x in lora["target_modules"]),
        ),
        training=TrainingConfig(
            seed=int(training["seed"]),
            micro_batch_size=int(training["micro_batch_size"]),
            gradient_accumulation_steps=int(
                training["gradient_accumulation_steps"]
            ),
            max_optimizer_steps=int(training["max_optimizer_steps"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_steps=int(training["warmup_steps"]),
            minimum_lr_ratio=float(training["minimum_lr_ratio"]),
            max_grad_norm=float(training["max_grad_norm"]),
            eval_interval=int(training["eval_interval"]),
            scaler_initial_scale=float(training["scaler_initial_scale"]),
        ),
        paths=PathsConfig(
            train_data=Path(paths["train_data"]),
            validation_data=Path(paths["validation_data"]),
            artifact_directory=Path(paths["artifact_directory"]),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(generation["max_new_tokens"]),
            prompts=tuple(generation["prompts"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("model.dtype 必须是 float16、bfloat16 或 float32。")
    if config.model.max_length <= 1:
        raise ValueError("max_length 必须大于 1。")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank 和 alpha 必须大于 0。")
    if not 0.0 <= config.lora.dropout < 1.0:
        raise ValueError("LoRA dropout 必须位于 [0, 1)。")
    if not config.lora.target_modules:
        raise ValueError("target_modules 不能为空。")
    train = config.training
    positive = (
        train.micro_batch_size,
        train.gradient_accumulation_steps,
        train.max_optimizer_steps,
        train.learning_rate,
        train.max_grad_norm,
        train.eval_interval,
        train.scaler_initial_scale,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("训练正值配置必须全部大于 0。")
    if not 0 <= train.warmup_steps < train.max_optimizer_steps:
        raise ValueError("warmup_steps 必须小于 max_optimizer_steps。")
    if not 0.0 <= train.minimum_lr_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio 必须位于 [0, 1]。")
