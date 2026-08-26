"""项目配置读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    """数据流水线配置。"""

    raw_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    tokenizer_path: Path

    train_ratio: float
    val_ratio: float

    block_size: int
    batch_size: int
    tokenizer_type: str = "character"
    tokenizer_vocab_size: int | None = None
    packing: bool = False
    sample_separator: str = "\n\n"


@dataclass(frozen=True)
class ModelConfig:
    """MiniGPT 模型配置。"""

    embedding_dim: int
    num_heads: int
    num_layers: int
    expansion_factor: int
    tie_weights: bool
    position_encoding: str = "learned"
    norm: str = "layernorm"
    activation: str = "gelu"
    attention_backend: str = "manual"
    qkv_projection: str = "separate"
    qk_norm: bool = False
    num_kv_heads: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """阶段三的训练配置。"""

    learning_rate: float
    weight_decay: float
    max_steps: int
    grad_clip_norm: float
    log_interval: int
    eval_interval: int
    eval_steps: int
    save_interval: int
    seed: int
    overfit_steps: int


def load_data_config(path: Path) -> DataConfig:
    """从 YAML 文件读取数据配置。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw_config, dict):
        raise ValueError("配置文件顶层必须是对象。")

    data_config = raw_config.get("data")

    if not isinstance(data_config, dict):
        raise ValueError("配置文件必须包含 data 区块。")

    project_root = path.resolve().parents[1]

    def resolve_path(name: str) -> Path:
        value = data_config.get(name)

        if not isinstance(value, str):
            raise ValueError(f"配置项 {name} 必须是字符串。")

        return project_root / value

    train_ratio = data_config.get("train_ratio")
    val_ratio = data_config.get("val_ratio")
    block_size = data_config.get("block_size")
    batch_size = data_config.get("batch_size")
    tokenizer_type = data_config.get("tokenizer_type", "character")
    tokenizer_vocab_size = data_config.get("tokenizer_vocab_size")
    packing = data_config.get("packing", False)
    sample_separator = data_config.get("sample_separator", "\n\n")

    if not isinstance(train_ratio, (int, float)):
        raise ValueError("train_ratio 必须是数字。")

    if not isinstance(val_ratio, (int, float)):
        raise ValueError("val_ratio 必须是数字。")

    if not isinstance(block_size, int):
        raise ValueError("block_size 必须是整数。")

    if not isinstance(batch_size, int):
        raise ValueError("batch_size 必须是整数。")

    if tokenizer_type not in {"character", "bpe"}:
        raise ValueError("tokenizer_type 必须是 character 或 bpe。")
    if tokenizer_vocab_size is not None and (
        type(tokenizer_vocab_size) is not int or tokenizer_vocab_size <= 0
    ):
        raise ValueError("tokenizer_vocab_size 必须是正整数或 null。")
    if tokenizer_type == "bpe" and tokenizer_vocab_size is None:
        raise ValueError("BPE 配置必须提供 tokenizer_vocab_size。")
    if type(packing) is not bool:
        raise ValueError("packing 必须是布尔值。")
    if packing and tokenizer_type != "bpe":
        raise ValueError("packing 当前只支持带 EOS/PAD 的 BPE Tokenizer。")
    if not isinstance(sample_separator, str) or not sample_separator:
        raise ValueError("sample_separator 必须是非空字符串。")

    if train_ratio <= 0:
        raise ValueError("train_ratio 必须大于 0。")

    if val_ratio <= 0:
        raise ValueError("val_ratio 必须大于 0。")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 之和必须小于 1。")

    if block_size <= 0:
        raise ValueError("block_size 必须大于 0。")

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    return DataConfig(
        raw_path=resolve_path("raw_path"),
        train_path=resolve_path("train_path"),
        val_path=resolve_path("val_path"),
        test_path=resolve_path("test_path"),
        tokenizer_path=resolve_path("tokenizer_path"),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        block_size=block_size,
        batch_size=batch_size,
        tokenizer_type=tokenizer_type,
        tokenizer_vocab_size=tokenizer_vocab_size,
        packing=packing,
        sample_separator=sample_separator,
    )


def load_model_config(path: Path) -> ModelConfig:
    """从 YAML 文件读取模型配置。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是对象。")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("配置文件必须包含 model 区块。")

    integer_names = (
        "embedding_dim",
        "num_heads",
        "num_layers",
        "expansion_factor",
    )
    values: dict[str, int] = {}
    for name in integer_names:
        value = model.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数。")
        values[name] = value

    tie_weights = model.get("tie_weights")
    if type(tie_weights) is not bool:
        raise ValueError("tie_weights 必须是布尔值。")

    if values["embedding_dim"] % values["num_heads"] != 0:
        raise ValueError("embedding_dim 必须能被 num_heads 整除。")

    qk_norm = model.get("qk_norm", False)
    if type(qk_norm) is not bool:
        raise ValueError("qk_norm 必须是布尔值。")
    num_kv_heads = model.get("num_kv_heads", values["num_heads"])
    if type(num_kv_heads) is not int or num_kv_heads <= 0:
        raise ValueError("num_kv_heads 必须是正整数。")
    if values["num_heads"] % num_kv_heads != 0:
        raise ValueError("num_heads 必须能被 num_kv_heads 整除。")

    string_choices = {
        "position_encoding": {"learned", "rope"},
        "norm": {"layernorm", "rmsnorm"},
        "activation": {"gelu", "swiglu"},
        "attention_backend": {"manual", "sdpa"},
        "qkv_projection": {"separate", "fused"},
    }
    selected: dict[str, str] = {}
    for name, allowed in string_choices.items():
        default = next(iter(allowed))
        defaults = {
            "position_encoding": "learned",
            "norm": "layernorm",
            "activation": "gelu",
            "attention_backend": "manual",
            "qkv_projection": "separate",
        }
        value = model.get(name, defaults.get(name, default))
        if not isinstance(value, str) or value not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(f"{name} 必须是以下值之一：{options}。")
        selected[name] = value
    if (
        selected["position_encoding"] == "rope"
        and values["embedding_dim"] // values["num_heads"] % 2 != 0
    ):
        raise ValueError("RoPE 要求 head_dim 为偶数。")

    return ModelConfig(
        embedding_dim=values["embedding_dim"],
        num_heads=values["num_heads"],
        num_layers=values["num_layers"],
        expansion_factor=values["expansion_factor"],
        tie_weights=tie_weights,
        position_encoding=selected["position_encoding"],
        norm=selected["norm"],
        activation=selected["activation"],
        attention_backend=selected["attention_backend"],
        qkv_projection=selected["qkv_projection"],
        qk_norm=qk_norm,
        num_kv_heads=num_kv_heads,
    )


def load_training_config(path: Path) -> TrainingConfig:
    """从 YAML 文件读取训练配置。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是对象。")

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("配置文件必须包含 training 区块。")

    learning_rate = training.get("learning_rate")
    weight_decay = training.get("weight_decay")
    max_steps = training.get("max_steps")
    grad_clip_norm = training.get("grad_clip_norm")
    log_interval = training.get("log_interval")
    eval_interval = training.get("eval_interval")
    eval_steps = training.get("eval_steps")
    save_interval = training.get("save_interval")
    seed = training.get("seed")
    overfit_steps = training.get("overfit_steps")

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate 必须是大于 0 的数字。")
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("weight_decay 必须是大于等于 0 的数字。")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps 必须是正整数。")
    if not isinstance(grad_clip_norm, (int, float)) or grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm 必须是大于 0 的数字。")
    if type(log_interval) is not int or log_interval <= 0:
        raise ValueError("log_interval 必须是正整数。")

    if type(eval_interval) is not int or eval_interval <= 0:
        raise ValueError("eval_interval 必须是正整数。")

    if type(eval_steps) is not int or eval_steps <= 0:
        raise ValueError("eval_steps 必须是正整数。")

    if type(save_interval) is not int or save_interval <= 0:
        raise ValueError("save_interval 必须是正整数。")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必须是非负整数。")

    if type(overfit_steps) is not int or overfit_steps <= 0:
        raise ValueError("overfit_steps 必须是正整数。")

    return TrainingConfig(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_steps=max_steps,
        grad_clip_norm=float(grad_clip_norm),
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        save_interval=save_interval,
        seed=seed,
        overfit_steps=overfit_steps,
    )
