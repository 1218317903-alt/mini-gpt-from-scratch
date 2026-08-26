"""真实 Causal LM 与 PEFT LoRA 装配。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    frozen: int
    trainable_percentage: float


def summarize_parameters(model: torch.nn.Module) -> ParameterSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ParameterSummary(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        trainable_percentage=100.0 * trainable / total,
    )


def load_tokenizer(source: str | Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer 没有聊天模板。")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer 同时缺少 PAD 和 EOS Token。")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    source: str | Path,
    *,
    dtype_name: str,
    device: torch.device,
    quantization: str = "none",
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": DTYPES[dtype_name],
        "low_cpu_mem_usage": True,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPES[dtype_name],
        )
        kwargs["device_map"] = {"": str(device)}
        return AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    if quantization != "none":
        raise ValueError(f"不支持的量化方式：{quantization}")
    model: Any = AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    return model.to(device)


def estimate_lora_parameters(
    model: torch.nn.Module, target_modules: tuple[str, ...], rank: int
) -> tuple[int, int]:
    targets = set(target_modules)
    matched = 0
    parameters = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            continue
        matched += 1
        parameters += rank * (in_features + out_features)
    if matched == 0:
        raise ValueError(f"没有匹配任何 LoRA 目标模块：{target_modules}")
    return matched, parameters


def inject_lora(
    model: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
    quantization: str = "none",
) -> Any:
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        init_lora_weights=True,
    )
    peft_model: Any = get_peft_model(model, config)
    peft_model.config.use_cache = False
    return peft_model
