"""真实 Causal LM 与 PEFT LoRA 装配。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    source: str | Path, *, dtype_name: str, device: torch.device
) -> Any:
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=DTYPES[dtype_name],
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def estimate_lora_parameters(
    model: torch.nn.Module, target_modules: tuple[str, ...], rank: int
) -> tuple[int, int]:
    targets = set(target_modules)
    matched = 0
    parameters = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        matched += 1
        parameters += rank * (module.in_features + module.out_features)
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
) -> Any:
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
