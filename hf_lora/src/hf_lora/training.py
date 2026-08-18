"""不依赖 Trainer 的 LoRA 训练循环。"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from hf_lora.config import TrainingConfig
from hf_lora.data import IGNORE_INDEX


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def infinite_batches(data_loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yielded = False
        for batch in data_loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("训练 DataLoader 为空。")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate_token_weighted_loss(
    model: torch.nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    weighted_sum = 0.0
    token_count = 0
    for cpu_batch in data_loader:
        batch = _move_batch(cpu_batch, device)
        supervised = int((batch["labels"] != IGNORE_INDEX).sum().item())
        if supervised == 0:
            raise ValueError("验证 Batch 没有监督 Token。")
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model(**batch).loss.float()
        if not torch.isfinite(loss):
            raise FloatingPointError("验证 Loss 出现 NaN 或 Inf。")
        weighted_sum += loss.item() * supervised
        token_count += supervised
    if was_training:
        model.train()
    if token_count == 0:
        raise ValueError("验证集为空。")
    return weighted_sum / token_count


def train_lora(
    model: torch.nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    validation_loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    """按有效监督 Token 加权累积梯度。"""
    model.config.use_cache = False
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("模型没有可训练参数。")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_optimizer_steps,
        minimum_ratio=config.minimum_lr_ratio,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16,
        init_scale=config.scaler_initial_scale,
        growth_interval=1000,
    )
    batches = infinite_batches(train_loader)
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []

    for step in range(1, config.max_optimizer_steps + 1):
        started = time.perf_counter()
        micro_batches: list[tuple[dict[str, torch.Tensor], int]] = []
        for _ in range(config.gradient_accumulation_steps):
            batch = _move_batch(next(batches), device)
            supervised = int((batch["labels"] != IGNORE_INDEX).sum().item())
            if supervised == 0:
                raise ValueError("训练 Batch 没有监督 Token。")
            micro_batches.append((batch, supervised))
        total_tokens = sum(count for _, count in micro_batches)
        weighted_loss = 0.0
        used_lr = float(optimizer.param_groups[0]["lr"])
        scale_before = float(scaler.get_scale())

        for batch, supervised in micro_batches:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                raw_loss = model(**batch).loss
                backward_loss = raw_loss * (supervised / total_tokens)
            if not torch.isfinite(raw_loss):
                raise FloatingPointError("训练 Loss 出现 NaN 或 Inf。")
            scaler.scale(backward_loss).backward()
            weighted_loss += raw_loss.detach().float().item() * supervised

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = float(scaler.get_scale()) >= scale_before
        optimizer.zero_grad(set_to_none=True)
        if optimizer_updated:
            scheduler.step()

        record: dict[str, Any] = {
            "optimizer_step": step,
            "train_loss": weighted_loss / total_tokens,
            "learning_rate": used_lr,
            "gradient_norm": float(gradient_norm),
            "supervised_tokens": total_tokens,
            "optimizer_updated": optimizer_updated,
            "grad_scale": scale_before,
            "step_seconds": time.perf_counter() - started,
            "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        if step % config.eval_interval == 0 or step == config.max_optimizer_steps:
            record["validation_loss"] = evaluate_token_weighted_loss(
                model,
                validation_loader,
                device=device,
                amp_dtype=amp_dtype,
            )
        history.append(record)
        print(record)
        model.train()
    return history
