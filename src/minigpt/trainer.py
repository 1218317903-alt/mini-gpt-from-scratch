from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def compute_language_model_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    inputs:  [B, T]
    targets: [B, T]
    logits:  [B, T, V]
    loss:    标量
    """

    model_output = model(inputs)
    logits = model_output[0] if isinstance(model_output, tuple) else model_output

    if logits.ndim != 3:
        raise ValueError("logits 必须是 [B, T, V]")

    batch_size, sequence_length, vocab_size = logits.shape

    if targets.shape != (batch_size, sequence_length):
        raise ValueError(
            "targets 的形状必须与 logits 的前两维一致"
        )

    flattened_logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    flattened_targets = targets.reshape(
        batch_size * sequence_length,
    ).long()

    loss = F.cross_entropy(
        flattened_logits,
        flattened_targets,
    )

    return logits, loss


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    grad_clip_norm: float,
) -> tuple[torch.Tensor, float]:
    """兼容原接口：使用一个 Micro Batch 执行一次参数更新。"""
    loss, gradient_norm, _ = train_accumulation_step(
        model=model,
        optimizer=optimizer,
        micro_batches=[(inputs, targets)],
        grad_clip_norm=grad_clip_norm,
    )
    return loss, gradient_norm


def _autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> Any:
    if amp_dtype is None:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("本阶段的混合精度训练只支持 CUDA。")
    if amp_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("amp_dtype 必须是 float16、bfloat16 或 None。")
    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    )


def create_cuda_grad_scaler() -> Any:
    """创建 FP16 GradScaler，并兼容新旧 PyTorch 命名空间。"""
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "GradScaler"):
        try:
            return amp_namespace.GradScaler("cuda")
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler()


def train_accumulation_step(
    model: nn.Module,
    optimizer: Optimizer,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    grad_clip_norm: float,
    *,
    amp_dtype: torch.dtype | None = None,
    scaler: Any | None = None,
) -> tuple[torch.Tensor, float, bool]:
    """累积多个 Micro Batch，并且只裁剪和更新一次参数。"""
    if grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm 必须大于 0。")
    if not micro_batches:
        raise ValueError("micro_batches 不能为空。")
    if amp_dtype == torch.float16 and scaler is None:
        raise ValueError("CUDA FP16 训练必须提供 GradScaler。")
    if amp_dtype != torch.float16 and scaler is not None:
        raise ValueError("GradScaler 只应在 FP16 路径中使用。")

    token_counts = [targets.numel() for _, targets in micro_batches]
    if any(count <= 0 for count in token_counts):
        raise ValueError("每个 Micro Batch 必须包含至少一个目标 Token。")
    total_token_count = sum(token_counts)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    reported_loss: torch.Tensor | None = None
    for (inputs, targets), token_count in zip(
        micro_batches,
        token_counts,
        strict=True,
    ):
        with _autocast_context(inputs.device, amp_dtype):
            _, micro_loss = compute_language_model_loss(
                model=model,
                inputs=inputs,
                targets=targets,
            )

        if micro_loss.ndim != 0:
            raise ValueError("Loss 必须是标量。")
        if not torch.isfinite(micro_loss).item():
            raise FloatingPointError("Loss 出现 NaN 或 Inf。")

        weight = token_count / total_token_count
        backward_loss = micro_loss * weight
        if scaler is None:
            backward_loss.backward()
        else:
            scaler.scale(backward_loss).backward()

        detached_contribution = micro_loss.detach().float() * weight
        reported_loss = (
            detached_contribution
            if reported_loss is None
            else reported_loss + detached_contribution
        )

    if scaler is not None:
        scaler.unscale_(optimizer)

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=grad_clip_norm,
    )

    if not torch.isfinite(gradient_norm).item():
        raise FloatingPointError("梯度范数出现 NaN 或 Inf。")

    optimizer_updated = True
    if scaler is None:
        optimizer.step()
    else:
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = float(scaler.get_scale()) >= scale_before

    assert reported_loss is not None
    return (
        reported_loss,
        float(gradient_norm.item()),
        optimizer_updated,
    )


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """计算验证集的平均 Loss。"""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches 必须大于 0。")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    batch_count = 0

    try:
        with torch.no_grad():
            for inputs, targets in data_loader:
                if (
                    max_batches is not None
                    and batch_count >= max_batches
                ):
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                with _autocast_context(device, amp_dtype):
                    _, loss = compute_language_model_loss(
                        model=model,
                        inputs=inputs,
                        targets=targets,
                    )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "验证 Loss 出现 NaN 或 Inf。"
                    )

                total_loss += loss.item()
                batch_count += 1
    finally:
        if was_training:
            model.train()

    if batch_count == 0:
        raise ValueError(
            "验证 DataLoader 没有提供任何 Batch。"
        )

    return total_loss / batch_count
