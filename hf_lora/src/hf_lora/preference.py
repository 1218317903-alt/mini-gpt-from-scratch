"""Preference data encoding and a transparent Direct Preference Optimization loop."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from hf_lora.data import IGNORE_INDEX, validate_messages


@dataclass(frozen=True)
class DPOTrainingConfig:
    """Small, explicit DPO optimizer configuration."""

    beta: float
    learning_rate: float
    weight_decay: float
    max_optimizer_steps: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    warmup_steps: int = 0
    minimum_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.beta,
            self.learning_rate,
            self.max_optimizer_steps,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "DPO positive configuration values must be greater than zero."
            )
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0 <= self.warmup_steps < self.max_optimizer_steps:
            raise ValueError("warmup_steps must be smaller than max_optimizer_steps.")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be between zero and one.")


def validate_preference_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one prompt with a preferred and rejected assistant response."""
    prompt = record.get("prompt")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError("preference prompt must be a message sequence.")
    prompt_messages = [dict(message) for message in prompt if isinstance(message, dict)]
    if len(prompt_messages) != len(prompt) or not prompt_messages:
        raise ValueError("preference prompt contains an invalid message.")
    if prompt_messages[-1].get("role") != "user":
        raise ValueError("preference prompt must end with a user message.")
    synthetic = [*prompt_messages, {"role": "assistant", "content": "placeholder"}]
    normalized = validate_messages(synthetic)[:-1]
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("chosen must be a non-empty string.")
    if not isinstance(rejected, str) or not rejected.strip():
        raise ValueError("rejected must be a non-empty string.")
    if chosen.strip() == rejected.strip():
        raise ValueError("chosen and rejected responses must differ.")
    return {
        "prompt": normalized,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def _chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], generation: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list) or not all(
        type(item) is int for item in input_ids
    ):
        raise TypeError("chat template input_ids must be a list of integers.")
    return list(input_ids)


def _encode_candidate(
    tokenizer: Any,
    prompt: list[dict[str, str]],
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = _chat_ids(tokenizer, prompt, generation=True)
    full_ids = _chat_ids(
        tokenizer,
        [*prompt, {"role": "assistant", "content": response}],
        generation=False,
    )
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant response tokens do not extend the prompt prefix.")
    if len(full_ids) > max_length:
        raise ValueError(
            f"preference sample exceeds max_length: {len(full_ids)} > {max_length}."
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("preference response contains no tokens.")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def encode_preference_example(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode chosen/rejected continuations with response-only labels."""
    if max_length <= 1:
        raise ValueError("max_length must be greater than one.")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer has no chat template.")
    normalized = validate_preference_record(record)
    prompt = normalized["prompt"]
    chosen = _encode_candidate(tokenizer, prompt, normalized["chosen"], max_length)
    rejected = _encode_candidate(tokenizer, prompt, normalized["rejected"], max_length)
    return {
        **{f"chosen_{key}": value for key, value in chosen.items()},
        **{f"rejected_{key}": value for key, value in rejected.items()},
    }


class PreferenceDataset(Dataset[dict[str, list[int]]]):
    """Pre-tokenized chosen/rejected pairs."""

    def __init__(
        self, records: list[dict[str, Any]], tokenizer: Any, max_length: int
    ) -> None:
        self.examples = [
            encode_preference_example(record, tokenizer, max_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class PreferenceDataCollator:
    """Pad chosen and rejected sequences independently into one paired batch."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id cannot be negative.")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty.")
        batch: dict[str, torch.Tensor] = {}
        for prefix in ("chosen", "rejected"):
            width = max(len(feature[f"{prefix}_input_ids"]) for feature in features)
            if self.pad_to_multiple_of:
                multiple = self.pad_to_multiple_of
                width = (width + multiple - 1) // multiple * multiple
            for field, padding_value in (
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", IGNORE_INDEX),
            ):
                rows = []
                for feature in features:
                    values = feature[f"{prefix}_{field}"]
                    rows.append(values + [padding_value] * (width - len(values)))
                batch[f"{prefix}_{field}"] = torch.tensor(rows, dtype=torch.long)
        return batch


def sequence_log_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Sum autoregressive log probabilities only over response labels."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have shapes [B,T,V] and [B,T].")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every preference sequence must supervise response tokens.")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = (
        shifted_logits.log_softmax(dim=-1)
        .gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return (token_logps * mask).sum(dim=-1)


@dataclass(frozen=True)
class DPOLossResult:
    """DPO loss and interpretable batch diagnostics."""

    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    preference_accuracy: torch.Tensor
    margin: torch.Tensor


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> DPOLossResult:
    """Compute the reference-relative logistic DPO objective."""
    if beta <= 0:
        raise ValueError("beta must be greater than zero.")
    shapes = {
        tuple(tensor.shape)
        for tensor in (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
    }
    if len(shapes) != 1:
        raise ValueError("all DPO log-probability tensors must have the same shape.")
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = chosen_reward - rejected_reward
    losses = -F.logsigmoid(margin)
    return DPOLossResult(
        loss=losses.mean(),
        chosen_reward=chosen_reward.detach().mean(),
        rejected_reward=rejected_reward.detach().mean(),
        preference_accuracy=margin.detach().gt(0).float().mean(),
        margin=margin.detach().mean(),
    )


def _candidate_batch(
    batch: dict[str, torch.Tensor],
    prefix: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        field: batch[f"{prefix}_{field}"].to(device, non_blocking=True)
        for field in ("input_ids", "attention_mask", "labels")
    }


def _model_logps(model: Any, candidate: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=candidate["input_ids"],
        attention_mask=candidate["attention_mask"],
    )
    return sequence_log_probabilities(outputs.logits, candidate["labels"])


def _infinite_batches(
    data_loader: Iterable[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yielded = False
        for batch in data_loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("DPO training DataLoader is empty.")


def _scheduler_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum_ratio + (1 - minimum_ratio) * cosine


@torch.no_grad()
def evaluate_dpo(
    policy_model: Any,
    reference_model: Any,
    data_loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    beta: float,
) -> dict[str, float]:
    """Evaluate the DPO objective without changing either model's train/eval state."""
    policy_was_training = policy_model.training
    reference_was_training = reference_model.training
    policy_model.eval()
    reference_model.eval()
    total_examples = 0
    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    try:
        for batch in data_loader:
            chosen = _candidate_batch(batch, "chosen", device)
            rejected = _candidate_batch(batch, "rejected", device)
            batch_size = int(chosen["input_ids"].shape[0])
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                result = dpo_loss(
                    _model_logps(policy_model, chosen),
                    _model_logps(policy_model, rejected),
                    _model_logps(reference_model, chosen),
                    _model_logps(reference_model, rejected),
                    beta=beta,
                )
            total_examples += batch_size
            total_loss += float(result.loss) * batch_size
            total_accuracy += float(result.preference_accuracy) * batch_size
            total_margin += float(result.margin) * batch_size
    finally:
        policy_model.train(policy_was_training)
        reference_model.train(reference_was_training)
    if total_examples == 0:
        raise ValueError("DPO evaluation DataLoader is empty.")
    return {
        "loss": total_loss / total_examples,
        "preference_accuracy": total_accuracy / total_examples,
        "reward_margin": total_margin / total_examples,
        "examples": float(total_examples),
    }


def train_dpo(
    policy_model: Any,
    reference_model: Any,
    data_loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: DPOTrainingConfig,
) -> list[dict[str, float | int]]:
    """Train a LoRA policy against a frozen reference model."""
    policy_model.train()
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    parameters = [
        parameter for parameter in policy_model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("policy model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_multiplier(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_optimizer_steps,
            minimum_ratio=config.minimum_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )
    batches = _infinite_batches(data_loader)
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.max_optimizer_steps + 1):
        started = time.perf_counter()
        scale_before = float(scaler.get_scale())
        accumulated_loss = 0.0
        accumulated_accuracy = 0.0
        accumulated_margin = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch = next(batches)
            chosen = _candidate_batch(batch, "chosen", device)
            rejected = _candidate_batch(batch, "rejected", device)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ),
            ):
                reference_chosen = _model_logps(reference_model, chosen)
                reference_rejected = _model_logps(reference_model, rejected)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                policy_chosen = _model_logps(policy_model, chosen)
                policy_rejected = _model_logps(policy_model, rejected)
                result = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta=config.beta,
                )
                backward_loss = result.loss / config.gradient_accumulation_steps
            scaler.scale(backward_loss).backward()
            accumulated_loss += float(result.loss.detach())
            accumulated_accuracy += float(result.preference_accuracy)
            accumulated_margin += float(result.margin)
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "DPO gradient norm is NaN or Inf; prefer BF16 or reduce the learning rate."
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = float(scaler.get_scale()) >= scale_before
        optimizer.zero_grad(set_to_none=True)
        if optimizer_updated:
            scheduler.step()
        history.append(
            {
                "optimizer_step": step,
                "loss": accumulated_loss / config.gradient_accumulation_steps,
                "preference_accuracy": accumulated_accuracy
                / config.gradient_accumulation_steps,
                "reward_margin": accumulated_margin
                / config.gradient_accumulation_steps,
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_updated": int(optimizer_updated),
                "step_seconds": time.perf_counter() - started,
            }
        )
    return history
