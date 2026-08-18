"""Autoregressive generation and basic sampling strategies."""

from __future__ import annotations

import math

import torch
from torch import nn


def _require_finite_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    """Validate that a tensor is floating-point and finite."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def select_last_logits(logits: torch.Tensor) -> torch.Tensor:
    """Select the last time step from logits with shape ``[B, T, V]``."""
    _require_finite_tensor(logits, name="logits")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V].")

    _, sequence_length, vocab_size = logits.shape
    if sequence_length <= 0:
        raise ValueError("logits must have a positive sequence length.")
    if vocab_size <= 0:
        raise ValueError("logits must have a positive vocabulary size.")

    return logits[:, -1, :]


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert a ``[B, V]`` logits matrix into row-wise probabilities."""
    _require_finite_tensor(logits, name="logits")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, V].")
    if logits.shape[0] <= 0 or logits.shape[1] <= 0:
        raise ValueError("logits must have positive batch and vocabulary sizes.")

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all().item():
        raise FloatingPointError("Softmax probabilities contain NaN or Inf.")
    return probabilities


def greedy_decode(last_logits: torch.Tensor) -> torch.Tensor:
    """Select the highest-logit Token ID for every batch item."""
    _require_finite_tensor(last_logits, name="last_logits")
    if last_logits.ndim != 2:
        raise ValueError("last_logits must have shape [B, V].")
    if last_logits.shape[0] <= 0 or last_logits.shape[1] <= 0:
        raise ValueError("last_logits must have positive batch and vocabulary sizes.")
    return torch.argmax(last_logits, dim=-1)


def validate_temperature(temperature: float) -> float:
    """Validate and normalize a Temperature value."""
    if isinstance(temperature, bool):
        raise TypeError("temperature must be a number, not bool.")
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be an int or float.")

    temperature = float(temperature)
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")
    return temperature


def apply_temperature(
    last_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Scale ``[B, V]`` logits by ``1 / temperature``."""
    _require_finite_tensor(last_logits, name="last_logits")
    if last_logits.ndim != 2:
        raise ValueError("last_logits must have shape [B, V].")
    if last_logits.shape[0] <= 0 or last_logits.shape[1] <= 0:
        raise ValueError("last_logits must have positive batch and vocabulary sizes.")

    temperature = validate_temperature(temperature)
    scaled_logits = last_logits / temperature
    _require_finite_tensor(scaled_logits, name="scaled_logits")
    return scaled_logits


def _validate_top_k_type(top_k: int | None) -> None:
    if top_k is None:
        return
    if isinstance(top_k, bool):
        raise TypeError("top_k must be an integer, not bool.")
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an int or None.")
    if top_k < 0:
        raise ValueError("top_k cannot be negative.")


def validate_top_k(
    top_k: int | None,
    *,
    vocab_size: int,
) -> int | None:
    """Validate Top-k and clamp it to the vocabulary size."""
    _validate_top_k_type(top_k)
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than 0.")
    if top_k is None or top_k == 0:
        return None
    return min(top_k, vocab_size)


def apply_top_k(
    last_logits: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """Keep the highest ``top_k`` logits in every batch row."""
    _require_finite_tensor(last_logits, name="last_logits")
    if last_logits.ndim != 2:
        raise ValueError("last_logits must have shape [B, V].")

    batch_size, vocab_size = last_logits.shape
    if batch_size <= 0 or vocab_size <= 0:
        raise ValueError("last_logits must have positive batch and vocabulary sizes.")

    effective_top_k = validate_top_k(
        top_k,
        vocab_size=vocab_size,
    )
    if effective_top_k is None:
        return last_logits.clone()

    top_k_indices = torch.topk(
        last_logits,
        k=effective_top_k,
        dim=-1,
    ).indices

    keep_mask = torch.zeros_like(
        last_logits,
        dtype=torch.bool,
    )
    keep_mask.scatter_(
        dim=-1,
        index=top_k_indices,
        value=True,
    )

    mask_value = torch.finfo(last_logits.dtype).min
    return last_logits.masked_fill(~keep_mask, mask_value)


def validate_top_p(top_p: float | None) -> float | None:
    """Validate Top-p and treat 1.0 as disabled filtering."""
    if top_p is None:
        return None
    if isinstance(top_p, bool):
        raise TypeError("top_p must be a number, not bool.")
    if not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be an int, float, or None.")

    top_p = float(top_p)
    if not math.isfinite(top_p):
        raise ValueError("top_p must be finite.")
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if top_p == 1.0:
        return None
    return top_p


def apply_top_p(
    last_logits: torch.Tensor,
    top_p: float | None = None,
) -> torch.Tensor:
    """Keep the smallest probability nucleus whose mass reaches ``top_p``."""
    _require_finite_tensor(last_logits, name="last_logits")
    if last_logits.ndim != 2:
        raise ValueError("last_logits must have shape [B, V].")

    batch_size, vocab_size = last_logits.shape
    if batch_size <= 0 or vocab_size <= 0:
        raise ValueError("last_logits must have positive batch and vocabulary sizes.")

    effective_top_p = validate_top_p(top_p)
    if effective_top_p is None:
        return last_logits.clone()

    sorted_logits, sorted_indices = torch.sort(
        last_logits,
        descending=True,
        dim=-1,
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    keep_counts = (cumulative_probabilities < effective_top_p).sum(dim=-1) + 1
    keep_counts = keep_counts.clamp(
        min=1,
        max=vocab_size,
    )

    ranks = torch.arange(
        vocab_size,
        device=last_logits.device,
    ).unsqueeze(0)
    keep_sorted = ranks < keep_counts.unsqueeze(-1)
    remove_sorted = ~keep_sorted

    remove_mask = torch.zeros_like(
        remove_sorted,
        dtype=torch.bool,
    )
    remove_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=remove_sorted,
    )

    mask_value = torch.finfo(last_logits.dtype).min
    return last_logits.masked_fill(remove_mask, mask_value)


def _validate_probability_matrix(
    probabilities: torch.Tensor,
) -> None:
    """Validate a row-wise probability matrix with shape ``[B, V]``."""
    _require_finite_tensor(probabilities, name="probabilities")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [B, V].")
    if probabilities.shape[0] <= 0 or probabilities.shape[1] <= 0:
        raise ValueError("probabilities must have positive batch and vocabulary sizes.")
    if (probabilities < 0).any().item():
        raise ValueError("probabilities cannot contain negative values.")

    row_sums = probabilities.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each probability row must sum to approximately 1.")


def sample_next_token(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Sample one Token ID per batch row from ``[B, V]`` probabilities."""
    _validate_probability_matrix(probabilities)
    return torch.multinomial(
        probabilities,
        num_samples=1,
    ).squeeze(-1)


def _validate_model_input(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long dtype.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T].")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids must have positive batch and sequence sizes.")


def predict_next_token_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one deterministic Greedy prediction step."""
    _validate_model_input(model, input_ids)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        return greedy_decode(select_last_logits(logits))


def predict_next_token_sampling(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Run one Temperature/Top-k/Top-p sampling prediction step."""
    _validate_model_input(model, input_ids)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = select_last_logits(logits)
        last_logits = apply_temperature(last_logits, temperature)
        last_logits = apply_top_k(last_logits, top_k)
        last_logits = apply_top_p(last_logits, top_p)
        return sample_next_token(logits_to_probabilities(last_logits))


def validate_max_new_tokens(max_new_tokens: int) -> int:
    """Validate the number of new Tokens to generate."""
    if type(max_new_tokens) is not int:
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")
    return max_new_tokens


def validate_block_size(block_size: int) -> int:
    """Validate a model context window length."""
    if type(block_size) is not int:
        raise TypeError("block_size must be an integer.")
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0.")
    return block_size


def _select_next_token(
    last_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Use the configured decoding policy for one ``[B,V]`` matrix."""
    if not do_sample:
        return greedy_decode(last_logits)
    filtered_logits = apply_temperature(last_logits, temperature)
    filtered_logits = apply_top_k(filtered_logits, top_k)
    filtered_logits = apply_top_p(filtered_logits, top_p)
    return sample_next_token(
        logits_to_probabilities(filtered_logits),
    )


def _generate_without_cache(
    model: nn.Module,
    generated: torch.Tensor,
    *,
    max_new_tokens: int,
    block_size: int,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Baseline generation that recomputes the visible context each step."""
    for _ in range(max_new_tokens):
        context = generated[:, -block_size:]
        logits = model(context)
        next_token = _select_next_token(
            select_last_logits(logits),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat(
            (generated, next_token.unsqueeze(1)),
            dim=1,
        )
    return generated


def _generate_with_cache(
    model: nn.Module,
    generated: torch.Tensor,
    *,
    max_new_tokens: int,
    block_size: int,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Generate by one Prompt prefill followed by one-Token decode steps."""
    total_length = generated.size(1) + max_new_tokens
    if total_length > block_size:
        raise ValueError(
            "KV Cache 要求 prompt_length + max_new_tokens 不超过 block_size。"
        )

    logits, past_key_values = model(generated, use_cache=True)
    for step in range(max_new_tokens):
        next_token = _select_next_token(
            select_last_logits(logits),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat(
            (generated, next_token.unsqueeze(1)),
            dim=1,
        )
        if step + 1 < max_new_tokens:
            logits, past_key_values = model(
                next_token.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
            )
    return generated


def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    block_size: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """Generate a sequence containing the prompt and new Token IDs."""
    _validate_model_input(model, input_ids)
    max_new_tokens = validate_max_new_tokens(max_new_tokens)
    block_size = validate_block_size(block_size)

    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a bool.")
    if not isinstance(use_kv_cache, bool):
        raise TypeError("use_kv_cache must be a bool.")

    # Validate sampling arguments even when the caller selects Greedy. This
    # makes invalid command-line parameters fail early and predictably.
    temperature = validate_temperature(temperature)
    top_p = validate_top_p(top_p)
    _validate_top_k_type(top_k)

    generated = input_ids.clone()
    model.eval()

    if max_new_tokens == 0:
        return generated

    with torch.no_grad():
        if use_kv_cache:
            return _generate_with_cache(
                model,
                generated,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        return _generate_without_cache(
            model,
            generated,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
