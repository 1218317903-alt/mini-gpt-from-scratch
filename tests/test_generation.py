"""Stage-four generation and sampling tests."""

import pytest
import torch
from torch import nn

from minigpt.generation import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    generate,
    greedy_decode,
    logits_to_probabilities,
    sample_next_token,
    select_last_logits,
)
from minigpt.tokenizer import CharacterTokenizer
from minigpt.utils import set_random_seed


class TrackingLogitModel(nn.Module):
    """Return fixed logits while recording inference behavior."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        if logits.ndim != 1:
            raise ValueError("logits must be [V].")
        self.register_buffer("fixed_logits", logits.float())
        self.context_lengths: list[int] = []
        self.training_flags: list[bool] = []
        self.grad_enabled_flags: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.context_lengths.append(input_ids.shape[1])
        self.training_flags.append(self.training)
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        batch_size, sequence_length = input_ids.shape
        return self.fixed_logits.view(1, 1, -1).expand(
            batch_size,
            sequence_length,
            -1,
        )


def test_select_last_logits_and_softmax() -> None:
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]]
    )

    last_logits = select_last_logits(logits)
    probabilities = logits_to_probabilities(last_logits)

    assert last_logits.shape == (1, 3)
    assert torch.equal(last_logits, logits[:, -1, :])
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(1),
    )


def test_greedy_decode_is_deterministic() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0]])

    assert greedy_decode(logits).tolist() == [1]
    assert greedy_decode(logits).tolist() == [1]


def test_temperature_changes_logits_and_rejects_invalid_values() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0]])

    assert torch.equal(apply_temperature(logits, 1.0), logits)
    assert apply_temperature(logits, 0.5)[0, 0] == 8.0
    assert apply_temperature(logits, 2.0)[0, 0] == 2.0

    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -1.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, float("nan"))


def test_top_k_clamps_to_vocabulary_size() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0]])

    filtered = apply_top_k(logits, top_k=100)
    assert torch.allclose(filtered, logits)

    filtered = apply_top_k(logits, top_k=1)
    probabilities = logits_to_probabilities(filtered)
    assert probabilities.argmax(dim=-1).tolist() == [0]
    assert probabilities[0, 1:].max().item() == 0.0

    with pytest.raises(ValueError):
        apply_top_k(logits, top_k=-1)
    with pytest.raises(TypeError):
        apply_top_k(logits, top_k=1.5)  # type: ignore[arg-type]


def test_top_p_keeps_at_least_one_and_validates_range() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0]])

    filtered = apply_top_p(logits, top_p=0.01)
    probabilities = logits_to_probabilities(filtered)
    assert probabilities[0, 0].item() == pytest.approx(1.0)
    assert probabilities[0, 1:].max().item() == 0.0

    with pytest.raises(ValueError):
        apply_top_p(logits, top_p=0.0)
    with pytest.raises(ValueError):
        apply_top_p(logits, top_p=1.01)
    with pytest.raises(ValueError):
        apply_top_p(logits, top_p=float("nan"))


def test_sampling_repeats_with_fixed_seed() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.7]])

    set_random_seed(42)
    first = torch.stack(
        [sample_next_token(probabilities) for _ in range(8)]
    )
    set_random_seed(42)
    second = torch.stack(
        [sample_next_token(probabilities) for _ in range(8)]
    )

    assert torch.equal(first, second)


def test_generate_greedy_adds_one_token_per_step_and_truncates_context() -> None:
    model = TrackingLogitModel(torch.tensor([0.0, 1.0, 5.0]))
    prompt = torch.tensor([[0, 1, 2, 1]])

    generated = generate(
        model,
        prompt,
        max_new_tokens=3,
        block_size=2,
        do_sample=False,
    )

    assert generated.shape == (1, 7)
    assert generated[0, -3:].tolist() == [2, 2, 2]
    assert model.context_lengths == [2, 2, 2]
    assert model.training_flags == [False, False, False]
    assert model.grad_enabled_flags == [False, False, False]


def test_generate_sampling_is_reproducible() -> None:
    model = TrackingLogitModel(torch.tensor([0.0, 1.0, 2.0]))
    prompt = torch.tensor([[0, 1]])

    set_random_seed(7)
    first = generate(
        model,
        prompt,
        max_new_tokens=12,
        block_size=4,
        do_sample=True,
        temperature=0.8,
        top_k=2,
        top_p=0.95,
    )

    set_random_seed(7)
    second = generate(
        model,
        prompt,
        max_new_tokens=12,
        block_size=4,
        do_sample=True,
        temperature=0.8,
        top_k=2,
        top_p=0.95,
    )

    assert torch.equal(first, second)


def test_generate_rejects_empty_prompt_and_supports_zero_new_tokens() -> None:
    model = TrackingLogitModel(torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="positive batch"):
        generate(
            model,
            torch.empty((1, 0), dtype=torch.long),
            max_new_tokens=1,
            block_size=4,
        )

    prompt = torch.tensor([[0, 1]])
    generated = generate(
        model,
        prompt,
        max_new_tokens=0,
        block_size=4,
    )
    assert torch.equal(generated, prompt)


def test_generated_ids_can_be_decoded() -> None:
    tokenizer = CharacterTokenizer("abc")
    model = TrackingLogitModel(
        torch.tensor([0.0, 1.0, 2.0, 3.0])
    )
    prompt = torch.tensor([tokenizer.encode("ab")])

    generated = generate(
        model,
        prompt,
        max_new_tokens=2,
        block_size=4,
        do_sample=False,
    )

    # Token ID 3 is <unk>, which is still a valid Tokenizer output.
    text = tokenizer.decode(generated[0].tolist())
    assert text.startswith("ab")

