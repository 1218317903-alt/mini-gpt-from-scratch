from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from hf_lora.preference import (
    DPOTrainingConfig,
    PreferenceDataCollator,
    dpo_loss,
    encode_preference_example,
    evaluate_dpo,
    sequence_log_probabilities,
    train_dpo,
    validate_preference_record,
)


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        assert tokenize and return_dict
        ids = [1, 2, 3]
        if add_generation_prompt:
            return {"input_ids": ids}
        else:
            ids.extend([10 + len(messages[-1]["content"]), 9])
        return {"input_ids": ids}


def record() -> dict:
    return {
        "prompt": [{"role": "user", "content": "解释 DDP"}],
        "chosen": "DDP 会在各进程同步梯度。",
        "rejected": "DDP 会自动扩大显存。",
    }


def test_preference_encoding_masks_prompt_for_both_candidates() -> None:
    encoded = encode_preference_example(record(), FakeTokenizer(), 16)

    assert encoded["chosen_labels"][:3] == [-100, -100, -100]
    assert encoded["rejected_labels"][:3] == [-100, -100, -100]
    assert encoded["chosen_labels"][-2:] != encoded["rejected_labels"][-2:]


def test_preference_validation_rejects_equal_or_non_user_prompt() -> None:
    invalid = record()
    invalid["rejected"] = invalid["chosen"]
    with pytest.raises(ValueError, match="must differ"):
        validate_preference_record(invalid)
    with pytest.raises(ValueError, match="end with a user"):
        validate_preference_record(
            {
                "prompt": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "chosen": "c",
                "rejected": "r",
            }
        )


def test_preference_collator_pads_chosen_and_rejected_independently() -> None:
    first = encode_preference_example(record(), FakeTokenizer(), 16)
    second_record = record()
    second_record["chosen"] = "更长的优选回答"
    second = encode_preference_example(second_record, FakeTokenizer(), 16)
    batch = PreferenceDataCollator(0, pad_to_multiple_of=None)([first, second])

    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_labels"].shape[0] == 2


def test_sequence_log_probabilities_uses_only_response_labels() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    result = sequence_log_probabilities(logits, labels)

    assert result.item() == pytest.approx(-2 * torch.log(torch.tensor(3.0)).item())


def test_dpo_loss_rewards_larger_policy_preference_margin() -> None:
    good = dpo_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=0.5,
    )
    bad = dpo_loss(
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=0.5,
    )

    assert good.loss < bad.loss
    assert good.preference_accuracy.item() == 1.0
    assert bad.preference_accuracy.item() == 0.0


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.output = torch.nn.Linear(8, 32)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def preference_batch() -> dict[str, torch.Tensor]:
    return {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "chosen_labels": torch.tensor([[-100, -100, 3, 4]]),
        "rejected_input_ids": torch.tensor([[1, 2, 5, 6]]),
        "rejected_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "rejected_labels": torch.tensor([[-100, -100, 5, 6]]),
    }


def test_dpo_evaluation_matches_identical_policy_and_reference() -> None:
    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    policy.train()
    reference.train()

    metrics = evaluate_dpo(
        policy,
        reference,
        [preference_batch()],
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        beta=0.1,
    )

    assert metrics["loss"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert metrics["preference_accuracy"] == 0.0
    assert metrics["reward_margin"] == 0.0
    assert metrics["examples"] == 1.0
    assert policy.training and reference.training


def test_dpo_training_loop_runs_on_cpu() -> None:
    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    history = train_dpo(
        policy,
        reference,
        [preference_batch()],
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        config=DPOTrainingConfig(
            beta=0.1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_optimizer_steps=2,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        ),
    )

    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in history)
