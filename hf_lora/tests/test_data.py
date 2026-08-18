from __future__ import annotations

import pytest
import torch

from hf_lora.data import SFTDataCollator, encode_sft_example, validate_messages


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        assert tokenize is True and return_dict is True
        if add_generation_prompt:
            return {"input_ids": [10, 11, 12, 13]}
        return {"input_ids": [10, 11, 12, 13, 20, 21, 9]}


def example() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "答案"},
        ]
    }


def test_encode_masks_only_prompt() -> None:
    encoded = encode_sft_example(example(), FakeTokenizer(), 16)
    assert encoded["input_ids"] == [10, 11, 12, 13, 20, 21, 9]
    assert encoded["attention_mask"] == [1] * 7
    assert encoded["labels"] == [-100, -100, -100, -100, 20, 21, 9]


def test_collator_pads_fields_with_distinct_values() -> None:
    collator = SFTDataCollator(pad_token_id=0, pad_to_multiple_of=None)
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            },
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
    )
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))


def test_invalid_roles_and_overlong_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_messages([{"role": "user", "content": "问题"}])
    with pytest.raises(ValueError, match="超过 max_length"):
        encode_sft_example(example(), FakeTokenizer(), 5)
