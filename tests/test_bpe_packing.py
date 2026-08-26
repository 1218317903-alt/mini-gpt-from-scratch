"""P8 tokenizer and sequence-packing tests."""

from pathlib import Path

import torch

from minigpt.bpe_tokenizer import BPETokenizer
from minigpt.dataset import PackedLanguageModelDataset, build_language_model_dataset
from minigpt.model import MiniGPT
from minigpt.tokenizer import decode_token_ids
from minigpt.trainer import compute_language_model_loss


def test_bpe_training_is_deterministic_and_roundtrips(tmp_path: Path) -> None:
    text = "abababab 中文中文 abab"
    first = BPETokenizer.train(text, vocab_size=len(set(text)) + 8)
    second = BPETokenizer.train(text, vocab_size=len(set(text)) + 8)
    assert first.itos == second.itos
    assert first.merges == second.merges
    encoded = first.encode(text)
    assert first.decode(encoded) == text
    assert len(encoded) < len(text)

    path = tmp_path / "bpe.json"
    first.save(path)
    restored = BPETokenizer.load(path)
    assert restored.encode(text) == encoded
    assert restored.decode(encoded) == text
    assert restored.pad_id is not None
    assert restored.eos_id is not None


def test_bpe_unknown_policy_is_explicit() -> None:
    tokenizer = BPETokenizer.train("abcabc", vocab_size=6)
    try:
        tokenizer.encode("z")
    except ValueError as error:
        assert "未知字符" in str(error)
    else:
        raise AssertionError("unknown character should fail")
    assert tokenizer.encode("z", allow_unknown=True) == [tokenizer.unk_id]


def test_packed_dataset_uses_eos_and_masks_padding() -> None:
    dataset = PackedLanguageModelDataset(
        [[1, 2], [3, 4, 5]], block_size=4, eos_id=6, pad_id=0
    )
    assert len(dataset) == 2
    first_inputs, first_targets = dataset[0]
    assert first_inputs.tolist() == [1, 2, 6, 3]
    assert first_targets.tolist() == [2, 6, 3, 4]
    _, final_targets = dataset[1]
    assert -100 in final_targets.tolist()
    assert 0 < dataset.utilization < 1


def test_language_model_loss_ignores_packed_padding_targets() -> None:
    model = MiniGPT(12, 4, 16, 4, 1)
    inputs = torch.tensor([[1, 2, 3, 0]])
    targets = torch.tensor([[2, 3, 4, -100]])
    _, loss = compute_language_model_loss(model, inputs, targets)
    assert torch.isfinite(loss)


def test_bpe_pipeline_inserts_eos_between_samples() -> None:
    tokenizer = BPETokenizer.train("abcd", vocab_size=7)
    dataset = build_language_model_dataset(
        "ab\n\ncd",
        tokenizer,
        block_size=4,
        packing=True,
        sample_separator="\n\n",
    )

    first_inputs, first_targets = dataset[0]
    assert tokenizer.eos_id in first_inputs.tolist() + first_targets.tolist()


def test_user_facing_bpe_decode_hides_control_tokens() -> None:
    tokenizer = BPETokenizer.train("abcd", vocab_size=7)
    assert tokenizer.eos_id is not None
    assert tokenizer.pad_id is not None

    text = decode_token_ids(
        tokenizer,
        [tokenizer.stoi["a"], tokenizer.eos_id, tokenizer.pad_id],
    )

    assert text == "a"
