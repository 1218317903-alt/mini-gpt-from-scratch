"""Dataset 单元测试。"""

import pytest
import torch
from torch.utils.data import DataLoader

from minigpt.dataset import (
    CharacterLanguageModelDataset,
)


def test_dataset_returns_shifted_sequences() -> None:
    """x 和 y 应该错开一个 Token。"""
    dataset = CharacterLanguageModelDataset(
        token_ids=[10, 11, 12, 13, 14],
        block_size=4,
    )

    x, y = dataset[0]

    assert torch.equal(
        x,
        torch.tensor(
            [10, 11, 12, 13],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        y,
        torch.tensor(
            [11, 12, 13, 14],
            dtype=torch.long,
        ),
    )


def test_dataset_shapes_and_dtype() -> None:
    """x 和 y 的形状及 dtype 应正确。"""
    dataset = CharacterLanguageModelDataset(
        token_ids=list(range(10)),
        block_size=4,
    )

    x, y = dataset[0]

    assert x.shape == (4,)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataloader_creates_batch() -> None:
    """DataLoader 应将 [T] 组合成 [B,T]。"""
    dataset = CharacterLanguageModelDataset(
        token_ids=list(range(10)),
        block_size=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(loader))

    assert x_batch.shape == (2, 4)
    assert y_batch.shape == (2, 4)
    assert x_batch.dtype == torch.long
    assert y_batch.dtype == torch.long


def test_dataset_rejects_short_tokens() -> None:
    """Token 数量不足时应该报错。"""
    with pytest.raises(ValueError):
        CharacterLanguageModelDataset(
            token_ids=[1, 2, 3],
            block_size=3,
        )
