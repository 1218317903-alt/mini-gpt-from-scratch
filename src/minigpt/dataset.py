"""字符级语言模型 Dataset。"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from minigpt.tokenizer import ProjectTokenizer


class CharacterLanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """将连续 Token ID 转换为 x 和 y。"""

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> None:
        """初始化 Dataset。"""
        if not token_ids:
            raise ValueError("token_ids 不能为空。")

        if block_size <= 0:
            raise ValueError("block_size 必须大于 0。")

        if len(token_ids) <= block_size:
            raise ValueError("token 数量必须大于 block_size。")

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.block_size = block_size

    def __len__(self) -> int:
        """返回可切分的样本数量。"""
        return self.token_ids.size(0) - self.block_size

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回一个 x 和 y。"""
        if not isinstance(index, int):
            raise TypeError("index 必须是 int。")

        if not 0 <= index < len(self):
            raise IndexError(f"index 超出范围：{index}。")

        window = self.token_ids[index : index + self.block_size + 1]

        x = window[:-1]
        y = window[1:]

        return x, y


# 保留旧名称，避免已有脚本立刻失效。
CausalLanguageModelingDataset = CharacterLanguageModelDataset


class PackedLanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """把多个变长样本用 EOS 拼成固定长度块，尾部目标用 -100 屏蔽。"""

    def __init__(
        self,
        sequences: list[list[int]],
        block_size: int,
        *,
        eos_id: int,
        pad_id: int,
    ) -> None:
        if not sequences or any(not item for item in sequences):
            raise ValueError("sequences 必须包含非空 Token 序列。")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size 必须是正整数。")
        if type(eos_id) is not int or eos_id < 0:
            raise ValueError("eos_id 必须是非负整数。")
        if type(pad_id) is not int or pad_id < 0:
            raise ValueError("pad_id 必须是非负整数。")

        stream: list[int] = []
        for sequence in sequences:
            stream.extend(sequence)
            stream.append(eos_id)
        self.blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        stride = block_size + 1
        for start in range(0, len(stream) - 1, block_size):
            window = stream[start : start + stride]
            valid_targets = max(0, len(window) - 1)
            if len(window) < stride:
                window.extend([pad_id] * (stride - len(window)))
            inputs = torch.tensor(window[:-1], dtype=torch.long)
            targets = torch.tensor(window[1:], dtype=torch.long)
            if valid_targets < block_size:
                targets[valid_targets:] = -100
            self.blocks.append((inputs, targets))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if type(index) is not int:
            raise TypeError("index 必须是 int。")
        return self.blocks[index]

    @property
    def utilization(self) -> float:
        """返回非 padding 目标 Token 占总槽位的比例。"""
        valid = sum(int((targets != -100).sum()) for _, targets in self.blocks)
        total = sum(targets.numel() for _, targets in self.blocks)
        return valid / total


LanguageModelDataset = CharacterLanguageModelDataset | PackedLanguageModelDataset


def split_text_samples(text: str, separator: str) -> list[str]:
    """Split a corpus into non-empty logical samples for sequence packing."""
    if not isinstance(text, str) or not text:
        raise ValueError("text 必须是非空字符串。")
    if not isinstance(separator, str) or not separator:
        raise ValueError("separator 必须是非空字符串。")
    samples = [sample for sample in text.split(separator) if sample]
    return samples or [text]


def build_language_model_dataset(
    text: str,
    tokenizer: ProjectTokenizer,
    *,
    block_size: int,
    packing: bool,
    sample_separator: str,
    allow_unknown: bool = False,
) -> LanguageModelDataset:
    """Build either the legacy continuous stream or EOS-aware packed data."""
    if not packing:
        return CharacterLanguageModelDataset(
            tokenizer.encode(text, allow_unknown=allow_unknown),
            block_size,
        )
    eos_id = getattr(tokenizer, "eos_id", None)
    pad_id = getattr(tokenizer, "pad_id", None)
    if eos_id is None or pad_id is None:
        raise ValueError("Packing requires a Tokenizer with EOS and PAD IDs.")
    sequences = [
        tokenizer.encode(sample, allow_unknown=allow_unknown)
        for sample in split_text_samples(text, sample_separator)
    ]
    sequences = [sequence for sequence in sequences if sequence]
    if not sequences:
        raise ValueError("Packing did not produce any non-empty Token sequences.")
    return PackedLanguageModelDataset(
        sequences,
        block_size,
        eos_id=eos_id,
        pad_id=pad_id,
    )
