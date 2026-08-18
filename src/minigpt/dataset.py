"""字符级语言模型 Dataset。"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


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
