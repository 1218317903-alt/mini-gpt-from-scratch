"""聊天 SFT 数据校验、Tokenize 与动态 Padding。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
VALID_ROLE_ORDERS = (
    ("user", "assistant"),
    ("system", "user", "assistant"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL，并在错误中保留行号。"""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON。") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
        records.append(record)
    if not records:
        raise ValueError(f"数据文件为空：{path}")
    return records


def validate_messages(messages: Any) -> list[dict[str, str]]:
    """第一版只接受单轮 user/assistant，可选 system。"""
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or not messages
    ):
        raise ValueError("messages 必须是非空消息序列。")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"第 {index} 条消息必须是字典。")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"第 {index} 条消息 role 非法：{role!r}。")
        if not isinstance(content, str):
            raise TypeError(f"第 {index} 条消息 content 必须是字符串。")
        content = content.strip()
        if not content:
            raise ValueError(f"第 {index} 条消息 content 不能为空。")
        normalized.append({"role": role, "content": content})
    roles = tuple(message["role"] for message in normalized)
    if roles not in VALID_ROLE_ORDERS:
        raise ValueError(f"第一版不支持角色顺序：{roles}。")
    return normalized


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list) or not all(type(x) is int for x in input_ids):
        raise TypeError("聊天模板的 input_ids 必须是整数列表。")
    return list(input_ids)


def encode_sft_example(
    example: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """构造只监督 Assistant Response 的单条 SFT 样本。"""
    if max_length <= 1:
        raise ValueError("max_length 必须大于 1。")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer 没有聊天模板。")
    messages = validate_messages(example.get("messages"))
    prompt_ids = _chat_ids(tokenizer, messages[:-1], generation=True)
    full_ids = _chat_ids(tokenizer, messages, generation=False)
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("完整序列没有 Assistant Response Token。")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt Token 不是完整训练序列的前缀。")
    if len(full_ids) > max_length:
        raise ValueError(
            f"样本超过 max_length：{len(full_ids)} > {max_length}。"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SFTDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = [
            encode_sft_example(record, tokenizer, max_length) for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class SFTDataCollator:
    """右侧动态 Padding，保留 response-only labels。"""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8) -> None:
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("pad_token_id 必须是非负整数。")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of 必须是正整数或 None。")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features 不能为空。")
        width = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            width = (width + multiple - 1) // multiple * multiple
        rows: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for index, feature in enumerate(features):
            lengths = {len(feature[key]) for key in rows}
            if len(lengths) != 1:
                raise ValueError(f"样本 {index} 的字段长度不一致。")
            if not any(label != IGNORE_INDEX for label in feature["labels"]):
                raise ValueError(f"样本 {index} 没有监督 Token。")
            padding = width - len(feature["input_ids"])
            rows["input_ids"].append(
                list(feature["input_ids"]) + [self.pad_token_id] * padding
            )
            rows["attention_mask"].append(
                list(feature["attention_mask"]) + [0] * padding
            )
            rows["labels"].append(
                list(feature["labels"]) + [IGNORE_INDEX] * padding
            )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()}
