"""聊天 SFT 数据校验、Tokenize 与动态 Padding。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
VALID_ROLES = {"system", "user", "assistant", "tool"}


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
    """接受多轮对话与工具轨迹，并拒绝不可能的角色转移。"""
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
        if role not in VALID_ROLES:
            raise ValueError(f"第 {index} 条消息 role 非法：{role!r}。")
        if not isinstance(content, str):
            raise TypeError(f"第 {index} 条消息 content 必须是字符串。")
        content = content.strip()
        if not content:
            raise ValueError(f"第 {index} 条消息 content 不能为空。")
        normalized_message = {"role": role, "content": content}
        for optional_key in ("name", "tool_call_id"):
            optional_value = message.get(optional_key)
            if optional_value is not None:
                if not isinstance(optional_value, str) or not optional_value.strip():
                    raise TypeError(
                        f"第 {index} 条消息 {optional_key} 必须是非空字符串。"
                    )
                normalized_message[optional_key] = optional_value.strip()
        normalized.append(normalized_message)
    roles = tuple(message["role"] for message in normalized)
    start = 1 if roles[0] == "system" else 0
    if start >= len(roles) or roles[start] != "user":
        raise ValueError(f"首个非 system 角色必须是 user：{roles}。")
    if roles[-1] != "assistant":
        raise ValueError(f"最后一个角色必须是 assistant：{roles}。")
    for index, role in enumerate(roles):
        if role == "system" and index != 0:
            raise ValueError(f"system 只能出现在第一条：{roles}。")
        if index <= start:
            continue
        previous = roles[index - 1]
        allowed_previous = {
            "user": {"assistant"},
            "assistant": {"user", "tool"},
            "tool": {"assistant"},
        }[role]
        if previous not in allowed_previous:
            raise ValueError(f"不支持角色转移 {previous} -> {role}：{roles}。")
    return normalized


def _chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, generation: bool
) -> list[int]:
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
    """构造多轮样本，只监督每一段 Assistant 输出。"""
    if max_length <= 1:
        raise ValueError("max_length 必须大于 1。")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer 没有聊天模板。")
    messages = validate_messages(example.get("messages"))
    full_ids = _chat_ids(tokenizer, messages, generation=False)
    if not full_ids:
        raise ValueError("完整训练序列没有 Token。")
    if len(full_ids) > max_length:
        raise ValueError(f"样本超过 max_length：{len(full_ids)} > {max_length}。")
    labels = [IGNORE_INDEX] * len(full_ids)
    supervised_tokens = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_ids = _chat_ids(tokenizer, messages[:index], generation=True)
        through_ids = _chat_ids(tokenizer, messages[: index + 1], generation=False)
        if not prompt_ids or len(through_ids) <= len(prompt_ids):
            raise ValueError(f"第 {index} 条 Assistant 没有 Response Token。")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Assistant Prompt Token 不是完整训练序列的前缀。")
        if full_ids[: len(through_ids)] != through_ids:
            raise ValueError("Assistant Response Token 不是完整训练序列的前缀。")
        for token_index in range(len(prompt_ids), len(through_ids)):
            labels[token_index] = full_ids[token_index]
            supervised_tokens += 1
    if supervised_tokens == 0:
        raise ValueError("完整序列没有 Assistant Response Token。")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SFTDataset(Dataset[dict[str, list[int]]]):
    def __init__(
        self, records: list[dict[str, Any]], tokenizer: Any, max_length: int
    ) -> None:
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
            rows["labels"].append(list(feature["labels"]) + [IGNORE_INDEX] * padding)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()
        }
