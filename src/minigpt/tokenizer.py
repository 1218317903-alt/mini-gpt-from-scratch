"""字符级 Tokenizer。"""

from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    """在字符和 Token ID 之间转换。"""

    UNK_TOKEN = "<unk>"

    def __init__(self, training_text: str) -> None:
        if not isinstance(training_text, str):
            raise TypeError("training_text 必须是 str。")
        if not training_text:
            raise ValueError("training_text 不能为空。")

        self.itos: list[str] = [
            self.UNK_TOKEN,
            *sorted(set(training_text)),
        ]
        self.stoi: dict[str, int] = {
            token: index for index, token in enumerate(self.itos)
        }
        self.unk_id = self.stoi[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """返回词表大小。"""
        return len(self.itos)

    def encode(
        self,
        text: str,
        *,
        allow_unknown: bool = False,
    ) -> list[int]:
        """将文本编码为 Token ID。"""
        if not isinstance(text, str):
            raise TypeError("text 必须是 str。")

        token_ids: list[int] = []
        for position, character in enumerate(text):
            token_id = self.stoi.get(character)
            if token_id is None:
                if not allow_unknown:
                    raise ValueError(f"未知字符 {character!r}，位置为 {position}。")
                token_id = self.unk_id
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """将 Token ID 解码为文本。"""
        characters: list[str] = []
        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("每个 token_id 必须是 int。")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"非法 token ID：{token_id}。")
            characters.append(self.itos[token_id])
        return "".join(characters)

    def save(self, path: Path) -> None:
        """将词表保存为 JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "unk_token": self.UNK_TOKEN,
            "itos": self.itos,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CharacterTokenizer:
        """从 JSON 加载词表。"""
        if not path.is_file():
            raise FileNotFoundError(f"找不到 Tokenizer 文件：{path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Tokenizer JSON 格式错误。") from error

        if not isinstance(payload, dict):
            raise ValueError("Tokenizer JSON 顶层必须是对象。")
        if payload.get("unk_token") != cls.UNK_TOKEN:
            raise ValueError("Tokenizer 的 unk_token 配置错误。")

        raw_itos = payload.get("itos")
        if not isinstance(raw_itos, list):
            raise ValueError("Tokenizer JSON 必须包含 itos 列表。")
        if not raw_itos or raw_itos[0] != cls.UNK_TOKEN:
            raise ValueError("itos 第一个 Token 必须是 <unk>。")
        if not all(isinstance(token, str) for token in raw_itos):
            raise ValueError("itos 中的内容必须全部是字符串。")
        if len(set(raw_itos)) != len(raw_itos):
            raise ValueError("itos 中不能有重复 Token。")

        tokenizer = cls.__new__(cls)
        tokenizer.itos = raw_itos
        tokenizer.stoi = {token: index for index, token in enumerate(raw_itos)}
        tokenizer.unk_id = tokenizer.stoi[cls.UNK_TOKEN]
        return tokenizer
