"""教学可读的确定性 Byte Pair Encoding Tokenizer。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """在 Unicode 字符序列上学习 BPE 合并规则。"""

    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocabulary: list[str], merges: list[tuple[str, str]]) -> None:
        if not vocabulary or vocabulary[0] != self.UNK_TOKEN:
            raise ValueError("BPE 词表第一个 Token 必须是 <unk>。")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("BPE 词表不能包含重复 Token。")
        self.itos = list(vocabulary)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.merges = list(merges)
        self.unk_id = 0
        self.pad_id = self.stoi.get(self.PAD_TOKEN)
        self.eos_id = self.stoi.get(self.EOS_TOKEN)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """用最高频相邻对迭代合并，频率相同时按字典序稳定选择。"""
        if not isinstance(text, str) or not text:
            raise ValueError("训练文本必须是非空字符串。")
        base_tokens = sorted(set(text))
        special_tokens = [cls.UNK_TOKEN, cls.PAD_TOKEN, cls.EOS_TOKEN]
        if vocab_size < len(base_tokens) + len(special_tokens):
            raise ValueError("vocab_size 小于字符基础词表。")

        sequence = list(text)
        vocabulary = [*special_tokens, *base_tokens]
        merges: list[tuple[str, str]] = []
        while len(vocabulary) < vocab_size and len(sequence) >= 2:
            counts = Counter(zip(sequence, sequence[1:], strict=False))
            if not counts:
                break
            max_count = max(counts.values())
            pair = min(pair for pair, count in counts.items() if count == max_count)
            merged_token = "".join(pair)
            if merged_token in vocabulary:
                break

            merged_sequence: list[str] = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and sequence[index] == pair[0]
                    and sequence[index + 1] == pair[1]
                ):
                    merged_sequence.append(merged_token)
                    index += 2
                else:
                    merged_sequence.append(sequence[index])
                    index += 1
            sequence = merged_sequence
            vocabulary.append(merged_token)
            merges.append(pair)
        return cls(vocabulary, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _tokenize(self, text: str, *, allow_unknown: bool) -> list[str]:
        tokens: list[str] = []
        for position, character in enumerate(text):
            if character not in self.stoi:
                if not allow_unknown:
                    raise ValueError(f"未知字符 {character!r}，位置为 {position}。")
                tokens.append(self.UNK_TOKEN)
            else:
                tokens.append(character)
        for left, right in self.merges:
            merged = left + right
            output: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == left
                    and tokens[index + 1] == right
                ):
                    output.append(merged)
                    index += 2
                else:
                    output.append(tokens[index])
                    index += 1
            tokens = output
        return tokens

    def encode(self, text: str, *, allow_unknown: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text 必须是 str。")
        return [
            self.stoi.get(token, self.unk_id)
            for token in self._tokenize(text, allow_unknown=allow_unknown)
        ]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("每个 token_id 必须是 int。")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"非法 token ID：{token_id}。")
            token = self.itos[token_id]
            if skip_special_tokens and token in {
                self.UNK_TOKEN,
                self.PAD_TOKEN,
                self.EOS_TOKEN,
            }:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"unk_token": self.UNK_TOKEN, "itos": self.itos, "merges": self.merges},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("unk_token") != cls.UNK_TOKEN:
            raise ValueError("BPE Tokenizer JSON 格式错误。")
        vocabulary = payload.get("itos")
        raw_merges = payload.get("merges")
        if not isinstance(vocabulary, list) or not all(
            isinstance(item, str) for item in vocabulary
        ):
            raise ValueError("BPE itos 必须是字符串列表。")
        if not isinstance(raw_merges, list):
            raise ValueError("BPE merges 必须是列表。")
        merges: list[tuple[str, str]] = []
        for item in raw_merges:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                raise ValueError("每条 BPE merge 必须包含两个字符串。")
            merges.append((item[0], item[1]))
        return cls(vocabulary, merges)
