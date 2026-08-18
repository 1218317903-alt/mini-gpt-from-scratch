"""CharacterTokenizer 测试。"""

import pytest

from minigpt.tokenizer import CharacterTokenizer


def test_vocab_and_round_trip() -> None:
    text = "我喜欢学习Python。"
    tokenizer = CharacterTokenizer(text)

    assert tokenizer.vocab_size == len(set(text)) + 1
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_character_requires_permission() -> None:
    tokenizer = CharacterTokenizer("abc")

    with pytest.raises(ValueError, match="未知字符"):
        tokenizer.encode("abd")


def test_unknown_character_uses_unk_when_allowed() -> None:
    tokenizer = CharacterTokenizer("abc")
    token_ids = tokenizer.encode("abd", allow_unknown=True)

    assert token_ids[-1] == tokenizer.unk_id
    assert tokenizer.decode(token_ids) == "ab<unk>"


def test_save_and_load(tmp_path) -> None:
    tokenizer = CharacterTokenizer("hello")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)

    assert loaded.itos == tokenizer.itos
    assert loaded.stoi == tokenizer.stoi


def test_invalid_token_id_raises() -> None:
    tokenizer = CharacterTokenizer("abc")

    with pytest.raises(ValueError, match="非法 token ID"):
        tokenizer.decode([999])
