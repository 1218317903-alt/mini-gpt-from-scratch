"""Tests for the backend-independent Hugging Face chat adapter."""

from types import SimpleNamespace

import pytest
import torch

from minigpt_service.chat_model import HuggingFaceChatModel
from minigpt_service.config import ChatServiceConfig
from minigpt_service.schemas import ChatMessage


class _FakeBatch(dict):
    def to(self, _device: torch.device) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    chat_template = "fake"

    def apply_chat_template(self, messages, **_: object) -> _FakeBatch:
        assert messages[-1]["role"] == "user"
        return _FakeBatch(input_ids=torch.tensor([[10, 11]]))

    def decode(self, token_ids, **_: object) -> str:
        assert list(token_ids) == [20, 99]
        return "  hello  "


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True, max_position_embeddings=32)
        self.training = False

    def eval(self) -> "_FakeModel":
        self.training = False
        return self

    def train(self) -> "_FakeModel":
        self.training = True
        return self

    def generate(self, **_: object) -> torch.Tensor:
        return torch.tensor([[10, 11, 20, 99]])


def _loaded_model(max_input_tokens: int = 8) -> HuggingFaceChatModel:
    backend = HuggingFaceChatModel(
        ChatServiceConfig(
            model_source="fake-model",
            device="cpu",
            max_input_tokens=max_input_tokens,
        ),
        torch.device("cpu"),
    )
    backend._tokenizer = _FakeTokenizer()
    backend._model = _FakeModel()
    return backend


def test_chat_model_generates_and_detects_eos() -> None:
    backend = _loaded_model()

    result = backend.chat(
        [ChatMessage(role="user", content="hello")],
        max_new_tokens=4,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )

    assert result.text == "hello"
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 2
    assert result.finish_reason == "stop"
    assert result.model == "fake-model"


def test_chat_model_rejects_input_over_budget() -> None:
    backend = _loaded_model(max_input_tokens=1)

    with pytest.raises(ValueError, match="max_input_tokens"):
        backend.chat(
            [ChatMessage(role="user", content="hello")],
            max_new_tokens=4,
            temperature=0.7,
            top_k=None,
            top_p=None,
            greedy=True,
            seed=None,
        )


def test_chat_model_requires_loaded_backend() -> None:
    backend = HuggingFaceChatModel(
        ChatServiceConfig(model_source="fake-model"),
        torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        backend.chat(
            [ChatMessage(role="user", content="hello")],
            max_new_tokens=4,
            temperature=0.7,
            top_k=None,
            top_p=None,
            greedy=True,
            seed=None,
        )
