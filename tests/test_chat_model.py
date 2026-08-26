"""Tests for the backend-independent Hugging Face chat adapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from agent.llm.huggingface_backend import HuggingFaceChatModel
from agent.messages import Message, ToolSchema
from minigpt_service.config import ChatServiceConfig


class _FakeBatch(dict):
    def to(self, _device: torch.device) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    unk_token_id = 1
    chat_template = "fake"
    last_messages: list[dict[str, Any]] | None = None
    last_tools: object = None

    def apply_chat_template(self, messages, **kwargs: object) -> _FakeBatch:
        self.last_messages = list(messages)
        self.last_tools = kwargs.get("tools")
        assert messages[-1]["role"] == "user"
        return _FakeBatch(input_ids=torch.tensor([[10, 11]]))

    def decode(self, token_ids, **_: object) -> str:
        values = [int(token_id) for token_id in token_ids]
        if values == [20, 99]:
            return "  hello  "
        if values == [30, 31]:
            return (
                '{"tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}}'
            )
        return "ok"

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|im_end|>":
            return 99
        return 1


class _FakeModel:
    def __init__(self, output_ids: list[int] | None = None) -> None:
        self.config = SimpleNamespace(use_cache=True, max_position_embeddings=32)
        self.training = False
        self.output_ids = output_ids or [10, 11, 20, 99]
        self.generate_calls = 0

    def eval(self) -> "_FakeModel":
        self.training = False
        return self

    def train(self) -> "_FakeModel":
        self.training = True
        return self

    def generate(self, **_: object) -> torch.Tensor:
        self.generate_calls += 1
        return torch.tensor([self.output_ids])


def _loaded_model(
    max_input_tokens: int = 8,
    output_ids: list[int] | None = None,
) -> HuggingFaceChatModel:
    backend = HuggingFaceChatModel(
        ChatServiceConfig(
            model_source="fake-model",
            device="cpu",
            max_input_tokens=max_input_tokens,
        ),
        torch.device("cpu"),
    )
    backend._tokenizer = _FakeTokenizer()
    backend._model = _FakeModel(output_ids)
    return backend


def test_chat_model_generates_and_detects_eos() -> None:
    backend = _loaded_model()

    result = backend.chat(
        [Message(role="user", content="hello")],
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
    assert result.duration_ms >= 0


def test_chat_model_keeps_three_turn_context() -> None:
    backend = _loaded_model()
    tokenizer = backend._tokenizer
    assert isinstance(tokenizer, _FakeTokenizer)

    backend.chat(
        [
            Message(role="system", content="be brief"),
            Message(role="user", content="1+1"),
            Message(role="assistant", content="2"),
            Message(role="user", content="plus 3?"),
            Message(role="assistant", content="5"),
            Message(role="user", content="and 4?"),
        ],
        max_new_tokens=4,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )

    assert tokenizer.last_messages is not None
    roles = [message["role"] for message in tokenizer.last_messages]
    contents = [message["content"] for message in tokenizer.last_messages]
    assert roles[-3:] == ["user", "assistant", "user"]
    assert "1+1" in contents
    assert "plus 3?" in contents
    assert "and 4?" in contents


def test_chat_model_parses_structured_json_tool_calls() -> None:
    backend = _loaded_model(output_ids=[10, 11, 30, 31])

    result = backend.chat(
        [Message(role="user", content="2+2")],
        [
            ToolSchema(
                name="calculator",
                description="evaluate arithmetic",
                parameters={"type": "object"},
            )
        ],
        max_new_tokens=16,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )

    assert result.tool_calls[0].name == "calculator"
    assert result.tool_calls[0].arguments == {"expression": "2+2"}
    tokenizer = backend._tokenizer
    assert isinstance(tokenizer, _FakeTokenizer)
    assert tokenizer.last_tools is not None


def test_chat_model_rejects_input_over_budget() -> None:
    backend = _loaded_model(max_input_tokens=1)

    with pytest.raises(ValueError, match="max_input_tokens"):
        backend.chat(
            [Message(role="user", content="hello")],
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
            [Message(role="user", content="hello")],
            max_new_tokens=4,
            temperature=0.7,
            top_k=None,
            top_p=None,
            greedy=True,
            seed=None,
        )


def test_chat_model_load_attaches_lora_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Tokenizer:
        chat_template = "template"
        pad_token_id = None
        eos_token_id = 2
        eos_token = "</s>"
        pad_token = None
        padding_side = "left"

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_cache=False)
            self.moved_to = None

        def to(self, device: torch.device) -> "_Model":
            self.moved_to = device
            return self

        def eval(self) -> "_Model":
            return self

    class _Adapter:
        def __init__(self, model: _Model) -> None:
            self.base = model
            self.config = model.config

        def eval(self) -> "_Adapter":
            return self

    def fake_tokenizer_from_pretrained(*args: object, **kwargs: object) -> _Tokenizer:
        captured["tokenizer_args"] = (args, kwargs)
        return _Tokenizer()

    def fake_model_from_pretrained(*args: object, **kwargs: object) -> _Model:
        captured["model_kwargs"] = kwargs
        return _Model()

    def fake_adapter_from_pretrained(
        model: _Model, path: str, **kwargs: object
    ) -> _Adapter:
        captured["adapter"] = (path, kwargs, model)
        return _Adapter(model)

    monkeypatch.setattr(
        "agent.llm.huggingface_backend.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "agent.llm.huggingface_backend.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "agent.llm.huggingface_backend.PeftModel.from_pretrained",
        fake_adapter_from_pretrained,
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_: "cached-model-snapshot",
    )

    backend = HuggingFaceChatModel(
        ChatServiceConfig(
            model_source="Qwen/Qwen2.5-0.5B-Instruct",
            adapter_path=__import__("pathlib").Path("adapter-dir"),
            device="cpu",
            local_files_only=True,
        ),
        torch.device("cpu"),
    )
    backend.load()
    backend.load()

    assert backend.is_ready
    assert captured["adapter"][0].endswith("adapter-dir")
    assert captured["adapter"][1]["is_trainable"] is False
    first_model = backend._model
    backend.close()
    assert not backend.is_ready
    assert first_model is not None


def test_chat_model_local_only_resolves_cached_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HuggingFaceChatModel(
        ChatServiceConfig(
            model_source="org/cached-model",
            local_files_only=True,
        ),
        torch.device("cpu"),
    )
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "local-snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert backend._resolved_model_source() == "local-snapshot"
    assert captured == {"repo_id": "org/cached-model", "local_files_only": True}
