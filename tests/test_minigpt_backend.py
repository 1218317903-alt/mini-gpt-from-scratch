"""Tests for the MiniGPT ChatModel backend."""

import torch

from agent.llm.minigpt_backend import MiniGPTChatModel, render_minigpt_prompt
from agent.messages import Message, ToolSchema
from minigpt.tokenizer import CharacterTokenizer

_VOCAB = 'abcdefghijklmnopqrstuvwxyz0123456789_:+{}[],." '


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.block_size = 16
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, 4)
        self.project = torch.nn.Linear(4, vocab_size)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.project(hidden)


class _FakeService:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer(_VOCAB)
        self.model = _TinyModel(self.tokenizer.vocab_size)
        self.block_size = 16
        self.device = torch.device("cpu")
        self.closed = False
        self.loaded = False

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True


def test_render_minigpt_prompt_includes_all_roles() -> None:
    prompt = render_minigpt_prompt(
        [
            Message(role="system", content="brief"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(
                role="tool",
                content="ok",
                name="calculator",
                tool_call_id="call-1",
            ),
        ]
    )

    assert "system: brief" in prompt
    assert "user: hi" in prompt
    assert "assistant: hello" in prompt
    assert "tool:calculator:call-1: ok" in prompt
    assert prompt.endswith("assistant:")


def test_minigpt_backend_generates_and_counts_tokens() -> None:
    service = _FakeService()
    backend = MiniGPTChatModel(service, owns_service=True)  # type: ignore[arg-type]
    backend.load()

    result = backend.chat(
        [Message(role="user", content="ab")],
        max_new_tokens=4,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=0,
    )
    backend.close()

    assert service.loaded is True
    assert service.closed is True
    assert result.model == "minigpt"
    assert result.finish_reason == "length"
    assert result.generated_tokens == 4
    assert result.prompt_tokens > 0
    assert result.duration_ms >= 0


def test_minigpt_backend_parses_tool_json_from_generated_text(
    monkeypatch: object,
) -> None:
    service = _FakeService()
    backend = MiniGPTChatModel(service)  # type: ignore[arg-type]

    def fake_generate(input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        continuation = service.tokenizer.encode(
            '{"tool_call":{"name":"calculator","arguments":{"expression":"1+1"}}}'
        )
        extra = torch.tensor([continuation], dtype=torch.long)
        return torch.cat([input_ids.cpu(), extra], dim=1)

    monkeypatch.setattr("agent.llm.minigpt_backend.generate", fake_generate)  # type: ignore[attr-defined]

    result = backend.chat(
        [Message(role="user", content="add")],
        [
            ToolSchema(
                name="calculator",
                description="math",
                parameters={"type": "object"},
            )
        ],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )

    assert result.tool_calls[0].name == "calculator"
    assert result.tool_calls[0].arguments == {"expression": "1+1"}
