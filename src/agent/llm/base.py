"""Backend-independent chat model contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agent.messages import Message, ToolSchema
from agent.tools.base import ToolCall

ChatFinishReason = Literal["length", "stop"]


@dataclass(frozen=True)
class ChatResult:
    """Backend-independent result returned by a chat model."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: ChatFinishReason
    model: str
    duration_ms: float = 0.0
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


class ChatModel(Protocol):
    """Lifecycle and inference contract used by Agent and HTTP layers."""

    @property
    def is_ready(self) -> bool:
        """Return whether the backend is loaded and ready."""

    def load(self) -> None:
        """Load model weights and tokenizer."""

    def close(self) -> None:
        """Release model resources."""

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> ChatResult:
        """Generate one assistant response, optionally with tools."""
