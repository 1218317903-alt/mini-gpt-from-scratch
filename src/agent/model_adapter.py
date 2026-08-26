"""Adapt any ChatModel backend to the Agent turn protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.llm.tool_format import parse_tool_calls, schemas_from_definitions
from agent.messages import Message, as_message
from agent.tools.base import ToolCall, ToolDefinition
from minigpt_service.schemas import ChatMessage

if TYPE_CHECKING:
    from agent.llm.base import ChatModel


@dataclass(frozen=True)
class AgentTurn:
    """One model turn after parsing any requested tool calls."""

    text: str
    raw_text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    generated_tokens: int
    model: str


class ChatModelAgentAdapter:
    """Use a backend-agnostic chat model with explicit tool schemas."""

    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    @property
    def is_ready(self) -> bool:
        """Return whether the wrapped chat model is ready."""
        return self.chat_model.is_ready

    def complete(
        self,
        messages: Sequence[ChatMessage | Message],
        tools: Sequence[ToolDefinition],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> AgentTurn:
        """Prompt the chat model and parse its optional JSON tool call."""
        prepared_messages = [as_message(message) for message in messages]
        schemas = schemas_from_definitions(tools)
        result = self.chat_model.chat(
            prepared_messages,
            schemas or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            seed=seed,
        )
        tool_calls = result.tool_calls or parse_tool_calls(result.text)
        return AgentTurn(
            text="" if tool_calls else result.text,
            raw_text=result.text,
            tool_calls=tool_calls,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            model=result.model,
        )


# Kept for tests and callers that imported the parser from this module.
__all__ = [
    "AgentTurn",
    "ChatModelAgentAdapter",
    "parse_tool_calls",
]
