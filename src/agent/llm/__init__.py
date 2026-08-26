"""Unified chat backends used by the Agent and HTTP service."""

from typing import Any

from agent.llm.base import ChatFinishReason, ChatModel, ChatResult, ChatStreamChunk
from agent.llm.tool_format import (
    parse_tool_calls,
    prepare_messages,
    schemas_from_definitions,
)

__all__ = [
    "ChatFinishReason",
    "ChatModel",
    "ChatResult",
    "ChatStreamChunk",
    "HuggingFaceChatModel",
    "MiniGPTChatModel",
    "create_huggingface_chat_model",
    "parse_tool_calls",
    "prepare_messages",
    "render_minigpt_prompt",
    "schemas_from_definitions",
]


def __getattr__(name: str) -> Any:
    if name in {"HuggingFaceChatModel", "create_huggingface_chat_model"}:
        from agent.llm.huggingface_backend import (
            HuggingFaceChatModel,
            create_huggingface_chat_model,
        )

        if name == "HuggingFaceChatModel":
            return HuggingFaceChatModel
        return create_huggingface_chat_model
    if name in {"MiniGPTChatModel", "render_minigpt_prompt"}:
        from agent.llm.minigpt_backend import MiniGPTChatModel, render_minigpt_prompt

        if name == "MiniGPTChatModel":
            return MiniGPTChatModel
        return render_minigpt_prompt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
