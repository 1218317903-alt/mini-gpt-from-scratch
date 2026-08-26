"""Compatibility exports for the Agent-layer chat backends."""

from typing import Any

from agent.llm.base import ChatFinishReason, ChatModel, ChatResult, ChatStreamChunk

__all__ = [  # noqa: F822
    "ChatFinishReason",
    "ChatModel",
    "ChatResult",
    "ChatStreamChunk",
    "HuggingFaceChatModel",
    "MiniGPTChatModel",
    "create_huggingface_chat_model",
]


def __getattr__(name: str) -> Any:
    if name in {"HuggingFaceChatModel", "create_huggingface_chat_model"}:
        from agent.llm.huggingface_backend import (
            HuggingFaceChatModel,
            create_huggingface_chat_model,
        )

        exports = {
            "HuggingFaceChatModel": HuggingFaceChatModel,
            "create_huggingface_chat_model": create_huggingface_chat_model,
        }
        return exports[name]
    if name == "MiniGPTChatModel":
        from agent.llm.minigpt_backend import MiniGPTChatModel

        return MiniGPTChatModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
