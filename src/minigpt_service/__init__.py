"""HTTP service layer for MiniGPT inference."""

from importlib import import_module
from typing import Any

__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "AgentRequest",
    "AgentResponse",
    "AgentTraceResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "ModelService",
    "ChatModel",
    "ChatResult",
    "HuggingFaceChatModel",
    "ReadyResponse",
    "ServiceConfig",
    "ChatServiceConfig",
    "app",
    "create_app",
    "create_app_from_environment",
]


def __getattr__(name: str) -> Any:
    """Load service components lazily so schema-only imports stay lightweight."""
    modules = {
        "GenerateRequest": "minigpt_service.schemas",
        "GenerateResponse": "minigpt_service.schemas",
        "AgentRequest": "minigpt_service.schemas",
        "AgentResponse": "minigpt_service.schemas",
        "AgentTraceResponse": "minigpt_service.schemas",
        "ChatMessage": "minigpt_service.schemas",
        "ChatRequest": "minigpt_service.schemas",
        "ChatResponse": "minigpt_service.schemas",
        "ErrorResponse": "minigpt_service.schemas",
        "HealthResponse": "minigpt_service.schemas",
        "ModelService": "minigpt_service.model_service",
        "ChatModel": "minigpt_service.chat_model",
        "ChatResult": "minigpt_service.chat_model",
        "HuggingFaceChatModel": "minigpt_service.chat_model",
        "ReadyResponse": "minigpt_service.schemas",
        "ServiceConfig": "minigpt_service.config",
        "ChatServiceConfig": "minigpt_service.config",
        "app": "minigpt_service.app",
        "create_app": "minigpt_service.app",
        "create_app_from_environment": "minigpt_service.app",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
