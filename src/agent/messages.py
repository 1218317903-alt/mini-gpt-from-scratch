"""Conversation types shared by every chat backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """One conversation turn accepted by a chat backend."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolSchema:
    """Model-facing tool description, independent of the HTTP layer."""

    name: str
    description: str
    parameters: dict[str, Any]


def as_message(item: Message | Any) -> Message:
    """Normalize a Message, ChatMessage, or mapping into ``Message``."""
    if isinstance(item, Message):
        return item
    role = getattr(item, "role", None)
    content = getattr(item, "content", None)
    if role is None and isinstance(item, dict):
        role = item.get("role")
        content = item.get("content")
        name = item.get("name")
        tool_call_id = item.get("tool_call_id")
    else:
        name = getattr(item, "name", None)
        tool_call_id = getattr(item, "tool_call_id", None)
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("message role must be system, user, assistant, or tool.")
    if not isinstance(content, str):
        raise TypeError("message content must be a string.")
    return Message(
        role=role,
        content=content,
        name=name if isinstance(name, str) else None,
        tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
    )


def message_payload(message: Message) -> dict[str, Any]:
    """Convert a message into a chat-template mapping."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload
