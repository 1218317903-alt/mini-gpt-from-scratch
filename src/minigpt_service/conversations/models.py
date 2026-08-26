"""Storage-independent conversation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.messages import MessageRole

MessageStatus = Literal[
    "complete",
    "generating",
    "cancelled",
    "error",
    "interrupted",
]


@dataclass(frozen=True)
class Conversation:
    """One persistent chat thread."""

    id: str
    title: str
    model: str | None
    system_prompt: str | None
    summary: str | None
    summarized_through_sequence: int
    created_at: str
    updated_at: str
    archived: bool = False


@dataclass(frozen=True)
class StoredMessage:
    """One ordered message belonging to a conversation."""

    id: str
    conversation_id: str
    sequence: int
    role: MessageRole
    content: str
    status: MessageStatus
    model: str | None
    prompt_tokens: int | None
    generated_tokens: int | None
    finish_reason: str | None
    request_id: str | None
    created_at: str
    updated_at: str
    name: str | None = None
    tool_call_id: str | None = None
