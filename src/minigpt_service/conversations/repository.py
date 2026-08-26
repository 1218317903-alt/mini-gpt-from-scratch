"""Conversation persistence contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent.messages import MessageRole
from minigpt_service.conversations.models import (
    Conversation,
    MessageStatus,
    StoredMessage,
)


class ConversationRepository(Protocol):
    """Storage operations required by the conversation HTTP layer."""

    def initialize(self) -> None:
        """Create or migrate storage and recover interrupted generations."""

    def close(self) -> None:
        """Release repository resources."""

    def create_conversation(
        self,
        *,
        title: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
        """Create a new conversation."""

    def list_conversations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """List recently updated non-archived conversations."""

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return one conversation or ``None``."""

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation | None:
        """Update mutable conversation metadata."""

    def update_summary(
        self,
        conversation_id: str,
        *,
        summary: str,
        summarized_through_sequence: int,
    ) -> Conversation:
        """Persist automatic context compression state."""

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""

    def append_message(
        self,
        conversation_id: str,
        *,
        role: MessageRole,
        content: str,
        status: MessageStatus = "complete",
        model: str | None = None,
        request_id: str | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
    ) -> StoredMessage:
        """Append one ordered message."""

    def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        status: MessageStatus | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> StoredMessage | None:
        """Update one message after or during generation."""

    def list_messages(self, conversation_id: str) -> Sequence[StoredMessage]:
        """Return messages in ascending sequence order."""

    def get_message_by_request_id(self, request_id: str) -> StoredMessage | None:
        """Return the message associated with an idempotency key."""
