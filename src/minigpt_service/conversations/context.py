"""Token-budgeted context building with persistent automatic compression."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent.messages import Message
from minigpt_service.conversations.models import Conversation, StoredMessage

TokenCounter = Callable[[Sequence[Message]], int]
SummaryFunction = Callable[[str | None, Sequence[StoredMessage], int], str]


@dataclass(frozen=True)
class ContextBuildResult:
    """Messages sent to the model plus updated compression metadata."""

    messages: tuple[Message, ...]
    estimated_tokens: int
    compressed: bool
    summary: str | None
    summarized_through_sequence: int
    omitted_messages: int


class ConversationContextBuilder:
    """Keep recent turns and summarize older complete messages automatically."""

    def __init__(
        self,
        *,
        max_input_tokens: int = 4096,
        reserved_output_tokens: int = 512,
        summary_token_budget: int = 768,
        recent_turns: int = 6,
        token_counter: TokenCounter | None = None,
        summarizer: SummaryFunction | None = None,
    ) -> None:
        if max_input_tokens < 256:
            raise ValueError("max_input_tokens must be at least 256.")
        if not 1 <= reserved_output_tokens < max_input_tokens:
            raise ValueError("reserved_output_tokens must fit inside the context.")
        if summary_token_budget < 64:
            raise ValueError("summary_token_budget is outside the valid range.")
        if recent_turns < 1:
            raise ValueError("recent_turns must be positive.")
        self.max_input_tokens = max_input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.summary_token_budget = min(
            summary_token_budget,
            max(64, (max_input_tokens - reserved_output_tokens) // 2),
        )
        self.recent_turns = recent_turns
        self._token_counter = token_counter or estimate_message_tokens
        self._summarizer = summarizer or deterministic_summary

    @property
    def input_budget(self) -> int:
        """Return tokens available before response generation."""
        return self.max_input_tokens - self.reserved_output_tokens

    def build(
        self,
        conversation: Conversation,
        stored_messages: Sequence[StoredMessage],
        *,
        reserved_output_tokens: int | None = None,
    ) -> ContextBuildResult:
        """Build bounded context and compress old messages when required."""
        reserved = (
            self.reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        if not 1 <= reserved < self.max_input_tokens:
            raise ValueError("reserved_output_tokens must fit inside the context.")
        input_budget = self.max_input_tokens - reserved
        complete = [
            message
            for message in stored_messages
            if message.status in {"complete", "cancelled", "interrupted"}
            and message.content.strip()
        ]
        system_messages = self._system_messages(conversation)
        unsummarized = [
            message
            for message in complete
            if message.sequence > conversation.summarized_through_sequence
        ]
        summary_message = self._summary_message(conversation.summary)
        candidate = [*system_messages]
        if summary_message is not None:
            candidate.append(summary_message)
        candidate.extend(self._as_message(message) for message in unsummarized)
        candidate_tokens = self._token_counter(candidate)
        if candidate_tokens <= input_budget:
            return ContextBuildResult(
                messages=tuple(candidate),
                estimated_tokens=candidate_tokens,
                compressed=False,
                summary=conversation.summary,
                summarized_through_sequence=conversation.summarized_through_sequence,
                omitted_messages=0,
            )

        keep_count = min(len(unsummarized), self.recent_turns * 2)
        recent = unsummarized[-keep_count:] if keep_count else []
        older = unsummarized[:-keep_count] if keep_count else unsummarized
        if not older and len(recent) > 1:
            older = recent[:1]
            recent = recent[1:]

        summary = self._summarizer(
            conversation.summary,
            older,
            self.summary_token_budget,
        )
        summarized_through = (
            older[-1].sequence if older else conversation.summarized_through_sequence
        )
        omitted_messages = len(older)
        output = [*system_messages]
        if summary:
            summary_message = self._summary_message(summary)
            assert summary_message is not None
            output.append(summary_message)
        output.extend(self._as_message(message) for message in recent)

        while len(recent) > 1 and self._token_counter(output) > input_budget:
            newly_older = recent.pop(0)
            summary = self._summarizer(
                summary,
                [newly_older],
                self.summary_token_budget,
            )
            summarized_through = newly_older.sequence
            omitted_messages += 1
            summary_message = self._summary_message(summary)
            assert summary_message is not None
            output = [*system_messages, summary_message]
            output.extend(self._as_message(message) for message in recent)

        while (
            summary and len(summary) > 32 and self._token_counter(output) > input_budget
        ):
            summary = summary[: max(32, int(len(summary) * 0.75))].rstrip()
            summary_message = self._summary_message(summary)
            assert summary_message is not None
            output = [*system_messages, summary_message]
            output.extend(self._as_message(message) for message in recent)

        estimated = self._token_counter(output)
        if estimated > input_budget:
            output = self._hard_bound(output, input_budget=input_budget)
            estimated = self._token_counter(output)

        return ContextBuildResult(
            messages=tuple(message for message in output if message is not None),
            estimated_tokens=estimated,
            compressed=True,
            summary=summary,
            summarized_through_sequence=summarized_through,
            omitted_messages=omitted_messages,
        )

    def _hard_bound(
        self,
        messages: list[Message],
        *,
        input_budget: int,
    ) -> list[Message]:
        output = list(messages)
        while len(output) > 2 and self._token_counter(output) > input_budget:
            removable = next(
                (
                    index
                    for index, message in enumerate(output[:-1])
                    if message.role != "system"
                ),
                None,
            )
            if removable is None:
                break
            output.pop(removable)
        if self._token_counter(output) > input_budget:
            raise ValueError("Latest message exceeds the configured context budget.")
        return output

    @staticmethod
    def _system_messages(conversation: Conversation) -> list[Message]:
        if not conversation.system_prompt:
            return []
        return [Message(role="system", content=conversation.system_prompt)]

    @staticmethod
    def _summary_message(summary: str | None) -> Message | None:
        if not summary:
            return None
        return Message(
            role="system",
            content=(
                "Earlier conversation summary; do not invent omitted details:\n"
                + summary
            ),
        )

    @staticmethod
    def _as_message(message: StoredMessage) -> Message:
        return Message(
            role=message.role,
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
        )


def estimate_message_tokens(messages: Sequence[Message]) -> int:
    """Conservatively estimate mixed Chinese/Latin chat-template token usage."""
    total = 8
    for message in messages:
        content = message.content
        cjk = len(re.findall(r"[\u3400-\u9fff]", content))
        non_cjk = max(0, len(content) - cjk)
        total += 8 + cjk + (non_cjk + 3) // 4
    return total


def deterministic_summary(
    existing_summary: str | None,
    messages: Sequence[StoredMessage],
    token_budget: int,
) -> str:
    """Create a reproducible extractive summary when no model summarizer is used."""
    character_budget = max(128, token_budget * 2)
    lines: list[str] = []
    if existing_summary:
        lines.append(existing_summary.strip())
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }
    for message in messages:
        normalized = " ".join(message.content.split())
        if len(normalized) > 360:
            normalized = normalized[:357].rstrip() + "..."
        lines.append(f"- {role_names[message.role]}: {normalized}")
    summary = "\n".join(line for line in lines if line)
    if len(summary) <= character_budget:
        return summary
    return summary[-character_budget:].lstrip()
