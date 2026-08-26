"""Deterministic reliability helpers for bounded Agent execution."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event
from typing import Any

from pydantic import ValidationError

from agent.messages import Message
from agent.tools.base import ToolCall, ToolExecutionResult
from agent.tools.registry import ToolRegistry

_ARGUMENT_ALIASES = {
    "expr": "expression",
    "formula": "expression",
    "time_zone": "timezone",
    "tz": "timezone",
    "top_k": "k",
    "limit": "k",
    "path": "filename",
    "file": "filename",
}
_DOC_CITATION = re.compile(r"(?<![\w-])doc-[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_EXPLICIT_CITATION = re.compile(r"\[\[citation:([^\]]+)\]\]", re.IGNORECASE)


class CancellationToken:
    """Thread-safe cooperative cancellation checked at Agent boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation without terminating shared model state."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()


@dataclass(frozen=True)
class CitationValidation:
    """Result of checking generated citations against retrieved documents."""

    allowed: tuple[str, ...]
    predicted: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.unsupported


class CitationGuard:
    """Collect search results and block document IDs not returned this run."""

    def __init__(self) -> None:
        self._allowed: set[str] = set()

    @property
    def allowed(self) -> tuple[str, ...]:
        return tuple(sorted(self._allowed))

    def record(self, call: ToolCall, result: ToolExecutionResult) -> None:
        """Add citations from one successful search result to the whitelist."""
        if call.name != "search_docs" or result.status != "ok":
            return
        try:
            payload = json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        results = payload.get("results")
        if not isinstance(results, list):
            return
        for item in results:
            if not isinstance(item, dict):
                continue
            citation = item.get("citation")
            if isinstance(citation, str) and citation.strip():
                self._allowed.add(citation.strip())

    def validate(self, text: str) -> CitationValidation:
        """Validate explicit citation tokens and doc-* identifiers in text."""
        predicted = set(_DOC_CITATION.findall(text))
        predicted.update(match.strip() for match in _EXPLICIT_CITATION.findall(text))
        normalized_allowed = {value.lower(): value for value in self._allowed}
        unsupported = sorted(
            value for value in predicted if value.lower() not in normalized_allowed
        )
        return CitationValidation(
            allowed=self.allowed,
            predicted=tuple(sorted(predicted)),
            unsupported=tuple(unsupported),
        )


class SchemaArgumentRepairer:
    """Apply one conservative alias/shape repair, then require schema validity."""

    def repair(self, call: ToolCall, registry: ToolRegistry) -> ToolCall | None:
        """Return a validated repaired call, or ``None`` when unsafe to infer."""
        tool = registry.get(call.name)
        if tool is None:
            return None
        try:
            tool.arguments_model.model_validate(call.arguments)
        except ValidationError:
            pass
        else:
            return call

        original = call.arguments
        candidate: dict[str, Any]
        nested = original.get("arguments")
        if len(original) == 1 and isinstance(nested, dict):
            candidate = dict(nested)
        else:
            candidate = dict(original)

        fields = tool.arguments_model.model_fields
        for alias, canonical in _ARGUMENT_ALIASES.items():
            if (
                alias in candidate
                and canonical in fields
                and canonical not in candidate
            ):
                candidate[canonical] = candidate.pop(alias)

        missing_required = [
            name
            for name, field in fields.items()
            if field.is_required() and name not in candidate
        ]
        unknown = [name for name in candidate if name not in fields]
        if len(missing_required) == 1 and len(unknown) == 1:
            candidate[missing_required[0]] = candidate.pop(unknown[0])

        if candidate == original:
            return None
        try:
            validated = tool.arguments_model.model_validate(candidate)
        except ValidationError:
            return None
        return ToolCall(
            call_id=call.call_id,
            name=call.name,
            arguments=validated.model_dump(),
        )


def compact_messages(
    messages: Sequence[Message],
    *,
    max_characters: int,
) -> tuple[Message, ...]:
    """Bound old context while preserving system instructions and recent turns."""
    if max_characters < 512:
        raise ValueError("max_characters must be at least 512.")
    total = sum(len(message.content) for message in messages)
    if total <= max_characters:
        return tuple(messages)

    system = next((message for message in messages if message.role == "system"), None)
    budget = max_characters - (len(system.content) if system else 0) - 240
    kept_reversed: list[Message] = []
    used = 0
    for message in reversed(messages):
        if message is system:
            continue
        size = len(message.content)
        if kept_reversed and used + size > budget:
            break
        kept_reversed.append(message)
        used += size
    kept = list(reversed(kept_reversed))
    omitted_count = len(messages) - len(kept) - (1 if system else 0)
    summary = Message(
        role="system",
        content=(
            f"Context budget applied: {omitted_count} earlier messages were omitted. "
            "Use the preserved recent turns and do not invent missing details."
        ),
    )
    output = ([system] if system else []) + [summary] + kept
    return tuple(message for message in output if message is not None)
