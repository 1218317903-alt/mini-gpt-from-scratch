"""Structured, request-local Agent tracing with conservative redaction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agent.messages import Message, message_payload
from agent.model_adapter import AgentTurn
from agent.tools.base import ToolCall, ToolExecutionResult

TracePhase = Literal["model", "tool"]
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}
_SECRET_PATTERN = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|authorization)\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def redact_text(value: str) -> str:
    """Redact common inline credential assignments without hiding token metrics."""
    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", value)


def redact(value: Any) -> Any:
    """Recursively redact sensitive mapping fields and credential-like strings."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower().replace("-", "_") in _SENSITIVE_KEYS
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def redact_content(value: str) -> str:
    """Redact either plain text or a serialized JSON tool result."""
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return redact_text(value)
    return json.dumps(redact(payload), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class AgentEvent:
    """One sanitized model or tool event from an Agent run."""

    step: int
    phase: TracePhase
    input: Any
    output: Any
    duration_ms: float
    error: str | None = None


class AgentTracer:
    """Collect sanitized events and optionally emit structured logs."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._events: list[AgentEvent] = []
        self._logger = logger or logging.getLogger("agent.trace")

    @property
    def events(self) -> tuple[AgentEvent, ...]:
        return tuple(self._events)

    def record_model(
        self,
        step: int,
        messages: list[Message],
        turn: AgentTurn | None,
        duration_ms: float,
        error: Exception | None = None,
    ) -> None:
        output: Any = None
        if turn is not None:
            output = {
                "text": turn.raw_text,
                "tool_calls": [
                    {"id": call.call_id, "name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
                "prompt_tokens": turn.prompt_tokens,
                "generated_tokens": turn.generated_tokens,
                "model": turn.model,
            }
        self._record(
            AgentEvent(
                step=step,
                phase="model",
                input=redact([message_payload(message) for message in messages]),
                output=redact(output),
                duration_ms=round(duration_ms, 3),
                error=redact_text(str(error)) if error else None,
            )
        )

    def record_tool(
        self,
        step: int,
        call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        error = redact_content(result.content) if result.status != "ok" else None
        self._record(
            AgentEvent(
                step=step,
                phase="tool",
                input=redact(
                    {"id": call.call_id, "name": call.name, "arguments": call.arguments}
                ),
                output=redact(
                    {
                        "status": result.status,
                        "content": redact_content(result.content),
                        "confirmation_id": result.confirmation_id,
                    }
                ),
                duration_ms=result.duration_ms,
                error=error,
            )
        )

    def _record(self, event: AgentEvent) -> None:
        self._events.append(event)
        self._logger.info(
            json.dumps(
                {"event": "agent_step", **asdict(event)},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
