"""Request-local conversation state and repeated-call detection."""

from __future__ import annotations

import json
from collections.abc import Sequence

from agent.messages import Message, as_message
from agent.tools.base import ToolCall, ToolExecutionResult


class AgentContext:
    """Own mutable state for one bounded Agent run."""

    def __init__(self, messages: Sequence[Message | object]) -> None:
        self.messages = [as_message(message) for message in messages]
        self._tool_call_attempts: dict[str, int] = {}
        self._successful_tool_calls: set[str] = set()

    def append_assistant(self, content: str) -> None:
        """Append the model's raw assistant turn."""
        self.messages.append(Message(role="assistant", content=content))

    def append_tool(self, call: ToolCall, result: ToolExecutionResult) -> None:
        """Append one normalized tool result for the next model turn."""
        self.messages.append(
            Message(
                role="tool",
                name=call.name,
                tool_call_id=call.call_id,
                content=result.content,
            )
        )

    def register_tool_call(self, call: ToolCall) -> bool:
        """Allow one retry after an error, but block successful or looping repeats."""
        signature = self._signature(call)
        attempts = self._tool_call_attempts.get(signature, 0)
        if signature in self._successful_tool_calls or attempts >= 2:
            return False
        self._tool_call_attempts[signature] = attempts + 1
        return True

    def record_tool_result(
        self,
        call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        """Remember successful calls so the model reuses their existing result."""
        if result.status == "ok":
            self._successful_tool_calls.add(self._signature(call))

    @staticmethod
    def _signature(call: ToolCall) -> str:
        return json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
