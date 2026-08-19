"""Validated, bounded execution for explicitly registered tools."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any

from pydantic import ValidationError

from agent.tools.base import ToolCall, ToolExecutionResult
from agent.tools.registry import ToolRegistry


class ToolExecutor:
    """Validate arguments and execute registry tools with a timeout."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-tool")

    def close(self) -> None:
        """Stop accepting new work and cancel futures that have not started."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def execute(self, call: ToolCall) -> ToolExecutionResult:
        """Execute one call and convert all failures to a stable result."""
        started = time.perf_counter()
        tool = self.registry.get(call.name)
        if tool is None:
            return self._error(call, "Unknown tool.", started)

        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except ValidationError as error:
            return self._error(call, f"Invalid arguments: {error}", started)

        future: Future[Any] = self._pool.submit(
            tool.handler,
            **arguments.model_dump(),
        )
        try:
            value = future.result(timeout=tool.timeout_seconds)
        except TimeoutError:
            future.cancel()
            return self._error(call, "Tool timed out.", started)
        except Exception as error:  # Tool errors must not crash the Agent loop.
            return self._error(call, f"Tool failed: {error}", started)

        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            status="ok",
            content=self._serialize(value),
            duration_ms=self._duration_ms(started),
        )

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _duration_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _error(
        self,
        call: ToolCall,
        content: str,
        started: float,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            status="error",
            content=content,
            duration_ms=self._duration_ms(started),
        )
