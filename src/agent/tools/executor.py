"""Validated, bounded execution for explicitly registered tools."""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable, Collection
from multiprocessing.connection import Connection
from threading import BoundedSemaphore
from typing import Any

from pydantic import ValidationError

from agent.tools.base import ToolCall, ToolExecutionResult, tool_confirmation_id
from agent.tools.registry import ToolRegistry
from agent.tracing import redact_text


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _run_tool(
    handler: Callable[..., Any],
    arguments: dict[str, Any],
    connection: Connection,
) -> None:
    """Run one handler in a child process and return a serialized envelope."""
    try:
        connection.send(("ok", _serialize(handler(**arguments))))
    except Exception as error:
        detail = redact_text(str(error))
        connection.send(("error", f"Tool failed ({type(error).__name__}): {detail}"))
    finally:
        connection.close()


class ToolExecutor:
    """Validate arguments and execute tools in terminable child processes."""

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 4) -> None:
        if type(max_concurrency) is not int or max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        self.registry = registry
        self._context = multiprocessing.get_context("spawn")
        self._slots = BoundedSemaphore(max_concurrency)
        self._closed = False

    def close(self) -> None:
        """Stop accepting new work; active child processes remain request-owned."""
        self._closed = True

    def execute(
        self,
        call: ToolCall,
        *,
        approved_side_effects: Collection[str] = (),
    ) -> ToolExecutionResult:
        """Execute one call and convert all failures to a stable result."""
        started = time.perf_counter()
        if self._closed:
            return self._error(call, "Tool executor is closed.", started)
        tool = self.registry.get(call.name)
        if tool is None:
            return self._error(call, "Unknown tool.", started)

        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except ValidationError as error:
            details = error.errors(include_url=False, include_input=False)
            return self._error(
                call,
                f"Invalid arguments: {_serialize(details)}",
                started,
            )

        confirmation_id = tool_confirmation_id(call)
        if tool.side_effect and confirmation_id not in approved_side_effects:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                status="confirmation_required",
                content="User confirmation is required before executing this tool.",
                duration_ms=self._duration_ms(started),
                confirmation_id=confirmation_id,
            )

        if not self._slots.acquire(timeout=tool.timeout_seconds):
            return self._error(call, "Tool executor is busy.", started)
        parent_connection, child_connection = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_run_tool,
            args=(tool.handler, arguments.model_dump(), child_connection),
            name=f"agent-tool-{tool.name}",
            daemon=True,
        )
        try:
            process.start()
            child_connection.close()
            if not parent_connection.poll(tool.timeout_seconds):
                process.terminate()
                process.join()
                return self._error(call, "Tool timed out.", started)
            status, content = parent_connection.recv()
            process.join()
        except Exception as error:  # Process startup/IPC failures are tool failures.
            if process.is_alive():
                process.terminate()
                process.join()
            return self._error(call, f"Tool failed to start: {error}", started)
        finally:
            parent_connection.close()
            child_connection.close()
            self._slots.release()

        if status == "error":
            return self._error(call, content, started)
        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            status="ok",
            content=content,
            duration_ms=self._duration_ms(started),
        )

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
