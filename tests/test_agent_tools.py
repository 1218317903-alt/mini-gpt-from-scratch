"""Tests for the explicit Agent tool allow-list and executor."""

import time

import pytest
from pydantic import BaseModel, Field

from agent.builtin_tools import build_default_registry, calculate, current_time
from agent.tools.base import ToolCall, ToolSpec
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry


def test_default_registry_exposes_only_safe_builtins() -> None:
    registry = build_default_registry()

    assert registry.names() == ("calculator", "current_time")
    assert {definition.name for definition in registry.definitions()} == {
        "calculator",
        "current_time",
    }


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("17 * 6", 102), ("(10 - 3) / 2", 3.5), ("2 ** 3", 8)],
)
def test_calculator_handles_basic_arithmetic(
    expression: str, expected: int | float
) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('whoami')", "2 ** 101", "1 / 0", "[1, 2]"],
)
def test_calculator_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_current_time_returns_iso_timestamp() -> None:
    value = current_time("UTC")

    assert value.endswith("+00:00")
    assert "T" in value


def test_registry_rejects_duplicate_names() -> None:
    class Arguments(BaseModel):
        value: int = Field(ge=0)

    tool = ToolSpec(
        name="duplicate",
        description="test",
        arguments_model=Arguments,
        handler=lambda value: value,
    )
    registry = ToolRegistry([tool])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_executor_validates_and_serializes_results() -> None:
    registry = build_default_registry()
    executor = ToolExecutor(registry)
    try:
        result = executor.execute(
            ToolCall(
                call_id="call-1",
                name="calculator",
                arguments={"expression": "17*6"},
            )
        )
        invalid = executor.execute(
            ToolCall(call_id="call-2", name="calculator", arguments={})
        )
        unknown = executor.execute(
            ToolCall(call_id="call-3", name="shell", arguments={})
        )
    finally:
        executor.close()

    assert (result.status, result.content) == ("ok", "102")
    assert invalid.status == "error"
    assert "Invalid arguments" in invalid.content
    assert (unknown.status, unknown.content) == ("error", "Unknown tool.")


def test_executor_enforces_timeout() -> None:
    class Arguments(BaseModel):
        seconds: float = Field(gt=0)

    def slow(seconds: float) -> str:
        time.sleep(seconds)
        return "done"

    registry = ToolRegistry(
        [
            ToolSpec(
                name="slow",
                description="slow test tool",
                arguments_model=Arguments,
                handler=slow,
                timeout_seconds=0.01,
            )
        ]
    )
    executor = ToolExecutor(registry)
    try:
        result = executor.execute(
            ToolCall(call_id="call-1", name="slow", arguments={"seconds": 0.1})
        )
    finally:
        executor.close()

    assert (result.status, result.content) == ("error", "Tool timed out.")
