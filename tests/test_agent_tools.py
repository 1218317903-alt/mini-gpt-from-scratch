"""Tests for the explicit Agent tool allow-list and executor."""

import time

import pytest
from pydantic import BaseModel, Field

from agent.builtin_tools import (
    CorpusDocumentSearch,
    LocalDocumentSearch,
    build_default_registry,
    calculate,
    current_time,
)
from agent.tools.base import ToolCall, ToolSpec, tool_confirmation_id
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry


class SlowArguments(BaseModel):
    seconds: float = Field(gt=0)


def slow_tool(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


class WriteArguments(BaseModel):
    value: str


def side_effect_tool(value: str) -> str:
    return f"wrote {value}"


class SecretArguments(BaseModel):
    password: int


def test_default_registry_exposes_only_safe_builtins() -> None:
    registry = build_default_registry()

    assert registry.names() == ("calculator", "current_time", "search_docs")
    assert {definition.name for definition in registry.definitions()} == {
        "calculator",
        "current_time",
        "search_docs",
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


def test_local_document_search_returns_ranked_citations(tmp_path) -> None:
    (tmp_path / "agent.md").write_text(
        "工具调用失败后应把错误作为 tool message 返回模型。", encoding="utf-8"
    )
    (tmp_path / "other.txt").write_text("无关的厨房记录。", encoding="utf-8")

    payload = LocalDocumentSearch(tmp_path)("工具失败 错误", k=1)

    assert payload["results"][0]["citation"] == "agent.md"
    assert "tool message" in payload["results"][0]["snippet"]


def test_corpus_document_search_preserves_eval_document_ids() -> None:
    search = CorpusDocumentSearch(
        [
            {"id": "doc-office", "text": "总部位于杭州云栖园区。"},
            {"id": "doc-wifi", "text": "访客网络名称是 Qingwu-Guest。"},
        ]
    )

    payload = search("总部 杭州", k=1)

    assert payload["results"][0]["citation"] == "doc-office"


def test_corpus_document_search_rejects_conflicting_ids() -> None:
    with pytest.raises(ValueError, match="Conflicting corpus document"):
        CorpusDocumentSearch(
            [
                {"id": "same", "text": "first"},
                {"id": "same", "text": "second"},
            ]
        )


def test_registry_accepts_injected_search_handler() -> None:
    def handler(query, k=5):
        return {"query": query, "results": [], "k": k}

    registry = build_default_registry(search_handler=handler)

    assert registry.get("search_docs").handler is handler


def test_registry_rejects_multiple_search_sources(tmp_path) -> None:
    with pytest.raises(ValueError, match="only one"):
        build_default_registry(tmp_path, documents=[])


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
    registry = ToolRegistry(
        [
            ToolSpec(
                name="slow",
                description="slow test tool",
                arguments_model=SlowArguments,
                handler=slow_tool,
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


def test_executor_remains_usable_after_timeout() -> None:
    registry = build_default_registry()
    registry.register(
        ToolSpec(
            name="slow",
            description="slow test tool",
            arguments_model=SlowArguments,
            handler=slow_tool,
            timeout_seconds=0.01,
        )
    )
    executor = ToolExecutor(registry)
    try:
        timed_out = executor.execute(
            ToolCall(call_id="call-1", name="slow", arguments={"seconds": 10})
        )
        healthy = executor.execute(
            ToolCall(
                call_id="call-2",
                name="calculator",
                arguments={"expression": "2+2"},
            )
        )
    finally:
        executor.close()

    assert timed_out.content == "Tool timed out."
    assert (healthy.status, healthy.content) == ("ok", "4")


def test_executor_requires_confirmation_for_side_effects() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_value",
                description="test-only side effect",
                arguments_model=WriteArguments,
                handler=side_effect_tool,
                side_effect=True,
            )
        ]
    )
    executor = ToolExecutor(registry)
    call = ToolCall("call-write", "write_value", {"value": "safe"})
    try:
        blocked = executor.execute(call)
        approved = executor.execute(
            call,
            approved_side_effects={tool_confirmation_id(call)},
        )
    finally:
        executor.close()

    assert blocked.status == "confirmation_required"
    assert blocked.confirmation_id == tool_confirmation_id(call)
    assert (approved.status, approved.content) == ("ok", "wrote safe")


def test_validation_errors_do_not_echo_sensitive_input_values() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="secret_test",
                description="test-only validation",
                arguments_model=SecretArguments,
                handler=lambda password: password,
            )
        ]
    )
    executor = ToolExecutor(registry)
    try:
        result = executor.execute(
            ToolCall("call-secret", "secret_test", {"password": "do-not-log"})
        )
    finally:
        executor.close()

    assert result.status == "error"
    assert "do-not-log" not in result.content
