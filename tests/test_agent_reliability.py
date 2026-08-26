"""Regression tests for P6 planning, repair, cancellation, and citations."""

import json
from collections.abc import Sequence
from typing import Any

from agent.builtin_tools import build_default_registry
from agent.messages import Message
from agent.model_adapter import AgentTurn
from agent.orchestrator import (
    AgentOrchestrator,
    AgentProgressEvent,
    AgentResult,
)
from agent.reliability import CancellationToken, CitationGuard, compact_messages
from agent.routing import HeuristicAgentRouter
from agent.tools.base import ToolCall, ToolDefinition, ToolExecutionResult
from minigpt_service.schemas import ChatMessage


class _Model:
    is_ready = True

    def __init__(self, turns: list[AgentTurn]) -> None:
        self.turns = turns
        self.messages: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        **kwargs: object,
    ) -> AgentTurn:
        del tools, kwargs
        self.messages.append(list(messages))
        return self.turns.pop(0)


def _turn(
    text: str = "",
    *,
    calls: tuple[ToolCall, ...] = (),
    prompt_tokens: int = 3,
    generated_tokens: int = 2,
) -> AgentTurn:
    return AgentTurn(
        text=text,
        raw_text=text or "tool-call",
        tool_calls=calls,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        model="fake-p6",
    )


def _run(
    orchestrator: AgentOrchestrator,
    text: str,
    **kwargs: Any,
) -> AgentResult:
    return orchestrator.run(
        [ChatMessage(role="user", content=text)],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
        **kwargs,
    )


def test_multi_step_plan_blocks_premature_final_until_all_tools_succeed() -> None:
    calculator = ToolCall("calc", "calculator", {"expression": "1+1"})
    search = ToolCall("search", "search_docs", {"query": "总部", "k": 1})
    model = _Model(
        [
            _turn(calls=(calculator,)),
            _turn("过早结束"),
            _turn(calls=(search,)),
            _turn("结果为 2，依据 doc-office。"),
        ]
    )
    registry = build_default_registry(
        documents=[{"id": "doc-office", "text": "总部信息：计算结果对应 2。"}]
    )
    orchestrator = AgentOrchestrator(
        model,
        registry,
        router=HeuristicAgentRouter(),
    )

    result = _run(orchestrator, "请查资料里的总部信息，再用计算器计算 1+1。")
    orchestrator.close()

    assert result.finish_reason == "stop"
    assert result.tool_calls == 2
    assert [trace.name for trace in result.trace] == ["calculator", "search_docs"]
    assert any("plan is incomplete" in item.content for item in model.messages[2])


def test_schema_argument_alias_is_repaired_once_before_execution() -> None:
    call = ToolCall("calc", "calculator", {"expr": "6*7"})
    model = _Model([_turn(calls=(call,)), _turn("42")])
    orchestrator = AgentOrchestrator(model, build_default_registry())

    result = _run(orchestrator, "请计算 6*7")
    orchestrator.close()

    assert result.trace[0].arguments == {"expression": "6*7"}
    assert result.trace[0].argument_repaired is True
    assert result.trace[0].content == "42"


def test_unsafe_or_ambiguous_argument_repair_is_not_guessed() -> None:
    call = ToolCall("calc", "calculator", {"left": 6, "right": 7})
    model = _Model([_turn(calls=(call,)), _turn("无法计算")])
    orchestrator = AgentOrchestrator(model, build_default_registry())

    result = _run(orchestrator, "请计算 6*7")
    orchestrator.close()

    assert result.trace[0].status == "error"
    assert result.trace[0].argument_repaired is False


def test_citation_guard_blocks_document_id_not_returned_this_run() -> None:
    model = _Model([_turn("根据 doc-invented，答案是 7。")])
    orchestrator = AgentOrchestrator(model, build_default_registry())

    result = _run(orchestrator, "直接回答")
    orchestrator.close()

    assert result.finish_reason == "citation_guard"
    assert result.citation is not None
    assert result.citation.unsupported == ("doc-invented",)


def test_citation_guard_accepts_only_search_result_ids() -> None:
    guard = CitationGuard()
    call = ToolCall("s", "search_docs", {"query": "x"})
    result = ToolExecutionResult(
        "s",
        "search_docs",
        "ok",
        json.dumps({"results": [{"citation": "doc-safe"}]}),
        1.0,
    )
    guard.record(call, result)

    validation = guard.validate("依据 doc-safe，拒绝 doc-fake。")

    assert validation.allowed == ("doc-safe",)
    assert validation.unsupported == ("doc-fake",)


def test_pre_cancelled_run_stops_before_model_or_tool() -> None:
    token = CancellationToken()
    token.cancel()
    model = _Model([_turn("不应执行")])
    orchestrator = AgentOrchestrator(model, build_default_registry())

    result = _run(orchestrator, "hello", cancellation_token=token)
    orchestrator.close()

    assert result.finish_reason == "cancelled"
    assert model.messages == []
    assert result.tool_calls == 0


def test_progress_and_usage_account_for_model_and_tool_events() -> None:
    call = ToolCall("calc", "calculator", {"expression": "2+3"})
    model = _Model([_turn(calls=(call,)), _turn("5")])
    orchestrator = AgentOrchestrator(model, build_default_registry())
    events: list[AgentProgressEvent] = []

    result = _run(orchestrator, "请计算 2+3", progress_callback=events.append)
    orchestrator.close()

    assert [event.kind for event in events] == [
        "route",
        "model",
        "tool",
        "model",
        "final",
    ]
    assert result.usage.prompt_tokens == 6
    assert result.usage.generated_tokens == 4
    assert result.usage.total_duration_ms >= result.usage.tool_duration_ms


def test_context_compaction_preserves_system_and_recent_user_turn() -> None:
    messages = [Message(role="system", content="规则")]
    messages.extend(
        Message(role="user" if index % 2 == 0 else "assistant", content="旧内容" * 100)
        for index in range(8)
    )
    messages.append(Message(role="user", content="最后的问题"))

    compacted = compact_messages(messages, max_characters=700)

    assert compacted[0].content == "规则"
    assert "omitted" in compacted[1].content
    assert compacted[-1].content == "最后的问题"


def test_context_compaction_rejects_unusable_budget() -> None:
    try:
        compact_messages([Message(role="user", content="x")], max_characters=100)
    except ValueError as error:
        assert "at least 512" in str(error)
    else:
        raise AssertionError("Expected an invalid context budget to fail")
