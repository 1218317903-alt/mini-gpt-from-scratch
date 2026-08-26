"""Tests for executing and persisting real-Agent evaluation predictions."""

from __future__ import annotations

import json
from pathlib import Path

from agent import build_default_registry
from agent.orchestrator import AgentResult, AgentTrace
from agent.tools.base import ToolCall
from agent.tracing import AgentEvent
from evals.agent_runner import (
    AgentEvalConfig,
    EvalToolExecutor,
    collect_corpus,
    prediction_from_result,
    run_cases_to_jsonl,
    select_cases,
)
from evals.metrics import load_cases

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals" / "cases"


def _result() -> AgentResult:
    return AgentResult(
        text="根据 doc-office，总部位于杭州。",
        model="fake",
        steps=2,
        tool_calls=1,
        finish_reason="stop",
        trace=(
            AgentTrace(
                step=1,
                call_id="call-1",
                name="search_docs",
                arguments={"query": "总部"},
                status="ok",
                content=json.dumps(
                    {"results": [{"citation": "doc-office", "snippet": "杭州"}]}
                ),
                duration_ms=1.0,
            ),
        ),
        events=(
            AgentEvent(1, "model", [], {"generated_tokens": 7}, 2.0),
            AgentEvent(1, "tool", {}, {}, 1.0),
            AgentEvent(2, "model", [], {"generated_tokens": 5}, 2.0),
        ),
    )


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *_args, **_kwargs) -> AgentResult:
        self.calls += 1
        return _result()


def test_collect_corpus_deduplicates_embedded_documents() -> None:
    cases = load_cases(CASES)

    corpus = collect_corpus(cases)

    ids = [document["id"] for document in corpus]
    assert "doc-office" in ids
    assert len(ids) == len(set(ids))


def test_prediction_collects_trace_tokens_retrieval_and_citations() -> None:
    case = next(case for case in load_cases(CASES) if case.id == "rt-001")

    prediction = prediction_from_result(case, _result(), duration_ms=12.5)

    assert prediction.tool_calls[0].name == "search_docs"
    assert prediction.retrieved_doc_ids == ("doc-office",)
    assert prediction.cited_doc_ids == ("doc-office",)
    assert prediction.generated_tokens == 12
    assert prediction.finish_reason == "stop"


def test_jsonl_run_is_incremental_and_resumable(tmp_path: Path) -> None:
    case = next(case for case in load_cases(CASES) if case.id == "rt-001")
    agent = _FakeAgent()
    output = tmp_path / "predictions.jsonl"

    first = run_cases_to_jsonl(
        [case],
        agent,  # type: ignore[arg-type]
        AgentEvalConfig(),
        output,
    )
    resumed = run_cases_to_jsonl(
        [case],
        agent,  # type: ignore[arg-type]
        AgentEvalConfig(),
        output,
        resume=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert (first, resumed, agent.calls) == (1, 0, 1)
    assert payload["case_id"] == "rt-001"


def test_select_cases_validates_ids_and_limit() -> None:
    cases = load_cases(CASES)

    assert [case.id for case in select_cases(cases, case_ids=["da-001"])] == ["da-001"]
    assert len(select_cases(cases, limit=3)) == 3


def test_eval_executor_injects_one_failure_then_runs_real_tool() -> None:
    case = next(case for case in load_cases(CASES) if case.id == "ms-011")
    executor = EvalToolExecutor(build_default_registry())
    call = ToolCall(
        call_id="call-1",
        name="calculator",
        arguments={"expression": "19*4"},
    )
    try:
        executor.begin_case(case)
        first = executor.execute(call)
        second = executor.execute(call)
    finally:
        executor.close()

    assert first.status == "error"
    assert "Injected evaluation failure" in first.content
    assert second.status == "ok"
    assert second.content == "76"
