"""Tests for P6 concurrency, cancellation, and aggregate metrics."""

from agent.orchestrator import AgentFinishReason, AgentResult, AgentUsage
from minigpt_service.agent_runtime import AgentRunManager


def _result(
    reason: AgentFinishReason = "stop",
    duration: float = 12.5,
) -> AgentResult:
    return AgentResult(
        text="done",
        model="fake",
        steps=1,
        tool_calls=0,
        finish_reason=reason,
        trace=(),
        usage=AgentUsage(total_duration_ms=duration),
    )


def test_manager_rejects_work_above_concurrency_limit() -> None:
    manager = AgentRunManager(max_concurrency=1)

    assert manager.start("one") is not None
    assert manager.start("two") is None
    assert manager.snapshot().rejected_busy == 1


def test_manager_cancels_and_releases_run() -> None:
    manager = AgentRunManager(max_concurrency=1)
    token = manager.start("one")
    assert token is not None

    assert manager.cancel("one") is True
    assert token.is_cancelled is True
    manager.finish("one", result=_result("cancelled"))

    snapshot = manager.snapshot()
    assert snapshot.running == 0
    assert snapshot.cancelled == 1


def test_manager_records_completed_average_duration() -> None:
    manager = AgentRunManager(max_concurrency=1)
    manager.start("one")
    manager.finish("one", result=_result(duration=20))

    snapshot = manager.snapshot()
    assert snapshot.completed == 1
    assert snapshot.avg_duration_ms == 20


def test_cancelling_unknown_run_returns_false() -> None:
    assert AgentRunManager().cancel("missing") is False
