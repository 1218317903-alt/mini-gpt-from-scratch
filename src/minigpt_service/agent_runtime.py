"""Concurrency, cancellation, and aggregate metrics for Agent requests."""

from __future__ import annotations

from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from agent.orchestrator import AgentResult
from agent.reliability import CancellationToken


@dataclass(frozen=True)
class AgentRuntimeSnapshot:
    """Small process-local service snapshot without prompt contents."""

    running: int
    accepted: int
    completed: int
    cancelled: int
    failed: int
    rejected_busy: int
    avg_duration_ms: float


class AgentRunManager:
    """Bound concurrent Agent work and expose cooperative cancellation."""

    def __init__(self, *, max_concurrency: int = 2) -> None:
        if type(max_concurrency) is not int or max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        self._slots = BoundedSemaphore(max_concurrency)
        self._lock = Lock()
        self._tokens: dict[str, CancellationToken] = {}
        self._accepted = 0
        self._completed = 0
        self._cancelled = 0
        self._failed = 0
        self._rejected_busy = 0
        self._duration_total_ms = 0.0

    def start(self, run_id: str) -> CancellationToken | None:
        """Reserve a slot without queueing indefinitely."""
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected_busy += 1
            return None
        token = CancellationToken()
        with self._lock:
            if run_id in self._tokens:
                self._slots.release()
                raise ValueError(f"Agent run already exists: {run_id}")
            self._tokens[run_id] = token
            self._accepted += 1
        return token

    def cancel(self, run_id: str) -> bool:
        """Request cancellation for one active run."""
        with self._lock:
            token = self._tokens.get(run_id)
        if token is None:
            return False
        token.cancel()
        return True

    def finish(
        self,
        run_id: str,
        *,
        result: AgentResult | None = None,
        failed: bool = False,
    ) -> None:
        """Release a run exactly once and update aggregate counters."""
        with self._lock:
            token = self._tokens.pop(run_id, None)
            if token is None:
                return
            if failed:
                self._failed += 1
            elif result is not None and result.finish_reason == "cancelled":
                self._cancelled += 1
            else:
                self._completed += 1
            if result is not None:
                self._duration_total_ms += result.usage.total_duration_ms
        self._slots.release()

    def snapshot(self) -> AgentRuntimeSnapshot:
        """Return aggregate operational metrics without sensitive content."""
        with self._lock:
            finished = self._completed + self._cancelled
            average = self._duration_total_ms / finished if finished else 0.0
            return AgentRuntimeSnapshot(
                running=len(self._tokens),
                accepted=self._accepted,
                completed=self._completed,
                cancelled=self._cancelled,
                failed=self._failed,
                rejected_busy=self._rejected_busy,
                avg_duration_ms=round(average, 3),
            )
