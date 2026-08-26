"""Concurrency and cancellation for persistent conversation generation."""

from __future__ import annotations

from threading import BoundedSemaphore, Lock

from agent.reliability import CancellationToken


class ConversationRunManager:
    """Allow one active generation per conversation within a bounded pool."""

    def __init__(self, *, max_concurrency: int = 2) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")
        self._slots = BoundedSemaphore(max_concurrency)
        self._lock = Lock()
        self._tokens: dict[str, CancellationToken] = {}
        self._conversation_runs: dict[str, str] = {}

    def start(
        self,
        run_id: str,
        conversation_id: str,
    ) -> CancellationToken | None:
        """Reserve a slot or reject duplicate/busy generation."""
        if not self._slots.acquire(blocking=False):
            return None
        token = CancellationToken()
        with self._lock:
            if run_id in self._tokens or conversation_id in self._conversation_runs:
                self._slots.release()
                raise ValueError("Conversation already has an active generation.")
            self._tokens[run_id] = token
            self._conversation_runs[conversation_id] = run_id
        return token

    def cancel(self, run_id: str) -> bool:
        """Request cooperative cancellation for an active run."""
        with self._lock:
            token = self._tokens.get(run_id)
        if token is None:
            return False
        token.cancel()
        return True

    def finish(self, run_id: str, conversation_id: str) -> None:
        """Release a run exactly once."""
        with self._lock:
            token = self._tokens.pop(run_id, None)
            self._conversation_runs.pop(conversation_id, None)
        if token is not None:
            self._slots.release()
