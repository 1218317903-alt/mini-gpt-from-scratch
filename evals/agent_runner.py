"""Execute evaluation cases against the real bounded Agent loop."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.messages import as_message
from agent.orchestrator import AgentOrchestrator, AgentResult
from agent.tools.base import ToolCall, ToolExecutionResult
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from evals.metrics import EvalCase, PredictedCall, Prediction


@dataclass(frozen=True)
class AgentEvalConfig:
    """Deterministic generation settings used for one baseline run."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int | None = None
    top_p: float | None = None
    greedy: bool = True
    seed: int | None = 42


class EvalToolExecutor(ToolExecutor):
    """Inject one deterministic tool failure for recovery evaluation cases."""

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self._failure_tool: str | None = None
        self._failure_injected = False

    def begin_case(self, case: EvalCase) -> None:
        """Reset case-local fault state before an Agent run."""
        self._failure_tool = (
            case.expected_tools[0]
            if case.expect_recovery and case.expected_tools
            else None
        )
        self._failure_injected = False

    def execute(
        self,
        call: ToolCall,
        *,
        approved_side_effects: Collection[str] = (),
    ) -> ToolExecutionResult:
        """Fail the first expected call, then use the production executor."""
        if call.name == self._failure_tool and not self._failure_injected:
            self._failure_injected = True
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                content=(
                    "Injected evaluation failure: the tool failed transiently. "
                    "Retry with corrected arguments or the same call."
                ),
                duration_ms=0.0,
            )
        return super().execute(
            call,
            approved_side_effects=approved_side_effects,
        )


def collect_corpus(cases: Sequence[EvalCase]) -> tuple[dict[str, str], ...]:
    """Merge self-contained retrieval corpora, rejecting conflicting IDs."""
    documents: dict[str, str] = {}
    for case in cases:
        if case.retrieval is None:
            continue
        for document in case.retrieval.corpus:
            doc_id = document["id"]
            text = document["text"]
            previous = documents.get(doc_id)
            if previous is not None and previous != text:
                raise ValueError(f"Conflicting eval document: {doc_id}")
            documents[doc_id] = text
    return tuple(
        {"id": doc_id, "text": documents[doc_id]} for doc_id in sorted(documents)
    )


def run_case(
    case: EvalCase,
    agent: AgentOrchestrator,
    config: AgentEvalConfig,
) -> Prediction:
    """Run one case and convert the Agent result to the metrics contract."""
    messages = [as_message(message) for message in case.messages]
    begin_case = getattr(getattr(agent, "executor", None), "begin_case", None)
    if callable(begin_case):
        begin_case(case)
    started = perf_counter()
    try:
        result = agent.run(
            messages,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            greedy=config.greedy,
            seed=config.seed,
            max_steps=case.max_steps,
        )
    except Exception as error:
        duration_ms = (perf_counter() - started) * 1000
        return Prediction(
            case_id=case.id,
            text="",
            duration_ms=round(duration_ms, 3),
            error=f"{type(error).__name__}: {error}",
        )
    duration_ms = (perf_counter() - started) * 1000
    return prediction_from_result(case, result, duration_ms=duration_ms)


def prediction_from_result(
    case: EvalCase,
    result: AgentResult,
    *,
    duration_ms: float,
) -> Prediction:
    calls = tuple(
        PredictedCall(
            name=trace.name,
            arguments=dict(trace.arguments),
            status=trace.status,
        )
        for trace in result.trace
    )
    retrieved = _retrieved_doc_ids(result)
    cited = tuple(sorted(set(re.findall(r"doc-[a-z0-9-]+", result.text))))
    generated_tokens = 0
    for event in result.events:
        if event.phase != "model" or not isinstance(event.output, dict):
            continue
        value = event.output.get("generated_tokens", 0)
        if isinstance(value, int):
            generated_tokens += value
    recovered = None
    if case.expect_recovery or case.success_mode == "recovery":
        recovered = any(
            first.status == "error"
            and any(
                later.name == first.name and later.status == "ok"
                for later in calls[index + 1 :]
            )
            for index, first in enumerate(calls)
        )
    return Prediction(
        case_id=case.id,
        text=result.text,
        tool_calls=calls,
        retrieved_doc_ids=retrieved,
        cited_doc_ids=cited,
        duration_ms=round(duration_ms, 3),
        generated_tokens=generated_tokens,
        recovered=recovered,
        finish_reason=result.finish_reason,
    )


def _retrieved_doc_ids(result: AgentResult) -> tuple[str, ...]:
    ids: list[str] = []
    for trace in result.trace:
        if trace.name != "search_docs" or trace.status != "ok":
            continue
        try:
            payload = json.loads(trace.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            citation = item.get("citation")
            if isinstance(citation, str) and citation not in ids:
                ids.append(citation)
    return tuple(ids)


def prediction_payload(prediction: Prediction) -> dict[str, Any]:
    """Convert one prediction to stable JSONL output."""
    return asdict(prediction)


def load_completed_ids(path: Path) -> set[str]:
    """Read existing result IDs for resumable runs."""
    if not path.exists():
        return set()
    completed: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("case_id"), str):
            completed.add(payload["case_id"])
    return completed


def run_cases_to_jsonl(
    cases: Sequence[EvalCase],
    agent: AgentOrchestrator,
    config: AgentEvalConfig,
    output: Path,
    *,
    resume: bool = False,
) -> int:
    """Run selected cases and persist each result immediately."""
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output) if resume else set()
    mode = "a" if resume else "w"
    written = 0
    with output.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, case in enumerate(cases, start=1):
            if case.id in completed:
                continue
            prediction = run_case(case, agent, config)
            handle.write(
                json.dumps(
                    prediction_payload(prediction),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            written += 1
            progress = {
                "event": "agent_eval_case",
                "index": index,
                "total": len(cases),
                "case_id": case.id,
                "tool_calls": len(prediction.tool_calls),
                "duration_ms": prediction.duration_ms,
                "error": prediction.error,
            }
            print(json.dumps(progress, ensure_ascii=False), file=sys.stderr)
    return written


def select_cases(
    cases: Sequence[EvalCase],
    *,
    case_ids: Iterable[str] = (),
    limit: int | None = None,
) -> list[EvalCase]:
    """Select deterministic subsets for smoke tests and resumable baselines."""
    requested = set(case_ids)
    selected = [case for case in cases if not requested or case.id in requested]
    if requested:
        missing = requested - {case.id for case in selected}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    return selected if limit is None else selected[:limit]
