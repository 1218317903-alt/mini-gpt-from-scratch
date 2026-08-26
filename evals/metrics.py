"""Agent evaluation metrics for the stage-two case set."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TARGETS: dict[str, float] = {
    "json_valid_rate": 0.95,
    "single_tool_success_rate": 0.85,
    "unnecessary_tool_rate": 0.10,
    "rag_recall_at_5": 0.85,
    "recovery_rate": 0.70,
}

INTERVAL_METRICS = (
    "tool_selection_accuracy",
    "json_valid_rate",
    "argument_accuracy",
    "task_success_rate",
    "single_tool_success_rate",
    "unnecessary_tool_rate",
    "retrieval_trigger_precision",
    "retrieval_trigger_recall",
    "end_to_end_rag_recall_at_5",
    "conditional_retrieval_recall_at_5",
    "citation_precision",
    "citation_recall",
    "unsupported_answer_rate",
    "recovery_rate",
)


@dataclass(frozen=True)
class RetrievalSpec:
    """Self-contained corpus used by one retrieval case."""

    query: str
    gold_doc_ids: tuple[str, ...]
    corpus: tuple[dict[str, str], ...]
    k: int = 5


@dataclass(frozen=True)
class EvalCase:
    """One human-reviewed Agent evaluation case."""

    id: str
    category: str
    messages: tuple[dict[str, str], ...]
    expect_tool: bool
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_arguments: dict[str, dict[str, Any]] | None = None
    gold_answer: str | None = None
    answer_contains: tuple[str, ...] = ()
    success_mode: str = "answer_contains"
    citations: tuple[str, ...] = ()
    retrieval: RetrievalSpec | None = None
    max_steps: int = 6
    min_tool_calls: int = 0
    expect_recovery: bool = False
    subcategory: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PredictedCall:
    """One tool invocation recorded by a model or fixture."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


@dataclass(frozen=True)
class Prediction:
    """Model output aligned to one evaluation case."""

    case_id: str
    text: str
    tool_calls: tuple[PredictedCall, ...] = ()
    retrieved_doc_ids: tuple[str, ...] = ()
    cited_doc_ids: tuple[str, ...] = ()
    duration_ms: float = 0.0
    generated_tokens: int = 0
    recovered: bool | None = None
    finish_reason: str | None = None
    error: str | None = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


def case_from_dict(payload: dict[str, Any]) -> EvalCase:
    """Parse one evaluation case from a JSON object."""
    retrieval_payload = payload.get("retrieval")
    retrieval = None
    if isinstance(retrieval_payload, dict):
        corpus_items = tuple(
            {"id": str(item["id"]), "text": str(item["text"])}
            for item in retrieval_payload.get("corpus", [])
        )
        retrieval = RetrievalSpec(
            query=str(retrieval_payload.get("query", "")),
            gold_doc_ids=tuple(
                str(item) for item in retrieval_payload.get("gold_doc_ids", [])
            ),
            corpus=corpus_items,
            k=int(retrieval_payload.get("k", 5)),
        )
    expected_arguments = payload.get("expected_arguments")
    return EvalCase(
        id=str(payload["id"]),
        category=str(payload["category"]),
        subcategory=(
            str(payload["subcategory"])
            if payload.get("subcategory") is not None
            else None
        ),
        messages=tuple(payload.get("messages", ())),
        expect_tool=bool(payload.get("expect_tool", False)),
        expected_tools=tuple(payload.get("expected_tools", ())),
        forbidden_tools=tuple(payload.get("forbidden_tools", ())),
        expected_arguments=(
            expected_arguments if isinstance(expected_arguments, dict) else None
        ),
        gold_answer=payload.get("gold_answer"),
        answer_contains=tuple(payload.get("answer_contains", ())),
        success_mode=str(payload.get("success_mode", "answer_contains")),
        citations=tuple(payload.get("citations", ())),
        retrieval=retrieval,
        max_steps=int(payload.get("max_steps", 6)),
        min_tool_calls=int(payload.get("min_tool_calls", 0)),
        expect_recovery=bool(payload.get("expect_recovery", False)),
        notes=str(payload.get("notes", "")),
    )


def prediction_from_dict(payload: dict[str, Any]) -> Prediction:
    """Parse one recorded prediction."""
    calls = []
    for item in payload.get("tool_calls", []):
        arguments = item.get("arguments", {})
        calls.append(
            PredictedCall(
                name=str(item.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
                status=str(item.get("status", "ok")),
            )
        )
    return Prediction(
        case_id=str(payload["case_id"]),
        text=str(payload.get("text", "")),
        tool_calls=tuple(calls),
        retrieved_doc_ids=tuple(payload.get("retrieved_doc_ids", ())),
        cited_doc_ids=tuple(payload.get("cited_doc_ids", ())),
        duration_ms=float(payload.get("duration_ms", 0.0)),
        generated_tokens=int(payload.get("generated_tokens", 0)),
        recovered=payload.get("recovered"),
        finish_reason=(
            str(payload["finish_reason"])
            if payload.get("finish_reason") is not None
            else None
        ),
        error=str(payload["error"]) if payload.get("error") is not None else None,
    )


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every JSONL case file in deterministic filename order."""
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        for payload in load_jsonl(path):
            cases.append(case_from_dict(payload))
    return cases


def keyword_retrieve(
    query: str,
    corpus: Sequence[Mapping[str, str]],
    *,
    k: int = 5,
) -> list[str]:
    """Rank documents by overlapping CJK/word tokens; used as Recall@5 baseline."""
    query_tokens = _tokenize(query)
    scored: list[tuple[float, str]] = []
    for document in corpus:
        doc_id = str(document["id"])
        text = str(document["text"])
        doc_tokens = _tokenize(text)
        if not query_tokens or not doc_tokens:
            score = 0.0
        else:
            overlap = len(query_tokens & doc_tokens)
            score = overlap / math.sqrt(len(query_tokens) * len(doc_tokens))
        scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for score, doc_id in scored[:k] if score > 0]


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    tokens.update(character for character in text if "\u4e00" <= character <= "\u9fff")
    return tokens


def _tool_names(prediction: Prediction) -> list[str]:
    return [call.name for call in prediction.tool_calls if call.name]


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("{", "[", "```", "<tool_call>"))


def json_is_valid(prediction: Prediction) -> bool | None:
    """Return JSON validity when the model attempted a structured call."""
    if prediction.tool_calls:
        return all(bool(call.name) for call in prediction.tool_calls)
    if _looks_like_json(prediction.text):
        try:
            json.loads(prediction.text.strip().removeprefix("```json").strip("` \n"))
        except json.JSONDecodeError:
            return False
        return True
    return None


def arguments_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Compare expected argument values after light normalization."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if _normalize(actual[key]) != _normalize(value):
            return False
    return True


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        compact = re.sub(r"\s+", "", value).lower()
        compact = compact.replace("x", "*").replace("\u00d7", "*")
        return compact
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def case_succeeded(case: EvalCase, prediction: Prediction) -> bool:
    """Apply the case-specific success rule."""
    names = _tool_names(prediction)
    if case.success_mode == "tool_only":
        return bool(case.expected_tools) and names[:1] == list(case.expected_tools[:1])
    if case.success_mode == "tool_and_answer":
        if case.expected_tools and not set(case.expected_tools).issubset(set(names)):
            return False
        if case.expected_arguments and not _arguments_ok(case, prediction):
            return False
        return _contains_all(prediction.text, case.answer_contains)
    if case.success_mode == "cite_and_answer":
        cited = set(prediction.cited_doc_ids) or _citations_from_text(prediction.text)
        gold = set(case.citations) or (
            set(case.retrieval.gold_doc_ids) if case.retrieval else set()
        )
        return bool(gold & cited) and _contains_all(
            prediction.text, case.answer_contains
        )
    if case.success_mode == "recovery":
        if prediction.recovered is not None:
            return bool(prediction.recovered)
        return any(call.status == "error" for call in prediction.tool_calls) and (
            any(call.status == "ok" for call in prediction.tool_calls)
            or _contains_all(prediction.text, case.answer_contains)
        )
    if case.expect_tool is False and names:
        return False
    return _contains_all(prediction.text, case.answer_contains)


def _arguments_ok(case: EvalCase, prediction: Prediction) -> bool:
    if not case.expected_arguments:
        return True
    by_name = {call.name: call.arguments for call in prediction.tool_calls}
    for name, expected in case.expected_arguments.items():
        actual = by_name.get(name)
        if actual is None or not arguments_match(expected, actual):
            return False
    return True


def _contains_all(text: str, needles: Sequence[str]) -> bool:
    if not needles:
        return True
    return all(needle in text for needle in needles)


def _citations_from_text(text: str) -> set[str]:
    return set(re.findall(r"doc-[a-z0-9-]+", text))


def compute_metrics(
    cases: Sequence[EvalCase],
    predictions: Mapping[str, Prediction],
) -> dict[str, Any]:
    """Compute the stage-two Agent metrics from aligned predictions."""
    selection_hits = 0
    selection_total = 0
    json_hits = 0
    json_total = 0
    argument_hits = 0
    argument_total = 0
    success_hits = 0
    success_total = 0
    single_tool_hits = 0
    single_tool_total = 0
    unnecessary = 0
    no_tool_total = 0
    recall_scores: list[float] = []
    recall_at_1_scores: list[float] = []
    recall_at_3_scores: list[float] = []
    conditional_recall_at_1_scores: list[float] = []
    conditional_recall_at_3_scores: list[float] = []
    conditional_recall_at_5_scores: list[float] = []
    baseline_recall_scores: list[float] = []
    retrieval_required = 0
    retrieval_triggered = 0
    retrieval_trigger_true_positives = 0
    citation_hits = 0
    citation_total = 0
    citation_true_positives = 0
    citation_predictions = 0
    citation_expected = 0
    ungrounded = 0
    retrieval_total = 0
    recovery_hits = 0
    recovery_total = 0
    multi_step_calls: list[int] = []
    durations: list[float] = []
    tokens: list[int] = []

    for case in cases:
        prediction = predictions.get(case.id)
        if prediction is None:
            prediction = Prediction(case_id=case.id, text="")
        names = _tool_names(prediction)
        search_triggered = "search_docs" in names
        if search_triggered:
            retrieval_triggered += 1
        durations.append(prediction.duration_ms)
        tokens.append(prediction.generated_tokens)

        validity = json_is_valid(prediction)
        if validity is not None:
            json_total += 1
            json_hits += int(validity)

        if case.expect_tool and case.expected_tools:
            selection_total += 1
            if case.category == "multi_step":
                selected = set(case.expected_tools).issubset(set(names))
            else:
                selected = names[:1] == [case.expected_tools[0]]
            selection_hits += int(selected)
        if not case.expect_tool:
            no_tool_total += 1
            unnecessary += int(bool(names))

        if case.expected_arguments:
            argument_total += 1
            argument_hits += int(_arguments_ok(case, prediction))

        success_total += 1
        succeeded = case_succeeded(case, prediction)
        success_hits += int(succeeded)
        if (
            case.category in {"tool_selection", "tool_arguments"}
            and case.expect_tool
            and case.subcategory != "distractor"
        ):
            single_tool_total += 1
            single_tool_hits += int(succeeded)

        if case.category == "multi_step":
            multi_step_calls.append(len(prediction.tool_calls))
        if case.expect_recovery or case.success_mode == "recovery":
            recovery_total += 1
            recovery_hits += int(case_succeeded(case, prediction))

        if case.retrieval is not None:
            retrieval_total += 1
            retrieval_required += 1
            retrieval_trigger_true_positives += int(search_triggered)
            retrieved = list(prediction.retrieved_doc_ids)
            gold = set(case.retrieval.gold_doc_ids)
            recall_at_1 = _retrieval_recall(gold, retrieved, k=1)
            recall_at_3 = _retrieval_recall(gold, retrieved, k=3)
            recall_at_5 = _retrieval_recall(
                gold,
                retrieved,
                k=min(5, case.retrieval.k),
            )
            recall_at_1_scores.append(recall_at_1)
            recall_at_3_scores.append(recall_at_3)
            recall_scores.append(recall_at_5)
            if search_triggered:
                conditional_recall_at_1_scores.append(recall_at_1)
                conditional_recall_at_3_scores.append(recall_at_3)
                conditional_recall_at_5_scores.append(recall_at_5)
            baseline_retrieved = keyword_retrieve(
                case.retrieval.query or _user_text(case),
                case.retrieval.corpus,
                k=case.retrieval.k,
            )
            baseline_hit = len(
                gold.intersection(baseline_retrieved[: case.retrieval.k])
            )
            baseline_recall_scores.append(baseline_hit / len(gold) if gold else 0.0)
            cited = set(prediction.cited_doc_ids) or _citations_from_text(
                prediction.text
            )
            expected_citations = set(case.citations) or gold
            citation_total += 1
            citation_hits += int(bool(expected_citations & cited))
            citation_true_positives += len(expected_citations & cited)
            citation_expected += len(expected_citations)
            if prediction.text.strip() and not (expected_citations & cited):
                ungrounded += 1
        cited = set(prediction.cited_doc_ids) or _citations_from_text(prediction.text)
        citation_predictions += len(cited)

    def _ratio(hits: int, total: int) -> float | None:
        if total == 0:
            return None
        return hits / total

    return {
        "cases_scored": success_total,
        "tool_selection_accuracy": _ratio(selection_hits, selection_total),
        "json_valid_rate": _ratio(json_hits, json_total),
        "argument_accuracy": _ratio(argument_hits, argument_total),
        "task_success_rate": _ratio(success_hits, success_total),
        "single_tool_success_rate": _ratio(single_tool_hits, single_tool_total),
        "unnecessary_tool_rate": _ratio(unnecessary, no_tool_total),
        "multi_step_avg_calls": (
            sum(multi_step_calls) / len(multi_step_calls) if multi_step_calls else None
        ),
        "retrieval_trigger_precision": _ratio(
            retrieval_trigger_true_positives,
            retrieval_triggered,
        ),
        "retrieval_trigger_recall": _ratio(
            retrieval_trigger_true_positives,
            retrieval_required,
        ),
        "end_to_end_rag_recall_at_1": _mean_or_none(recall_at_1_scores),
        "end_to_end_rag_recall_at_3": _mean_or_none(recall_at_3_scores),
        "end_to_end_rag_recall_at_5": _mean_or_none(recall_scores),
        "conditional_retrieval_recall_at_1": _mean_or_none(
            conditional_recall_at_1_scores
        ),
        "conditional_retrieval_recall_at_3": _mean_or_none(
            conditional_recall_at_3_scores
        ),
        "conditional_retrieval_recall_at_5": _mean_or_none(
            conditional_recall_at_5_scores
        ),
        "rag_recall_at_5": (
            sum(recall_scores) / len(recall_scores) if recall_scores else None
        ),
        "baseline_rag_recall_at_5": (
            sum(baseline_recall_scores) / len(baseline_recall_scores)
            if baseline_recall_scores
            else None
        ),
        "citation_accuracy": _ratio(citation_hits, citation_total),
        "citation_precision": _ratio(
            citation_true_positives,
            citation_predictions,
        ),
        "citation_recall": _ratio(citation_true_positives, citation_expected),
        "ungrounded_rate": _ratio(ungrounded, retrieval_total),
        "unsupported_answer_rate": _ratio(ungrounded, retrieval_total),
        "recovery_rate": _ratio(recovery_hits, recovery_total),
        "avg_duration_ms": sum(durations) / len(durations) if durations else None,
        "avg_generated_tokens": sum(tokens) / len(tokens) if tokens else None,
        "targets": TARGETS,
    }


def _retrieval_recall(gold: set[str], retrieved: Sequence[str], *, k: int) -> float:
    """Compute recall for one retrieval case at a fixed cutoff."""
    if not gold:
        return 0.0
    return len(gold.intersection(retrieved[:k])) / len(gold)


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, preserving an empty metric as unavailable."""
    if not values:
        return None
    return sum(values) / len(values)


def bootstrap_confidence_intervals(
    cases: Sequence[EvalCase],
    predictions: Mapping[str, Prediction],
    *,
    metric_names: Sequence[str] = INTERVAL_METRICS,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int | None]]:
    """Estimate deterministic case-bootstrap intervals for selected metrics."""
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    point_metrics = compute_metrics(cases, predictions)
    distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    if cases:
        generator = random.Random(seed)
        for _ in range(samples):
            sampled = [cases[generator.randrange(len(cases))] for _ in cases]
            sampled_metrics = compute_metrics(sampled, predictions)
            for name in metric_names:
                value = sampled_metrics.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    distributions[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = sorted(distributions[name])
        point = point_metrics.get(name)
        intervals[name] = {
            "point": (
                float(point)
                if isinstance(point, int | float) and not isinstance(point, bool)
                else None
            ),
            "lower": _percentile(values, alpha) if values else None,
            "upper": _percentile(values, 1.0 - alpha) if values else None,
            "confidence": confidence,
            "valid_samples": len(values),
            "requested_samples": samples,
        }
    return intervals


def bootstrap_metric_delta_intervals(
    cases: Sequence[EvalCase],
    baseline_predictions: Mapping[str, Prediction],
    candidate_predictions: Mapping[str, Prediction],
    *,
    metric_names: Sequence[str] = INTERVAL_METRICS,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int | None]]:
    """Estimate paired-bootstrap intervals for candidate-minus-baseline deltas."""
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    baseline_point = compute_metrics(cases, baseline_predictions)
    candidate_point = compute_metrics(cases, candidate_predictions)
    distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    if cases:
        generator = random.Random(seed)
        for _ in range(samples):
            sampled = [cases[generator.randrange(len(cases))] for _ in cases]
            baseline_metrics = compute_metrics(sampled, baseline_predictions)
            candidate_metrics = compute_metrics(sampled, candidate_predictions)
            for name in metric_names:
                before = baseline_metrics.get(name)
                after = candidate_metrics.get(name)
                if (
                    isinstance(before, int | float)
                    and not isinstance(before, bool)
                    and isinstance(after, int | float)
                    and not isinstance(after, bool)
                ):
                    distributions[name].append(float(after) - float(before))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = sorted(distributions[name])
        before = baseline_point.get(name)
        after = candidate_point.get(name)
        point = (
            float(after) - float(before)
            if isinstance(before, int | float)
            and not isinstance(before, bool)
            and isinstance(after, int | float)
            and not isinstance(after, bool)
            else None
        )
        intervals[name] = {
            "point_delta": point,
            "lower": _percentile(values, alpha) if values else None,
            "upper": _percentile(values, 1.0 - alpha) if values else None,
            "confidence": confidence,
            "valid_samples": len(values),
            "requested_samples": samples,
        }
    return intervals


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolate one percentile from a sorted numeric sequence."""
    if not values:
        raise ValueError("values must not be empty")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def evaluate_targets(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Compare computed metrics with the first-version targets."""
    checks = {
        "json_valid_rate": lambda value: (
            value is not None and value >= TARGETS["json_valid_rate"]
        ),
        "single_tool_success_rate": lambda value: (
            value is not None and value >= TARGETS["single_tool_success_rate"]
        ),
        "unnecessary_tool_rate": lambda value: (
            value is not None and value <= TARGETS["unnecessary_tool_rate"]
        ),
        "rag_recall_at_5": lambda value: (
            value is not None and value >= TARGETS["rag_recall_at_5"]
        ),
        "recovery_rate": lambda value: (
            value is not None and value >= TARGETS["recovery_rate"]
        ),
    }
    return {name: predicate(metrics.get(name)) for name, predicate in checks.items()}


def _user_text(case: EvalCase) -> str:
    for message in case.messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def training_user_texts(paths: Iterable[Path]) -> set[str]:
    """Collect normalized user utterances from SFT JSONL files."""
    texts: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for payload in load_jsonl(path):
            for message in payload.get("messages", []):
                if message.get("role") == "user":
                    texts.add(_normalize_text(str(message.get("content", ""))))
    return texts


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def overlap_with_training(
    cases: Sequence[EvalCase], training_texts: set[str]
) -> list[str]:
    """Return case IDs whose user text matches a training/validation prompt."""
    overlapped: list[str] = []
    for case in cases:
        user_text = _normalize_text(_user_text(case))
        if user_text and user_text in training_texts:
            overlapped.append(case.id)
    return overlapped
