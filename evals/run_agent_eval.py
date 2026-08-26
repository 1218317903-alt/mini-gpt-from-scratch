"""Validate and score the stage-two Agent evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.inventory import load_inventory, validate_inventory  # noqa: E402
from evals.metrics import (  # noqa: E402
    EvalCase,
    Prediction,
    arguments_match,
    bootstrap_confidence_intervals,
    bootstrap_metric_delta_intervals,
    case_succeeded,
    compute_metrics,
    evaluate_targets,
    keyword_retrieve,
    load_cases,
    load_jsonl,
    overlap_with_training,
    prediction_from_dict,
    training_user_texts,
)

DEFAULT_CASES = Path(__file__).resolve().parent / "cases"
TRAIN_PATHS = (
    ROOT / "hf_lora" / "data" / "train.jsonl",
    ROOT / "hf_lora" / "data" / "validation.jsonl",
)
EXPECTED_COUNTS = {
    "direct_answer": 20,
    "tool_selection": 35,
    "tool_arguments": 15,
    "retrieval": 15,
    "multi_step": 15,
}


def validate_cases(cases_dir: Path) -> dict[str, Any]:
    """Check inventory, independence from SFT data, and retrieval quality."""
    cases = load_cases(cases_dir)
    inventory = load_inventory(cases_dir)
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    counts = Counter(case.category for case in cases)
    expected_counts = inventory.expected_counts if inventory else EXPECTED_COUNTS
    count_errors = {
        category: {"expected": expected, "actual": counts.get(category, 0)}
        for category, expected in expected_counts.items()
        if counts.get(category, 0) != expected
    }
    unexpected_categories = sorted(set(counts) - set(expected_counts))
    overlapped = overlap_with_training(cases, training_user_texts(TRAIN_PATHS))

    retrieval_errors: list[str] = []
    recall_failures: list[str] = []
    for case in cases:
        if case.retrieval is None:
            continue
        corpus_ids = {item["id"] for item in case.retrieval.corpus}
        missing = [
            doc_id for doc_id in case.retrieval.gold_doc_ids if doc_id not in corpus_ids
        ]
        if missing:
            retrieval_errors.append(f"{case.id}: missing gold docs {missing}")
            continue
        retrieved = keyword_retrieve(
            case.retrieval.query
            or next(
                (
                    message.get("content", "")
                    for message in case.messages
                    if message.get("role") == "user"
                ),
                "",
            ),
            case.retrieval.corpus,
            k=case.retrieval.k,
        )
        if not set(case.retrieval.gold_doc_ids).issubset(set(retrieved)):
            recall_failures.append(case.id)

    empty_user = [
        case.id
        for case in cases
        if not any(message.get("role") == "user" for message in case.messages)
    ]
    minimum_cases = inventory.minimum_cases if inventory else 80
    maximum_cases = inventory.maximum_cases if inventory else 120
    manifest_errors = validate_inventory(cases_dir, inventory) if inventory else []
    ok = not (
        duplicates
        or count_errors
        or unexpected_categories
        or overlapped
        or retrieval_errors
        or recall_failures
        or empty_user
        or manifest_errors
        or not (minimum_cases <= len(cases) <= maximum_cases)
    )
    return {
        "ok": ok,
        "case_count": len(cases),
        "counts": dict(counts),
        "duplicates": duplicates,
        "count_errors": count_errors,
        "unexpected_categories": unexpected_categories,
        "training_overlap": overlapped,
        "retrieval_errors": retrieval_errors,
        "recall_failures": recall_failures,
        "empty_user": empty_user,
        "manifest": (
            {
                "split": inventory.split,
                "version": inventory.version,
                "purpose": inventory.purpose,
                "frozen": inventory.frozen,
                "fingerprint": inventory.fingerprint,
                "review_status": inventory.review_status,
                "errors": manifest_errors,
            }
            if inventory
            else None
        ),
    }


def score_predictions(
    cases_dir: Path,
    predictions_path: Path,
    *,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Score a recorded prediction file against the eval cases."""
    cases = load_cases(cases_dir)
    if case_ids:
        cases = [case for case in cases if case.id in case_ids]
        missing_case_ids = sorted(case_ids - {case.id for case in cases})
        if missing_case_ids:
            raise ValueError(f"Unknown case IDs: {missing_case_ids}")
    rows = load_jsonl(predictions_path)
    prediction_ids = [str(item.get("case_id", "")) for item in rows]
    duplicates = sorted(
        case_id
        for case_id, count in Counter(prediction_ids).items()
        if case_id and count > 1
    )
    case_ids = {case.id for case in cases}
    unknown = sorted(set(prediction_ids) - case_ids - {""})
    predictions: dict[str, Prediction] = {}
    for item in rows:
        prediction = prediction_from_dict(item)
        predictions.setdefault(prediction.case_id, prediction)
    missing = [case.id for case in cases if case.id not in predictions]
    metrics = compute_metrics(cases, predictions)
    return {
        "ok": not duplicates and not unknown and not missing,
        "metrics": metrics,
        "confidence_intervals": bootstrap_confidence_intervals(cases, predictions),
        "targets": evaluate_targets(metrics),
        "failure_analysis": analyze_predictions(cases, predictions),
        "missing_predictions": missing,
        "duplicate_predictions": duplicates,
        "unknown_predictions": unknown,
    }


def compare_prediction_files(
    cases_dir: Path,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Compare two complete prediction files with metric and case-level deltas."""
    cases = load_cases(cases_dir)
    baseline = _prediction_map(baseline_path)
    candidate = _prediction_map(candidate_path)
    expected_ids = {case.id for case in cases}
    baseline_missing = sorted(expected_ids - set(baseline))
    candidate_missing = sorted(expected_ids - set(candidate))
    if baseline_missing or candidate_missing:
        return {
            "ok": False,
            "baseline_missing": baseline_missing,
            "candidate_missing": candidate_missing,
        }
    baseline_metrics = compute_metrics(cases, baseline)
    candidate_metrics = compute_metrics(cases, candidate)
    deltas: dict[str, float] = {}
    for name, baseline_value in baseline_metrics.items():
        candidate_value = candidate_metrics.get(name)
        if (
            isinstance(baseline_value, int | float)
            and not isinstance(baseline_value, bool)
            and isinstance(candidate_value, int | float)
            and not isinstance(candidate_value, bool)
        ):
            deltas[name] = float(candidate_value) - float(baseline_value)

    improved: list[str] = []
    regressed: list[str] = []
    unchanged_pass: list[str] = []
    unchanged_fail: list[str] = []
    for case in cases:
        before = case_succeeded(case, baseline[case.id])
        after = case_succeeded(case, candidate[case.id])
        if not before and after:
            improved.append(case.id)
        elif before and not after:
            regressed.append(case.id)
        elif before:
            unchanged_pass.append(case.id)
        else:
            unchanged_fail.append(case.id)

    baseline_analysis = analyze_predictions(cases, baseline)
    candidate_analysis = analyze_predictions(cases, candidate)
    failure_transitions: dict[str, dict[str, list[str]]] = {}
    baseline_failures = baseline_analysis["failure_case_ids"]
    candidate_failures = candidate_analysis["failure_case_ids"]
    for name in sorted(set(baseline_failures) | set(candidate_failures)):
        before_failures = set(baseline_failures.get(name, []))
        after_failures = set(candidate_failures.get(name, []))
        failure_transitions[name] = {
            "resolved": sorted(before_failures - after_failures),
            "introduced": sorted(after_failures - before_failures),
            "persistent": sorted(before_failures & after_failures),
        }
    return {
        "ok": True,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": deltas,
        "delta_confidence_intervals": bootstrap_metric_delta_intervals(
            cases,
            baseline,
            candidate,
        ),
        "case_transitions": {
            "improved": improved,
            "regressed": regressed,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
        },
        "failure_transitions": failure_transitions,
    }


def _prediction_map(path: Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for payload in load_jsonl(path):
        prediction = prediction_from_dict(payload)
        if prediction.case_id in predictions:
            raise ValueError(f"Duplicate prediction ID in {path}: {prediction.case_id}")
        predictions[prediction.case_id] = prediction
    return predictions


def analyze_predictions(
    cases: list[EvalCase],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Group baseline failures into actionable, deterministic case lists."""
    category_totals: Counter[str] = Counter()
    category_successes: Counter[str] = Counter()
    failures: dict[str, list[str]] = {
        "wrong_first_tool": [],
        "missing_tool": [],
        "unnecessary_tool": [],
        "argument_mismatch": [],
        "retrieval_miss": [],
        "citation_miss": [],
        "recovery_failure": [],
        "runtime_error": [],
    }
    finish_reasons: Counter[str] = Counter()
    for case in cases:
        prediction = predictions.get(case.id, Prediction(case_id=case.id, text=""))
        category_totals[case.category] += 1
        category_successes[case.category] += int(case_succeeded(case, prediction))
        names = [call.name for call in prediction.tool_calls if call.name]
        finish_reasons[prediction.finish_reason or "missing"] += 1
        if prediction.error:
            failures["runtime_error"].append(case.id)
        if case.expect_tool and case.expected_tools:
            if not names:
                failures["missing_tool"].append(case.id)
            elif names[0] != case.expected_tools[0]:
                failures["wrong_first_tool"].append(case.id)
        elif not case.expect_tool and names:
            failures["unnecessary_tool"].append(case.id)
        if case.expected_arguments:
            by_name = {call.name: call.arguments for call in prediction.tool_calls}
            if any(
                name not in by_name or not arguments_match(expected, by_name[name])
                for name, expected in case.expected_arguments.items()
            ):
                failures["argument_mismatch"].append(case.id)
        if case.retrieval is not None:
            gold = set(case.retrieval.gold_doc_ids)
            if not gold.intersection(prediction.retrieved_doc_ids[: case.retrieval.k]):
                failures["retrieval_miss"].append(case.id)
            if not gold.intersection(prediction.cited_doc_ids):
                failures["citation_miss"].append(case.id)
        if (case.expect_recovery or case.success_mode == "recovery") and not (
            prediction.recovered
        ):
            failures["recovery_failure"].append(case.id)

    category_summary = {
        category: {
            "cases": total,
            "successes": category_successes[category],
            "success_rate": category_successes[category] / total,
        }
        for category, total in sorted(category_totals.items())
    }
    return {
        "category_success": category_summary,
        "failure_case_ids": failures,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=DEFAULT_CASES,
        help="Directory containing JSONL evaluation cases.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional recorded prediction JSONL to score.",
    )
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--candidate-predictions", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate the case set; do not score predictions.",
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help=(
            "Fail when any configured quality target is missed. By default, "
            "score mode only validates prediction inventory integrity."
        ),
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Load MINIGPT_CHAT_* backend, execute cases, and write predictions.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("evals/results/predictions.jsonl"),
        help="JSONL destination used by --run-agent.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing case IDs to an existing predictions output.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run a specific case ID; repeat the option to select multiple cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected cases for a smoke test.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routing",
        choices=("free", "heuristic"),
        default="free",
        help="Optionally restrict model-visible tools with the deterministic router.",
    )
    return parser


def run_real_agent(
    cases: list[EvalCase],
    *,
    predictions_output: Path,
    resume: bool,
    max_new_tokens: int,
    temperature: float,
    greedy: bool,
    seed: int | None,
    routing: str,
) -> dict[str, Any]:
    """Load the configured backend once and execute selected eval cases."""
    from agent import AgentOrchestrator, ChatModelAgentAdapter, build_default_registry
    from agent.llm import create_huggingface_chat_model
    from agent.routing import HeuristicAgentRouter
    from evals.agent_runner import (
        AgentEvalConfig,
        EvalToolExecutor,
        collect_corpus,
        load_completed_ids,
        run_cases_to_jsonl,
    )
    from minigpt_service.config import ChatServiceConfig
    from minigpt_service.model_service import select_device

    config = ChatServiceConfig.from_environment()
    if config is None:
        raise ValueError("Set MINIGPT_CHAT_MODEL_SOURCE before using --run-agent.")
    backend = create_huggingface_chat_model(
        config,
        select_device(config.device),
    )
    registry = build_default_registry(documents=collect_corpus(cases))
    executor = EvalToolExecutor(registry)
    agent = AgentOrchestrator(
        ChatModelAgentAdapter(backend),
        registry,
        executor=executor,
        max_steps=max(case.max_steps for case in cases),
        router=HeuristicAgentRouter() if routing == "heuristic" else None,
    )
    generation = AgentEvalConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        greedy=greedy,
        seed=seed,
    )
    try:
        backend.load()
        written = run_cases_to_jsonl(
            cases,
            agent,
            generation,
            predictions_output,
            resume=resume,
        )
    finally:
        agent.close()
        executor.close()
        backend.close()
    return {
        "model": config.model_source,
        "adapter_path": str(config.adapter_path) if config.adapter_path else None,
        "device": config.device,
        "dtype": config.dtype,
        "quantization": config.quantization,
        "routing": routing,
        "selected_cases": len(cases),
        "predictions_written": written,
        "predictions_available": len(load_completed_ids(predictions_output)),
        "predictions_output": str(predictions_output),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "greedy": greedy,
            "seed": seed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.run_agent and args.predictions is not None:
        parser.error("Use either --run-agent or --predictions, not both.")
    comparing = (
        args.baseline_predictions is not None or args.candidate_predictions is not None
    )
    if comparing and not (
        args.baseline_predictions is not None and args.candidate_predictions is not None
    ):
        parser.error("Comparison requires both prediction files.")
    if comparing and (args.run_agent or args.predictions is not None):
        parser.error("Comparison cannot be combined with run or score mode.")
    report: dict[str, Any] = {"validation": validate_cases(args.cases_dir)}
    if comparing and not args.check_only:
        report["comparison"] = compare_prediction_files(
            args.cases_dir,
            args.baseline_predictions,
            args.candidate_predictions,
        )
    elif args.run_agent and not args.check_only:
        from evals.agent_runner import load_completed_ids, select_cases

        all_cases = load_cases(args.cases_dir)
        selected = select_cases(
            all_cases,
            case_ids=args.case_id,
            limit=args.limit,
        )
        if not selected:
            parser.error("No evaluation cases were selected.")
        report["run"] = run_real_agent(
            selected,
            predictions_output=args.predictions_output,
            resume=args.resume,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            greedy=args.greedy,
            seed=args.seed,
            routing=args.routing,
        )
        completed = load_completed_ids(args.predictions_output)
        all_ids = {case.id for case in all_cases}
        if all_ids.issubset(completed):
            report["scoring"] = score_predictions(
                args.cases_dir,
                args.predictions_output,
            )
        else:
            report["run"]["remaining_cases"] = len(all_ids - completed)
    elif args.predictions is not None and not args.check_only:
        report["scoring"] = score_predictions(
            args.cases_dir,
            args.predictions,
            case_ids=set(args.case_id) or None,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    validation_ok = bool(report["validation"]["ok"])
    scoring = report.get("scoring")
    scoring_ok = scoring_meets_requirements(
        scoring,
        require_targets=args.require_targets,
    )
    comparison = report.get("comparison")
    comparison_ok = comparison is None or bool(comparison["ok"])
    return 0 if validation_ok and scoring_ok and comparison_ok else 1


def scoring_meets_requirements(
    scoring: Mapping[str, Any] | None,
    *,
    require_targets: bool,
) -> bool:
    """Separate prediction-integrity checks from optional quality gates."""
    if scoring is None:
        return not require_targets
    if not bool(scoring["ok"]):
        return False
    if not require_targets:
        return True
    targets = scoring.get("targets")
    return isinstance(targets, Mapping) and bool(targets) and all(targets.values())


if __name__ == "__main__":
    raise SystemExit(main())
