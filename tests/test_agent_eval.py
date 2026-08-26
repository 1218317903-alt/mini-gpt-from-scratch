"""Tests for the stage-two Agent evaluation set and metrics."""

from __future__ import annotations

from pathlib import Path

from agent.builtin_tools import calculate, current_time
from evals.inventory import compare_split_texts, load_inventory, validate_inventory
from evals.metrics import (
    EvalCase,
    PredictedCall,
    Prediction,
    RetrievalSpec,
    bootstrap_confidence_intervals,
    bootstrap_metric_delta_intervals,
    case_succeeded,
    compute_metrics,
    evaluate_targets,
    keyword_retrieve,
    load_cases,
    overlap_with_training,
    training_user_texts,
)
from evals.run_agent_eval import (
    EXPECTED_COUNTS,
    analyze_predictions,
    compare_prediction_files,
    score_predictions,
    scoring_meets_requirements,
    validate_cases,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals" / "cases"
FROZEN_CASES = ROOT / "evals" / "frozen_cases"


def test_eval_inventory_is_independent_and_complete() -> None:
    report = validate_cases(CASES)

    assert report["ok"] is True
    assert 80 <= report["case_count"] <= 120
    assert report["counts"] == EXPECTED_COUNTS
    assert report["training_overlap"] == []
    assert report["duplicates"] == []
    assert report["recall_failures"] == []
    assert report["manifest"]["split"] == "dev"
    assert report["manifest"]["errors"] == []


def test_frozen_inventory_is_versioned_independent_and_complete() -> None:
    report = validate_cases(FROZEN_CASES)

    assert report["ok"] is True
    assert report["case_count"] == 150
    assert report["counts"] == {
        "direct_answer": 30,
        "multi_step": 20,
        "retrieval": 30,
        "tool_arguments": 25,
        "tool_selection": 45,
    }
    assert report["training_overlap"] == []
    assert report["manifest"]["frozen"] is True
    assert report["manifest"]["version"] == "agent-frozen-test-v1"
    assert report["manifest"]["review_status"] == "approved"
    inventory = load_inventory(FROZEN_CASES)
    assert inventory is not None
    assert validate_inventory(FROZEN_CASES, inventory) == []


def test_dev_and_frozen_prompts_do_not_overlap() -> None:
    overlap = compare_split_texts(CASES, FROZEN_CASES)

    assert overlap == {"exact": [], "near": []}


def test_frozen_expected_arguments_execute_and_gold_is_supported() -> None:
    cases = load_cases(FROZEN_CASES)

    for case in cases:
        arguments = case.expected_arguments or {}
        calculator = arguments.get("calculator")
        if calculator is not None:
            result = str(calculate(str(calculator["expression"])))
            if case.answer_contains:
                assert result in case.answer_contains, case.id
        clock = arguments.get("current_time")
        if clock is not None:
            assert current_time(str(clock["timezone"])), case.id
        if case.retrieval is not None and case.expected_tools == ("search_docs",):
            gold_documents = {
                document["id"]: document["text"] for document in case.retrieval.corpus
            }
            supported_text = " ".join(
                gold_documents[doc_id] for doc_id in case.retrieval.gold_doc_ids
            )
            assert all(answer in supported_text for answer in case.answer_contains), (
                case.id
            )


def test_eval_questions_do_not_copy_sft_prompts() -> None:
    cases = load_cases(CASES)
    training = training_user_texts(
        [
            ROOT / "hf_lora" / "data" / "train.jsonl",
            ROOT / "hf_lora" / "data" / "validation.jsonl",
        ]
    )

    assert overlap_with_training(cases, training) == []
    assert "什么是lora？" in training or any("lora" in text for text in training)


def test_keyword_retriever_finds_gold_document() -> None:
    corpus = [
        {"id": "doc-office", "text": "青梧实验室总部位于杭州云栖园区 7 号楼。"},
        {"id": "doc-wifi", "text": "访客 Wi-Fi 名称是 Qingwu-Guest。"},
        {"id": "doc-budget", "text": "2026 年算力预算为 36 万元。"},
    ]

    retrieved = keyword_retrieve("青梧实验室总部 杭州", corpus, k=5)

    assert retrieved[0] == "doc-office"


def test_metrics_cover_tool_json_rag_and_recovery() -> None:
    cases = load_cases(CASES)
    selected = {
        "da-003": Prediction(case_id="da-003", text="首都是北京。", generated_tokens=4),
        "ts-001": Prediction(
            case_id="ts-001",
            text="888",
            tool_calls=(
                PredictedCall(name="calculator", arguments={"expression": "37*24"}),
            ),
            generated_tokens=8,
            duration_ms=12.0,
        ),
        "ts-d-001": Prediction(
            case_id="ts-d-001",
            text="乘法交换律是指 a*b=b*a。",
            generated_tokens=6,
        ),
        "ta-002": Prediction(
            case_id="ta-002",
            text="1240",
            tool_calls=(
                PredictedCall(name="calculator", arguments={"expression": "1234+6"}),
            ),
        ),
        "rt-001": Prediction(
            case_id="rt-001",
            text="根据 doc-office，总部在杭州。",
            tool_calls=(
                PredictedCall(name="search_docs", arguments={"query": "总部"}),
            ),
            retrieved_doc_ids=("doc-office", "doc-wifi"),
            cited_doc_ids=("doc-office",),
        ),
        "ms-011": Prediction(
            case_id="ms-011",
            text="102",
            tool_calls=(
                PredictedCall(
                    name="calculator",
                    arguments={"expression": "17 x 6"},
                    status="error",
                ),
                PredictedCall(name="calculator", arguments={"expression": "17*6"}),
            ),
            recovered=True,
        ),
    }
    by_id = {case.id: case for case in cases}
    used = [by_id[case_id] for case_id in selected]
    metrics = compute_metrics(used, selected)

    assert metrics["json_valid_rate"] == 1.0
    assert metrics["unnecessary_tool_rate"] == 0.0
    assert metrics["single_tool_success_rate"] == 1.0
    assert metrics["rag_recall_at_5"] == 1.0
    assert metrics["retrieval_trigger_precision"] == 1.0
    assert metrics["retrieval_trigger_recall"] == 1.0
    assert metrics["conditional_retrieval_recall_at_5"] == 1.0
    assert metrics["end_to_end_rag_recall_at_5"] == 1.0
    assert metrics["recovery_rate"] == 1.0
    assert metrics["citation_accuracy"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert case_succeeded(by_id["ts-d-001"], selected["ts-d-001"]) is True
    assert evaluate_targets(metrics)["json_valid_rate"] is True


def test_missing_predictions_count_as_failures() -> None:
    cases = load_cases(CASES)[:2]

    metrics = compute_metrics(
        cases,
        {cases[0].id: Prediction(case_id=cases[0].id, text="correct")},
    )

    assert metrics["cases_scored"] == 2
    assert metrics["task_success_rate"] <= 0.5


def test_retrieval_metric_does_not_credit_baseline_to_system() -> None:
    retrieval_case = next(case for case in load_cases(CASES) if case.retrieval)

    metrics = compute_metrics(
        [retrieval_case],
        {retrieval_case.id: Prediction(case_id=retrieval_case.id, text="")},
    )

    assert metrics["rag_recall_at_5"] == 0.0
    assert metrics["baseline_rag_recall_at_5"] == 1.0
    assert metrics["retrieval_trigger_recall"] == 0.0
    assert metrics["conditional_retrieval_recall_at_5"] is None


def test_rag_metrics_separate_trigger_retrieval_and_citation() -> None:
    retrieval = RetrievalSpec(
        query="policy",
        gold_doc_ids=("doc-gold",),
        corpus=({"id": "doc-gold", "text": "policy"},),
    )
    cases = [
        EvalCase(
            id="retrieval-hit",
            category="retrieval",
            messages=({"role": "user", "content": "find policy"},),
            expect_tool=True,
            expected_tools=("search_docs",),
            retrieval=retrieval,
            citations=("doc-gold",),
        ),
        EvalCase(
            id="retrieval-missed-trigger",
            category="retrieval",
            messages=({"role": "user", "content": "find policy again"},),
            expect_tool=True,
            expected_tools=("search_docs",),
            retrieval=retrieval,
            citations=("doc-gold",),
        ),
        EvalCase(
            id="direct-false-trigger",
            category="direct_answer",
            messages=({"role": "user", "content": "say hello"},),
            expect_tool=False,
        ),
    ]
    predictions = {
        "retrieval-hit": Prediction(
            case_id="retrieval-hit",
            text="supported [doc-gold]",
            tool_calls=(PredictedCall(name="search_docs"),),
            retrieved_doc_ids=("doc-gold",),
            cited_doc_ids=("doc-gold",),
        ),
        "retrieval-missed-trigger": Prediction(
            case_id="retrieval-missed-trigger",
            text="unsupported answer",
        ),
        "direct-false-trigger": Prediction(
            case_id="direct-false-trigger",
            text="irrelevant [doc-fake]",
            tool_calls=(PredictedCall(name="search_docs"),),
            cited_doc_ids=("doc-fake",),
        ),
    }

    metrics = compute_metrics(cases, predictions)

    assert metrics["retrieval_trigger_precision"] == 0.5
    assert metrics["retrieval_trigger_recall"] == 0.5
    assert metrics["conditional_retrieval_recall_at_1"] == 1.0
    assert metrics["end_to_end_rag_recall_at_1"] == 0.5
    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_recall"] == 0.5
    assert metrics["unsupported_answer_rate"] == 0.5


def test_bootstrap_intervals_are_deterministic_and_include_point() -> None:
    cases = [
        EvalCase(
            id="case-pass",
            category="direct_answer",
            messages=({"role": "user", "content": "pass"},),
            expect_tool=False,
            answer_contains=("yes",),
        ),
        EvalCase(
            id="case-fail",
            category="direct_answer",
            messages=({"role": "user", "content": "fail"},),
            expect_tool=False,
            answer_contains=("yes",),
        ),
    ]
    predictions = {
        "case-pass": Prediction(case_id="case-pass", text="yes"),
        "case-fail": Prediction(case_id="case-fail", text="no"),
    }

    first = bootstrap_confidence_intervals(
        cases,
        predictions,
        metric_names=("task_success_rate",),
        samples=200,
        seed=7,
    )
    second = bootstrap_confidence_intervals(
        cases,
        predictions,
        metric_names=("task_success_rate",),
        samples=200,
        seed=7,
    )

    assert first == second
    interval = first["task_success_rate"]
    assert interval["point"] == 0.5
    assert interval["lower"] == 0.0
    assert interval["upper"] == 1.0
    assert interval["valid_samples"] == 200


def test_paired_bootstrap_delta_uses_same_resampled_cases() -> None:
    cases = [
        EvalCase(
            id="one",
            category="direct_answer",
            messages=({"role": "user", "content": "one"},),
            expect_tool=False,
            answer_contains=("yes",),
        ),
        EvalCase(
            id="two",
            category="direct_answer",
            messages=({"role": "user", "content": "two"},),
            expect_tool=False,
            answer_contains=("yes",),
        ),
    ]
    baseline = {
        "one": Prediction(case_id="one", text="no"),
        "two": Prediction(case_id="two", text="no"),
    }
    candidate = {
        "one": Prediction(case_id="one", text="yes"),
        "two": Prediction(case_id="two", text="yes"),
    }

    result = bootstrap_metric_delta_intervals(
        cases,
        baseline,
        candidate,
        metric_names=("task_success_rate",),
        samples=100,
    )

    interval = result["task_success_rate"]
    assert interval["point_delta"] == 1.0
    assert interval["lower"] == 1.0
    assert interval["upper"] == 1.0


def test_prediction_comparison_reports_case_transitions(tmp_path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "direct_answer.jsonl").write_text(
        "\n".join(
            [
                '{"id":"one","category":"direct_answer","messages":[{"role":"user","content":"one"}],"expect_tool":false,"answer_contains":["yes"]}',
                '{"id":"two","category":"direct_answer","messages":[{"role":"user","content":"two"}],"expect_tool":false,"answer_contains":["yes"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        '{"case_id":"one","text":"no"}\n{"case_id":"two","text":"yes"}\n',
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        '{"case_id":"one","text":"yes"}\n{"case_id":"two","text":"no"}\n',
        encoding="utf-8",
    )

    comparison = compare_prediction_files(cases_dir, baseline, candidate)

    assert comparison["ok"] is True
    assert comparison["case_transitions"]["improved"] == ["one"]
    assert comparison["case_transitions"]["regressed"] == ["two"]
    assert comparison["metric_deltas"]["task_success_rate"] == 0.0


def test_prediction_inventory_reports_missing_duplicate_and_unknown(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                '{"case_id":"da-001","text":"a"}',
                '{"case_id":"da-001","text":"b"}',
                '{"case_id":"unknown","text":"c"}',
            ]
        ),
        encoding="utf-8",
    )

    report = score_predictions(CASES, predictions)

    assert report["ok"] is False
    assert report["duplicate_predictions"] == ["da-001"]
    assert report["unknown_predictions"] == ["unknown"]
    assert "da-002" in report["missing_predictions"]


def test_strict_scoring_gate_is_distinct_from_inventory_integrity() -> None:
    scoring = {
        "ok": True,
        "targets": {"json_valid_rate": True, "recovery_rate": False},
    }

    assert scoring_meets_requirements(scoring, require_targets=False) is True
    assert scoring_meets_requirements(scoring, require_targets=True) is False
    assert scoring_meets_requirements(None, require_targets=False) is True
    assert scoring_meets_requirements(None, require_targets=True) is False
    assert (
        scoring_meets_requirements(
            {"ok": True, "targets": {"json_valid_rate": True}},
            require_targets=True,
        )
        is True
    )


def test_failure_analysis_groups_actionable_case_ids() -> None:
    case = next(case for case in load_cases(CASES) if case.id == "da-001")
    predictions = {
        case.id: Prediction(
            case_id=case.id,
            text="",
            tool_calls=(PredictedCall(name="current_time"),),
            finish_reason="stop",
        )
    }

    analysis = analyze_predictions([case], predictions)

    assert analysis["category_success"]["direct_answer"]["success_rate"] == 0.0
    assert analysis["failure_case_ids"]["unnecessary_tool"] == ["da-001"]
    assert analysis["finish_reason_counts"] == {"stop": 1}
