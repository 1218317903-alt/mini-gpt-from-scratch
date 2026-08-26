"""Benchmark lexical, dense and hybrid retrieval on a versioned case split."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.retrieval import (  # noqa: E402
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    TransformerMeanPoolEmbedder,
)
from evals.agent_runner import collect_corpus  # noqa: E402
from evals.inventory import load_inventory  # noqa: E402
from evals.metrics import EvalCase, load_cases  # noqa: E402


def _query(case: EvalCase) -> str:
    if case.retrieval is None:
        raise ValueError(f"Case does not define retrieval: {case.id}")
    if case.retrieval.query:
        return case.retrieval.query
    return next(
        message["content"]
        for message in reversed(case.messages)
        if message["role"] == "user"
    )


def benchmark(name: str, retriever: Any, cases: list[EvalCase]) -> dict[str, Any]:
    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        assert case.retrieval is not None
        started = perf_counter()
        hits = retriever.search(_query(case), k=5)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        citations = [hit.citation for hit in hits]
        gold = set(case.retrieval.gold_doc_ids)
        first_rank = next(
            (
                rank
                for rank, citation in enumerate(citations, start=1)
                if citation in gold
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in recall_hits:
            recall_hits[k] += int(bool(gold.intersection(citations[:k])))
        rows.append(
            {
                "case_id": case.id,
                "gold": sorted(gold),
                "retrieved": citations,
                "first_relevant_rank": first_rank,
                "latency_ms": round(latency_ms, 3),
            }
        )
    count = len(cases)
    return {
        "name": name,
        "case_count": count,
        "recall_at_1": recall_hits[1] / count,
        "recall_at_3": recall_hits[3] / count,
        "recall_at_5": recall_hits[5] / count,
        "mrr": statistics.fmean(reciprocal_ranks),
        "avg_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * count) - 1)],
        "predictions": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    parser.add_argument(
        "--query-prefix",
        default="为这个句子生成表示以用于检索相关文章：",
    )
    args = parser.parse_args(argv)

    all_cases = load_cases(args.cases_dir)
    cases = [case for case in all_cases if case.retrieval is not None]
    documents = collect_corpus(cases)
    lexical = LexicalRetriever(documents)
    strategies: list[tuple[str, Any]] = [("lexical", lexical)]
    if args.embedding_model:
        embedder = TransformerMeanPoolEmbedder(
            args.embedding_model,
            device=args.device,
            local_files_only=True,
        )
        dense = DenseRetriever(
            documents,
            embedder,
            query_prefix=args.query_prefix,
        )
        strategies.extend(
            [
                ("dense", dense),
                (
                    "hybrid_rrf",
                    HybridRetriever(
                        lexical,
                        dense,
                        lexical_weight=args.lexical_weight,
                        dense_weight=1.0 - args.lexical_weight,
                    ),
                ),
            ]
        )
    inventory = load_inventory(args.cases_dir)
    report = {
        "split": inventory.split if inventory else str(args.cases_dir),
        "version": inventory.version if inventory else None,
        "fingerprint": inventory.fingerprint if inventory else None,
        "embedding_model": args.embedding_model,
        "document_count": len(documents),
        "strategies": [
            benchmark(name, retriever, cases) for name, retriever in strategies
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
