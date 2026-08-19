"""Run a small concurrent HTTP load test against the MiniGPT service."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class RequestResult:
    """Outcome and timing for one HTTP request."""

    status_code: int | None
    latency_ms: float
    error: str | None = None


def _request_once(url: str, timeout: float) -> RequestResult:
    started = perf_counter()
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            status_code = response.status
    except HTTPError as error:
        status_code = error.code
        error.close()
    except (OSError, URLError) as error:
        return RequestResult(
            status_code=None,
            latency_ms=(perf_counter() - started) * 1000,
            error=f"{type(error).__name__}: {error}",
        )

    return RequestResult(
        status_code=status_code,
        latency_ms=(perf_counter() - started) * 1000,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    rank = max(1, ceil(percentile / 100 * len(values)))
    return sorted(values)[rank - 1]


def summarize_results(
    results: list[RequestResult],
    *,
    url: str,
    concurrency: int,
    timeout: float,
    elapsed_s: float,
) -> dict[str, object]:
    """Convert request outcomes into a JSON-serializable performance record."""
    latencies = [result.latency_ms for result in results]
    successful = sum(
        result.status_code is not None and 200 <= result.status_code < 400
        for result in results
    )
    status_counts = Counter(
        str(result.status_code) for result in results if result.status_code is not None
    )
    errors = Counter(result.error for result in results if result.error is not None)
    total = len(results)
    return {
        "url": url,
        "requests": total,
        "concurrency": concurrency,
        "timeout_s": timeout,
        "elapsed_s": round(elapsed_s, 6),
        "throughput_rps": round(total / elapsed_s if elapsed_s > 0 else 0.0, 3),
        "successes": successful,
        "failures": total - successful,
        "success_rate": round(successful / total if total else 0.0, 6),
        "status_counts": dict(sorted(status_counts.items())),
        "errors": dict(sorted(errors.items())),
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def run_load_test(
    url: str,
    *,
    requests: int = 100,
    concurrency: int = 8,
    timeout: float = 5.0,
) -> dict[str, object]:
    """Issue concurrent GET requests and return a performance record."""
    if requests <= 0:
        raise ValueError("requests must be greater than 0.")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than 0.")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0.")

    started = perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=min(requests, concurrency)) as executor:
        futures = [
            executor.submit(_request_once, url, timeout) for _ in range(requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    return summarize_results(
        results,
        url=url,
        concurrency=concurrency,
        timeout=timeout,
        elapsed_s=perf_counter() - started,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the performance record.",
    )
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        help="Fail when the p95 latency exceeds this threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = run_load_test(
        args.url,
        requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    rendered = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if record["failures"]:
        return 1
    latency = record["latency_ms"]
    if args.max_p95_ms is not None and isinstance(latency, dict):
        if float(latency["p95"]) > args.max_p95_ms:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
