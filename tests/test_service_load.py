"""Tests for the service load-test metrics."""

from scripts import load_test_service


def test_summarize_results_calculates_percentiles_and_statuses() -> None:
    results = [
        load_test_service.RequestResult(200, 1.0),
        load_test_service.RequestResult(200, 2.0),
        load_test_service.RequestResult(503, 3.0),
        load_test_service.RequestResult(None, 4.0, "timeout"),
    ]

    record = load_test_service.summarize_results(
        results,
        url="http://service/health",
        concurrency=2,
        timeout=5.0,
        elapsed_s=0.1,
    )

    assert record["requests"] == 4
    assert record["successes"] == 2
    assert record["failures"] == 2
    assert record["status_counts"] == {"200": 2, "503": 1}
    assert record["errors"] == {"timeout": 1}
    assert record["latency_ms"] == {
        "min": 1.0,
        "p50": 2.0,
        "p95": 4.0,
        "p99": 4.0,
        "max": 4.0,
    }


def test_run_load_test_rejects_invalid_parameters() -> None:
    for kwargs, message in [
        ({"requests": 0}, "requests"),
        ({"concurrency": 0}, "concurrency"),
        ({"timeout": 0}, "timeout"),
    ]:
        try:
            load_test_service.run_load_test("http://service/health", **kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Expected invalid load-test arguments to fail")


def test_run_load_test_uses_concurrent_request_worker(monkeypatch) -> None:
    monkeypatch.setattr(
        load_test_service,
        "_request_once",
        lambda _url, _timeout: load_test_service.RequestResult(200, 1.0),
    )

    record = load_test_service.run_load_test(
        "http://service/health",
        requests=5,
        concurrency=2,
    )

    assert record["requests"] == 5
    assert record["success_rate"] == 1.0
