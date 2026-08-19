"""Tests for the stage-eight FastAPI application skeleton."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from minigpt_service.app import (
    SERVICE_NAME,
    SERVICE_VERSION,
    app,
    create_app,
    create_app_from_environment,
)
from minigpt_service.schemas import GenerateRequest


def test_create_app_returns_configured_fastapi_application() -> None:
    created_app = create_app()

    assert isinstance(created_app, FastAPI)
    assert created_app.title == SERVICE_NAME
    assert created_app.version == SERVICE_VERSION


def test_create_app_returns_independent_instances() -> None:
    first_app = create_app()
    second_app = create_app()

    assert first_app is not second_app
    assert app is not first_app


def test_environment_factory_starts_health_only_without_model_paths(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MINIGPT_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MINIGPT_CHECKPOINT_PATH", raising=False)

    created_app = create_app_from_environment()

    assert isinstance(created_app, FastAPI)
    assert "/health" in created_app.openapi()["paths"]


def test_openapi_schema_exposes_service_metadata() -> None:
    schema = app.openapi()

    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == SERVICE_VERSION
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/generate" in schema["paths"]


class _FakeModelService:
    is_ready = True

    def generate_text(self, prompt: str, **_: object) -> str:
        return prompt + "abc"


def test_generate_route_forwards_request_and_counts_character_tokens() -> None:
    service_app = create_app(_FakeModelService())  # type: ignore[arg-type]
    route = next(route for route in service_app.routes if route.path == "/generate")

    response = route.endpoint(GenerateRequest(prompt="Hi", max_new_tokens=3))

    assert response.generated_text == "Hiabc"
    assert response.generated_tokens == 3
    assert response.finish_reason == "length"


def test_health_route_reports_alive() -> None:
    service_app = create_app()
    route = next(route for route in service_app.routes if route.path == "/health")

    response = route.endpoint()

    assert response.status == "ok"
    assert response.service == SERVICE_NAME


def test_ready_route_reports_unavailable_without_model() -> None:
    service_app = create_app()
    route = next(route for route in service_app.routes if route.path == "/ready")

    try:
        route.endpoint()
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected /ready to reject an unloaded service")


def test_generate_route_returns_service_unavailable_without_model() -> None:
    service_app = create_app()
    route = next(route for route in service_app.routes if route.path == "/generate")

    try:
        route.endpoint(GenerateRequest(prompt="Hi"))
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected /generate to reject an unloaded service")


def test_request_id_is_propagated_and_structured_request_is_logged(caplog) -> None:
    service_app = create_app()
    caplog.set_level("INFO", logger="minigpt_service.http")

    with TestClient(service_app) as client:
        response = client.get("/health", headers={"X-Request-ID": "trace-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-123"
    assert response.json()["status"] == "ok"
    assert '"event": "http_request"' in caplog.text
    assert '"request_id": "trace-123"' in caplog.text


def test_errors_include_request_id_and_use_common_envelope() -> None:
    service_app = create_app()

    with TestClient(service_app) as client:
        response = client.get("/ready", headers={"X-Request-ID": "not-ready"})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "not-ready"
    assert response.json() == {
        "detail": "Model service is not ready.",
        "request_id": "not-ready",
    }


def test_unexpected_errors_return_internal_error_envelope() -> None:
    service_app = create_app()

    def boom() -> None:
        raise ValueError("boom")

    service_app.add_api_route("/boom", boom)
    with TestClient(service_app, raise_server_exceptions=False) as client:
        response = client.get("/boom", headers={"X-Request-ID": "boom-123"})

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "boom-123"
    assert response.json() == {
        "detail": "Internal server error.",
        "request_id": "boom-123",
    }
