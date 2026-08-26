"""Tests for the stage-eight FastAPI application skeleton."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent.orchestrator import AgentResult, AgentTrace
from minigpt_service.app import (
    SERVICE_NAME,
    SERVICE_VERSION,
    app,
    create_app,
    create_app_from_environment,
)
from minigpt_service.chat_model import ChatResult
from minigpt_service.schemas import AgentRequest, ChatRequest, GenerateRequest


def test_create_app_returns_configured_fastapi_application() -> None:
    created_app = create_app()

    assert isinstance(created_app, FastAPI)
    assert created_app.title == SERVICE_NAME
    assert created_app.version == "0.5.0"


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
    monkeypatch.delenv("MINIGPT_CHAT_MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MINIGPT_CHAT_ADAPTER_PATH", raising=False)

    created_app = create_app_from_environment()

    assert isinstance(created_app, FastAPI)
    assert "/health" in created_app.openapi()["paths"]


def test_environment_factory_configures_chat_backend_without_minigpt_paths(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MINIGPT_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MINIGPT_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv(
        "MINIGPT_CHAT_MODEL_SOURCE",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    monkeypatch.setenv("MINIGPT_CHAT_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("MINIGPT_DEVICE", "cpu")

    created_app = create_app_from_environment()

    assert "/chat" in created_app.openapi()["paths"]
    assert "/agent/chat" in created_app.openapi()["paths"]


def test_openapi_schema_exposes_service_metadata() -> None:
    schema = app.openapi()

    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == SERVICE_VERSION
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/generate" in schema["paths"]
    assert "/chat" in schema["paths"]
    assert "/agent/chat" in schema["paths"]
    assert "/agent/chat/stream" in schema["paths"]
    assert "/agent/cancel/{run_id}" in schema["paths"]
    assert "/agent/metrics" in schema["paths"]


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


class _FakeChatModel:
    is_ready = True

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def chat(self, messages, **_: object) -> ChatResult:
        return ChatResult(
            text=f"reply to: {messages[-1].content}",
            prompt_tokens=4,
            generated_tokens=3,
            finish_reason="stop",
            model="fake-chat",
        )


class _FakeAgent:
    is_ready = True

    def close(self) -> None:
        pass

    def run(self, messages, **_: object) -> AgentResult:
        return AgentResult(
            text=f"agent reply to: {messages[-1].content}",
            model="fake-agent",
            steps=2,
            tool_calls=1,
            finish_reason="stop",
            trace=(
                AgentTrace(
                    step=1,
                    call_id="call-1",
                    name="calculator",
                    arguments={"expression": "17*6"},
                    status="ok",
                    content="102",
                    duration_ms=0.2,
                ),
            ),
        )


def test_chat_route_forwards_messages_and_returns_usage() -> None:
    service_app = create_app(chat_model=_FakeChatModel())  # type: ignore[arg-type]
    route = next(route for route in service_app.routes if route.path == "/chat")

    response = route.endpoint(
        ChatRequest(
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ]
        )
    )

    assert response.generated_text == "reply to: hello"
    assert response.message.role == "assistant"
    assert response.generated_tokens == 3
    assert response.model == "fake-chat"


def test_chat_route_returns_service_unavailable_without_chat_model() -> None:
    service_app = create_app()
    route = next(route for route in service_app.routes if route.path == "/chat")

    try:
        route.endpoint(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected /chat to reject an unloaded chat model")


def test_agent_route_returns_trace_and_final_answer() -> None:
    service_app = create_app(agent=_FakeAgent())  # type: ignore[arg-type]
    route = next(route for route in service_app.routes if route.path == "/agent/chat")

    response = route.endpoint(
        AgentRequest(messages=[{"role": "user", "content": "17乘以6？"}])
    )

    assert response.generated_text == "agent reply to: 17乘以6？"
    assert response.tool_calls == 1
    assert response.trace[0].content == "102"


def test_agent_route_returns_service_unavailable_without_agent() -> None:
    service_app = create_app()
    route = next(route for route in service_app.routes if route.path == "/agent/chat")

    try:
        route.endpoint(AgentRequest(messages=[{"role": "user", "content": "hello"}]))
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected /agent/chat to reject an unavailable agent")


def test_agent_sse_stream_emits_start_and_result_events() -> None:
    service_app = create_app(agent=_FakeAgent())  # type: ignore[arg-type]

    with TestClient(service_app) as client:
        response = client.post(
            "/agent/chat/stream",
            headers={"X-Request-ID": "stream-123"},
            json={"messages": [{"role": "user", "content": "17乘以6？"}]},
        )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: result" in response.text
    assert "stream-123" in response.text


def test_agent_metrics_and_unknown_cancellation_are_observable() -> None:
    service_app = create_app(agent=_FakeAgent())  # type: ignore[arg-type]

    with TestClient(service_app) as client:
        response = client.post(
            "/agent/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        metrics = client.get("/agent/metrics")
        cancellation = client.post("/agent/cancel/missing")

    assert response.status_code == 200
    assert metrics.json()["completed"] == 1
    assert metrics.json()["running"] == 0
    assert cancellation.json() == {
        "run_id": "missing",
        "cancellation_requested": False,
    }


def test_demo_page_exposes_local_observable_agent_ui() -> None:
    service_app = create_app()

    with TestClient(service_app) as client:
        response = client.get("/demo")

    assert response.status_code == 200
    assert "MiniGPT 可观测 Agent" in response.text
    assert "/agent/chat/stream" in response.text


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
        "detail": "No model backend is ready.",
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
