"""Tests for complete Agent events and credential redaction."""

import json

from agent.messages import Message
from agent.model_adapter import AgentTurn
from agent.tools.base import ToolCall, ToolExecutionResult
from agent.tracing import AgentTracer, redact, redact_content


def test_redaction_handles_nested_fields_plain_text_and_json() -> None:
    assert redact({"api_key": "abc", "generated_tokens": 3}) == {
        "api_key": "[REDACTED]",
        "generated_tokens": 3,
    }
    assert "hunter2" not in redact_content("password=hunter2")
    serialized = redact_content('{"password":"hunter2","value":1}')
    assert json.loads(serialized) == {"password": "[REDACTED]", "value": 1}


def test_tracer_records_sanitized_model_and_tool_events() -> None:
    tracer = AgentTracer()
    call = ToolCall("call-1", "lookup", {"token": "private"})
    turn = AgentTurn(
        text="",
        raw_text="password=private",
        tool_calls=(call,),
        prompt_tokens=2,
        generated_tokens=3,
        model="fake",
    )

    tracer.record_model(
        1,
        [Message(role="user", content="api_key=private")],
        turn,
        1.25,
    )
    tracer.record_tool(
        1,
        call,
        ToolExecutionResult(
            call_id="call-1",
            name="lookup",
            status="error",
            content='{"secret":"private"}',
            duration_ms=0.5,
        ),
    )

    rendered = json.dumps(
        [event.__dict__ for event in tracer.events], ensure_ascii=False
    )
    assert "private" not in rendered
    assert [event.phase for event in tracer.events] == ["model", "tool"]
