"""Tests for the JSON tool-call adapter."""

from agent.model_adapter import ChatModelAgentAdapter, parse_tool_calls
from agent.tools.base import ToolDefinition
from minigpt_service.chat_model import ChatResult
from minigpt_service.schemas import ChatMessage


def test_parse_tool_calls_supports_wrapped_and_native_forms() -> None:
    wrapped = parse_tool_calls(
        '{"tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}}'
    )
    native = parse_tool_calls(
        '<tool_call>{"name":"current_time","arguments":{}}</tool_call>'
    )
    many = parse_tool_calls(
        '{"tool_calls":[{"name":"calculator","arguments":{"expression":"1+1"}},'
        '{"function":{"name":"current_time","arguments":"{}"}}]}'
    )

    assert wrapped[0].name == "calculator"
    assert wrapped[0].arguments == {"expression": "2+2"}
    assert native[0].name == "current_time"
    assert len(many) == 2


def test_parse_tool_calls_returns_empty_for_malformed_or_normal_text() -> None:
    assert parse_tool_calls("普通回答，不需要工具。") == ()
    assert parse_tool_calls('{"tool_call":{"name":"calculator"}}') == ()
    assert parse_tool_calls('{"tool_call":{"name":"calculator","arguments":[]}}') == ()


class _FakeChatModel:
    is_ready = True

    def __init__(self) -> None:
        self.messages: list[object] = []
        self.tools: list[object] = []

    def chat(self, messages, tools=None, **_: object) -> ChatResult:
        self.messages = list(messages)
        self.tools = list(tools or [])
        return ChatResult(
            text='{"tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}}',
            prompt_tokens=4,
            generated_tokens=10,
            finish_reason="length",
            model="fake",
        )


def test_adapter_injects_tool_definitions_and_parses_response() -> None:
    backend = _FakeChatModel()
    adapter = ChatModelAgentAdapter(backend)  # type: ignore[arg-type]

    turn = adapter.complete(
        [ChatMessage(role="user", content="calculate")],
        [
            ToolDefinition(
                name="calculator",
                description="calculate",
                parameters={"type": "object"},
            )
        ],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )

    assert turn.text == ""
    assert turn.tool_calls[0].name == "calculator"
    assert backend.tools[0].name == "calculator"
