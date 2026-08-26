"""Tests for bounded model/tool orchestration."""

from collections.abc import Sequence

from pydantic import BaseModel

from agent.builtin_tools import build_default_registry
from agent.model_adapter import AgentTurn
from agent.orchestrator import AgentOrchestrator
from agent.routing import HeuristicAgentRouter
from agent.tools.base import ToolCall, ToolExecutionResult, ToolSpec
from agent.tools.registry import ToolRegistry
from minigpt_service.schemas import ChatMessage


class _FakeAgentModel:
    is_ready = True

    def __init__(self, turns: Sequence[AgentTurn]) -> None:
        self.turns = list(turns)
        self.seen_messages: list[list[ChatMessage]] = []
        self.seen_tools: list[list[str]] = []

    def complete(self, messages, tools, **_: object) -> AgentTurn:
        self.seen_messages.append(list(messages))
        self.seen_tools.append([tool.name for tool in tools])
        return self.turns.pop(0)


class _TransientExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, call: ToolCall, **_: object) -> ToolExecutionResult:
        self.calls += 1
        if self.calls == 1:
            return ToolExecutionResult(
                call.call_id,
                call.name,
                "error",
                "Tool timed out.",
                1.0,
            )
        return ToolExecutionResult(call.call_id, call.name, "ok", "102", 1.0)


def _tool_turn() -> AgentTurn:
    return AgentTurn(
        text="",
        raw_text='{"tool_call":{"name":"calculator","arguments":{"expression":"17*6"}}}',
        tool_calls=(
            ToolCall(
                call_id="call-1",
                name="calculator",
                arguments={"expression": "17*6"},
            ),
        ),
        prompt_tokens=5,
        generated_tokens=8,
        model="fake-agent",
    )


def _final_turn(text: str = "结果是 102。") -> AgentTurn:
    return AgentTurn(
        text=text,
        raw_text=text,
        tool_calls=(),
        prompt_tokens=10,
        generated_tokens=4,
        model="fake-agent",
    )


class _WriteArguments(BaseModel):
    value: str


def _write_value(value: str) -> str:
    return value


def test_orchestrator_executes_tool_and_returns_final_answer() -> None:
    model = _FakeAgentModel([_tool_turn(), _final_turn()])
    orchestrator = AgentOrchestrator(model, build_default_registry())  # type: ignore[arg-type]

    result = orchestrator.run(
        [ChatMessage(role="user", content="17乘以6是多少？")],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )
    orchestrator.close()

    assert result.text == "结果是 102。"
    assert result.finish_reason == "stop"
    assert (result.steps, result.tool_calls) == (2, 1)
    assert result.trace[0].content == "102"
    assert model.seen_messages[1][-1].role == "tool"
    assert model.seen_messages[1][-1].tool_call_id == "call-1"


def test_orchestrator_preserves_ids_for_multiple_tool_calls() -> None:
    first_turn = AgentTurn(
        text="",
        raw_text="tool calls",
        tool_calls=(
            ToolCall("call-a", "calculator", {"expression": "1+1"}),
            ToolCall("call-b", "calculator", {"expression": "2+2"}),
        ),
        prompt_tokens=5,
        generated_tokens=8,
        model="fake-agent",
    )
    model = _FakeAgentModel([first_turn, _final_turn("2 and 4")])
    orchestrator = AgentOrchestrator(model, build_default_registry())  # type: ignore[arg-type]

    result = orchestrator.run(
        [ChatMessage(role="user", content="calculate both")],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )
    orchestrator.close()

    tool_messages = [
        message for message in model.seen_messages[1] if message.role == "tool"
    ]
    assert result.tool_calls == 2
    assert [message.tool_call_id for message in tool_messages] == ["call-a", "call-b"]


def test_orchestrator_stops_at_configured_max_steps() -> None:
    model = _FakeAgentModel([_tool_turn(), _tool_turn(), _tool_turn()])
    orchestrator = AgentOrchestrator(model, build_default_registry(), max_steps=2)

    result = orchestrator.run(
        [ChatMessage(role="user", content="repeat")],
        max_new_tokens=8,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )
    orchestrator.close()

    assert result.finish_reason == "max_steps"
    assert (result.steps, result.tool_calls) == (2, 2)


def test_orchestrator_rejects_invalid_step_override() -> None:
    model = _FakeAgentModel([_final_turn()])
    orchestrator = AgentOrchestrator(model, ToolRegistry(), max_steps=2)

    try:
        orchestrator.run(
            [ChatMessage(role="user", content="hello")],
            max_new_tokens=8,
            temperature=0.7,
            top_k=None,
            top_p=None,
            greedy=True,
            seed=None,
            max_steps=3,
        )
    except ValueError as error:
        assert "between 1 and 2" in str(error)
    else:
        raise AssertionError("Expected max_steps validation to fail")
    finally:
        orchestrator.close()


def test_orchestrator_blocks_repeated_identical_calls_and_records_events() -> None:
    model = _FakeAgentModel([_tool_turn(), _tool_turn(), _final_turn("done")])
    orchestrator = AgentOrchestrator(model, build_default_registry(), max_steps=3)

    result = orchestrator.run(
        [ChatMessage(role="user", content="repeat")],
        max_new_tokens=8,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )
    orchestrator.close()

    assert result.text == "done"
    assert [trace.status for trace in result.trace] == ["ok", "error"]
    assert "Repeated tool call blocked" in result.trace[1].content
    assert [event.phase for event in result.events] == [
        "model",
        "tool",
        "model",
        "tool",
        "model",
    ]


def test_orchestrator_stops_before_unconfirmed_side_effect() -> None:
    call = ToolCall("call-write", "write_value", {"value": "hello"})
    turn = AgentTurn(
        text="",
        raw_text="write",
        tool_calls=(call,),
        prompt_tokens=1,
        generated_tokens=1,
        model="fake-agent",
    )
    model = _FakeAgentModel([turn])
    registry = ToolRegistry(
        [
            ToolSpec(
                name="write_value",
                description="test-only write",
                arguments_model=_WriteArguments,
                handler=_write_value,
                side_effect=True,
            )
        ]
    )
    orchestrator = AgentOrchestrator(model, registry)

    result = orchestrator.run(
        [ChatMessage(role="user", content="write hello")],
        max_new_tokens=8,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )
    orchestrator.close()

    assert result.finish_reason == "confirmation_required"
    assert result.trace[0].status == "confirmation_required"
    assert result.trace[0].confirmation_id is not None


def test_orchestrator_allows_one_identical_retry_after_transient_error() -> None:
    model = _FakeAgentModel([_tool_turn(), _final_turn()])
    executor = _TransientExecutor()
    orchestrator = AgentOrchestrator(
        model,
        build_default_registry(),
        max_steps=3,
        executor=executor,  # type: ignore[arg-type]
    )

    result = orchestrator.run(
        [ChatMessage(role="user", content="retry after timeout")],
        max_new_tokens=8,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=None,
    )

    assert result.finish_reason == "stop"
    assert [trace.status for trace in result.trace] == ["ok"]
    assert result.trace[0].attempts == 2
    assert executor.calls == 2


def test_router_limits_model_visible_tools() -> None:
    model = _FakeAgentModel([_tool_turn(), _final_turn()])
    orchestrator = AgentOrchestrator(
        model,
        build_default_registry(),
        router=HeuristicAgentRouter(),
    )

    result = orchestrator.run(
        [ChatMessage(role="user", content="请计算 17 乘以 6。")],
        max_new_tokens=32,
        temperature=0.7,
        top_k=None,
        top_p=None,
        greedy=True,
        seed=42,
    )
    orchestrator.close()

    assert model.seen_tools == [["calculator"], ["calculator"]]
    assert result.route is not None
    assert result.route.kind == "calculator"
