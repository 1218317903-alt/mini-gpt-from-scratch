"""Tests for bounded model/tool orchestration."""

from collections.abc import Sequence

from agent.builtin_tools import build_default_registry
from agent.model_adapter import AgentTurn
from agent.orchestrator import AgentOrchestrator
from agent.tools.base import ToolCall
from agent.tools.registry import ToolRegistry
from minigpt_service.schemas import ChatMessage


class _FakeAgentModel:
    is_ready = True

    def __init__(self, turns: Sequence[AgentTurn]) -> None:
        self.turns = list(turns)
        self.seen_messages: list[list[ChatMessage]] = []

    def complete(self, messages, _tools, **_: object) -> AgentTurn:
        self.seen_messages.append(list(messages))
        return self.turns.pop(0)


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
