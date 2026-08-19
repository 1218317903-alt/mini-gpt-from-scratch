"""Bounded Agent loop that connects a chat model to registered tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.model_adapter import AgentTurn
from agent.tools.base import ToolCall, ToolDefinition, ToolExecutionResult
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from minigpt_service.schemas import ChatMessage

AgentFinishReason = Literal["stop", "max_steps"]


class AgentModel(Protocol):
    """Model contract required by the orchestrator."""

    @property
    def is_ready(self) -> bool:
        """Return whether the underlying chat model is ready."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
    ) -> AgentTurn:
        """Generate a model turn with optional tool calls."""


@dataclass(frozen=True)
class AgentTrace:
    """One tool execution recorded in an Agent run."""

    step: int
    call_id: str
    name: str
    arguments: dict[str, object]
    status: Literal["ok", "error"]
    content: str
    duration_ms: float


@dataclass(frozen=True)
class AgentResult:
    """Final answer plus bounded execution trace."""

    text: str
    model: str
    steps: int
    tool_calls: int
    finish_reason: AgentFinishReason
    trace: tuple[AgentTrace, ...]


class AgentOrchestrator:
    """Run a model/tool loop with explicit limits and no hidden tools."""

    def __init__(
        self,
        model: AgentModel,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        executor: ToolExecutor | None = None,
    ) -> None:
        if not 1 <= max_steps <= 12:
            raise ValueError("max_steps must be between 1 and 12.")
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.executor = executor or ToolExecutor(registry)
        self._owns_executor = executor is None

    @property
    def is_ready(self) -> bool:
        """Return whether the underlying model can answer requests."""
        return self.model.is_ready

    def close(self) -> None:
        """Close the internally owned executor."""
        if self._owns_executor:
            self.executor.close()

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
        max_steps: int | None = None,
    ) -> AgentResult:
        """Answer directly or execute tools until a final answer is produced."""
        if not self.is_ready:
            raise RuntimeError("Agent model is not ready.")
        step_limit = self.max_steps if max_steps is None else max_steps
        if not 1 <= step_limit <= self.max_steps:
            raise ValueError(f"max_steps must be between 1 and {self.max_steps}.")

        conversation = list(messages)
        definitions = self.registry.definitions()
        traces: list[AgentTrace] = []
        total_tool_calls = 0
        last_model = "agent"

        for step in range(1, step_limit + 1):
            turn = self.model.complete(
                conversation,
                definitions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                seed=seed,
            )
            last_model = turn.model
            if not turn.tool_calls:
                return AgentResult(
                    text=turn.text,
                    model=last_model,
                    steps=step,
                    tool_calls=total_tool_calls,
                    finish_reason="stop",
                    trace=tuple(traces),
                )

            conversation.append(ChatMessage(role="assistant", content=turn.raw_text))
            for call in turn.tool_calls:
                total_tool_calls += 1
                execution = self.executor.execute(call)
                traces.append(self._trace(step, call, execution))
                conversation.append(
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        content=execution.content,
                    )
                )

        return AgentResult(
            text="达到最大工具调用轮数，未能完成任务。",
            model=last_model,
            steps=step_limit,
            tool_calls=total_tool_calls,
            finish_reason="max_steps",
            trace=tuple(traces),
        )

    @staticmethod
    def _trace(
        step: int,
        call: ToolCall,
        execution: ToolExecutionResult,
    ) -> AgentTrace:
        return AgentTrace(
            step=step,
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            status=execution.status,
            content=execution.content,
            duration_ms=execution.duration_ms,
        )
