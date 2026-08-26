"""Bounded Agent loop that connects a chat model to registered tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

from agent.context import AgentContext
from agent.messages import Message, as_message
from agent.model_adapter import AgentTurn
from agent.prompts import (
    CANCELLED_MESSAGE,
    CITATION_GUARD_MESSAGE,
    CONFIRMATION_MESSAGE,
    DEFAULT_MAX_STEPS,
    INCOMPLETE_PLAN_MESSAGE,
    MAX_ALLOWED_STEPS,
    MAX_STEPS_MESSAGE,
    REPEATED_CALL_MESSAGE,
)
from agent.reliability import (
    CancellationToken,
    CitationGuard,
    CitationValidation,
    SchemaArgumentRepairer,
    compact_messages,
)
from agent.routing import AgentRouter, RouteDecision
from agent.tools.base import ToolCall, ToolDefinition, ToolExecutionResult
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from agent.tracing import AgentEvent, AgentTracer, redact, redact_content
from minigpt_service.schemas import ChatMessage

AgentFinishReason = Literal[
    "stop",
    "max_steps",
    "confirmation_required",
    "cancelled",
    "citation_guard",
]


@dataclass(frozen=True)
class AgentUsage:
    """Request-level token and latency accounting."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    model_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0
    total_duration_ms: float = 0.0


@dataclass(frozen=True)
class AgentProgressEvent:
    """Small event emitted during an Agent run for SSE and local observers."""

    kind: Literal["route", "model", "tool", "plan", "final"]
    step: int
    payload: dict[str, object]


class AgentModel(Protocol):
    """Model contract required by the orchestrator."""

    @property
    def is_ready(self) -> bool:
        """Return whether the underlying chat model is ready."""

    def complete(
        self,
        messages: Sequence[Message],
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
    status: Literal["ok", "error", "confirmation_required"]
    content: str
    duration_ms: float
    confirmation_id: str | None = None
    argument_repaired: bool = False
    attempts: int = 1


@dataclass(frozen=True)
class AgentResult:
    """Final answer plus bounded execution trace."""

    text: str
    model: str
    steps: int
    tool_calls: int
    finish_reason: AgentFinishReason
    trace: tuple[AgentTrace, ...]
    events: tuple[AgentEvent, ...] = ()
    route: RouteDecision | None = None
    usage: AgentUsage = AgentUsage()
    citation: CitationValidation | None = None


class AgentOrchestrator:
    """Run a model/tool loop with explicit limits and no hidden tools."""

    def __init__(
        self,
        model: AgentModel,
        registry: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        executor: ToolExecutor | None = None,
        tracer_factory: type[AgentTracer] = AgentTracer,
        router: AgentRouter | None = None,
        argument_repairer: SchemaArgumentRepairer | None = None,
        max_context_characters: int = 65_536,
    ) -> None:
        if not 1 <= max_steps <= MAX_ALLOWED_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_ALLOWED_STEPS}.")
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.executor = executor or ToolExecutor(registry)
        self._owns_executor = executor is None
        self._tracer_factory = tracer_factory
        self.router = router
        self.argument_repairer = argument_repairer or SchemaArgumentRepairer()
        if max_context_characters < 512:
            raise ValueError("max_context_characters must be at least 512.")
        self.max_context_characters = max_context_characters

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
        messages: Sequence[ChatMessage | Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        seed: int | None,
        max_steps: int | None = None,
        approved_side_effects: Sequence[str] = (),
        cancellation_token: CancellationToken | None = None,
        progress_callback: Callable[[AgentProgressEvent], None] | None = None,
    ) -> AgentResult:
        """Answer directly or execute tools until a final answer is produced."""
        if not self.is_ready:
            raise RuntimeError("Agent model is not ready.")
        step_limit = self.max_steps if max_steps is None else max_steps
        if not 1 <= step_limit <= self.max_steps:
            raise ValueError(f"max_steps must be between 1 and {self.max_steps}.")

        run_started = perf_counter()
        context = AgentContext(messages)
        definitions = self.registry.definitions()
        route = None
        if self.router is not None:
            route = self.router.route(
                [as_message(message) for message in messages],
                self.registry.names(),
            )
            allowed = set(route.allowed_tools)
            definitions = tuple(
                definition for definition in definitions if definition.name in allowed
            )
        self._progress(
            progress_callback,
            AgentProgressEvent(
                "route",
                0,
                {
                    "kind": route.kind if route else "auto",
                    "allowed_tools": list(route.allowed_tools)
                    if route
                    else list(self.registry.names()),
                },
            ),
        )
        traces: list[AgentTrace] = []
        tracer = self._tracer_factory()
        total_tool_calls = 0
        last_model = "agent"
        prompt_tokens = 0
        generated_tokens = 0
        model_duration_ms = 0.0
        tool_duration_ms = 0.0
        required_tools = set(
            route.allowed_tools if route and route.kind == "multi_step" else ()
        )
        citation_guard = CitationGuard()

        for step in range(1, step_limit + 1):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                return self._result(
                    text=CANCELLED_MESSAGE,
                    model=last_model,
                    steps=max(1, step - 1),
                    tool_calls=total_tool_calls,
                    finish_reason="cancelled",
                    traces=traces,
                    tracer=tracer,
                    route=route,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    model_duration_ms=model_duration_ms,
                    tool_duration_ms=tool_duration_ms,
                    run_started=run_started,
                    citation=None,
                    callback=progress_callback,
                )
            started = perf_counter()
            try:
                turn = self.model.complete(
                    compact_messages(
                        context.messages,
                        max_characters=self.max_context_characters,
                    ),
                    definitions,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy,
                    seed=seed,
                )
            except Exception as error:
                tracer.record_model(
                    step,
                    list(context.messages),
                    None,
                    (perf_counter() - started) * 1000,
                    error,
                )
                raise
            tracer.record_model(
                step,
                list(context.messages),
                turn,
                (perf_counter() - started) * 1000,
            )
            duration = (perf_counter() - started) * 1000
            model_duration_ms += duration
            prompt_tokens += turn.prompt_tokens
            generated_tokens += turn.generated_tokens
            self._progress(
                progress_callback,
                AgentProgressEvent(
                    "model",
                    step,
                    {
                        "model": turn.model,
                        "prompt_tokens": turn.prompt_tokens,
                        "generated_tokens": turn.generated_tokens,
                        "tool_calls": [call.name for call in turn.tool_calls],
                        "duration_ms": round(duration, 3),
                    },
                ),
            )
            last_model = turn.model
            if not turn.tool_calls:
                if required_tools and step < step_limit:
                    remaining = ", ".join(sorted(required_tools))
                    context.messages.append(
                        Message(
                            role="system",
                            content=INCOMPLETE_PLAN_MESSAGE.format(tools=remaining),
                        )
                    )
                    self._progress(
                        progress_callback,
                        AgentProgressEvent(
                            "plan",
                            step,
                            {"remaining_tools": sorted(required_tools)},
                        ),
                    )
                    continue
                citation = citation_guard.validate(turn.text)
                finish_reason: AgentFinishReason = "stop"
                text = turn.text
                if not citation.is_valid:
                    finish_reason = "citation_guard"
                    text = CITATION_GUARD_MESSAGE
                return self._result(
                    text=text,
                    model=last_model,
                    steps=step,
                    tool_calls=total_tool_calls,
                    finish_reason=finish_reason,
                    traces=traces,
                    tracer=tracer,
                    route=route,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    model_duration_ms=model_duration_ms,
                    tool_duration_ms=tool_duration_ms,
                    run_started=run_started,
                    citation=citation,
                    callback=progress_callback,
                )

            context.append_assistant(turn.raw_text)
            for original_call in turn.tool_calls:
                total_tool_calls += 1
                repaired_call = self.argument_repairer.repair(
                    original_call,
                    self.registry,
                )
                call = repaired_call or original_call
                argument_repaired = call.arguments != original_call.arguments
                if not context.register_tool_call(call):
                    execution = ToolExecutionResult(
                        call_id=call.call_id,
                        name=call.name,
                        status="error",
                        content=REPEATED_CALL_MESSAGE,
                        duration_ms=0.0,
                    )
                else:
                    if (
                        cancellation_token is not None
                        and cancellation_token.is_cancelled
                    ):
                        return self._result(
                            text=CANCELLED_MESSAGE,
                            model=last_model,
                            steps=step,
                            tool_calls=total_tool_calls - 1,
                            finish_reason="cancelled",
                            traces=traces,
                            tracer=tracer,
                            route=route,
                            prompt_tokens=prompt_tokens,
                            generated_tokens=generated_tokens,
                            model_duration_ms=model_duration_ms,
                            tool_duration_ms=tool_duration_ms,
                            run_started=run_started,
                            citation=None,
                            callback=progress_callback,
                        )
                    execution = self._execute_with_retry(
                        call,
                        approved_side_effects=approved_side_effects,
                    )
                    context.record_tool_result(call, execution)
                tool_duration_ms += execution.duration_ms
                citation_guard.record(call, execution)
                if execution.status == "ok":
                    required_tools.discard(call.name)
                tracer.record_tool(step, call, execution)
                traces.append(
                    self._trace(
                        step,
                        call,
                        execution,
                        argument_repaired=argument_repaired,
                    )
                )
                context.append_tool(call, execution)
                self._progress(
                    progress_callback,
                    AgentProgressEvent(
                        "tool",
                        step,
                        {
                            "name": call.name,
                            "status": execution.status,
                            "duration_ms": execution.duration_ms,
                            "argument_repaired": argument_repaired,
                            "attempts": execution.attempts,
                        },
                    ),
                )
                if execution.status == "confirmation_required":
                    return self._result(
                        text=CONFIRMATION_MESSAGE,
                        model=last_model,
                        steps=step,
                        tool_calls=total_tool_calls,
                        finish_reason="confirmation_required",
                        traces=traces,
                        tracer=tracer,
                        route=route,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=generated_tokens,
                        model_duration_ms=model_duration_ms,
                        tool_duration_ms=tool_duration_ms,
                        run_started=run_started,
                        citation=None,
                        callback=progress_callback,
                    )

        return self._result(
            text=MAX_STEPS_MESSAGE,
            model=last_model,
            steps=step_limit,
            tool_calls=total_tool_calls,
            finish_reason="max_steps",
            traces=traces,
            tracer=tracer,
            route=route,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            model_duration_ms=model_duration_ms,
            tool_duration_ms=tool_duration_ms,
            run_started=run_started,
            citation=None,
            callback=progress_callback,
        )

    @staticmethod
    def _trace(
        step: int,
        call: ToolCall,
        execution: ToolExecutionResult,
        *,
        argument_repaired: bool = False,
    ) -> AgentTrace:
        return AgentTrace(
            step=step,
            call_id=call.call_id,
            name=call.name,
            arguments=redact(call.arguments),
            status=execution.status,
            content=redact_content(execution.content),
            duration_ms=execution.duration_ms,
            confirmation_id=execution.confirmation_id,
            argument_repaired=argument_repaired,
            attempts=execution.attempts,
        )

    def _execute_with_retry(
        self,
        call: ToolCall,
        *,
        approved_side_effects: Sequence[str],
    ) -> ToolExecutionResult:
        first = self.executor.execute(
            call,
            approved_side_effects=approved_side_effects,
        )
        transient = first.content in {
            "Tool timed out.",
            "Tool executor is busy.",
        } or first.content.startswith("Tool failed to start:")
        tool = self.registry.get(call.name)
        if first.status != "error" or not transient or tool is None or tool.side_effect:
            return first
        second = self.executor.execute(
            call,
            approved_side_effects=approved_side_effects,
        )
        return ToolExecutionResult(
            call_id=second.call_id,
            name=second.name,
            status=second.status,
            content=second.content,
            duration_ms=round(first.duration_ms + second.duration_ms, 3),
            confirmation_id=second.confirmation_id,
            attempts=2,
        )

    @staticmethod
    def _progress(
        callback: Callable[[AgentProgressEvent], None] | None,
        event: AgentProgressEvent,
    ) -> None:
        if callback is not None:
            callback(event)

    @classmethod
    def _result(
        cls,
        *,
        text: str,
        model: str,
        steps: int,
        tool_calls: int,
        finish_reason: AgentFinishReason,
        traces: list[AgentTrace],
        tracer: AgentTracer,
        route: RouteDecision | None,
        prompt_tokens: int,
        generated_tokens: int,
        model_duration_ms: float,
        tool_duration_ms: float,
        run_started: float,
        citation: CitationValidation | None,
        callback: Callable[[AgentProgressEvent], None] | None,
    ) -> AgentResult:
        usage = AgentUsage(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            model_duration_ms=round(model_duration_ms, 3),
            tool_duration_ms=round(tool_duration_ms, 3),
            total_duration_ms=round((perf_counter() - run_started) * 1000, 3),
        )
        result = AgentResult(
            text=text,
            model=model,
            steps=steps,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            trace=tuple(traces),
            events=tracer.events,
            route=route,
            usage=usage,
            citation=citation,
        )
        cls._progress(
            callback,
            AgentProgressEvent(
                "final",
                steps,
                {
                    "finish_reason": finish_reason,
                    "total_duration_ms": usage.total_duration_ms,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                },
            ),
        )
        return result
