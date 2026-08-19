"""Adapt the stage-one chat backend to a small JSON tool-call protocol."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.tools.base import ToolCall, ToolDefinition
from minigpt_service.chat_model import ChatModel
from minigpt_service.schemas import ChatMessage


@dataclass(frozen=True)
class AgentTurn:
    """One model turn after parsing any requested tool calls."""

    text: str
    raw_text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    generated_tokens: int
    model: str


class ChatModelAgentAdapter:
    """Use a normal chat model with explicit JSON tool instructions."""

    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    @property
    def is_ready(self) -> bool:
        """Return whether the wrapped chat model is ready."""
        return self.chat_model.is_ready

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
        """Prompt the chat model and parse its optional JSON tool call."""
        prepared_messages = self._add_tool_instructions(messages, tools)
        result = self.chat_model.chat(
            prepared_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            seed=seed,
        )
        tool_calls = parse_tool_calls(result.text)
        return AgentTurn(
            text="" if tool_calls else result.text,
            raw_text=result.text,
            tool_calls=tool_calls,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            model=result.model,
        )

    @staticmethod
    def _add_tool_instructions(
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> list[ChatMessage]:
        if not tools:
            return list(messages)
        definitions = json.dumps(
            [tool.as_dict() for tool in tools],
            ensure_ascii=False,
            sort_keys=True,
        )
        instruction = (
            "You are a tool-using agent. Available tools are described below.\n"
            f"{definitions}\n"
            "If a tool is required, output only JSON in this exact shape: "
            '{"tool_call":{"name":"tool_name","arguments":{}}}. '
            "Do not use Markdown fences. If no tool is needed, answer normally."
        )
        prepared = list(messages)
        if prepared and prepared[0].role == "system":
            first = prepared[0]
            prepared[0] = ChatMessage(
                role="system",
                content=f"{first.content}\n\n{instruction}",
                name=first.name,
            )
        else:
            prepared.insert(0, ChatMessage(role="system", content=instruction))
        return prepared


def parse_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Parse the supported JSON and ``<tool_call>`` response forms."""
    candidate = text.strip()
    if not candidate.startswith(("{", "[", "```", "<tool_call>")):
        return ()
    if "<tool_call>" in candidate:
        candidate = (
            candidate.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        )
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()

    payload = _decode_json(candidate)
    if payload is None:
        return ()
    if isinstance(payload, dict) and "tool_call" in payload:
        payload = payload["tool_call"]
    if isinstance(payload, dict) and "tool_calls" in payload:
        payload = payload["tool_calls"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return ()

    calls: list[ToolCall] = []
    for index, item in enumerate(payload, start=1):
        call = _parse_one_call(item, index)
        if call is None:
            return ()
        calls.append(call)
    return tuple(calls)


def _decode_json(candidate: str) -> Any | None:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            return value
    return None


def _parse_one_call(item: Any, index: int) -> ToolCall | None:
    if not isinstance(item, dict):
        return None
    function = item.get("function")
    if isinstance(function, dict):
        item = function
    name = item.get("name")
    if "arguments" not in item:
        return None
    arguments: Any = item["arguments"]
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    call_id = item.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    return ToolCall(call_id=call_id, name=name, arguments=arguments)
