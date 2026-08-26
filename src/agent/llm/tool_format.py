"""Shared JSON tool-call protocol used by every chat backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent.messages import Message, ToolSchema, as_message
from agent.tools.base import ToolCall, ToolDefinition


def schemas_from_definitions(
    tools: Sequence[ToolDefinition | ToolSchema],
) -> tuple[ToolSchema, ...]:
    """Normalize registry definitions or backend schemas."""
    schemas: list[ToolSchema] = []
    for tool in tools:
        if isinstance(tool, ToolSchema):
            schemas.append(tool)
            continue
        schemas.append(
            ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        )
    return tuple(schemas)


def tool_instruction(tools: Sequence[ToolSchema]) -> str:
    """Build the explicit JSON tool protocol used by small local models."""
    definitions = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "You are a tool-using agent. Available tools are described below.\n"
        f"{definitions}\n"
        "If a tool is required, output only JSON in this exact shape: "
        '{"tool_call":{"name":"tool_name","arguments":{}}}. '
        "Do not use Markdown fences. If no tool is needed, answer normally. "
        "After using search_docs, include the exact citation value for every "
        "document relied on in the final answer."
    )


def prepare_messages(
    messages: Sequence[Message | Any],
    tools: Sequence[ToolSchema] | None = None,
    *,
    inject_instruction: bool = True,
) -> list[Message]:
    """Normalize messages and optionally prepend tool instructions."""
    prepared = [as_message(message) for message in messages]
    if not tools or not inject_instruction:
        return prepared
    instruction = tool_instruction(tools)
    if prepared and prepared[0].role == "system":
        first = prepared[0]
        prepared[0] = Message(
            role="system",
            content=f"{first.content}\n\n{instruction}",
            name=first.name,
            tool_call_id=first.tool_call_id,
        )
    else:
        prepared.insert(0, Message(role="system", content=instruction))
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
