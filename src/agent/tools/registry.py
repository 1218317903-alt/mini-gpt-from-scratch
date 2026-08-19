"""Deterministic registry for explicitly allowed Agent tools."""

from __future__ import annotations

from collections.abc import Iterable

from agent.tools.base import ToolDefinition, ToolSpec


class ToolRegistry:
    """Own the allow-list of tools available to an Agent."""

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        """Register one tool, rejecting accidental name replacement."""
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        """Return a registered tool or ``None`` for an unknown name."""
        return self._tools.get(name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return stable, name-sorted definitions for model prompts."""
        return tuple(self._tools[name].definition() for name in sorted(self._tools))

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""
        return tuple(sorted(self._tools))
