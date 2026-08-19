"""Typed contracts shared by Agent tools and executors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

ToolStatus = Literal["ok", "error"]


@dataclass(frozen=True)
class ToolDefinition:
    """The model-facing description of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolCall:
    """A model-requested tool invocation."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result returned by the executor."""

    call_id: str
    name: str
    status: ToolStatus
    content: str
    duration_ms: float


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool and its Pydantic argument model."""

    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[..., Any]
    timeout_seconds: float = 10.0
    side_effect: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("Tool name must be non-empty and contain no whitespace.")
        if not self.description.strip():
            raise ValueError("Tool description must not be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("Tool timeout_seconds must be greater than zero.")

    def definition(self) -> ToolDefinition:
        """Build the model-facing JSON schema from the argument model."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.arguments_model.model_json_schema(),
        )
