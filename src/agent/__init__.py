"""Small, framework-independent Agent orchestration layer."""

from agent.builtin_tools import build_default_registry
from agent.model_adapter import ChatModelAgentAdapter
from agent.orchestrator import AgentOrchestrator, AgentResult, AgentTrace
from agent.tools.base import ToolCall, ToolDefinition, ToolExecutionResult, ToolSpec
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry

__all__ = [
    "AgentOrchestrator",
    "AgentResult",
    "AgentTrace",
    "ChatModelAgentAdapter",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
]
