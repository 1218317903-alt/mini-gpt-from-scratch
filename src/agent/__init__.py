"""Small, framework-independent Agent orchestration layer."""

from agent.builtin_tools import (
    CorpusDocumentSearch,
    LocalDocumentSearch,
    build_default_registry,
)
from agent.context import AgentContext
from agent.document_tools import DocumentWorkspace, register_document_tools
from agent.domain_adapter import (
    DomainAdapterController,
    DomainAdapterManifest,
    DomainAdapterRegistry,
    validate_domain_dataset,
)
from agent.messages import Message, ToolSchema
from agent.model_adapter import ChatModelAgentAdapter
from agent.multimodal import (
    AgentTextResponder,
    MultimodalConversationPipeline,
    MultimodalWorkspace,
    Pyttsx3TTSBackend,
    TransformersTrOCRBackend,
    TransformersWhisperBackend,
    register_multimodal_tools,
)
from agent.orchestrator import (
    AgentOrchestrator,
    AgentProgressEvent,
    AgentResult,
    AgentTrace,
    AgentUsage,
)
from agent.reliability import (
    CancellationToken,
    CitationGuard,
    CitationValidation,
    SchemaArgumentRepairer,
    compact_messages,
)
from agent.retrieval import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    RetrievalSearchHandler,
    TransformerMeanPoolEmbedder,
)
from agent.routing import AgentRouter, HeuristicAgentRouter, RouteDecision
from agent.tools.base import ToolCall, ToolDefinition, ToolExecutionResult, ToolSpec
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from agent.tracing import AgentEvent, AgentTracer

__all__ = [
    "AgentOrchestrator",
    "AgentProgressEvent",
    "AgentContext",
    "AgentEvent",
    "AgentResult",
    "AgentTrace",
    "AgentUsage",
    "AgentTextResponder",
    "AgentTracer",
    "ChatModelAgentAdapter",
    "CancellationToken",
    "CitationGuard",
    "CitationValidation",
    "CorpusDocumentSearch",
    "DenseRetriever",
    "DocumentWorkspace",
    "DomainAdapterController",
    "DomainAdapterManifest",
    "DomainAdapterRegistry",
    "HybridRetriever",
    "HeuristicAgentRouter",
    "LexicalRetriever",
    "Message",
    "MultimodalWorkspace",
    "MultimodalConversationPipeline",
    "AgentRouter",
    "RetrievalSearchHandler",
    "RouteDecision",
    "SchemaArgumentRepairer",
    "Pyttsx3TTSBackend",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSchema",
    "ToolSpec",
    "TransformerMeanPoolEmbedder",
    "TransformersTrOCRBackend",
    "TransformersWhisperBackend",
    "LocalDocumentSearch",
    "build_default_registry",
    "compact_messages",
    "register_document_tools",
    "register_multimodal_tools",
    "validate_domain_dataset",
]
