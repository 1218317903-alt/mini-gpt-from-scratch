"""Small built-in tools with deliberately narrow capabilities."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.base import ToolSpec
from agent.tools.registry import ToolRegistry


class CalculatorArguments(BaseModel):
    """Arguments accepted by the safe arithmetic calculator."""

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=256)


class CurrentTimeArguments(BaseModel):
    """Arguments accepted by the current-time tool."""

    model_config = ConfigDict(extra="forbid")

    timezone: str = Field(default="UTC", min_length=1, max_length=64)


class DocumentSearchArguments(BaseModel):
    """Arguments accepted by the bounded local-document retriever."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=5, ge=1, le=5)


class LocalDocumentSearch:
    """Small lexical retriever over a fixed, read-only local directory."""

    _SUFFIXES = {".md", ".rst", ".txt"}

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = 500,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise ValueError(f"Document root is not a directory: {resolved}")
        self.root = resolved
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes

    def __call__(self, query: str, k: int = 5) -> dict[str, Any]:
        documents: list[tuple[str, str]] = []
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self._SUFFIXES
        )[: self.max_files]
        for path in paths:
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            citation = path.relative_to(self.root).as_posix()
            documents.append((citation, text))
        return _search_document_pairs(query, documents, k=k)


class CorpusDocumentSearch:
    """Picklable lexical retriever over an explicit in-memory eval corpus."""

    def __init__(self, documents: Sequence[Mapping[str, str]]) -> None:
        by_id: dict[str, str] = {}
        for document in documents:
            doc_id = str(document["id"])
            text = str(document["text"])
            if not doc_id or not text:
                raise ValueError("Corpus documents require non-empty id and text.")
            previous = by_id.get(doc_id)
            if previous is not None and previous != text:
                raise ValueError(f"Conflicting corpus document: {doc_id}")
            by_id[doc_id] = text
        self.documents = tuple(sorted(by_id.items()))

    def __call__(self, query: str, k: int = 5) -> dict[str, Any]:
        return _search_document_pairs(query, self.documents, k=k)


def _search_document_pairs(
    query: str,
    documents: Sequence[tuple[str, str]],
    *,
    k: int,
) -> dict[str, Any]:
    query_tokens = _search_tokens(query)
    scored: list[tuple[float, str, str]] = []
    for citation, text in documents:
        tokens = _search_tokens(text)
        overlap = len(query_tokens & tokens)
        if overlap == 0:
            continue
        score = overlap / math.sqrt(max(1, len(query_tokens) * len(tokens)))
        scored.append((score, citation, _best_snippet(text, query_tokens)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {
        "query": query,
        "results": [
            {"citation": citation, "score": round(score, 6), "snippet": snippet}
            for score, citation, snippet in scored[:k]
        ],
    }


def _search_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    tokens.update(character for character in text if "\u4e00" <= character <= "\u9fff")
    return tokens


def _best_snippet(text: str, query_tokens: set[str]) -> str:
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    if not candidates:
        return ""
    best = max(
        candidates,
        key=lambda line: (len(query_tokens & _search_tokens(line)), -len(line)),
    )
    return best[:500]


def _safe_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only integer and decimal numeric literals are allowed.")
    if not math.isfinite(float(value)):
        raise ValueError("The number must be finite.")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _safe_number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100:
                raise ValueError("Exponent is too large.")
            result = left**right
        else:
            raise ValueError("Only basic arithmetic operators are allowed.")
        try:
            result_as_float = float(result)
        except OverflowError as error:
            raise ValueError(
                "Arithmetic result is outside the safe numeric range."
            ) from error
        if not math.isfinite(result_as_float) or abs(result_as_float) > 1e100:
            raise ValueError("Arithmetic result is outside the safe numeric range.")
        return result
    raise ValueError("Only numeric literals and basic arithmetic are allowed.")


def calculate(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression without executing code."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid arithmetic expression: {error}") from error
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def current_time(timezone: str = "UTC") -> str:
    """Return an ISO-8601 timestamp for an IANA timezone."""
    if timezone in {"UTC", "Etc/UTC"}:
        return datetime.now(UTC).isoformat(timespec="seconds")
    try:
        zone = ZoneInfo(timezone)
    except Exception as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error
    return datetime.now(zone).isoformat(timespec="seconds")


def build_default_registry(
    document_root: Path | None = None,
    *,
    documents: Sequence[Mapping[str, str]] | None = None,
    search_handler: Any | None = None,
    document_workspace: Path | None = None,
    multimodal_workspace: Any | None = None,
) -> ToolRegistry:
    """Build the safe default tool allow-list for the local Agent."""
    configured = sum(
        value is not None for value in (document_root, documents, search_handler)
    )
    if configured > 1:
        raise ValueError("Set only one of document_root, documents, or search_handler.")
    if search_handler is not None:
        document_search = search_handler
    else:
        if document_root is None:
            document_root = Path(__file__).resolve().parents[2] / "docs"
        document_search = (
            CorpusDocumentSearch(documents)
            if documents is not None
            else LocalDocumentSearch(document_root)
        )
    registry = ToolRegistry(
        [
            ToolSpec(
                name="calculator",
                description="Evaluate a simple arithmetic expression safely.",
                arguments_model=CalculatorArguments,
                handler=calculate,
            ),
            ToolSpec(
                name="current_time",
                description="Return the current ISO-8601 time in an IANA timezone.",
                arguments_model=CurrentTimeArguments,
                handler=current_time,
            ),
            ToolSpec(
                name="search_docs",
                description=(
                    "Search the configured read-only local documentation and return "
                    "ranked snippets with citations. Cite the exact citation values "
                    "used in the final answer."
                ),
                arguments_model=DocumentSearchArguments,
                handler=document_search,
                timeout_seconds=5.0,
            ),
        ]
    )
    if document_workspace is not None:
        from agent.document_tools import DocumentWorkspace, register_document_tools

        register_document_tools(registry, DocumentWorkspace(document_workspace))
    if multimodal_workspace is not None:
        from agent.multimodal import register_multimodal_tools

        register_multimodal_tools(registry, multimodal_workspace)
    return registry
