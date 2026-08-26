"""Lexical, dense and hybrid retrieval with a shared auditable result contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievalHit:
    citation: str
    text: str
    score: float


class TextEmbedder(Protocol):
    """Minimal embedding contract so retrieval tests stay offline."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text."""


def search_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    tokens.update(character for character in text if "\u4e00" <= character <= "\u9fff")
    return tokens


def best_snippet(text: str, query_tokens: set[str]) -> str:
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    if not candidates:
        return ""
    best = max(
        candidates,
        key=lambda line: (len(query_tokens & search_tokens(line)), -len(line)),
    )
    return best[:500]


def _documents(documents: Sequence[Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
    by_id: dict[str, str] = {}
    for document in documents:
        citation = str(document["id"])
        text = str(document["text"])
        if not citation or not text:
            raise ValueError("Retrieval documents require non-empty id and text.")
        previous = by_id.get(citation)
        if previous is not None and previous != text:
            raise ValueError(f"Conflicting retrieval document: {citation}")
        by_id[citation] = text
    return tuple(sorted(by_id.items()))


class LexicalRetriever:
    """Deterministic overlap baseline used for ablations and fallback."""

    def __init__(self, documents: Sequence[Mapping[str, str]]) -> None:
        self.documents = _documents(documents)

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        query_tokens = search_tokens(query)
        scored: list[RetrievalHit] = []
        for citation, text in self.documents:
            tokens = search_tokens(text)
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(max(1, len(query_tokens) * len(tokens)))
            scored.append(RetrievalHit(citation, text, score))
        scored.sort(key=lambda hit: (-hit.score, hit.citation))
        return scored[:k]


class DenseRetriever:
    """Cosine-similarity retriever backed by an injected local embedder."""

    def __init__(
        self,
        documents: Sequence[Mapping[str, str]],
        embedder: TextEmbedder,
        *,
        query_prefix: str = "",
    ) -> None:
        self.documents = _documents(documents)
        self.embedder = embedder
        self.query_prefix = query_prefix
        self._vectors = tuple(
            self._normalize(vector)
            for vector in embedder.encode([text for _, text in self.documents])
        )
        if len(self._vectors) != len(self.documents):
            raise ValueError("Embedder returned the wrong number of document vectors.")

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        encoded = self.embedder.encode([self.query_prefix + query])
        if len(encoded) != 1:
            raise ValueError("Embedder must return exactly one query vector.")
        query_vector = self._normalize(encoded[0])
        scored = [
            RetrievalHit(citation, text, self._dot(query_vector, vector))
            for (citation, text), vector in zip(
                self.documents, self._vectors, strict=True
            )
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.citation))
        return scored[:k]

    @staticmethod
    def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if not values:
            raise ValueError("Embedding vectors cannot be empty.")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("Embedding vectors must have a finite non-zero norm.")
        return tuple(value / norm for value in values)

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vector dimensions do not match.")
        return sum(a * b for a, b in zip(left, right, strict=True))


class HybridRetriever:
    """Fuse lexical and dense ranks using weighted reciprocal-rank fusion."""

    def __init__(
        self,
        lexical: LexicalRetriever,
        dense: DenseRetriever,
        *,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_constant: int = 60,
    ) -> None:
        if lexical.documents != dense.documents:
            raise ValueError("Hybrid retrievers must index the same documents.")
        if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
            raise ValueError("Hybrid weights must be non-negative with a positive sum.")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive.")
        total = lexical_weight + dense_weight
        self.lexical = lexical
        self.dense = dense
        self.lexical_weight = lexical_weight / total
        self.dense_weight = dense_weight / total
        self.rrf_constant = rrf_constant

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        candidate_count = len(self.lexical.documents)
        lexical_hits = self.lexical.search(query, candidate_count)
        dense_hits = self.dense.search(query, candidate_count)
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        for weight, hits in (
            (self.lexical_weight, lexical_hits),
            (self.dense_weight, dense_hits),
        ):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.citation] = scores.get(hit.citation, 0.0) + weight / (
                    self.rrf_constant + rank
                )
                texts[hit.citation] = hit.text
        fused = [
            RetrievalHit(citation, texts[citation], score)
            for citation, score in scores.items()
        ]
        fused.sort(key=lambda hit: (-hit.score, hit.citation))
        return fused[:k]


class TransformerMeanPoolEmbedder:
    """Small local Transformer encoder with attention-mask mean pooling."""

    def __init__(
        self,
        model_source: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_source, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        torch = self._torch
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs: Any = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors


class RetrievalSearchHandler:
    """Adapt a retriever to the existing search_docs tool payload."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def __call__(self, query: str, k: int = 5) -> dict[str, Any]:
        query_tokens = search_tokens(query)
        return {
            "query": query,
            "results": [
                {
                    "citation": hit.citation,
                    "score": round(hit.score, 6),
                    "snippet": best_snippet(hit.text, query_tokens),
                }
                for hit in self.retriever.search(query, k)
            ],
        }
