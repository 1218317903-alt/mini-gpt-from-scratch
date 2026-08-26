"""Tests for lexical, dense and hybrid retrieval backends."""

from agent.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever


class _FakeEmbedder:
    _vectors = {
        "苹果公司总部": (1.0, 0.0),
        "水果香蕉": (0.0, 1.0),
        "苹果的办公地点": (1.0, 0.1),
    }

    def encode(self, texts):
        return [self._vectors[text] for text in texts]


def _documents():
    return [
        {"id": "doc-company", "text": "苹果公司总部"},
        {"id": "doc-fruit", "text": "水果香蕉"},
    ]


def test_dense_retriever_uses_cosine_similarity() -> None:
    retriever = DenseRetriever(_documents(), _FakeEmbedder())

    hits = retriever.search("苹果的办公地点", k=1)

    assert hits[0].citation == "doc-company"


def test_hybrid_retriever_fuses_both_rankings_deterministically() -> None:
    lexical = LexicalRetriever(_documents())
    dense = DenseRetriever(_documents(), _FakeEmbedder())
    hybrid = HybridRetriever(lexical, dense)

    hits = hybrid.search("苹果的办公地点", k=2)

    assert [hit.citation for hit in hits] == ["doc-company", "doc-fruit"]


def test_hybrid_retriever_rejects_mismatched_indexes() -> None:
    lexical = LexicalRetriever(_documents())
    dense = DenseRetriever([{"id": "other", "text": "苹果公司总部"}], _FakeEmbedder())

    try:
        HybridRetriever(lexical, dense)
    except ValueError as error:
        assert "same documents" in str(error)
    else:
        raise AssertionError("Expected mismatched hybrid indexes to fail")
