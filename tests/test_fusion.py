from __future__ import annotations

import numpy as np
import pytest

from hybrid_rag.fusion import HybridRetriever, reciprocal_rank_fusion
from hybrid_rag.models import Chunk, ScoredChunk
from hybrid_rag.reranking import IdentityReranker


def scored(chunk_id: str, score: float, stage: str) -> ScoredChunk:
    chunk = Chunk(chunk_id=chunk_id, source="x.md", text=chunk_id, start=0, end=1)
    return ScoredChunk(chunk=chunk, score=score, stage=stage)


def test_rrf_fuses_duplicates_and_uses_weights() -> None:
    lexical = [scored("a", 10, "bm25"), scored("b", 8, "bm25")]
    dense = [scored("b", 0.9, "dense"), scored("c", 0.8, "dense")]

    fused = reciprocal_rank_fusion([lexical, dense], weights=[0.99, 0.01], k=60)

    assert [item.chunk.chunk_id for item in fused] == ["a", "b", "c"]
    assert len({item.chunk.chunk_id for item in fused}) == 3
    assert fused[0].details["bm25"] > fused[1].details["bm25"]


def test_rrf_breaks_ties_by_chunk_id() -> None:
    result = reciprocal_rank_fusion(
        [[scored("b", 1, "dense"), scored("a", 1, "dense")]],
        weights=[1.0],
        k=60,
    )
    assert [item.chunk.chunk_id for item in result] == ["b", "a"]


@pytest.mark.parametrize(
    ("rankings", "weights", "k"),
    [([], [], 60), ([[scored("a", 1, "x")]], [1, 2], 60), ([[scored("a", 1, "x")]], [1], 0)],
)
def test_rrf_rejects_invalid_parameters(rankings, weights, k) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, weights=weights, k=k)


def test_hybrid_retriever_returns_up_to_requested_count_with_positive_evidence() -> None:
    chunks = [
        Chunk("a", "x.md", "错误码 E101 与身份令牌", 0, 16),
        Chunk("b", "x.md", "向量语义检索", 20, 26),
        Chunk("c", "x.md", "数据库连接", 30, 35),
    ]
    retriever = HybridRetriever.from_chunks(chunks, reranker=IdentityReranker())

    results = retriever.retrieve("E101 令牌", top_k=2)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"
    assert results[0].stage == "rerank"
    assert "local-hashing-char-ngram-v1" in retriever.backend


def test_hybrid_retriever_abstains_without_positive_sparse_or_dense_evidence() -> None:
    class ZeroEmbedder:
        name = "zero-test"
        dimensions = 2

        def encode(self, texts):
            return np.zeros((len(texts), self.dimensions), dtype=np.float32)

    chunks = [Chunk("a", "x.md", "数据库连接池配置", 0, 7)]
    retriever = HybridRetriever.from_chunks(
        chunks,
        embedder=ZeroEmbedder(),
        reranker=IdentityReranker(),
        min_dense_score=0.0,
    )

    assert retriever.retrieve("火星天气", top_k=3) == []


def test_retriever_exposes_separate_agent_tools() -> None:
    chunks = [
        Chunk("a", "x.md", "错误码 E101", 0, 8),
        Chunk("b", "x.md", "向量检索原理", 10, 16),
    ]
    retriever = HybridRetriever.from_chunks(chunks, prefer_faiss=False)

    lexical = retriever.retrieve("E101", strategy="lexical")
    dense = retriever.retrieve("向量检索", strategy="dense")

    assert [item.chunk.chunk_id for item in lexical] == ["a"]
    assert dense
    with pytest.raises(ValueError, match="strategy"):
        retriever.retrieve("问题", strategy="unknown")


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_hybrid_retriever_rejects_non_finite_dense_threshold(threshold: float) -> None:
    chunks = [Chunk("a", "x.md", "文本", 0, 2)]

    with pytest.raises(ValueError, match="min_dense_score"):
        HybridRetriever.from_chunks(chunks, min_dense_score=threshold)
