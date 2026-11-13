from __future__ import annotations

import numpy as np

import pytest

from hybrid_rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from hybrid_rag.models import Chunk
from hybrid_rag.retrieval import BM25Index, DenseIndex


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="manual.md", text=text, start=0, end=len(text))


def test_hashing_embedder_is_deterministic_normalized_and_finite() -> None:
    embedder = HashingEmbedder(dimensions=64)
    first = embedder.encode(["向量检索", "错误码 E101"])
    second = embedder.encode(["向量检索", "错误码 E101"])

    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), [1.0, 1.0], atol=1e-6)


def test_bm25_prioritizes_exact_error_code() -> None:
    chunks = [
        make_chunk("a", "服务返回错误码 E101 时检查令牌"),
        make_chunk("b", "向量服务运行缓慢"),
    ]

    results = BM25Index(chunks).search("如何处理 E101", k=2)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"


def test_bm25_returns_no_candidate_without_lexical_evidence() -> None:
    chunks = [make_chunk("a", "数据库连接池配置")]

    assert BM25Index(chunks).search("火星天气", k=3) == []


def test_dense_index_retrieves_related_character_ngrams_and_reports_backend() -> None:
    chunks = [
        make_chunk("a", "混合检索结合关键词与语义向量"),
        make_chunk("b", "数据库连接池配置"),
    ]
    index = DenseIndex(chunks, HashingEmbedder(dimensions=128), prefer_faiss=False)

    results = index.search("语义向量检索", k=1)

    assert results[0].chunk.chunk_id == "a"
    assert index.backend == "numpy"


def test_empty_query_returns_no_results() -> None:
    chunks = [make_chunk("a", "文本")]

    assert BM25Index(chunks).search("  ", k=3) == []
    assert DenseIndex(chunks, prefer_faiss=False).search("  ", k=3) == []


def test_model_adapter_rejects_nan_before_indexing() -> None:
    class NaNModel:
        def encode(self, texts, **kwargs):
            del texts, kwargs
            return np.array([[np.nan, 1.0]], dtype=np.float32)

    embedder = SentenceTransformerEmbedder(model=NaNModel())

    with pytest.raises(ValueError, match="非有限向量"):
        embedder.encode(["故障注入"])
