from __future__ import annotations

import pytest

from hybrid_rag.fusion import HybridRetriever
from hybrid_rag.generation import ExtractiveGenerator, extract_citations, validate_citations
from hybrid_rag.models import Chunk
from hybrid_rag.pipeline import RAGPipeline


class StaticGenerator:
    backend = "static-test"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, question: str, context: str) -> str:
        assert question
        assert context
        return self.text


def chunks() -> list[Chunk]:
    return [
        Chunk("aaaaaaaaaaaaaaaa", "auth.md", "# 标题\n\nE101 表示访问令牌已经过期。", 0, 24),
        Chunk("bbbbbbbbbbbbbbbb", "search.md", "RRF 使用排名而不是原始分数做融合。", 0, 20),
    ]


def test_extractive_pipeline_returns_traceable_citations() -> None:
    pipeline = RAGPipeline(HybridRetriever.from_chunks(chunks()), ExtractiveGenerator(), top_k=2)

    answer = pipeline.ask("E101 是什么意思？")

    assert answer.citations_valid is True
    assert answer.evidence
    assert "令牌已经过期" in answer.text
    assert set(extract_citations(answer.text)) <= {item.chunk.chunk_id for item in answer.evidence}
    assert answer.generator_backend == "extractive-offline"
    assert "bm25+" in answer.retrieval_backend


def test_pipeline_flags_hallucinated_citation() -> None:
    generator = StaticGenerator("答案来自不存在的证据 [cccccccccccccccc]")
    pipeline = RAGPipeline(HybridRetriever.from_chunks(chunks()), generator, top_k=1)

    answer = pipeline.ask("E101 是什么意思？")

    assert answer.citations_valid is False


def test_validate_citations_requires_at_least_one_valid_reference() -> None:
    assert validate_citations("无引用", {"aaaaaaaaaaaaaaaa"}) is False
    assert validate_citations("依据 [aaaaaaaaaaaaaaaa]", {"aaaaaaaaaaaaaaaa"}) is True


def test_pipeline_rejects_empty_question() -> None:
    pipeline = RAGPipeline(HybridRetriever.from_chunks(chunks()), ExtractiveGenerator())

    with pytest.raises(ValueError, match="问题"):
        pipeline.ask("  ")
