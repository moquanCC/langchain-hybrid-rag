from __future__ import annotations

import pytest

from hybrid_rag.agent import RetrievalToolCall, RuleBasedQueryPlanner
from hybrid_rag.fusion import HybridRetriever
from hybrid_rag.generation import ExtractiveGenerator
from hybrid_rag.models import Chunk
from hybrid_rag.pipeline import AgenticRAGPipeline


def test_planner_routes_exact_semantic_and_general_queries() -> None:
    planner = RuleBasedQueryPlanner()

    assert planner.plan("E101 如何处理？") == "lexical"
    assert planner.plan("RRF 为什么按排名融合？") == "dense"
    assert planner.plan("介绍混合检索的配置与使用") == "hybrid"


def test_retrieval_tool_call_validates_schema() -> None:
    call = RetrievalToolCall(name="lexical_search", query="E101", top_k=3)

    assert call.strategy == "lexical"
    assert call.arguments == {"query": "E101", "top_k": 3}
    with pytest.raises(ValueError, match="不能为空"):
        RetrievalToolCall(name="dense_search", query=" ", top_k=3)
    with pytest.raises(ValueError, match="top_k"):
        RetrievalToolCall(name="hybrid_search", query="RRF", top_k=0)


def test_agent_stops_after_sufficient_first_pass_and_records_trace() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 12)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False),
        ExtractiveGenerator(),
    )

    answer = pipeline.ask("E101 如何处理？")

    assert answer.stop_reason == "evidence_sufficient"
    assert len(answer.agent_trace) == 1
    assert answer.agent_trace[0].route == "lexical"
    assert answer.agent_trace[0].action == "answer"
    assert answer.agent_trace[0].evidence_count == 1
    assert answer.agent_trace[0].tool_name == "lexical_search"
    assert answer.agent_trace[0].tool_arguments == {"query": "E101 如何处理？", "top_k": 5}
    assert answer.agent_trace[0].latency_ms >= 0.0
    assert answer.citations_valid is True


def test_agent_rewrites_failed_query_then_retries() -> None:
    class RetryPlanner:
        def plan(self, question: str) -> str:
            return "dense" if question.startswith("请问") else "lexical"

    chunks = [Chunk("aaaaaaaaaaaaaaaa", "rrf.md", "RRF 使用排名进行融合。", 0, 12)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False, min_dense_score=2.0),
        ExtractiveGenerator(),
        planner=RetryPlanner(),
    )

    answer = pipeline.ask("请问 RRF 如何融合？")

    assert len(answer.agent_trace) == 2
    assert answer.agent_trace[0].action == "rewrite"
    assert answer.agent_trace[1].query == "RRF 如何融合"
    assert answer.stop_reason == "evidence_sufficient"


def test_agent_abstains_when_query_cannot_be_improved() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "db.md", "数据库连接池", 0, 6)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False, min_dense_score=2.0),
        ExtractiveGenerator(),
        max_iterations=2,
    )

    answer = pipeline.ask("火星天气")

    assert answer.evidence == ()
    assert answer.text == "未检索到足够证据，已停止回答。"
    assert answer.stop_reason == "query_could_not_be_improved"
    assert answer.agent_trace[-1].action == "abstain"


def test_agent_rewrites_and_expands_specialized_route_before_abstaining() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "db.md", "数据库连接池", 0, 6)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False, min_dense_score=2.0),
        ExtractiveGenerator(),
    )

    answer = pipeline.ask("E999 如何处理？")

    assert [step.route for step in answer.agent_trace] == ["lexical", "hybrid"]
    assert answer.agent_trace[0].action == "rewrite"
    assert answer.agent_trace[1].action == "abstain"
    assert answer.stop_reason == "max_iterations_reached"


def test_exact_identifier_constraint_is_preserved_during_hybrid_retry() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 12)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False),
        ExtractiveGenerator(),
    )

    answer = pipeline.ask("E999 如何处理？")

    assert [step.tool_name for step in answer.agent_trace] == [
        "lexical_search",
        "hybrid_search",
    ]
    assert answer.evidence == ()
    assert answer.agent_trace[-1].action == "abstain"


def test_exact_filename_can_be_satisfied_by_chunk_source_metadata() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "chunking.md", "切片大小为 420。", 0, 10)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False),
        ExtractiveGenerator(),
    )

    answer = pipeline.ask("chunking.md 记录了什么？")

    assert answer.stop_reason == "evidence_sufficient"
    assert answer.evidence[0].chunk.source == "chunking.md"
