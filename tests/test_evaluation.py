from __future__ import annotations

import pytest

from hybrid_rag.evaluation import (
    EvaluationCase,
    compare_pipelines,
    evaluate_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    required_term_coverage,
)
from hybrid_rag.fusion import HybridRetriever
from hybrid_rag.generation import ExtractiveGenerator
from hybrid_rag.models import Chunk
from hybrid_rag.pipeline import AgenticRAGPipeline, RAGPipeline


def test_ranking_metrics_have_known_values() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(
        (1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156)
    )


def test_source_level_metrics_ignore_repeated_chunks_from_the_same_source() -> None:
    retrieved = ["a", "a"]

    assert recall_at_k(retrieved, {"a"}, 2) == 1.0
    assert reciprocal_rank(retrieved, {"a"}) == 1.0
    assert ndcg_at_k(retrieved, {"a"}, 2) == 1.0


def test_required_term_coverage() -> None:
    assert required_term_coverage("令牌已经过期", ["令牌", "过期"]) == 1.0
    assert required_term_coverage("只提到令牌", ["令牌", "过期"]) == 0.5


def test_evaluate_cases_reports_per_case_and_aggregate_metrics() -> None:
    chunks = [
        Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 13),
        Chunk("bbbbbbbbbbbbbbbb", "rrf.md", "RRF 按排名融合。", 0, 10),
    ]
    pipeline = RAGPipeline(HybridRetriever.from_chunks(chunks), ExtractiveGenerator(), top_k=1)
    cases = [EvaluationCase("q1", "E101 是什么？", ("auth.md",), ("令牌", "过期"))]

    report = evaluate_cases(pipeline, cases)

    assert report["summary"]["recall_at_1"] == 1.0
    assert report["summary"]["citation_validity"] == 1.0
    assert report["cases"][0]["retrieved_sources"] == ["auth.md"]
    assert report["summary"]["average_agent_iterations"] == 0.0


def test_evaluate_cases_reports_agent_routes_iterations_and_abstention() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 13)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False),
        ExtractiveGenerator(),
        top_k=1,
    )

    report = evaluate_cases(
        pipeline,
        [EvaluationCase("q1", "E101 是什么？", ("auth.md",), ("令牌",))],
    )

    assert report["summary"]["average_agent_iterations"] == 1.0
    assert report["summary"]["agent_abstention_rate"] == 0.0
    assert report["summary"]["agent_route_distribution"] == {"lexical": 1}
    assert report["summary"]["task_success_rate"] == 1.0
    assert report["cases"][0]["agent"] == {
        "iterations": 1,
        "routes": ["lexical"],
        "actions": ["answer"],
        "stop_reason": "evidence_sufficient",
    }


def test_evaluate_cases_scores_route_and_safe_abstention() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 13)]
    pipeline = AgenticRAGPipeline(
        HybridRetriever.from_chunks(chunks, prefer_faiss=False),
        ExtractiveGenerator(),
        top_k=1,
    )
    cases = [
        EvaluationCase(
            "answerable",
            "E101 是什么？",
            ("auth.md",),
            ("令牌",),
            expected_route="lexical",
            answerable=True,
        ),
        EvaluationCase(
            "absent",
            "E999 是什么？",
            (),
            (),
            expected_route="lexical",
            answerable=False,
        ),
    ]

    report = evaluate_cases(pipeline, cases)

    assert report["summary"]["tool_selection_accuracy"] == 1.0
    assert report["summary"]["task_success_rate"] == 1.0
    assert report["summary"]["abstention_precision"] == 1.0
    assert report["summary"]["abstention_recall"] == 1.0
    assert report["summary"]["abstention_f1"] == 1.0


def test_compare_pipelines_reports_signed_agent_deltas() -> None:
    chunks = [Chunk("aaaaaaaaaaaaaaaa", "auth.md", "E101 表示令牌过期。", 0, 13)]
    retriever = HybridRetriever.from_chunks(chunks, prefer_faiss=False)
    fixed = RAGPipeline(retriever, ExtractiveGenerator(), top_k=1)
    agentic = AgenticRAGPipeline(retriever, ExtractiveGenerator(), top_k=1)
    cases = [
        EvaluationCase("absent", "E999 是什么？", (), answerable=False),
    ]

    report = compare_pipelines(fixed, agentic, cases)

    assert report["dataset"]["human_gold"] is False
    assert report["fixed_pipeline"]["summary"]["case_count"] == 1
    assert report["agentic_pipeline"]["summary"]["case_count"] == 1
    assert report["agent_minus_fixed"]["task_success_rate"] == 1.0
