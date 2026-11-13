from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from .pipeline import RAGPipeline


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    relevant_sources: tuple[str, ...]
    required_terms: tuple[str, ...] = ()
    expected_route: str = ""
    answerable: bool = True


def unique_in_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    unique = unique_in_order(retrieved)
    return len(set(unique[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, item in enumerate(unique_in_order(retrieved), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    unique = unique_in_order(retrieved)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(unique[:k], start=1)
        if item in relevant
    )
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def required_term_coverage(answer: str, required_terms: Sequence[str]) -> float:
    if not required_terms:
        return 1.0
    lowered = answer.lower()
    return sum(term.lower() in lowered for term in required_terms) / len(required_terms)


def load_cases(path: str | Path) -> list[EvaluationCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            case_id=row["id"],
            question=row["question"],
            relevant_sources=tuple(row["relevant_sources"]),
            required_terms=tuple(row.get("required_terms", [])),
            expected_route=row.get("expected_route", ""),
            answerable=bool(row.get("answerable", True)),
        )
        for row in rows
    ]


def evaluate_cases(pipeline: RAGPipeline, cases: Sequence[EvaluationCase]) -> dict[str, object]:
    if not cases:
        raise ValueError("评测集不能为空")
    rows: list[dict[str, object]] = []
    route_counts: Counter[str] = Counter()
    total_agent_iterations = 0
    agentic_case_count = 0
    abstention_count = 0
    for case in cases:
        answer = pipeline.ask(case.question)
        trace = answer.agent_trace
        if trace:
            agentic_case_count += 1
            total_agent_iterations += len(trace)
            route_counts.update(step.route for step in trace)
            abstention_count += answer.stop_reason != "evidence_sufficient"
        retrieved_sources = unique_in_order([item.chunk.source for item in answer.evidence])
        relevant = set(case.relevant_sources)
        recall = recall_at_k(retrieved_sources, relevant, pipeline.top_k)
        term_coverage = required_term_coverage(answer.text, case.required_terms)
        abstained = (
            answer.stop_reason != "evidence_sufficient"
            if trace
            else not bool(answer.evidence)
        )
        task_success = (
            recall > 0.0 and term_coverage == 1.0 and answer.citations_valid
            if case.answerable
            else abstained
        )
        first_route = trace[0].route if trace else ""
        rows.append(
            {
                "id": case.case_id,
                "question": case.question,
                "retrieved_sources": retrieved_sources,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank(retrieved_sources, relevant),
                "ndcg": ndcg_at_k(retrieved_sources, relevant, pipeline.top_k),
                "required_term_coverage": term_coverage,
                "citation_valid": answer.citations_valid,
                "citation_safe": answer.citations_valid or abstained,
                "answerable": case.answerable,
                "abstained": abstained,
                "task_success": task_success,
                "expected_route": case.expected_route,
                "route_correct": (
                    first_route == case.expected_route
                    if case.expected_route and trace
                    else None
                ),
                "agent": {
                    "iterations": len(trace),
                    "routes": [step.route for step in trace],
                    "actions": [step.action for step in trace],
                    "stop_reason": answer.stop_reason,
                },
            }
        )
    predicted_abstentions = sum(bool(row["abstained"]) for row in rows)
    actual_unanswerable = sum(not bool(row["answerable"]) for row in rows)
    correct_abstentions = sum(
        bool(row["abstained"]) and not bool(row["answerable"]) for row in rows
    )
    abstention_precision = (
        correct_abstentions / predicted_abstentions if predicted_abstentions else 0.0
    )
    abstention_recall = (
        correct_abstentions / actual_unanswerable if actual_unanswerable else 0.0
    )
    abstention_f1 = (
        2 * abstention_precision * abstention_recall
        / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall
        else 0.0
    )
    route_rows = [row for row in rows if row["route_correct"] is not None]
    retry_rows = [row for row in rows if int(row["agent"]["iterations"]) > 1]
    summary = {
        f"recall_at_{pipeline.top_k}": fmean(float(row["recall"]) for row in rows),
        "mrr": fmean(float(row["reciprocal_rank"]) for row in rows),
        f"ndcg_at_{pipeline.top_k}": fmean(float(row["ndcg"]) for row in rows),
        "required_term_coverage": fmean(float(row["required_term_coverage"]) for row in rows),
        "citation_validity": fmean(float(bool(row["citation_safe"])) for row in rows),
        "case_count": len(rows),
        "task_success_rate": fmean(float(row["task_success"]) for row in rows),
        "tool_selection_accuracy": (
            fmean(float(row["route_correct"]) for row in route_rows) if route_rows else 0.0
        ),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "retry_success_rate": (
            fmean(
                float(row["agent"]["stop_reason"] == "evidence_sufficient")
                for row in retry_rows
            )
            if retry_rows
            else 0.0
        ),
        "retry_case_count": len(retry_rows),
        "max_step_termination_rate": (
            sum(row["agent"]["stop_reason"] == "max_iterations_reached" for row in rows)
            / agentic_case_count
            if agentic_case_count
            else 0.0
        ),
        "average_agent_iterations": (
            total_agent_iterations / agentic_case_count if agentic_case_count else 0.0
        ),
        "agent_abstention_rate": (
            abstention_count / agentic_case_count if agentic_case_count else 0.0
        ),
        "agent_route_distribution": dict(sorted(route_counts.items())),
    }
    return {
        "backends": {
            "retrieval": pipeline.retriever.backend,
            "generator": pipeline.generator.backend,
        },
        "summary": summary,
        "cases": rows,
    }


def compare_pipelines(
    fixed: RAGPipeline,
    agentic: RAGPipeline,
    cases: Sequence[EvaluationCase],
) -> dict[str, object]:
    fixed_report = evaluate_cases(fixed, cases)
    agent_report = evaluate_cases(agentic, cases)
    fixed_summary = fixed_report["summary"]
    agent_summary = agent_report["summary"]
    delta_fields = (
        "task_success_rate",
        "citation_validity",
        "abstention_f1",
        f"recall_at_{agentic.top_k}",
        f"ndcg_at_{agentic.top_k}",
    )
    return {
        "dataset": {
            "synthetic": True,
            "human_gold": False,
            "case_count": len(cases),
            "purpose": "bounded_agent_behavior_regression",
        },
        "fixed_pipeline": fixed_report,
        "agentic_pipeline": agent_report,
        "agent_minus_fixed": {
            key: float(agent_summary[key]) - float(fixed_summary[key])
            for key in delta_fields
        },
        "limitations": [
            "The questions and reference sources are a synthetic regression benchmark, not human gold.",
            "The deterministic Agent policy is bounded to configured retrieval tools and is not an open-ended planner.",
            "Hashing comparison results must not be mixed with the separate BGE-M3/CrossEncoder smoke run.",
        ],
    }
