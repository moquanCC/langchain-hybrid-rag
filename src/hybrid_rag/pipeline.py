from __future__ import annotations

from time import perf_counter

from .agent import (
    EvidenceJudge,
    QueryPlanner,
    QueryRewriter,
    RuleBasedQueryPlanner,
    RuleBasedQueryRewriter,
    RetrievalToolCall,
    RetrievalToolRegistry,
    exact_terms,
)
from .fusion import HybridRetriever
from .generation import EVIDENCE_SEPARATOR, Generator, validate_citations
from .models import AgentStep, Answer, ScoredChunk


def format_context(evidence: list[ScoredChunk]) -> str:
    return EVIDENCE_SEPARATOR.join(
        f"[{item.chunk.chunk_id}] {item.chunk.source}:{item.chunk.start}-{item.chunk.end}\n"
        f"{item.chunk.text}"
        for item in evidence
    )


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator, *, top_k: int = 5) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("问题不能为空")
        evidence = self.retriever.retrieve(question, self.top_k)
        if evidence:
            text = self.generator.generate(question, format_context(evidence))
        else:
            text = "未检索到相关证据。"
        allowed_ids = {item.chunk.chunk_id for item in evidence}
        return Answer(
            question=question,
            text=text,
            evidence=tuple(evidence),
            citations_valid=validate_citations(text, allowed_ids),
            generator_backend=self.generator.backend,
            retrieval_backend=self.retriever.backend,
        )


class AgenticRAGPipeline(RAGPipeline):
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        *,
        top_k: int = 5,
        max_iterations: int = 2,
        planner: QueryPlanner | None = None,
        evidence_judge: EvidenceJudge | None = None,
        rewriter: QueryRewriter | None = None,
        tool_registry: RetrievalToolRegistry | None = None,
    ) -> None:
        super().__init__(retriever, generator, top_k=top_k)
        if max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0")
        self.max_iterations = max_iterations
        self.planner = planner or RuleBasedQueryPlanner()
        self.evidence_judge = evidence_judge or EvidenceJudge()
        self.rewriter = rewriter or RuleBasedQueryRewriter()
        self.tool_registry = tool_registry or RetrievalToolRegistry(retriever)

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("问题不能为空")
        query = question.strip()
        trace: list[AgentStep] = []
        final_evidence: list[ScoredChunk] = []
        stop_reason = "max_iterations_reached"
        required_identifiers = exact_terms(question)

        for iteration in range(1, self.max_iterations + 1):
            route = self.planner.plan(query) if iteration == 1 else "hybrid"
            call = RetrievalToolCall(
                name=f"{route}_search",
                query=query,
                top_k=self.top_k,
            )
            started = perf_counter()
            evidence = self.tool_registry.execute(call)
            latency_ms = (perf_counter() - started) * 1000
            if required_identifiers:
                evidence = [
                    item
                    for item in evidence
                    if any(
                        identifier in item.chunk.text.lower()
                        or identifier in item.chunk.source.lower()
                        for identifier in required_identifiers
                    )
                ]
            sufficient, reason = self.evidence_judge.evaluate(query, evidence, route)
            final_evidence = evidence
            if sufficient:
                action = "answer"
                stop_reason = "evidence_sufficient"
            elif iteration >= self.max_iterations:
                action = "abstain"
            else:
                rewritten = self.rewriter.rewrite(question, query)
                if rewritten == query:
                    if route != "hybrid":
                        action = "expand_retrieval"
                    else:
                        action = "abstain"
                        stop_reason = "query_could_not_be_improved"
                else:
                    action = "rewrite"

            trace.append(
                AgentStep(
                    iteration=iteration,
                    query=query,
                    route=route,
                    evidence_count=len(evidence),
                    evidence_sufficient=sufficient,
                    reason=reason,
                    action=action,
                    evidence_ids=tuple(item.chunk.chunk_id for item in evidence),
                    tool_name=call.name,
                    tool_arguments=call.arguments,
                    latency_ms=latency_ms,
                )
            )
            if sufficient or action == "abstain":
                break
            query = rewritten

        if final_evidence and stop_reason == "evidence_sufficient":
            text = self.generator.generate(question, format_context(final_evidence))
        else:
            final_evidence = []
            text = "未检索到足够证据，已停止回答。"
        allowed_ids = {item.chunk.chunk_id for item in final_evidence}
        return Answer(
            question=question,
            text=text,
            evidence=tuple(final_evidence),
            citations_valid=validate_citations(text, allowed_ids),
            generator_backend=self.generator.backend,
            retrieval_backend=self.retriever.backend,
            agent_trace=tuple(trace),
            stop_reason=stop_reason,
        )
