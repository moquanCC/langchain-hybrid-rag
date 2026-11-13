from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from .models import ScoredChunk


EXACT_PATTERN = re.compile(
    r"(?:\b[A-Za-z]+[-_]?\d{2,}\b|\bv?\d+\.\d+(?:\.\d+)?\b|(?:[/\\][\w.~-]+){2,}|\b[\w-]+\.(?:py|json|ya?ml|md|txt)\b)",
    re.IGNORECASE,
)
SEMANTIC_HINTS = ("什么意思", "含义", "原理", "为什么", "区别", "概念", "相似", "作用")


def exact_terms(question: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in EXACT_PATTERN.finditer(question))


class QueryPlanner(Protocol):
    def plan(self, question: str) -> str: ...


class QueryRewriter(Protocol):
    def rewrite(self, original_question: str, current_query: str) -> str: ...


class StrategyRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        strategy: str = "hybrid",
    ) -> list[ScoredChunk]: ...


RetrievalToolName = Literal["lexical_search", "dense_search", "hybrid_search"]


@dataclass(frozen=True)
class RetrievalToolCall:
    name: RetrievalToolName
    query: str
    top_k: int

    def __post_init__(self) -> None:
        if self.name not in {"lexical_search", "dense_search", "hybrid_search"}:
            raise ValueError("未知检索工具")
        if not self.query.strip():
            raise ValueError("工具查询不能为空")
        if self.top_k <= 0:
            raise ValueError("工具 top_k 必须大于 0")

    @property
    def strategy(self) -> str:
        return self.name.removesuffix("_search")

    @property
    def arguments(self) -> dict[str, str | int]:
        return {"query": self.query, "top_k": self.top_k}


class RetrievalToolRegistry:
    """Executes only validated retrieval tools against the configured retriever."""

    def __init__(self, retriever: StrategyRetriever) -> None:
        self.retriever = retriever

    def execute(self, call: RetrievalToolCall) -> list[ScoredChunk]:
        return self.retriever.retrieve(call.query, call.top_k, strategy=call.strategy)


class RuleBasedQueryPlanner:
    """Deterministic offline policy that chooses one of three retrieval tools."""

    def plan(self, question: str) -> str:
        if exact_terms(question):
            return "lexical"
        if any(hint in question for hint in SEMANTIC_HINTS):
            return "dense"
        return "hybrid"


class EvidenceJudge:
    def __init__(self, *, min_evidence_count: int = 1) -> None:
        if min_evidence_count <= 0:
            raise ValueError("min_evidence_count 必须大于 0")
        self.min_evidence_count = min_evidence_count

    def evaluate(
        self,
        question: str,
        evidence: Sequence[ScoredChunk],
        route: str,
    ) -> tuple[bool, str]:
        del question, route
        if len(evidence) < self.min_evidence_count:
            return False, f"evidence_count={len(evidence)}<{self.min_evidence_count}"
        return True, f"evidence_count={len(evidence)}>={self.min_evidence_count}"


class RuleBasedQueryRewriter:
    """Removes conversational filler and switches a failed query to hybrid retrieval."""

    _FILLER = re.compile(r"^(?:请问|请帮我|能否|可以|麻烦|我想知道)\s*")

    def rewrite(self, original_question: str, current_query: str) -> str:
        del original_question
        rewritten = self._FILLER.sub("", current_query.strip()).strip(" ？?。！!")
        if rewritten == current_query.strip():
            return current_query.strip()
        return rewritten
