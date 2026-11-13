from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .embeddings import Embedder
from .models import Chunk, ScoredChunk
from .reranking import IdentityReranker, Reranker
from .retrieval import BM25Index, DenseIndex


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    weights: Sequence[float],
    k: int = 60,
) -> list[ScoredChunk]:
    if not rankings:
        raise ValueError("rankings 不能为空")
    if len(rankings) != len(weights) or k <= 0 or any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("RRF 参数不合法")
    totals: defaultdict[str, float] = defaultdict(float)
    details: defaultdict[str, dict[str, float]] = defaultdict(dict)
    chunks: dict[str, Chunk] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            if chunk_id not in first_seen:
                first_seen[chunk_id] = seen_counter
                seen_counter += 1
            chunks[chunk_id] = item.chunk
            contribution = weight / (k + rank)
            totals[chunk_id] += contribution
            details[chunk_id][item.stage] = contribution
    fused = [
        ScoredChunk(chunks[chunk_id], score, "rrf", details[chunk_id])
        for chunk_id, score in totals.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, first_seen[item.chunk.chunk_id], item.chunk.chunk_id))


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        *,
        reranker: Reranker | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        rrf_k: int = 60,
        min_dense_score: float = 0.0,
    ) -> None:
        if isinstance(min_dense_score, bool) or not isinstance(min_dense_score, (int, float)):
            raise ValueError("min_dense_score 必须是有限数值")
        if not math.isfinite(float(min_dense_score)):
            raise ValueError("min_dense_score 必须是有限数值")
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker or IdentityReranker()
        self.weights = weights
        self.rrf_k = rrf_k
        self.min_dense_score = float(min_dense_score)

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        prefer_faiss: bool = True,
        min_dense_score: float = 0.0,
    ) -> "HybridRetriever":
        return cls(
            BM25Index(chunks),
            DenseIndex(chunks, embedder, prefer_faiss=prefer_faiss),
            reranker=reranker,
            min_dense_score=min_dense_score,
        )

    @property
    def backend(self) -> str:
        return (
            f"bm25+{self.dense.embedder.name}+{self.dense.backend}+"
            f"{self.reranker.name}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        strategy: str = "hybrid",
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if strategy not in {"lexical", "dense", "hybrid"}:
            raise ValueError("strategy 必须是 lexical、dense 或 hybrid")
        if not query.strip():
            return []
        candidate_k = max(top_k * 3, top_k)
        lexical = self.bm25.search(query, candidate_k) if strategy != "dense" else []
        dense = (
            [
                item
                for item in self.dense.search(query, candidate_k)
                if item.score > self.min_dense_score
            ]
            if strategy != "lexical"
            else []
        )
        if strategy == "lexical":
            rankings, weights = [lexical], [1.0]
        elif strategy == "dense":
            rankings, weights = [dense], [1.0]
        else:
            rankings, weights = [lexical, dense], self.weights
        fused = reciprocal_rank_fusion(rankings, weights=weights, k=self.rrf_k)
        return self.reranker.rerank(query, fused, top_k)
