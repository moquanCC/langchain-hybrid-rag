from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .models import ScoredChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class IdentityReranker:
    name = "identity"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        del query
        return [
            ScoredChunk(item.chunk, item.score, "rerank", {**item.details, "rrf": item.score})
            for item in candidates[:top_k]
        ]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model: object | None = None,
        *,
        local_files_only: bool = False,
    ) -> None:
        self.name = model_name
        self._model = model
        self.local_files_only = local_files_only

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, local_files_only=self.local_files_only)
        return self._model

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []
        scores = np.asarray(
            self._load_model().predict([(query, item.chunk.text) for item in candidates]),
            dtype=np.float32,
        ).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("CrossEncoder 返回数量错误或非有限分数")
        reranked = [
            ScoredChunk(item.chunk, float(score), "rerank", {**item.details, "rrf": item.score})
            for item, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
