from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from .embeddings import Embedder, HashingEmbedder
from .models import Chunk, ScoredChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_.:/-]+|[\u3400-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self.chunks)
        self._idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks or k <= 0:
            return []
        results: list[ScoredChunk] = []
        for chunk, tokens, length in zip(self.chunks, self._tokens, self._lengths, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                normalization = 1 - self.b + self.b * length / max(self._avg_length, 1.0)
                score += self._idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
                )
            results.append(ScoredChunk(chunk=chunk, score=score, stage="bm25"))
        positive = [item for item in results if item.score > 0.0]
        return sorted(positive, key=lambda item: (-item.score, item.chunk.chunk_id))[:k]


class DenseIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        *,
        prefer_faiss: bool = True,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or HashingEmbedder()
        self.vectors = (
            self.embedder.encode([chunk.text for chunk in self.chunks])
            if self.chunks
            else np.empty((0, self.embedder.dimensions), dtype=np.float32)
        )
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks):
            raise ValueError("向量数量或形状与切片不一致")
        if not np.isfinite(self.vectors).all():
            raise ValueError("向量索引包含非有限值")
        self.backend = "numpy"
        self._faiss_index = None
        if prefer_faiss and self.chunks:
            try:
                import faiss
            except ImportError:
                pass
            else:
                self._faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss_index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
                self.backend = "faiss"

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not query.strip() or not self.chunks or k <= 0:
            return []
        query_vector = self.embedder.encode([query])
        count = min(k, len(self.chunks))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector, count)
            pairs = zip(indices[0].tolist(), scores[0].tolist(), strict=True)
        else:
            raw_scores = self.vectors @ query_vector[0]
            indices = sorted(range(len(self.chunks)), key=lambda index: (-float(raw_scores[index]), self.chunks[index].chunk_id))[:count]
            pairs = ((index, float(raw_scores[index])) for index in indices)
        return [
            ScoredChunk(chunk=self.chunks[index], score=float(score), stage="dense")
            for index, score in pairs
            if index >= 0
        ]
