from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dimensions: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic signed character n-gram embeddings for offline tests."""

    name = "local-hashing-char-ngram-v1"

    def __init__(self, dimensions: int = 512, ngram_range: tuple[int, int] = (2, 4)) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须大于 0")
        if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range 不合法")
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, original in enumerate(texts):
            text = " ".join(original.lower().split())
            if not text:
                raise ValueError("待编码文本不能为空")
            features: list[str] = []
            for size in range(self.ngram_range[0], self.ngram_range[1] + 1):
                features.extend(text[index : index + size] for index in range(max(0, len(text) - size + 1)))
            if not features:
                features = [text]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, bucket] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(vectors).all():
            raise ValueError("Embedding 返回了零向量或非有限向量")
        return vectors / norms


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter; defaults to BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "cpu",
        batch_size: int = 8,
        model: object | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model
        self.local_files_only = local_files_only
        self.dimensions = 0

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        model = self._load_model()
        vectors = np.asarray(
            model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding 返回形状不正确")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding 返回了非有限向量")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Embedding 返回了零向量")
        self.dimensions = vectors.shape[1]
        return vectors / norms[:, None]
