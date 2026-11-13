from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from hybrid_rag.api import create_app
from hybrid_rag.models import ScoredChunk


def test_health_index_and_ask_flow() -> None:
    client = TestClient(create_app())
    data_path = Path(__file__).parents[1] / "data" / "demo"

    health_before = client.get("/health")
    indexed = client.post(
        "/index",
        json={"path": str(data_path), "top_k": 3, "generator": "extractive"},
    )
    answered = client.post("/ask", json={"question": "E101 应该如何处理？"})

    assert health_before.json() == {"status": "ok", "indexed": False}
    assert indexed.status_code == 200
    index_payload = indexed.json()
    assert index_payload["documents"] == 5
    assert index_payload["chunks"] == 5
    assert index_payload["retrieval_backend"] in {
        "bm25+local-hashing-char-ngram-v1+numpy+identity",
        "bm25+local-hashing-char-ngram-v1+faiss+identity",
    }
    assert index_payload["generator_backend"] == "extractive-offline"
    assert index_payload["agentic"] is True
    assert index_payload["max_agent_iterations"] == 2
    assert answered.status_code == 200
    payload = answered.json()
    assert payload["citations_valid"] is True
    assert payload["evidence"][0]["source"] == "auth.md"
    assert payload["stop_reason"] == "evidence_sufficient"
    assert payload["agent_trace"][0]["route"] == "lexical"
    assert payload["agent_trace"][0]["action"] == "answer"


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding": "unknown"},
        {"generator": "unknown"},
        {"rerank": "yes"},
        {"batch_size": 0},
        {"min_dense_score": "high"},
        {"min_dense_score": True},
        {"max_agent_iterations": 0},
        {"max_agent_iterations": True},
    ],
)
def test_index_rejects_invalid_pipeline_options(overrides: dict[str, object]) -> None:
    client = TestClient(create_app())
    data_path = Path(__file__).parents[1] / "data" / "demo"

    response = client.post("/index", json={"path": str(data_path), **overrides})

    assert response.status_code == 400


def test_index_applies_dense_abstention_threshold() -> None:
    client = TestClient(create_app())
    data_path = Path(__file__).parents[1] / "data" / "demo"

    indexed = client.post(
        "/index",
        json={"path": str(data_path), "min_dense_score": 2.0},
    )
    answered = client.post("/ask", json={"question": "ZZZ_NO_MATCH_9847"})

    assert indexed.status_code == 200
    assert answered.status_code == 200
    assert answered.json()["evidence"] == []
    assert answered.json()["answer"] == "未检索到足够证据，已停止回答。"
    assert answered.json()["agent_trace"][-1]["action"] == "abstain"


def test_index_configures_model_backends_without_loading_real_models(monkeypatch) -> None:
    import hybrid_rag.api as api_module

    class FakeEmbedder:
        name = "fake-bge-m3"
        dimensions = 4

        def encode(self, texts):
            vectors = np.ones((len(texts), self.dimensions), dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    class FakeReranker:
        name = "fake-reranker"

        def rerank(self, query, candidates, top_k):
            del query
            return [
                ScoredChunk(item.chunk, item.score, "rerank", item.details)
                for item in candidates[:top_k]
            ]

    monkeypatch.setattr(api_module, "SentenceTransformerEmbedder", lambda *args, **kwargs: FakeEmbedder())
    monkeypatch.setattr(api_module, "CrossEncoderReranker", lambda *args, **kwargs: FakeReranker())
    client = TestClient(create_app())
    data_path = Path(__file__).parents[1] / "data" / "demo"

    response = client.post(
        "/index",
        json={
            "path": str(data_path),
            "embedding": "bge-m3",
            "embedding_model": "fake-bge-m3",
            "rerank": True,
            "reranker_model": "fake-reranker",
            "local_files_only": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieval_backend"] in {
        "bm25+fake-bge-m3+numpy+fake-reranker",
        "bm25+fake-bge-m3+faiss+fake-reranker",
    }
