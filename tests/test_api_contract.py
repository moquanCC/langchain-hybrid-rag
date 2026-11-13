from __future__ import annotations

from hybrid_rag.api import answer_to_dict
from hybrid_rag.models import AgentStep, Answer, Chunk, ScoredChunk


def test_answer_contract_exposes_backend_and_evidence_offsets() -> None:
    chunk = Chunk("aaaaaaaaaaaaaaaa", "auth.md", "证据", 3, 5)
    answer = Answer(
        question="问题",
        text="回答 [aaaaaaaaaaaaaaaa]",
        evidence=(ScoredChunk(chunk, 0.5, "rerank"),),
        citations_valid=True,
        generator_backend="extractive-offline",
        retrieval_backend="bm25+local-hashing-char-ngram-v1+numpy+identity",
        agent_trace=(
            AgentStep(1, "问题", "hybrid", 1, True, "enough", "answer", ("aaaaaaaaaaaaaaaa",)),
        ),
        stop_reason="evidence_sufficient",
    )

    payload = answer_to_dict(answer)

    assert payload["citations_valid"] is True
    assert payload["stop_reason"] == "evidence_sufficient"
    assert payload["agent_trace"][0]["route"] == "hybrid"
    assert payload["agent_trace"][0]["action"] == "answer"
    assert payload["agent_trace"][0]["evidence_ids"] == ["aaaaaaaaaaaaaaaa"]
    assert payload["evidence"][0] == {
        "chunk_id": "aaaaaaaaaaaaaaaa",
        "source": "auth.md",
        "start": 3,
        "end": 5,
        "text": "证据",
        "score": 0.5,
        "stage": "rerank",
    }
