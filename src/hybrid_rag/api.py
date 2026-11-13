from __future__ import annotations

import math
from typing import Any

from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .fusion import HybridRetriever
from .generation import ExtractiveGenerator, LangChainOllamaGenerator
from .ingestion import load_documents, split_documents
from .models import Answer
from .pipeline import AgenticRAGPipeline, RAGPipeline
from .reranking import CrossEncoderReranker, IdentityReranker


def _choice(payload: dict[str, Any], key: str, default: str, choices: set[str]) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{key} 必须是以下值之一：{', '.join(sorted(choices))}")
    return value


def _positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} 必须是正整数")
    return value


def _boolean(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须是布尔值")
    return value


def _string(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串")
    return value.strip()


def _finite_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} 必须是有限数值")
    return result


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "answer": answer.text,
        "citations_valid": answer.citations_valid,
        "generator_backend": answer.generator_backend,
        "retrieval_backend": answer.retrieval_backend,
        "stop_reason": answer.stop_reason,
        "agent_trace": [
            {
                "iteration": step.iteration,
                "query": step.query,
                "route": step.route,
                "evidence_count": step.evidence_count,
                "evidence_sufficient": step.evidence_sufficient,
                "reason": step.reason,
                "action": step.action,
                "evidence_ids": list(step.evidence_ids),
                "tool_name": step.tool_name,
                "tool_arguments": step.tool_arguments,
                "latency_ms": step.latency_ms,
            }
            for step in answer.agent_trace
        ],
        "evidence": [
            {
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "start": item.chunk.start,
                "end": item.chunk.end,
                "text": item.chunk.text,
                "score": item.score,
                "stage": item.stage,
            }
            for item in answer.evidence
        ],
    }


def create_app() -> Any:
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI 未安装；请运行 `pip install -e .[api]`") from exc

    app = FastAPI(title="Traceable Agentic RAG", version="0.2.0")
    state: dict[str, RAGPipeline | None] = {"pipeline": None}

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "indexed": state["pipeline"] is not None}

    @app.post("/index")
    def index(payload: dict[str, Any] = Body(...)) -> dict[str, object]:
        try:
            path = _string(payload, "path", "")
            chunk_size = _positive_int(payload, "chunk_size", 420)
            chunk_overlap = payload.get("chunk_overlap", 60)
            if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                raise ValueError("chunk_overlap 必须是非负整数")
            top_k = _positive_int(payload, "top_k", 5)
            generator_name = _choice(payload, "generator", "extractive", {"extractive", "ollama"})
            embedding_name = _choice(payload, "embedding", "hashing", {"hashing", "bge-m3"})
            rerank = _boolean(payload, "rerank", False)
            local_files_only = _boolean(payload, "local_files_only", False)
            batch_size = _positive_int(payload, "batch_size", 8)
            min_dense_score = _finite_float(payload, "min_dense_score", 0.0)
            max_agent_iterations = _positive_int(payload, "max_agent_iterations", 2)

            documents = load_documents(path)
            chunks = split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            generator = (
                LangChainOllamaGenerator(_string(payload, "model", "qwen2.5:7b"))
                if generator_name == "ollama"
                else ExtractiveGenerator()
            )
            embedder = (
                SentenceTransformerEmbedder(
                    _string(payload, "embedding_model", "BAAI/bge-m3"),
                    device=_string(payload, "device", "cpu"),
                    batch_size=batch_size,
                    local_files_only=local_files_only,
                )
                if embedding_name == "bge-m3"
                else HashingEmbedder()
            )
            reranker = (
                CrossEncoderReranker(
                    _string(payload, "reranker_model", "BAAI/bge-reranker-v2-m3"),
                    local_files_only=local_files_only,
                )
                if rerank
                else IdentityReranker()
            )
            pipeline = AgenticRAGPipeline(
                HybridRetriever.from_chunks(
                    chunks,
                    embedder=embedder,
                    reranker=reranker,
                    min_dense_score=min_dense_score,
                ),
                generator,
                top_k=top_k,
                max_iterations=max_agent_iterations,
            )
            state["pipeline"] = pipeline
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "retrieval_backend": pipeline.retriever.backend,
            "generator_backend": pipeline.generator.backend,
            "agentic": True,
            "max_agent_iterations": pipeline.max_iterations,
        }

    @app.post("/ask")
    def ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pipeline = state["pipeline"]
        if pipeline is None:
            raise HTTPException(status_code=409, detail="请先调用 /index")
        try:
            return answer_to_dict(pipeline.ask(str(payload["question"])))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
