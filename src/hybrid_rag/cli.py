from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import answer_to_dict
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .evaluation import compare_pipelines, evaluate_cases, load_cases
from .fusion import HybridRetriever
from .generation import ExtractiveGenerator, LangChainOllamaGenerator
from .ingestion import load_documents, split_documents
from .pipeline import AgenticRAGPipeline, RAGPipeline
from .reranking import CrossEncoderReranker, IdentityReranker


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    documents = load_documents(args.data)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    generator = (
        LangChainOllamaGenerator(args.model) if args.generator == "ollama" else ExtractiveGenerator()
    )
    embedder = (
        SentenceTransformerEmbedder(
            args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        if args.embedding == "bge-m3"
        else HashingEmbedder()
    )
    reranker = (
        CrossEncoderReranker(
            args.reranker_model,
            local_files_only=args.local_files_only,
        )
        if args.rerank
        else IdentityReranker()
    )
    return AgenticRAGPipeline(
        HybridRetriever.from_chunks(
            chunks,
            embedder=embedder,
            reranker=reranker,
            min_dense_score=args.min_dense_score,
        ),
        generator,
        top_k=args.top_k,
        max_iterations=args.max_agent_iterations,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可追溯的受约束 Agentic RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ask", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--data", default="data/demo")
        child.add_argument("--top-k", type=int, default=3)
        child.add_argument("--chunk-size", type=int, default=420)
        child.add_argument("--chunk-overlap", type=int, default=60)
        child.add_argument("--generator", choices=("extractive", "ollama"), default="extractive")
        child.add_argument("--model", default="qwen2.5:7b")
        child.add_argument("--embedding", choices=("hashing", "bge-m3"), default="hashing")
        child.add_argument("--embedding-model", default="BAAI/bge-m3")
        child.add_argument("--device", default="cpu")
        child.add_argument("--batch-size", type=int, default=8)
        child.add_argument("--local-files-only", action="store_true")
        child.add_argument("--rerank", action="store_true")
        child.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
        child.add_argument("--min-dense-score", type=float, default=0.0)
        child.add_argument("--max-agent-iterations", type=int, default=2)
    subparsers.choices["ask"].add_argument("question")
    subparsers.choices["evaluate"].add_argument(
        "--questions", default="data/evaluation/questions.json"
    )
    subparsers.choices["evaluate"].add_argument("--output")
    subparsers.choices["evaluate"].add_argument("--compare-fixed", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    pipeline = build_pipeline(args)
    if args.command == "ask":
        payload = answer_to_dict(pipeline.ask(args.question))
    else:
        cases = load_cases(args.questions)
        payload = (
            compare_pipelines(
                RAGPipeline(pipeline.retriever, pipeline.generator, top_k=pipeline.top_k),
                pipeline,
                cases,
            )
            if args.compare_fixed
            else evaluate_cases(pipeline, cases)
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
