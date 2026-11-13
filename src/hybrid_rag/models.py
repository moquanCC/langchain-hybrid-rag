from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source 不能为空")
        if not self.text.strip():
            raise ValueError("文档文本不能为空")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id 不能为空")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("字符位置不合法")


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    stage: str
    details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStep:
    iteration: int
    query: str
    route: str
    evidence_count: int
    evidence_sufficient: bool
    reason: str
    action: str
    evidence_ids: tuple[str, ...] = ()
    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    evidence: tuple[ScoredChunk, ...]
    citations_valid: bool
    generator_backend: str
    retrieval_backend: str
    agent_trace: tuple[AgentStep, ...] = ()
    stop_reason: str = "fixed_pipeline_complete"
