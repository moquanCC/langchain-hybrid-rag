from __future__ import annotations

import re
from typing import Protocol


CITATION_PATTERN = re.compile(r"\[([0-9a-f]{16})\]")
CONTEXT_HEADER_PATTERN = re.compile(r"^\[([0-9a-f]{16})\]\s+.+$", re.MULTILINE)
EVIDENCE_SEPARATOR = "\n\n<<<EVIDENCE>>>\n\n"


class Generator(Protocol):
    backend: str

    def generate(self, question: str, context: str) -> str: ...


def extract_citations(text: str) -> list[str]:
    return CITATION_PATTERN.findall(text)


def validate_citations(text: str, allowed_chunk_ids: set[str]) -> bool:
    citations = extract_citations(text)
    return bool(citations) and set(citations) <= allowed_chunk_ids


class ExtractiveGenerator:
    """No-model fallback that exposes retrieval evidence without inventing facts."""

    backend = "extractive-offline"

    def generate(self, question: str, context: str) -> str:
        del question
        blocks = [block.strip() for block in context.split(EVIDENCE_SEPARATOR) if block.strip()]
        statements: list[str] = []
        for block in blocks[:3]:
            lines = block.splitlines()
            if len(lines) < 2:
                continue
            match = CONTEXT_HEADER_PATTERN.match(lines[0])
            if not match:
                continue
            content = " ".join(line.strip() for line in lines[1:] if line.strip())
            statements.append(f"- {content} [{match.group(1)}]")
        if not statements:
            return "未找到可以引用的证据。"
        return "根据检索到的文档片段：\n" + "\n".join(statements)


class LangChainOllamaGenerator:
    """Optional LangChain/Ollama generator loaded only when invoked."""

    def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.backend = f"langchain-ollama:{model}"

    def generate(self, question: str, context: str) -> str:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "LangChain/Ollama 依赖未安装；请安装 `pip install -e .[langchain]`，"
                "或使用 ExtractiveGenerator"
            ) from exc
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你只能依据提供的证据回答。每个事实后必须使用方括号标注原始 chunk ID；"
                    "证据不足时明确说不知道，不得编造引用。",
                ),
                ("human", "问题：{question}\n\n证据：\n{context}"),
            ]
        )
        chain = prompt | ChatOllama(model=self.model, temperature=self.temperature)
        response = chain.invoke({"question": question, "context": context})
        return str(response.content)
