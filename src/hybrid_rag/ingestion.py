from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import Chunk, Document


SUPPORTED_SUFFIXES = {".md", ".txt", ".jsonl"}


def load_documents(path: str | Path) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"文档路径不存在: {root}")
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型: {suffix}")
        display_path = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        raw = file_path.read_text(encoding="utf-8-sig")
        if suffix in {".md", ".txt"}:
            if raw.strip():
                documents.append(Document(source=display_path, text=raw.strip()))
            continue
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{display_path} 第 {line_number} 行不是合法 JSON") from exc
            text = row.get("text") or row.get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{display_path} 第 {line_number} 行必须包含 text 或 content")
            metadata = {key: value for key, value in row.items() if key not in {"text", "content"}}
            documents.append(
                Document(
                    source=f"{display_path}#L{line_number}",
                    text=text.strip(),
                    metadata=metadata,
                )
            )
    return documents


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 420,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    for document in documents:
        for start in range(0, len(document.text), step):
            end = min(start + chunk_size, len(document.text))
            text = document.text[start:end]
            if not text.strip():
                if end == len(document.text):
                    break
                continue
            digest_input = f"{document.source}\0{start}\0{end}\0{text}".encode("utf-8")
            chunk_id = hashlib.sha256(digest_input).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=document.source,
                    text=text,
                    start=start,
                    end=end,
                    metadata=dict(document.metadata),
                )
            )
            if end == len(document.text):
                break
    return chunks
