from __future__ import annotations

import json

import pytest

from hybrid_rag.ingestion import load_documents, split_documents
from hybrid_rag.models import Document


def test_loads_markdown_txt_and_jsonl(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("# 指南\n\n混合检索。", encoding="utf-8")
    (tmp_path / "note.txt").write_text("错误码 E101。", encoding="utf-8")
    rows = [{"text": "第一条", "title": "a"}, {"content": "第二条"}]
    (tmp_path / "items.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert [doc.source for doc in documents] == [
        "guide.md",
        "items.jsonl#L1",
        "items.jsonl#L2",
        "note.txt",
    ]
    assert [doc.text for doc in documents][1:3] == ["第一条", "第二条"]


def test_splitter_preserves_offsets_and_stable_ids() -> None:
    document = Document(source="manual.md", text="甲乙丙丁戊己庚辛壬癸")

    first = split_documents([document], chunk_size=6, chunk_overlap=2)
    second = split_documents([document], chunk_size=6, chunk_overlap=2)

    assert first == second
    assert [(chunk.start, chunk.end) for chunk in first] == [(0, 6), (4, 10)]
    assert [chunk.text for chunk in first] == ["甲乙丙丁戊己", "戊己庚辛壬癸"]
    assert all(len(chunk.chunk_id) == 16 for chunk in first)
    assert all(chunk.source == "manual.md" for chunk in first)


def test_splitter_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_documents([], chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([], chunk_size=10, chunk_overlap=10)


def test_loader_rejects_invalid_jsonl(tmp_path) -> None:
    (tmp_path / "bad.jsonl").write_text('{"title": "missing text"}', encoding="utf-8")

    with pytest.raises(ValueError, match="text.*content"):
        load_documents(tmp_path)
