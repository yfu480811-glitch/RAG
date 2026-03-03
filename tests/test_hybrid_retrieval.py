from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from tracerag.ingest import ingest_source
from tracerag.retrieval import HybridRetriever
from tracerag.storage import SQLiteStore


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vocab = ["trace", "rag", "python", "database", "fastapi", "chunk", "query", "source"]
        vectors: list[list[float]] = []
        for text in texts:
            low = text.lower()
            vec = [float(low.count(tok)) for tok in vocab]
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


def test_hybrid_retrieval_returns_relevant_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    store = SQLiteStore(db_path)
    store.init_schema()

    doc1 = tmp_path / "a.txt"
    doc2 = tmp_path / "b.txt"
    doc1.write_text("TraceRAG uses FastAPI and query retrieval from source chunks.", encoding="utf-8")
    doc2.write_text("This document is about cooking pasta and tomatoes.", encoding="utf-8")

    ingest_source(str(doc1), store)
    ingest_source(str(doc2), store)

    retriever = HybridRetriever(
        store=store,
        provider=FakeEmbeddingProvider(),
        index_path=tmp_path / "index.faiss",
        mapping_path=tmp_path / "index_map.json",
    )
    retriever.rebuild()

    results = retriever.search("fastapi query source", semantic_top_k=3, bm25_top_k=3, rrf_k=60)
    assert results
    top = results[0]
    assert "chunk_id" in top
    assert "source" in top.get("metadata", {}) or "source" in top
