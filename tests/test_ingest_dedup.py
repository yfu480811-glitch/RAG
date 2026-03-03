import json
from pathlib import Path

from tracerag.ingest import ingest_source
from tracerag.storage import SQLiteStore


def test_incremental_ingest_dedup(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("TraceRAG test content " * 100, encoding="utf-8")

    store = SQLiteStore(db_path)
    store.init_schema()

    first = ingest_source(str(txt_path), store, chunk_size=200, overlap=50)
    second = ingest_source(str(txt_path), store, chunk_size=200, overlap=50)

    assert first["status"] == "ingested"
    assert first["chunks"] > 0
    assert second["status"] == "skipped"

    doc_id = first["doc_id"]
    assert store.count_chunks_for_doc(doc_id) == first["chunks"]


def test_markdown_heading_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        "# Intro\nAlpha content here.\n## Details\nBeta content here.",
        encoding="utf-8",
    )

    store = SQLiteStore(db_path)
    store.init_schema()
    result = ingest_source(str(md_path), store, chunk_size=40, overlap=10)

    metas = store.fetch_chunk_metadata_for_doc(result["doc_id"])
    headings = {m.get("heading") for m in metas}
    assert "Intro" in headings or "Details" in headings
    assert all("source" in m for m in metas)

    # metadata_json is valid JSON persisted in DB
    with store.connect() as conn:
        row = conn.execute("SELECT metadata_json FROM chunks LIMIT 1").fetchone()
        assert isinstance(json.loads(row["metadata_json"]), dict)
