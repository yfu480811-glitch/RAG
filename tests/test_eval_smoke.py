from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from eval import run_eval
from tracerag.config import settings
from tracerag.ingest import ingest_source
from tracerag.storage import SQLiteStore


def test_eval_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.db"
    settings.db_path = db_path
    settings.vector_index_path = tmp_path / "index.faiss"
    settings.vector_map_path = tmp_path / "index_map.json"

    store = SQLiteStore(db_path)
    store.init_schema()

    doc = tmp_path / "doc.txt"
    doc.write_text("TraceRAG uses BM25 and RRF for retrieval with citations [1].", encoding="utf-8")
    ingest_source(str(doc), store)

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        '{"query":"What retrieval method is used?","expected_source":"' + str(doc) + '"}\n',
        encoding="utf-8",
    )

    report_path = tmp_path / "report.json"
    report = run_eval(eval_path, report_path, top_k=3)

    assert "citation_coverage" in report
    assert report_path.exists()
