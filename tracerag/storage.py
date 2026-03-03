from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tracerag.models import Chunk, Document


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    title TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );

                CREATE TABLE IF NOT EXISTS ingest_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    used_provider TEXT NOT NULL,
                    parse_ms REAL NOT NULL,
                    embedding_ms REAL NOT NULL,
                    retrieval_ms REAL NOT NULL,
                    generation_ms REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON chat_logs(created_at);
                """
            )

    def log_ingest_run(self, source: str, status: str, detail: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO ingest_runs (source, status, detail) VALUES (?, ?, ?)",
                (source, status, detail),
            )

    def insert_chat_log(
        self,
        *,
        request_id: str,
        query: str,
        top_k: int,
        used_provider: str,
        parse_ms: float,
        embedding_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        latency_ms: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_logs (
                    request_id, query, top_k, used_provider,
                    parse_ms, embedding_ms, retrieval_ms, generation_ms, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    query,
                    top_k,
                    used_provider,
                    parse_ms,
                    embedding_ms,
                    retrieval_ms,
                    generation_ms,
                    latency_ms,
                ),
            )

    def get_document_by_source(self, source: str) -> Document | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT doc_id, source, source_type, content_hash, title FROM documents WHERE source = ?",
                (source,),
            ).fetchone()
            if row is None:
                return None
            return Document(
                doc_id=row["doc_id"],
                source=row["source"],
                source_type=row["source_type"],
                content_hash=row["content_hash"],
                title=row["title"],
            )

    def insert_document(self, doc: Document) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO documents (source, source_type, content_hash, title)
                VALUES (?, ?, ?, ?)
                """,
                (doc.source, doc.source_type, doc.content_hash, doc.title),
            )
            return int(cur.lastrowid)

    def update_document(self, doc_id: int, content_hash: str, title: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET content_hash = ?, title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
                """,
                (content_hash, title, doc_id),
            )

    def replace_chunks(self, doc_id: int, chunks: list[Chunk]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.chunk_index,
                        c.text,
                        json.dumps(c.metadata, ensure_ascii=False),
                    )
                    for c in chunks
                ],
            )

    def fetch_all_chunks(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.metadata_json,
                       d.source, d.source_type, d.title
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY c.doc_id, c.chunk_index
                """
            ).fetchall()
            items: list[dict] = []
            for row in rows:
                metadata = json.loads(row["metadata_json"])
                items.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "chunk_index": row["chunk_index"],
                        "text": row["text"],
                        "metadata": metadata,
                        "source": row["source"],
                        "source_type": row["source_type"],
                        "title": row["title"],
                    }
                )
            return items

    def count_chunks_for_doc(self, doc_id: int) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
            return int(row["n"])

    def fetch_chunk_metadata_for_doc(self, doc_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT metadata_json FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
            return [json.loads(row["metadata_json"]) for row in rows]
