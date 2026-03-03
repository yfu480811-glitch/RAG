from __future__ import annotations

import json
from pathlib import Path


from typing import Protocol

class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
from tracerag.storage import SQLiteStore


def rrf_fuse(list1: list[str], list2: list[str], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(list1, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(list2, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._rows_by_id: dict[str, dict] = {}
        self._bm25 = None

    def build(self, chunks: list[dict]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is not installed.") from exc

        self._ids = [c["chunk_id"] for c in chunks]
        self._rows_by_id = {c["chunk_id"]: c for c in chunks}
        corpus = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25:
            return []
        q = query.lower().split()
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        out: list[dict] = []
        for idx in ranked:
            chunk_id = self._ids[int(idx)]
            row = self._rows_by_id[chunk_id]
            out.append({"chunk_id": chunk_id, "score": float(scores[int(idx)]), **row})
        return out


class FAISSVectorIndex:
    """Simple strategy: rebuild full index from SQLite each query/update.

    Reason: Milestone 3 prioritizes correctness and traceability over deletion complexity;
    full rebuild avoids stale vectors after document updates.
    """

    def __init__(self, index_path: Path, mapping_path: Path) -> None:
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index = None
        self.mapping: list[str] = []
        self.rows_by_id: dict[str, dict] = {}

    def build(self, chunks: list[dict], provider: EmbeddingProvider) -> None:
        if not chunks:
            self.index = None
            self.mapping = []
            self.rows_by_id = {}
            return

        import numpy as np

        vectors = np.array(provider.embed([c["text"] for c in chunks]), dtype="float32")
        dim = vectors.shape[1]
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is not installed.") from exc

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        self.mapping = [c["chunk_id"] for c in chunks]
        self.rows_by_id = {c["chunk_id"]: c for c in chunks}
        self.mapping_path.write_text(json.dumps(self.mapping, ensure_ascii=False), encoding="utf-8")
        self.index = index

    def load_if_exists(self) -> bool:
        if not self.index_path.exists() or not self.mapping_path.exists():
            return False
        try:
            import faiss
        except ImportError:
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        return True

    def search(self, query: str, provider: EmbeddingProvider, top_k: int) -> list[dict]:
        if self.index is None:
            return []
        import numpy as np

        qv = np.array(provider.embed([query]), dtype="float32")
        scores, idxs = self.index.search(qv, top_k)
        out: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.mapping):
                continue
            chunk_id = self.mapping[int(idx)]
            row = self.rows_by_id.get(chunk_id, {"chunk_id": chunk_id})
            out.append({"chunk_id": chunk_id, "score": float(score), **row})
        return out


class HybridRetriever:
    def __init__(
        self,
        store: SQLiteStore,
        provider: EmbeddingProvider,
        index_path: Path,
        mapping_path: Path,
    ) -> None:
        self.store = store
        self.provider = provider
        self.bm25 = BM25Index()
        self.vector = FAISSVectorIndex(index_path=index_path, mapping_path=mapping_path)

    def rebuild(self) -> None:
        chunks = self.store.fetch_all_chunks()
        self.bm25.build(chunks)
        self.vector.build(chunks, self.provider)

    def search(self, query: str, semantic_top_k: int, bm25_top_k: int, rrf_k: int) -> list[dict]:
        semantic = self.vector.search(query, self.provider, semantic_top_k)
        lexical = self.bm25.search(query, bm25_top_k)

        sem_ids = [x["chunk_id"] for x in semantic]
        lex_ids = [x["chunk_id"] for x in lexical]
        fused = rrf_fuse(sem_ids, lex_ids, k=rrf_k)

        by_id = {x["chunk_id"]: x for x in semantic}
        by_id.update({x["chunk_id"]: x for x in lexical})

        results: list[dict] = []
        for chunk_id, score in fused:
            row = by_id.get(chunk_id, {"chunk_id": chunk_id})
            results.append({**row, "rrf_score": score})
        return results
