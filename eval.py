from __future__ import annotations

import json
import re
from pathlib import Path

from tracerag.citations import assign_citations
from tracerag.config import settings
from tracerag.generator import OfflineGenerator
from tracerag.retrieval import HybridRetriever
from tracerag.storage import SQLiteStore


class SimpleEmbeddingProvider:
    """Dependency-light embedding for offline evaluation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vocab = [
            "rag",
            "trace",
            "retrieval",
            "chunk",
            "citation",
            "source",
            "fastapi",
            "bm25",
            "faiss",
            "markdown",
        ]
        vectors: list[list[float]] = []
        for text in texts:
            low = text.lower()
            vec = [float(low.count(tok)) for tok in vocab]
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


def _extract_citation_count(answer: str) -> int:
    return len(re.findall(r"\[\d+\]", answer))


def _citation_coverage(answer: str) -> float:
    paragraphs = [p for p in answer.split("\n") if p.strip()]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[\d+\]", p))
    return cited / len(paragraphs)


def run_eval(eval_path: Path, out_path: Path, top_k: int = 6) -> dict:
    store = SQLiteStore(settings.db_path)
    store.init_schema()

    retriever = HybridRetriever(
        store=store,
        provider=SimpleEmbeddingProvider(),
        index_path=settings.vector_index_path,
        mapping_path=settings.vector_map_path,
    )
    retriever.rebuild()

    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    hit_count = 0
    mrr_total = 0.0
    source_cases = 0
    cov_total = 0.0
    cite_count_total = 0

    for item in rows:
        q = item["query"]
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])

        results = retriever.search(q, semantic_top_k=top_k, bm25_top_k=top_k, rrf_k=settings.rrf_k)[:top_k]
        cited_chunks, sources = assign_citations(results)
        answer = OfflineGenerator().generate(q, cited_chunks, sources)

        cov_total += _citation_coverage(answer)
        cite_count_total += _extract_citation_count(answer)

        if expected_source:
            source_cases += 1
            ranked_sources = [s.get("source", "") for s in sources]
            if expected_source in ranked_sources:
                hit_count += 1
                rank = ranked_sources.index(expected_source) + 1
                mrr_total += 1.0 / rank

        if expected_keywords:
            _ = [kw for kw in expected_keywords if kw.lower() in answer.lower()]

    recall_at_k = (hit_count / source_cases) if source_cases else None
    mrr = (mrr_total / source_cases) if source_cases else None
    citation_coverage = cov_total / len(rows) if rows else 0.0
    avg_citations = cite_count_total / len(rows) if rows else 0.0

    report = {
        "total": len(rows),
        "top_k": top_k,
        "source_labeled": source_cases,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "citation_coverage": citation_coverage,
        "avg_citation_markers": avg_citations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== TraceRAG Eval Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"saved: {out_path}")

    return report


if __name__ == "__main__":
    run_eval(Path("data/eval_set.jsonl"), Path("reports/eval_report.json"), top_k=6)
