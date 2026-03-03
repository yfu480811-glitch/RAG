from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from tracerag.citations import assign_citations, format_sources_text
from tracerag.config import settings
from tracerag.embeddings import get_embedding_provider
from tracerag.generator import OfflineGenerator
from tracerag.ingest import collect_paths, ingest_source
from tracerag.retrieval import HybridRetriever
from tracerag.storage import SQLiteStore

app = typer.Typer(help="TraceRAG CLI")


@app.command()
def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    run_host = host or settings.host
    run_port = port or settings.port
    uvicorn.run("tracerag.api:app", host=run_host, port=run_port, reload=reload)


@app.command()
def ingest(
    path: Path | None = typer.Option(None, "--path", help="File path or folder path"),
    url: str | None = typer.Option(None, "--url", help="Remote HTML URL"),
) -> None:
    if (path is None and url is None) or (path is not None and url is not None):
        raise typer.BadParameter("Provide exactly one of --path or --url")

    store = SQLiteStore(settings.db_path)
    store.init_schema()

    results: list[dict] = []
    if url is not None:
        results.append(ingest_source(url, store, settings.chunk_size, settings.chunk_overlap))
    else:
        files = collect_paths(path)
        if not files:
            raise typer.BadParameter(f"No supported files found in: {path}")
        for file in files:
            results.append(ingest_source(str(file), store, settings.chunk_size, settings.chunk_overlap))

    typer.echo({"count": len(results), "results": results})


@app.command()
def query(question: str, top_k: int = 6) -> None:
    store = SQLiteStore(settings.db_path)
    store.init_schema()

    provider = get_embedding_provider(settings.embedding_provider)
    retriever = HybridRetriever(
        store=store,
        provider=provider,
        index_path=settings.vector_index_path,
        mapping_path=settings.vector_map_path,
    )
    retriever.rebuild()

    rows = retriever.search(
        question,
        semantic_top_k=max(top_k, settings.semantic_top_k),
        bm25_top_k=max(top_k, settings.bm25_top_k),
        rrf_k=settings.rrf_k,
    )[:top_k]

    cited_chunks, sources = assign_citations(rows)
    answer = OfflineGenerator().generate(question, cited_chunks, sources)

    typer.echo(answer)
    typer.echo("\nSources:")
    typer.echo(format_sources_text(sources))


if __name__ == "__main__":
    app()
