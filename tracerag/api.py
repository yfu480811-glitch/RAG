from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from tracerag.citations import assign_citations
from tracerag.config import settings
from tracerag.embeddings import get_embedding_provider
from tracerag.generator import OfflineGenerator
from tracerag.retrieval import HybridRetriever
from tracerag.schemas import ChatRequest, ChatSyncResponse
from tracerag.storage import SQLiteStore


def _retrieve(query: str, top_k: int) -> tuple[str, list[dict], dict[str, float], str]:
    t0 = time.perf_counter()
    store = SQLiteStore(settings.db_path)
    store.init_schema()
    parse_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    provider_name = settings.embedding_provider
    provider = get_embedding_provider(provider_name)
    embedding_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    retriever = HybridRetriever(
        store=store,
        provider=provider,
        index_path=settings.vector_index_path,
        mapping_path=settings.vector_map_path,
    )
    retriever.rebuild()
    results = retriever.search(
        query,
        semantic_top_k=max(top_k, settings.semantic_top_k),
        bm25_top_k=max(top_k, settings.bm25_top_k),
        rrf_k=settings.rrf_k,
    )[:top_k]
    retrieval_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    cited_chunks, sources = assign_citations(results)
    answer = OfflineGenerator().generate(query, cited_chunks, sources)
    generation_ms = (time.perf_counter() - t3) * 1000

    metrics = {
        "parse_ms": parse_ms,
        "embedding_ms": embedding_ms,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "latency_ms": parse_ms + embedding_ms + retrieval_ms + generation_ms,
    }
    return answer, sources, metrics, provider_name


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse("static/index.html")

    @app.post("/chat_sync", response_model=ChatSyncResponse)
    async def chat_sync(req: ChatRequest) -> ChatSyncResponse:
        answer, sources, _, _ = _retrieve(req.query, req.top_k)
        return ChatSyncResponse(answer=answer, sources=sources)

    @app.post("/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        async def event_stream() -> AsyncGenerator[str, None]:
            request_id = str(uuid.uuid4())
            yield f"event: status\ndata: {json.dumps({'stage': 'retrieving', 'request_id': request_id})}\n\n"
            await asyncio.sleep(0.02)

            answer, sources, metrics, provider_name = _retrieve(req.query, req.top_k)
            store = SQLiteStore(settings.db_path)
            store.init_schema()
            store.insert_chat_log(
                request_id=request_id,
                query=req.query,
                top_k=req.top_k,
                used_provider=provider_name,
                **metrics,
            )

            yield f"event: status\ndata: {json.dumps({'stage': 'generating', 'request_id': request_id})}\n\n"
            await asyncio.sleep(0.02)

            for i in range(0, len(answer), 48):
                delta = answer[i : i + 48]
                payload = {"delta": delta, "request_id": request_id}
                yield f"event: delta\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            yield f"event: sources\ndata: {json.dumps({'request_id': request_id, 'sources': sources}, ensure_ascii=False)}\n\n"
            yield f"event: status\ndata: {json.dumps({'stage': 'done', 'request_id': request_id, **metrics})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
