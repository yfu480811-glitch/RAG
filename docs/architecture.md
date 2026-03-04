# TraceRAG Architecture

## 模块分层

1. **Ingestion Layer**
   - 输入：PDF / Markdown / TXT / URL(HTML)
   - 职责：解析、chunking、metadata 提取（page/heading/source）

2. **Storage Layer (SQLite)**
   - `documents`：source/hash/title/时间戳
   - `chunks`：chunk 文本和 metadata_json
   - `chat_logs`：请求日志与耗时分解

3. **Retrieval Layer**
   - BM25（lexical）
   - FAISS（semantic）
   - RRF 融合输出 top-k 候选

4. **Generation & Citation Layer**
   - citation 编号分配 `[1][2]...`
   - OfflineGenerator（默认）/ LLMGenerator（可扩展）
   - 输出 Sources 列表

5. **Serving Layer**
   - CLI：ingest/query/serve
   - FastAPI：`/chat_sync` + `/chat`(SSE)
   - 静态 UI：实时显示流式答案与来源

6. **Evaluation Layer**
   - `eval.py`：Recall@k、MRR、citation coverage
   - 结果写入 `reports/eval_report.json`

## 关键数据流

Ingest -> SQLite(chunks) -> Build BM25/FAISS -> RRF -> Citation Assignment -> Generator -> Answer+Sources -> API/CLI
