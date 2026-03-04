# Design Decisions & Extensibility

## 1) Chunking 参数
- 采用 `chunk_size=1000`、`overlap=180`。
- 目标：减少上下文断裂，同时保证检索粒度。

## 2) Hybrid Retrieval + RRF
- BM25 和向量检索分数不可直接比较，RRF 用 rank 融合更稳。
- 当前先保留轻量方案，不引入复杂学习排序。

## 3) OfflineGenerator 作为默认生成器
- 原因：无外部依赖、可离线测试、便于 CI 稳定复现。
- LLMGenerator 保留接口，后续可接 OpenAI/本地模型。

## 4) 向量索引更新策略
- 目前采用“全量重建”以保证一致性和实现简洁。
- 若数据规模变大，可演进为增量 append + lazy deletion。

## 5) 可扩展点

### Rerank
- 增加 reranker 接口（cross-encoder）对融合结果二次排序。
- 适用：提升 top-1/top-3 精度。

### Qdrant / pgvector
- 将 FAISS 本地索引迁移到向量数据库，提升多实例与在线更新能力。

### LLM Provider
- 完成 LLMGenerator provider 抽象：OpenAI / vLLM / Ollama。
- 增加结构化输出校验，确保 citation 编号合法。
