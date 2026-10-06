# RAG API

`rag.rag_pipeline.RAGPipeline` is the public entry point:

```python
pipeline = RAGPipeline(persist_directory=None)
policy = pipeline.ingest(pdf_bytes_or_path_or_binary_file, filename=None)
chunks = pipeline.retrieve(policy.policy_id, "Is renal dialysis covered?", k=6)
same_policy = pipeline.get_policy(policy.policy_id)
```

- `persist_directory=None` uses `<Settings.data_dir>/chroma`. Chroma telemetry is disabled.
- `ingest(...) -> PolicyDocument` uses a SHA-256 content ID and is safe to repeat.
- `retrieve(policy_id, query, k=6) -> list[Chunk]` always filters by `policy_id`.
- `get_policy(policy_id) -> PolicyDocument` reloads durable metadata without an LLM.
- `load_pdf(source, filename=None) -> list[Page]`; use `load_pdf_details` for scan warnings and source bytes.
- Page numbers are 1-based physical PDF pages. Empty pages remain in `PolicyDocument.pages`, but make no chunks.
- `SentenceTransformerEmbeddings` loads `all-MiniLM-L6-v2` only on first use and supports injected models/factories.

All domain objects are dataclasses and can be serialized with `dataclasses.asdict`.
