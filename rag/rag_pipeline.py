"""Public policy ingestion and semantic retrieval API."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import BinaryIO
from functools import lru_cache

from config.constants import CLINICAL_SYNONYM_GROUPS
from config.settings import Settings, get_settings

from .embeddings import Embeddings, SentenceTransformerEmbeddings
from .ingest import build_policy_document
from .models import Chunk, PolicyDocument
from .vector_store import ChromaVectorStore
from .hybrid import LocalBM25, reciprocal_rank_fusion


@lru_cache(maxsize=2)
def _reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None


def rerank_chunks(query: str, chunks: list[Chunk], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[Chunk]:
    """Rerank fused candidates with an optional local cross-encoder."""
    if len(chunks) < 2:
        return chunks
    reranker = _reranker(model_name)
    if reranker is None:
        return chunks
    try:
        scores = reranker.predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda item: float(item[0]), reverse=True)
        return [chunk for _, chunk in ranked]
    except Exception:
        return chunks


def claim_retrieval_query(bill_fields: dict) -> str:
    """One query builder shared by live claims and gold-extraction retrieval tests.

    Avoid polluting semantic search with invoice amounts, dates and hospital names.
    Synonyms are added by RAGPipeline.retrieve, not by evaluation gold labels.
    """
    procedure = bill_fields.get("procedure_or_diagnosis", {}).get("value")
    if not isinstance(procedure, str) or not procedure.strip():
        from .models import PolicyLensError
        raise PolicyLensError("A procedure or diagnosis is required for claim retrieval.")
    return procedure.strip() + ". Relevant policy terms, conditions, limits and waiting periods."


def expand_clinical_query(query: str) -> str:
    """Append matching clinical synonyms while preserving the original query."""

    lowered = query.casefold()
    related: list[str] = []
    for group in CLINICAL_SYNONYM_GROUPS:
        if any(term.casefold() in lowered for term in group):
            related.extend(term for term in group if term.casefold() not in lowered)
    if not related:
        return query
    unique = list(dict.fromkeys(related))
    return f"{query}\nRelated clinical terms: {', '.join(unique)}"


class RAGPipeline:
    """Persistent local RAG pipeline. It does not create or require an LLM."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        *,
        embeddings: Embeddings | None = None,
        settings: Settings | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or SentenceTransformerEmbeddings(self.settings.embedding_model)
        directory = Path(persist_directory) if persist_directory is not None else self.settings.data_dir / "chroma"
        self.vector_store = vector_store or ChromaVectorStore(directory, self.embeddings)

    def ingest(
        self,
        source: bytes | bytearray | PathLike[str] | str | BinaryIO,
        filename: str | None = None,
    ) -> PolicyDocument:
        policy = build_policy_document(source, filename, settings=self.settings)
        self.vector_store.index(policy)
        return policy

    def retrieve(self, policy_id: str, query: str, k: int = 6) -> list[Chunk]:
        expanded = expand_clinical_query(query)
        vector_chunks = self.vector_store.retrieve(policy_id, expanded, max(k * 3, k))
        policy = self.vector_store.get_policy(policy_id)
        keyword_chunks = LocalBM25(policy.chunks).retrieve(expanded, max(k * 3, k))
        fused = reciprocal_rank_fusion([vector_chunks, keyword_chunks])[: max(k * 2, k)]
        reranked = rerank_chunks(expanded, fused, self.settings.reranker_model)
        return reranked[:k]

    def get_policy(self, policy_id: str) -> PolicyDocument:
        return self.vector_store.get_policy(policy_id)
