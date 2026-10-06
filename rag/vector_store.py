"""Policy-isolated Chroma storage and durable policy manifests."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .embeddings import Embeddings
from .models import Chunk, Page, PolicyDocument, PolicyLensError


class ChromaVectorStore:
    """A single local collection with mandatory policy filters on retrieval."""

    def __init__(self, persist_directory: str | Path, embeddings: Embeddings) -> None:
        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self._lock = threading.RLock()
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - environment configuration
            raise PolicyLensError("Vector storage is unavailable; install the pinned chromadb dependency") from exc
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                "policylens_chunks", metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise PolicyLensError(f"Could not initialize local vector storage: {exc}") from exc
        self._manifest_dir = self.persist_directory / "policies"
        self._manifest_dir.mkdir(exist_ok=True)

    def _manifest_path(self, policy_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", policy_id):
            raise PolicyLensError("Invalid policy ID")
        return self._manifest_dir / f"{policy_id}.json"

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int]:
        return {
            "policy_id": chunk.policy_id,
            "page": chunk.page,
            "section": chunk.section,
            "source": chunk.source,
        }

    def index(self, policy: PolicyDocument) -> None:
        """Upsert a complete policy and atomically replace its manifest."""

        if not policy.chunks:
            raise PolicyLensError("Policy contains no text chunks to index")
        ids = [chunk.id for chunk in policy.chunks]
        if len(ids) != len(set(ids)):
            raise PolicyLensError("Policy contains duplicate chunk IDs")
        try:
            vectors = self.embeddings.embed_documents([chunk.text for chunk in policy.chunks])
            if len(vectors) != len(policy.chunks):
                raise PolicyLensError("Embedding provider returned an unexpected vector count")
            with self._lock:
                for start in range(0, len(policy.chunks), 256):
                    batch = policy.chunks[start : start + 256]
                    self._collection.upsert(
                        ids=[chunk.id for chunk in batch],
                        documents=[chunk.text for chunk in batch],
                        metadatas=[self._metadata(chunk) for chunk in batch],
                        embeddings=vectors[start : start + 256],
                    )

                existing = self._collection.get(where={"policy_id": policy.policy_id})
                stale_ids = set(existing.get("ids", [])) - set(ids)
                if stale_ids:
                    self._collection.delete(ids=sorted(stale_ids))

                destination = self._manifest_path(policy.policy_id)
                temporary = destination.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(asdict(policy), ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, destination)
        except PolicyLensError:
            raise
        except Exception as exc:
            raise PolicyLensError(f"Could not index policy: {exc}") from exc

    def retrieve(self, policy_id: str, query: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        if not query.strip():
            raise PolicyLensError("Retrieval query cannot be empty")
        # Validate before executing a query and return a predictable not-found error.
        self._manifest_path(policy_id)
        if not self.has_policy(policy_id):
            raise PolicyLensError(f"Unknown policy ID: {policy_id}")
        try:
            vector = self.embeddings.embed_query(query)
            result: dict[str, Any] = self._collection.query(
                query_embeddings=[vector],
                n_results=k,
                where={"policy_id": policy_id},
                include=["documents", "metadatas"],
            )
            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            chunks: list[Chunk] = []
            for chunk_id, document, metadata in zip(ids, documents, metadatas):
                if str(metadata.get("policy_id", "")) != policy_id:
                    raise PolicyLensError("Vector storage returned context from a different policy")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        policy_id=policy_id,
                        text=document,
                        page=int(metadata["page"]),
                        section=str(metadata.get("section", "")),
                        source=str(metadata.get("source", "")),
                    )
                )
            return chunks
        except PolicyLensError:
            raise
        except Exception as exc:
            raise PolicyLensError(f"Could not retrieve policy context: {exc}") from exc

    def has_policy(self, policy_id: str) -> bool:
        return self._manifest_path(policy_id).is_file()

    def get_policy(self, policy_id: str) -> PolicyDocument:
        path = self._manifest_path(policy_id)
        if not path.is_file():
            raise PolicyLensError(f"Unknown policy ID: {policy_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return PolicyDocument(
                policy_id=payload["policy_id"],
                filename=payload["filename"],
                pages=[Page(**page) for page in payload.get("pages", [])],
                chunks=[Chunk(**chunk) for chunk in payload.get("chunks", [])],
                warnings=list(payload.get("warnings", [])),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PolicyLensError(f"Stored policy metadata is invalid: {policy_id}") from exc
