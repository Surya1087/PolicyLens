"""Lazy local embeddings with explicit test dependency injection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from config.constants import DEFAULT_EMBEDDING_MODEL

from .models import PolicyLensError


class Embeddings(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Load sentence-transformers only on the first embedding call.

    ``model`` or ``model_factory`` may be supplied to keep tests deterministic
    and offline. The injected model must expose ``encode``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._model_factory = model_factory

    def _get_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.model_name)
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - environment configuration
                    raise PolicyLensError(
                        "Embedding support is unavailable; install sentence-transformers"
                    ) from exc
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]
