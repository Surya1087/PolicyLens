"""Free local hybrid retrieval helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .models import Chunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.casefold()) if len(token) > 1]


class LocalBM25:
    """Small in-memory BM25 implementation with no hosted dependency."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.documents = [_tokens(chunk.text) for chunk in self.chunks]
        frequencies = [Counter(document) for document in self.documents]
        self.frequencies = frequencies
        self.lengths = [len(document) for document in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for document in frequencies:
            document_frequency.update(document.keys())
        self.document_frequency = document_frequency

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        if k <= 0 or not self.chunks:
            return []
        query_terms = set(_tokens(query))
        scores: list[tuple[int, float]] = []
        total_documents = len(self.chunks)
        for index, (document, frequencies) in enumerate(zip(self.documents, self.frequencies)):
            length_normalization = 1.0 - 0.75 + 0.75 * (self.lengths[index] / self.average_length)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                inverse_document_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                score += inverse_document_frequency * (frequency * 2.2) / (frequency + 1.2 * length_normalization)
            if score:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [self.chunks[index] for index, _ in scores[:k]]


def reciprocal_rank_fusion(ranked_lists: Iterable[Sequence[Chunk]], k: int = 60) -> list[Chunk]:
    """Fuse ranked lists without trusting their different score scales."""
    combined: dict[str, tuple[float, Chunk]] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            score = 1.0 / (k + rank)
            existing = combined.get(chunk.id)
            combined[chunk.id] = (existing[0] + score if existing else score, chunk)
    return [chunk for _, chunk in sorted(combined.values(), key=lambda item: item[0], reverse=True)]
