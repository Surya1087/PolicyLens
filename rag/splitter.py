"""Page-local chunking that never loses physical-page provenance."""

from __future__ import annotations

import hashlib
import re

from config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

from .models import Chunk, Page, PolicyLensError

_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 &'(),/\-]{2,80}:?$")


def _section_for(text: str) -> str:
    for line in text.splitlines()[:5]:
        candidate = " ".join(line.split()).strip()
        if candidate and _HEADING_RE.fullmatch(candidate) and len(candidate.split()) <= 12:
            return candidate.rstrip(":")
    return ""


def split_pages(
    pages: list[Page],
    policy_id: str,
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page independently using LangChain's recursive splitter."""

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise PolicyLensError("Invalid chunk size or overlap")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover - environment configuration
        raise PolicyLensError("Chunking support is unavailable; install langchain-text-splitters") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for ordinal, text in enumerate(splitter.split_text(page.text)):
            clean_text = text.strip()
            if not clean_text:
                continue
            digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{policy_id}:p{page.page}:c{ordinal}:{digest}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    policy_id=policy_id,
                    text=clean_text,
                    page=page.page,
                    section=_section_for(clean_text),
                    source=source,
                )
            )
    return chunks
