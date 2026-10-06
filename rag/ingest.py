"""PDF-to-policy ingestion orchestration."""

from __future__ import annotations

import hashlib
from os import PathLike
from typing import BinaryIO

from config.settings import Settings

from .loader import load_pdf_details
from .models import PolicyDocument
from .splitter import split_pages


def build_policy_document(
    source: bytes | bytearray | PathLike[str] | str | BinaryIO,
    filename: str | None = None,
    *,
    settings: Settings,
) -> PolicyDocument:
    """Load and split one PDF without indexing it."""

    loaded = load_pdf_details(source, filename, max_bytes=settings.max_pdf_bytes)
    policy_id = hashlib.sha256(loaded.content).hexdigest()
    chunks = split_pages(
        loaded.pages,
        policy_id,
        loaded.filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return PolicyDocument(
        policy_id=policy_id,
        filename=loaded.filename,
        pages=loaded.pages,
        chunks=chunks,
        warnings=loaded.warnings,
    )
