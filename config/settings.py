"""Environment-backed settings with no import-time network or model work."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from .constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_PDF_BYTES,
    DEFAULT_RETRIEVAL_K,
)


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("POLICYLENS_DATA_DIR", str(PROJECT_ROOT / ".local_data"))).expanduser().resolve()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("POLICYLENS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    max_pdf_bytes: int = field(
        default_factory=lambda: _positive_int("POLICYLENS_MAX_PDF_BYTES", DEFAULT_MAX_PDF_BYTES)
    )
    chunk_size: int = field(
        default_factory=lambda: _positive_int("POLICYLENS_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _positive_int("POLICYLENS_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    )
    retrieval_k: int = field(
        default_factory=lambda: _positive_int("POLICYLENS_RETRIEVAL_K", DEFAULT_RETRIEVAL_K)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings. Call ``get_settings.cache_clear()`` in tests."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    return Settings()
