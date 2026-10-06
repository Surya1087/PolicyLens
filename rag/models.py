"""Serializable shared domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


class PolicyLensError(Exception):
    """A user-actionable PolicyLens ingestion or retrieval error."""


@dataclass(frozen=True, slots=True)
class Page:
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    policy_id: str
    text: str
    page: int
    section: str
    source: str


@dataclass(slots=True)
class PolicyDocument:
    policy_id: str
    filename: str
    pages: list[Page] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
