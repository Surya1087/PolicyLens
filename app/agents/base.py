from __future__ import annotations

import json
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm import GroqClient
from rag.models import Chunk, PolicyLensError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Citation(StrictModel):
    id: str
    quote: str = Field(min_length=1, max_length=1500)


class Fact(StrictModel):
    statement: str
    citations: list[Citation] = Field(min_length=1)


class FactResponse(StrictModel):
    items: list[Fact] = Field(max_length=30)


def llm_or_default(llm: Any | None) -> Any:
    return llm if llm is not None else GroqClient()


def chunk_dict(chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "policy_id": chunk.policy_id,
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        "source": chunk.source,
    }


def unique_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.id not in seen:
            seen.add(chunk.id)
            result.append(chunk)
    return result


def limit_context(chunks: Iterable[Chunk], *, max_chunks: int = 8, max_chars: int = 16000) -> list[Chunk]:
    """Keep prompts below free-tier request limits while preserving retrieval order."""
    selected: list[Chunk] = []
    total = 0
    for chunk in chunks:
        if len(selected) >= max_chunks:
            break
        remaining = max_chars - total
        if remaining <= 0:
            break
        text = chunk.text if len(chunk.text) <= remaining else chunk.text[:remaining].rsplit(" ", 1)[0]
        if not text.strip():
            break
        selected.append(chunk if text == chunk.text else Chunk(chunk.id, chunk.policy_id, text, chunk.page, chunk.section, chunk.source))
        total += len(text)
    return selected


def excerpts(chunks: Iterable[Chunk], labels: dict[str, str] | None = None) -> str:
    sections = []
    for chunk in chunks:
        label = labels.get(chunk.id, "") if labels else ""
        sections.append(
            f'<excerpt id="{chunk.id}" policy="{label}" page="{chunk.page}" section="{chunk.section}">\n'
            f"{chunk.text}\n</excerpt>"
        )
    return "\n\n".join(sections)


def model_call(llm: Any, model: type[StrictModel], system: str, user: str) -> StrictModel:
    raw = llm.complete_json(system, user, model.model_json_schema())
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise PolicyLensError("AI response did not match the required structure.") from exc


def validate_facts(items: list[Fact], chunks: Iterable[Chunk]) -> tuple[list[Fact], list[str], list[Chunk]]:
    by_id = {chunk.id: chunk for chunk in chunks}
    valid: list[Fact] = []
    warnings: list[str] = []
    used: dict[str, Chunk] = {}
    for item in items:
        item_ok = bool(item.statement.strip())
        for citation in item.citations:
            chunk = by_id.get(citation.id)
            quote = citation.quote.strip()
            if chunk is None or not quote or _normalize(quote) not in _normalize(chunk.text):
                item_ok = False
                break
        if not item_ok:
            warnings.append("An unsupported generated statement was removed.")
            continue
        valid.append(item)
        for citation in item.citations:
            used[citation.id] = by_id[citation.id]
    return valid, warnings, list(used.values())


def render_facts(items: Iterable[Fact]) -> str:
    return "\n".join(f"- {item.statement} " + " ".join(f"[{c.id}]" for c in item.citations) for item in items)


def fact_result(items: list[Fact], chunks: Iterable[Chunk], *, empty: str, extras: dict | None = None, retrieved: Iterable[Chunk] | None = None) -> dict:
    chunk_list = list(chunks)
    valid, warnings, used = validate_facts(items, chunk_list)
    result = {
        "answer": render_facts(valid) if valid else empty,
        "sources": [chunk_dict(chunk) for chunk in used],
        "retrieved_contexts": [chunk_dict(chunk) for chunk in unique_chunks(retrieved if retrieved is not None else chunk_list)],
    }
    if warnings:
        result["warnings"] = warnings
    if extras:
        result.update(extras)
    return result


def json_user(instruction: str, chunks: Iterable[Chunk], **context: Any) -> str:
    return instruction + "\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False) + "\nEXCERPTS:\n" + excerpts(chunks)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
