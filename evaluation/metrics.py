"""Transparent retrieval/citation metrics plus optional RAGAS scoring.

All functions consume live records.  There are deliberately no default answers or
contexts in this module.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any, Iterable


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def anchor_hits(chunks: Iterable[dict[str, Any]], anchors: Iterable[dict[str, Any]]) -> set[int]:
    texts = [(int(c.get("page", -1)), normalize(str(c.get("text", "")))) for c in chunks]
    hits: set[int] = set()
    for i, anchor in enumerate(anchors):
        needle = normalize(str(anchor["anchor"]))
        pages = {int(anchor["page"]), *map(int, anchor.get("equivalent_pages", []))}
        if needle and any(p in pages and needle in text for p, text in texts):
            hits.add(i)
    return hits


def retrieval_scores(chunks: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, float | None]:
    """Anchor-level recall and chunk-level precision, independently labelled."""
    if not anchors:
        return {"retrieval_precision": None, "retrieval_recall": None}
    hits = anchor_hits(chunks, anchors)
    relevant_chunks = sum(bool(anchor_hits([chunk], anchors)) for chunk in chunks)
    return {
        "retrieval_precision": relevant_chunks / len(chunks) if chunks else 0.0,
        "retrieval_recall": len(hits) / len(anchors),
    }


def citation_scores(citations: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, float | None]:
    """Score only citations emitted by the agent, never the retrieved gold context."""
    if not anchors:
        return {"citation_precision": None, "citation_recall": None}
    hits = anchor_hits(citations, anchors)
    supported = sum(bool(anchor_hits([citation], anchors)) for citation in citations)
    return {
        "citation_precision": supported / len(citations) if citations else 0.0,
        "citation_recall": len(hits) / len(anchors),
    }


def bill_field_accuracy(actual: dict[str, Any], expected: dict[str, Any]) -> float | None:
    """Field-level exact match for the extractor's nested value/page/evidence shape."""
    if not expected:
        return None
    matches = []
    for key, value in expected.items():
        got = actual.get(key)
        if not isinstance(value, dict) or not isinstance(got, dict):
            matches.append(False)
            continue
        expected_value, got_value = value.get("value"), got.get("value")
        if isinstance(expected_value, (int, float)):
            try: value_matches = float(got_value) == float(expected_value)
            except (TypeError, ValueError): value_matches = False
        elif expected_value is None:
            value_matches = got_value is None
        else:
            value_matches = normalize(str(got_value or "")) == normalize(str(expected_value))
        matches.append(value_matches and got.get("page") == value.get("page") and normalize(str(got.get("evidence") or "")) == normalize(str(value.get("evidence") or "")))
    return sum(matches) / len(matches)


def aggregate(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"cases": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        result[field] = mean(values) if values else None
        result[field + "_n"] = len(values)
    return result


def ragas_scores(question: str, answer: str, contexts: list[str], reference: str, llm: Any, embeddings: Any) -> dict[str, float]:
    """Run RAGAS 0.2.x for one successful live generation (lazy imports)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    ds = Dataset.from_dict({"question": [question], "answer": [answer], "contexts": [contexts], "ground_truth": [reference]})
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                      llm=llm, embeddings=embeddings, raise_exceptions=False,
                      run_config=RunConfig(max_workers=1, max_retries=1, timeout=90), show_progress=False)
    row = result.to_pandas().iloc[0].to_dict()
    return {key: float(value) for key, value in row.items() if key in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"} and value == value}
