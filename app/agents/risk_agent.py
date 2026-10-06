from __future__ import annotations

import hashlib
import re
from typing import Iterable

from prompts.prompt import AMBIGUITY_TASK, RISK_BASELINE_SYSTEM, RISK_BASELINE_TASK, RISK_SYSTEM
from .base import Citation, Fact, FactResponse, StrictModel, chunk_dict, excerpts, fact_result, limit_context, llm_or_default, model_call, unique_chunks, validate_facts

RISK_FACETS = (
    "coverage gaps omitted benefits and conditions",
    "exclusions and circumstances not insured",
    "waiting periods and time restrictions",
    "deductibles copayments sublimits and cost limits",
    "duties limitations claim notice and documentation requirements",
)
AMBIGUITY_BATCH_CHUNKS = 6


class AmbiguityAssessment(StrictModel):
    candidate_id: str
    confirmed: bool
    excerpt: str | None
    why: str
    citations: list[Citation]


class AmbiguityResponse(StrictModel):
    assessments: list[AmbiguityAssessment]


class RiskAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, query: str = "Identify notable policy risks and ambiguities") -> dict:
        baseline_chunks = unique_chunks(
            chunk for facet in RISK_FACETS for chunk in self.pipeline.retrieve(policy_id, facet, k=4)
        )
        baseline_chunks = limit_context(baseline_chunks, max_chunks=8, max_chars=14000)
        if baseline_chunks:
            response = model_call(
                self.llm, FactResponse, RISK_BASELINE_SYSTEM,
                RISK_BASELINE_TASK + " User focus: " + query + "\nEXCERPTS:\n" + excerpts(baseline_chunks),
            )
            baseline = fact_result(
                response.items, baseline_chunks, empty="No supported baseline risk findings were produced.", retrieved=baseline_chunks,
            )
        else:
            baseline = {"answer": "No relevant policy excerpts were found for baseline risk review.", "sources": [], "retrieved_contexts": []}
        ambiguity = self.flag_ambiguities(policy_id, query=query)
        sources = _unique_dicts([*baseline["sources"], *ambiguity["sources"]])
        contexts = _unique_dicts([*baseline["retrieved_contexts"], *ambiguity["retrieved_contexts"]])
        answer = baseline["answer"]
        if ambiguity["answer"]:
            answer += "\n\nAmbiguity scan:\n" + ambiguity["answer"]
        result = {
            "answer": answer, "sources": sources, "retrieved_contexts": contexts,
            "findings": ambiguity["findings"], "ambiguity_flags": ambiguity["ambiguity_flags"],
            "scan": ambiguity["scan"],
        }
        warnings = [*baseline.get("warnings", []), *ambiguity.get("warnings", [])]
        if warnings:
            result["warnings"] = warnings
        return result

    def flag_ambiguities(self, policy_id: str, query: str = "") -> dict:
        policy = self.pipeline.get_policy(policy_id)
        patterns = _configured_patterns()
        candidates = _scan_candidates(policy.chunks, patterns)
        if not candidates:
            failures = list(policy.warnings)
            if not patterns:
                failures.append("No configured ambiguity patterns were available.")
            return {
                "answer": "No configured ambiguity language was confirmed.", "sources": [], "retrieved_contexts": [],
                "findings": [], "ambiguity_flags": [], "warnings": failures,
                "scan": {"chunks_scanned": len(policy.chunks), "candidate_chunks": 0, "candidates": 0,
                         "unassessed_candidates": [], "partial": bool(failures), "failures": failures},
            }
        by_chunk: dict[str, list[dict]] = {}
        chunks = {chunk.id: chunk for chunk in policy.chunks}
        for candidate in candidates:
            by_chunk.setdefault(candidate["chunk_id"], []).append(candidate)
        chunk_ids = list(by_chunk)
        confirmed, used, provided, assessed = [], {}, {}, set()
        failures = list(policy.warnings)
        for offset in range(0, len(chunk_ids), AMBIGUITY_BATCH_CHUNKS):
            batch_ids = chunk_ids[offset:offset + AMBIGUITY_BATCH_CHUNKS]
            batch_chunks = [chunks[chunk_id] for chunk_id in batch_ids]
            batch_candidates = [item for chunk_id in batch_ids for item in by_chunk[chunk_id]]
            provided.update({chunk.id: chunk for chunk in batch_chunks})
            candidate_map = {item["candidate_id"]: {k: v for k, v in item.items() if k != "context_key"} for item in batch_candidates}
            try:
                response = model_call(
                    self.llm, AmbiguityResponse, RISK_SYSTEM,
                    AMBIGUITY_TASK + " User focus: " + query + "\nCANDIDATES:\n"
                    + str(candidate_map) + "\nEXCERPTS:\n" + excerpts(batch_chunks),
                )
            except Exception as exc:
                failures.append(f"Ambiguity batch {offset // AMBIGUITY_BATCH_CHUNKS + 1} failed: {type(exc).__name__}.")
                continue
            candidate_by_id = {item["candidate_id"]: item for item in batch_candidates}
            for assessment in response.assessments:
                candidate = candidate_by_id.get(assessment.candidate_id)
                if candidate is None or assessment.candidate_id in assessed:
                    failures.append("An unknown or duplicate ambiguity assessment was removed.")
                    continue
                assessed.add(assessment.candidate_id)
                if not assessment.confirmed:
                    continue
                if not assessment.citations:
                    failures.append("An ambiguity finding without citations was removed.")
                    continue
                fact = Fact(statement=assessment.why, citations=assessment.citations)
                facts, _, cited = validate_facts([fact], batch_chunks)
                exact = (assessment.excerpt or "").strip()
                source_ok = any(chunk.id == candidate["chunk_id"] for chunk in cited)
                excerpt_ok = _norm(exact) == _norm(candidate["match"]) and source_ok
                if not facts or not excerpt_ok:
                    failures.append("An unsupported ambiguity finding was removed.")
                    continue
                finding = {
                    "candidate_id": assessment.candidate_id, "pattern": candidate["label"],
                    "excerpt": exact, "why": assessment.why,
                    "citations": [citation.model_dump() for citation in assessment.citations],
                }
                confirmed.append(finding)
                used.update({chunk.id: chunk for chunk in cited})
        missing = sorted({item["candidate_id"] for item in candidates} - assessed)
        if missing:
            failures.append(f"The model did not assess {len(missing)} ambiguity candidate(s).")
        answer = "\n".join(
            f"- {finding['why']} " + " ".join(f"[{citation['id']}]" for citation in finding["citations"])
            for finding in confirmed
        ) or "No configured ambiguity language was confirmed."
        return {
            "answer": answer, "sources": [chunk_dict(chunk) for chunk in used.values()],
            "retrieved_contexts": [chunk_dict(chunk) for chunk in provided.values()],
            "findings": confirmed, "ambiguity_flags": sorted({item["pattern"] for item in confirmed}),
            "warnings": failures,
            "scan": {"chunks_scanned": len(policy.chunks), "candidate_chunks": len(by_chunk), "candidates": len(candidates),
                     "unassessed_candidates": missing, "partial": bool(failures), "failures": failures},
        }


def _configured_patterns() -> list[tuple[str, str]]:
    import config.constants as constants
    configured = getattr(constants, "AMBIGUITY_PATTERNS", None)
    if isinstance(configured, dict):
        result = []
        for label, value in configured.items():
            if isinstance(value, dict):
                display = str(value.get("label", label))
                value = value.get("patterns", value.get("regex", value.get("pattern", [])))
                values = value if isinstance(value, (list, tuple, set)) else [value]
                result.extend((display, pattern) for pattern in values if isinstance(pattern, str))
                continue
            if isinstance(value, str) and _looks_regex(str(label)) and not _looks_regex(value):
                result.append((value, str(label)))
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            result.extend((str(label), pattern) for pattern in values if isinstance(pattern, str))
        return result
    if isinstance(configured, (list, tuple, set)):
        return [(pattern, pattern) for pattern in configured if isinstance(pattern, str)]
    configured = getattr(constants, "AMBIGUITY_REGEX_PATTERNS", ())
    return [(pattern, pattern) for pattern in configured if isinstance(pattern, str)]


def _scan_candidates(chunks: Iterable, patterns: list[tuple[str, str]]) -> list[dict]:
    candidates, seen = [], set()
    for chunk in chunks:
        for label, pattern in patterns:
            try:
                matches = re.finditer(pattern, chunk.text, re.IGNORECASE)
            except re.error:
                continue
            for match in matches:
                start = max(chunk.text.rfind("\n", 0, match.start()), chunk.text.rfind(".", 0, match.start())) + 1
                ends = [position for marker in ("\n", ".") if (position := chunk.text.find(marker, match.end())) >= 0]
                end = min(ends) + 1 if ends else min(len(chunk.text), match.end() + 80)
                context = _norm(chunk.text[start:end])
                context_key = (chunk.page, label, _norm(match.group(0)), context)
                if context_key in seen:
                    continue
                seen.add(context_key)
                digest = hashlib.sha256(":".join(map(str, context_key)).encode()).hexdigest()[:12]
                candidates.append({
                    "candidate_id": f"amb-{digest}", "chunk_id": chunk.id, "label": label,
                    "match": match.group(0), "context_key": context,
                })
    return candidates


def _unique_dicts(items: list[dict]) -> list[dict]:
    return list({item["id"]: item for item in items}.values())


def _norm(value: str) -> str:
    return " ".join(value.split()).casefold()


def _looks_regex(value: str) -> bool:
    return any(token in value for token in (r"\b", "(?:", "(?=", "[", "\\s", "|", ".*", "+"))
