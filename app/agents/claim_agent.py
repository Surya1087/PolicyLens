from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from prompts.prompt import CLAIM_SYSTEM, CLAIM_TASK
from rag.bill_extractor import BillExtractor
from rag.models import PolicyLensError
from rag.rag_pipeline import claim_retrieval_query

from .base import Citation, StrictModel, chunk_dict, excerpts, llm_or_default, model_call, unique_chunks

DISCLAIMER = "This is a factual document cross-check, not a coverage or claim decision. Only the insurer can assess the claim under the complete policy and submitted records."
Topic = Literal[
    "waiting_period", "cost_limit", "deductible_or_copay", "network_requirement",
    "exclusion", "documentation", "timing_requirement", "definition", "other_limitation",
]
TOPIC_LABELS = {
    "waiting_period": "waiting-period", "cost_limit": "cost-limit", "deductible_or_copay": "deductible or copayment",
    "network_requirement": "network-requirement", "exclusion": "exclusion", "documentation": "documentation-requirement",
    "timing_requirement": "timing-requirement", "definition": "definition", "other_limitation": "other limitation",
}


class ClauseSelection(StrictModel):
    topic: Topic
    citations: list[Citation] = Field(min_length=1, max_length=3)


class ClaimResponse(StrictModel):
    selections: list[ClauseSelection] = Field(max_length=12)


class ClaimAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)
        self.bill_extractor = BillExtractor(self.llm)

    def run(self, policy_id: str, query: str = "Cross-check this bill against relevant policy language", bill_source=None, policy_start_date: str | None = None) -> dict:
        if bill_source is None:
            raise PolicyLensError("A bill_source is required for Claim Check.")
        extracted = self.bill_extractor.extract(bill_source)
        bill_warnings = list(extracted.pop("_warnings", self.bill_extractor.last_warnings))
        bill_fields = extracted
        procedure = bill_fields["procedure_or_diagnosis"]
        if procedure["value"] is None:
            raise PolicyLensError("A cited procedure or diagnosis is required before policy cross-checking.")
        search = claim_retrieval_query(bill_fields)
        chunks = unique_chunks(self.pipeline.retrieve(policy_id, search, k=6))
        interval = _interval(policy_start_date, bill_fields["treatment_date"]["value"])
        base = {
            "bill_fields": bill_fields, "bill_warnings": bill_warnings, "policy_date_context": interval,
            "claim_status": "not_assessed", "disclaimer": DISCLAIMER,
            "retrieved_contexts": [chunk_dict(c) for c in chunks],
        }
        if not chunks:
            return {"answer": _bill_intro(bill_fields, interval) + "\n\n" + DISCLAIMER, "sources": [], "findings": [], **base}
        response = model_call(
            self.llm, ClaimResponse, CLAIM_SYSTEM,
            CLAIM_TASK + "\n"
            "BILL_FIELDS (untrusted):\n" + str(bill_fields) + "\nPOLICY_DATE_CONTEXT:\n" + str(interval)
            + "\nUSER_FOCUS:\n" + query + "\nEXCERPTS:\n" + excerpts(chunks),
        )
        by_id = {chunk.id: chunk for chunk in chunks}
        findings, used, warnings = [], {}, []
        seen: set[tuple[str, str, str]] = set()
        for selection in response.selections:
            for citation in selection.citations:
                chunk = by_id.get(citation.id)
                quote = citation.quote.strip()
                key = (selection.topic, citation.id, " ".join(quote.split()).casefold())
                if chunk is None or not quote or key in seen or key[2] not in " ".join(chunk.text.split()).casefold():
                    warnings.append("An unsupported or duplicate clause selection was removed.")
                    continue
                seen.add(key)
                used[chunk.id] = chunk
                label = TOPIC_LABELS[selection.topic]
                statement = f'A potentially relevant {label} passage appears on page {chunk.page}: “{quote}” [{chunk.id}]. Please confirm applicability with the insurer.'
                findings.append({
                    "topic": selection.topic, "assessment": "not_assessed", "statement": statement,
                    "citation": {"id": chunk.id, "quote": quote}, "page": chunk.page,
                })
        lines = [finding["statement"] for finding in findings]
        answer = _bill_intro(bill_fields, interval)
        if lines:
            answer += "\n\n" + "\n".join(f"- {line}" for line in lines)
        answer += "\n\n" + DISCLAIMER
        result = {"answer": answer, "sources": [chunk_dict(c) for c in used.values()], "findings": findings, **base}
        if warnings:
            result["warnings"] = warnings
        return result


def _bill_intro(fields: dict, interval: dict) -> str:
    page = fields["procedure_or_diagnosis"]["page"]
    parts = [f"The bill lists a procedure or diagnosis on page {page}."]
    if fields["amount"]["value"] is not None:
        parts.append(f"The bill lists an amount on page {fields['amount']['page']}.")
    if interval["elapsed_days"] is not None:
        days = interval["elapsed_days"]
        relation = f"{days} day(s) later" if days >= 0 else f"{abs(days)} day(s) earlier"
        parts.append(f"Using the unverified user-supplied policy start date, the treatment date is {relation}.")
    return " ".join(parts)


def _interval(start: str | None, treatment: object) -> dict:
    if not start:
        return {"policy_start_date": None, "verified": False, "elapsed_days": None}
    try:
        start_date = date.fromisoformat(start)
        treatment_date = date.fromisoformat(str(treatment)) if treatment else None
    except ValueError:
        raise PolicyLensError("Dates must use ISO format YYYY-MM-DD.")
    return {
        "policy_start_date": start, "verified": False,
        "source": "user supplied; not inferred from the policy brochure",
        "elapsed_days": (treatment_date - start_date).days if treatment_date else None,
    }
