from __future__ import annotations

from prompts.prompt import SUMMARY_SYSTEM, SUMMARY_TASK

from .base import FactResponse, fact_result, json_user, limit_context, llm_or_default, model_call, unique_chunks


SUMMARY_FACETS = (
    "coverage benefits inclusions insured events",
    "optional add-on covers riders purchased options",
    "policy number sum insured premium policy period key amounts",
    "exclusions limitations waiting periods",
    "limits sublimits deductibles copayments",
    "claim notification documents process deadlines",
    "renewal cancellation duties definitions",
)


class SummaryAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, query: str = "Executive summary") -> dict:
        chunks = unique_chunks(
            chunk for facet in SUMMARY_FACETS for chunk in self.pipeline.retrieve(policy_id, facet, k=4)
        )
        chunks = limit_context(chunks, max_chunks=8, max_chars=14000)
        if not chunks:
            return {"answer": "No policy excerpts were found for a scoped summary.", "sources": [], "retrieved_contexts": [], "scope": "No excerpts reviewed."}
        response = model_call(
            self.llm,
            FactResponse,
            SUMMARY_SYSTEM,
            json_user(SUMMARY_TASK, chunks, requested_focus=query),
        )
        return fact_result(
            response.items,
            chunks,
            empty="No supported summary items could be produced.",
            retrieved=chunks,
            extras={"scope": "Scoped summary from retrieved coverage, limitation, cost-sharing, claims, and lifecycle facets; not an exhaustive full-policy review."},
        )
