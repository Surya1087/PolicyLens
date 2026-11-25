from __future__ import annotations

from prompts.prompt import SUMMARY_SYSTEM, SUMMARY_TASK

from .base import SummaryResponse, fact_result, json_user, limit_context, llm_or_default, model_call, unique_chunks


SUMMARY_FACETS = (
    "policy coverage benefits inclusions insured events optional covers riders sum insured premium policy period",
    "exclusions limitations waiting periods limits sublimits deductibles copayments",
    "claim notification documents process deadlines renewal cancellation duties definitions",
)
SUMMARY_TOPIC_ORDER = (
    "Coverage and benefits",
    "Costs and limits",
    "Exclusions and waiting periods",
    "Claims and next steps",
    "Policy details",
)


class SummaryAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, query: str = "Executive summary") -> dict:
        chunks = unique_chunks(
            chunk for facet in SUMMARY_FACETS for chunk in self.pipeline.retrieve(policy_id, facet, k=3, rerank=False)
        )
        chunks = limit_context(chunks, max_chunks=5, max_chars=6500)
        if not chunks:
            return {"answer": "No policy excerpts were found for a scoped summary.", "sources": [], "retrieved_contexts": [], "scope": "No excerpts reviewed."}
        response = model_call(
            self.llm,
            SummaryResponse,
            SUMMARY_SYSTEM,
            json_user(SUMMARY_TASK, chunks, requested_focus=query),
        )
        response.items.sort(key=lambda item: SUMMARY_TOPIC_ORDER.index(item.topic))
        return fact_result(
            response.items,
            chunks,
            empty="No supported summary items could be produced.",
            retrieved=chunks,
            extras={"scope": "Scoped summary from retrieved coverage, limitation, cost-sharing, claims, and lifecycle facets; not an exhaustive full-policy review."},
        )
