from __future__ import annotations

from prompts.prompt import COMPARISON_SYSTEM, COMPARISON_TASK

from .base import FactResponse, fact_result, limit_context, llm_or_default, model_call, unique_chunks, excerpts
from .summary_agent import SUMMARY_FACETS


class ComparisonAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, policy_id_b: str, query: str = "Compare these policies") -> dict:
        chunks_a = unique_chunks(c for facet in SUMMARY_FACETS for c in self.pipeline.retrieve(policy_id, facet, k=3))
        chunks_b = unique_chunks(c for facet in SUMMARY_FACETS for c in self.pipeline.retrieve(policy_id_b, facet, k=3))
        chunks = limit_context(unique_chunks([*chunks_a, *chunks_b]), max_chunks=10, max_chars=14000)
        if not chunks_a or not chunks_b:
            return {"answer": "Both policies need relevant excerpts before they can be compared.", "sources": [], "retrieved_contexts": [], "policy_ids": [policy_id, policy_id_b]}
        labels = {c.id: "A" for c in chunks_a} | {c.id: "B" for c in chunks_b}
        response = model_call(
            self.llm,
            FactResponse,
            COMPARISON_SYSTEM,
            COMPARISON_TASK + " Request: " + query + "\nEXCERPTS:\n" + excerpts(chunks, labels),
        )
        # A comparison statement is accepted only with evidence from both policies.
        response.items = [
            item for item in response.items
            if {labels.get(citation.id) for citation in item.citations} >= {"A", "B"}
        ]
        return fact_result(
            response.items, chunks, empty="No supported comparison could be produced.",
            retrieved=chunks,
            extras={"policy_ids": [policy_id, policy_id_b], "scope": "Comparison is limited to retrieved excerpts and is not a recommendation."},
        )
