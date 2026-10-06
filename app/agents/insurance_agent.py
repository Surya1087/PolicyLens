from __future__ import annotations

from prompts.prompt import QA_SYSTEM, QA_TASK

from .base import FactResponse, fact_result, json_user, llm_or_default, model_call


class InsuranceAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, query: str = "What does this policy say?") -> dict:
        chunks = self.pipeline.retrieve(policy_id, query, k=6)
        if not chunks:
            return {"answer": "No relevant policy excerpts were found.", "sources": [], "retrieved_contexts": []}
        response = model_call(
            self.llm,
            FactResponse,
            QA_SYSTEM,
            json_user(QA_TASK, chunks, question=query),
        )
        return fact_result(response.items, chunks, empty="No supported answer could be produced from the retrieved excerpts.", retrieved=chunks)


PolicyAgent = InsuranceAgent
