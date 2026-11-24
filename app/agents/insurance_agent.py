from __future__ import annotations

from prompts.prompt import QA_SYSTEM, QA_TASK

from .base import FactResponse, evidence_confidence, fact_result, json_user, llm_or_default, model_call


class InsuranceAgent:
    def __init__(self, pipeline, llm=None):
        self.pipeline = pipeline
        self.llm = llm_or_default(llm)

    def run(self, policy_id: str, query: str = "What does this policy say?") -> dict:
        chunks = self.pipeline.retrieve(policy_id, query, k=6)
        if not chunks:
            confidence = evidence_confidence(chunks, query=query)
            return {
                "answer": "No relevant policy excerpts were found.",
                "sources": [],
                "retrieved_contexts": [],
                "evidence_confidence": confidence,
            }
        response = model_call(
            self.llm,
            FactResponse,
            QA_SYSTEM,
            json_user(QA_TASK, chunks, question=query),
        )
        result = fact_result(response.items, chunks, empty="No supported answer could be produced from the retrieved excerpts.", retrieved=chunks)
        confidence = evidence_confidence(chunks, query=query)
        result["evidence_confidence"] = confidence
        if confidence["level"] == "low":
            result["warnings"] = [*result.get("warnings", []), "Evidence confidence is low. The complete policy should be checked before relying on this answer."]
        return result


PolicyAgent = InsuranceAgent
