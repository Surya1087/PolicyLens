"""Opt-in local integration: actual pretrained embeddings, PDFs and Chroma.

RUN_EMBEDDING_TESTS=1 python -m pytest -q tests/test_live_retrieval.py
No hosted LLM is called; downloading public model weights may require internet.
"""
import os
from pathlib import Path

import pytest

from rag.rag_pipeline import RAGPipeline, claim_retrieval_query
from evaluation.test_cases import load_claim_cases
from evaluation.metrics import retrieval_scores


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_EMBEDDING_TESTS") != "1", reason="Opt-in real embedding download")
def test_actual_embeddings_pdf_and_policy_isolation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pipeline = RAGPipeline(persist_directory=tmp_path / "real-vectors")
    cases = load_claim_cases()
    cataract = next(case for case in cases if case["id"] == "claim-health-cataract")
    health = pipeline.ingest(root / "data/insurance" / cataract["policy_file"])
    travel = pipeline.ingest(root / "data/insurance/travel_insurance_brochure-721286560295.pdf")
    result = pipeline.retrieve(health.policy_id, claim_retrieval_query(cataract["expected_bill_fields"]), k=6)
    assert all(chunk.policy_id == health.policy_id for chunk in result)
    assert retrieval_scores([{"page": c.page, "text": c.text} for c in result], cataract["anchors"])["retrieval_recall"] == 1
    reloaded = RAGPipeline(persist_directory=tmp_path / "real-vectors", embeddings=pipeline.embeddings)
    assert reloaded.get_policy(health.policy_id).filename == health.filename
    travel_results = reloaded.retrieve(travel.policy_id, "cataract", k=6)
    assert all(chunk.policy_id == travel.policy_id for chunk in travel_results)
