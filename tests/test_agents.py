from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents import ComparisonAgent, InsuranceAgent, RiskAgent, Router
from app.agents.claim_agent import ClaimAgent, DISCLAIMER
from rag.models import Chunk, Page, PolicyDocument, PolicyLensError
from rag.bill_extractor import BillExtractor


def chunk(identifier="c1", text="The deductible is INR 5,000.", policy="p1", page=2):
    return Chunk(identifier, policy, text, page, "Benefits", "policy.pdf")


class Pipeline:
    def __init__(self, chunks=None, warnings=None):
        self.chunks = chunks or [chunk()]
        self.calls = []
        self.warnings = warnings or []

    def retrieve(self, policy_id, query, k=6):
        self.calls.append((policy_id, query, k))
        return [c for c in self.chunks if c.policy_id == policy_id]

    def get_policy(self, policy_id):
        selected = [c for c in self.chunks if c.policy_id == policy_id]
        return PolicyDocument(policy_id, "policy.pdf", [Page(1, "text")], selected, self.warnings)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        return self.responses.pop(0)


def test_question_keeps_only_exact_supported_citations():
    llm = FakeLLM([{"items": [
        {"statement": "A deductible applies.", "citations": [{"id": "c1", "quote": "deductible is INR 5,000"}]},
        {"statement": "Unlimited benefit.", "citations": [{"id": "missing", "quote": "unlimited"}]},
    ]}])
    result = InsuranceAgent(Pipeline(), llm).run("p1", "deductible?")
    assert "A deductible applies" in result["answer"]
    assert "Unlimited" not in result["answer"]
    assert [source["id"] for source in result["sources"]] == ["c1"]
    assert [context["id"] for context in result["retrieved_contexts"]] == ["c1"]
    assert result["warnings"]


def test_summary_retrieves_multiple_facets():
    llm = FakeLLM([{"items": []}])
    pipeline = Pipeline()
    Router(pipeline, llm).run("Executive Summary", policy_id="p1")
    from app.agents.summary_agent import SUMMARY_FACETS
    assert len(pipeline.calls) == len(SUMMARY_FACETS)
    assert {call[1] for call in pipeline.calls} == set(SUMMARY_FACETS)


def test_risk_scans_every_chunk_and_reports_partial_state():
    chunks = [
        chunk("c1", "Benefits are subject to reasonable charges."),
        chunk("c2", "This clause applies at our discretion.", page=3),
        chunk("c3", "Plain text.", page=4),
    ]
    from app.agents.risk_agent import _configured_patterns, _scan_candidates
    expected_id = next(item["candidate_id"] for item in _scan_candidates(chunks, _configured_patterns()) if item["chunk_id"] == "c2")
    class RiskLLM(FakeLLM):
        def complete_json(self, system, user, schema):
            self.calls.append((system, user, schema))
            if "assessments" not in schema.get("properties", {}):
                return {"items": []}
            return {"assessments": [{
                "candidate_id": expected_id, "confirmed": True, "excerpt": "at our discretion",
                "why": "The wording leaves discretion to the insurer.",
                "citations": [{"id": "c2", "quote": "at our discretion"}],
            }]}
    llm = RiskLLM([])
    result = RiskAgent(Pipeline(chunks, ["page omitted"]), llm).run("p1")
    assert result["scan"]["chunks_scanned"] == 3
    assert result["scan"]["partial"] is True
    assert result["sources"][0]["id"] == "c2"


def test_claim_blocks_decision_like_generated_prose(monkeypatch):
    llm = FakeLLM([{"selections": [{"topic": "deductible_or_copay", "citations": [{"id": "c1", "quote": "deductible is INR 5,000"}]}]}])
    agent = ClaimAgent(Pipeline(), llm)
    monkeypatch.setattr(agent.bill_extractor, "extract", lambda source: {
        **{name: {"value": None, "page": None, "evidence": None} for name in ("treatment_date", "amount", "currency", "hospital", "network_type")},
        "procedure_or_diagnosis": {"value": "cataract surgery", "page": 1, "evidence": "cataract surgery"},
    })
    result = agent.run("p1", bill_source=b"pdf")
    assert "approved" not in result["answer"].lower()
    assert result["claim_status"] == "not_assessed"
    assert "A potentially relevant deductible or copayment passage" in result["answer"]
    assert "model" not in result["answer"].lower()


def test_claim_requires_verified_procedure_or_diagnosis(monkeypatch):
    agent = ClaimAgent(Pipeline(), FakeLLM([]))
    monkeypatch.setattr(agent.bill_extractor, "extract", lambda source: {
        **{name: {"value": None, "page": None, "evidence": None} for name in ("procedure_or_diagnosis", "treatment_date", "amount", "currency", "hospital", "network_type")},
    })
    with pytest.raises(PolicyLensError, match="procedure or diagnosis"):
        agent.run("p1", bill_source=b"pdf")


def test_unknown_router_route_is_explicit():
    with pytest.raises(PolicyLensError, match="Unknown route"):
        Router(Pipeline(), FakeLLM([])).run("Magic")


def test_comparison_requires_citations_from_both_policies():
    chunks = [chunk("a", "Policy A has a deductible.", "a"), chunk("b", "Policy B has no stated deductible.", "b")]
    llm = FakeLLM([{"items": [
        {"statement": "Unsupported one-sided comparison.", "citations": [{"id": "a", "quote": "has a deductible"}]},
        {"statement": "The documents use different deductible language.", "citations": [
            {"id": "a", "quote": "has a deductible"}, {"id": "b", "quote": "no stated deductible"},
        ]},
    ]}])
    result = ComparisonAgent(Pipeline(chunks), llm).run("a", "b")
    assert "one-sided" not in result["answer"]
    assert "different deductible" in result["answer"]


def test_bill_extraction_verifies_page_evidence(monkeypatch):
    monkeypatch.setattr("rag.loader.load_pdf_details", lambda source, filename=None: SimpleNamespace(
        pages=[Page(1, "Cataract surgery Total INR 1200 at City Hospital")], warnings=["scan warning"]
    ))
    null = {"value": None, "evidence": None}
    llm = FakeLLM([{
        "procedure_or_diagnosis": {"value": "Cataract surgery", "evidence": {"page": 1, "quote": "Cataract surgery"}},
        "treatment_date": null,
        "amount": {"value": 1200.0, "evidence": {"page": 1, "quote": "INR 1200"}},
        "currency": {"value": "INR", "evidence": {"page": 9, "quote": "INR"}},
        "hospital": {"value": "City Hospital", "evidence": {"page": 1, "quote": "City Hospital"}},
        "network_type": null,
    }])
    result = BillExtractor(llm).extract(b"fake")
    assert result["amount"] == {"value": 1200.0, "page": 1, "evidence": "INR 1200"}
    assert result["currency"]["value"] is None
    assert result["_warnings"] == ["scan warning"]


def test_ambiguity_scan_batches_six_chunks_and_exposes_batch_failure():
    chunks = [chunk(f"c{i}", f"Term {i} is subject to review.", page=i) for i in range(1, 8)]

    class BatchLLM:
        def __init__(self):
            self.calls = []

        def complete_json(self, system, user, schema):
            self.calls.append(user)
            if len(self.calls) == 1:
                raise PolicyLensError("temporary model failure")
            return {"assessments": []}

    llm = BatchLLM()
    result = RiskAgent(Pipeline(chunks), llm).flag_ambiguities("p1")
    assert len(llm.calls) == 2
    assert all(call.count("<excerpt id=") <= 6 for call in llm.calls)
    assert result["scan"]["partial"] is True
    assert len(result["retrieved_contexts"]) == 7
    assert any("batch 1 failed" in failure.lower() for failure in result["scan"]["failures"])


def test_ambiguity_patterns_accept_mapping_shape(monkeypatch):
    import config.constants as constants
    from app.agents.risk_agent import _configured_patterns, _scan_candidates

    monkeypatch.setattr(constants, "AMBIGUITY_PATTERNS", {"discretion": r"\bat our discretion\b"})
    matches = _scan_candidates([chunk("c1", "Applied at our discretion.")], _configured_patterns())
    assert matches[0]["label"] == "discretion"


def test_bill_value_must_match_its_own_evidence(monkeypatch):
    monkeypatch.setattr("rag.loader.load_pdf_details", lambda source, filename=None: SimpleNamespace(
        pages=[Page(1, "Cataract surgery; total INR 1200; date 02/01/2026")], warnings=[]
    ))
    null = {"value": None, "evidence": None}
    result = BillExtractor(FakeLLM([{
        "procedure_or_diagnosis": {"value": "Cataract surgery", "evidence": {"page": 1, "quote": "Cataract surgery"}},
        "treatment_date": {"value": "2026-01-02", "evidence": {"page": 1, "quote": "date 02/01/2026"}},
        "amount": {"value": 999.0, "evidence": {"page": 1, "quote": "total INR 1200"}},
        "currency": {"value": "INR", "evidence": {"page": 1, "quote": "INR 1200"}},
        "hospital": null, "network_type": null,
    }])).extract(b"fake")
    assert result["treatment_date"]["value"] == "2026-01-02"
    assert result["amount"]["value"] is None
