"""Cross-module regression checks; no live hosted API or fake scores."""
from io import BytesIO

import pytest
from reportlab.pdfgen.canvas import Canvas

from rag.bill_extractor import BillExtractor, _valid_field
from rag.models import PolicyLensError
from app.agents.claim_agent import ClaimAgent, ClaimResponse
from rag.rag_pipeline import claim_retrieval_query
from config.constants import AMBIGUITY_PATTERNS


def test_network_negation_is_not_in_network():
    quote = "Hospital: non-network hospital"
    evidence = {"page": 1, "quote": quote}
    assert not _valid_field("network_type", "in_network", evidence, {1: quote})
    assert _valid_field("network_type", "out_of_network", evidence, {1: quote})


def test_company_sole_discretion_pattern():
    import re
    assert any(re.search(p, "at the Company's sole discretion", re.I) for p in AMBIGUITY_PATTERNS)


def test_claim_schema_cannot_accept_freeform_verdict():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ClaimResponse.model_validate({"selections": [], "decision": "insurer will pay"})
    with pytest.raises(ValidationError):
        ClaimResponse.model_validate({"selections": [{"topic": "approved", "citations": []}]})


def test_claim_queries_do_not_leak_person_names_or_amounts():
    query = claim_retrieval_query({"procedure_or_diagnosis": {"value": "phacoemulsification"}, "hospital": {"value": "Personal Hospital"}, "amount": {"value": 99999}})
    assert "phacoemulsification" in query
    assert "Personal" not in query and "99999" not in query


def test_bill_values_require_evidence_from_real_pdf():
    data = BytesIO()
    pdf = Canvas(data)
    pdf.drawString(30, 700, "Procedure: Cataract surgery")
    pdf.drawString(30, 680, "Date: 2026-05-20; Amount: INR 48000.00")
    pdf.save()

    class FakeLLM:
        def complete_json(self, system, user, schema):
            assert "Cataract surgery" in user and "claim decision" in system
            empty = {"value": None, "evidence": None}
            return {
                "procedure_or_diagnosis": {"value": "Cataract surgery", "evidence": {"page": 1, "quote": "Procedure: Cataract surgery"}},
                "treatment_date": {"value": "2026-05-20", "evidence": {"page": 1, "quote": "Date: 2026-05-20"}},
                "amount": {"value": 99000, "evidence": {"page": 1, "quote": "Amount: INR 48000.00"}},
                "currency": empty, "hospital": empty, "network_type": empty,
            }

    fields = BillExtractor(FakeLLM()).extract(data.getvalue())
    assert fields["procedure_or_diagnosis"]["value"] == "Cataract surgery"
    assert fields["amount"]["value"] is None
    assert fields["treatment_date"]["value"] == "2026-05-20"


def test_empty_bill_stops_before_api():
    class NeverLLM:
        def complete_json(self, *args):
            raise AssertionError("Network should not be called")
    with pytest.raises(PolicyLensError, match="empty"):
        BillExtractor(NeverLLM()).extract(b"")
