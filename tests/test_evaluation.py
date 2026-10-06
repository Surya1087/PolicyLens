import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.evaluator import load_evaluation_settings, run
from evaluation.metrics import aggregate, bill_field_accuracy, citation_scores, retrieval_scores
from evaluation.test_cases import load_agent_cases, load_claim_cases


def test_label_corpus_shape_and_physical_anchors():
    agent, claims = load_agent_cases(), load_claim_cases()
    assert {x["kind"] for x in agent} == {"qa", "summary", "risk"}
    assert len(claims) == 12
    assert any("phacoemulsification" in x["bill_document"]["description"].lower() for x in claims)
    assert all(set(x["expected_bill_fields"]) == {"procedure_or_diagnosis", "treatment_date", "amount", "currency", "hospital", "network_type"} for x in claims)
    assert sum(x["label"] == "insufficient_evidence" for x in claims) == 2
    assert all(a["page"] >= 1 for x in agent + claims for a in x["anchors"])


def test_retrieval_and_citation_are_scored_separately():
    anchors = [{"page": 3, "anchor": "covered after twenty four months"}]
    retrieved = [{"page": 3, "text": "Covered after twenty-four months."}, {"page": 8, "text": "unrelated"}]
    assert retrieval_scores(retrieved, anchors) == {"retrieval_precision": 0.5, "retrieval_recall": 1.0}
    assert citation_scores([], anchors) == {"citation_precision": 0.0, "citation_recall": 0.0}


def test_wrong_page_does_not_hit_and_empty_gold_is_na():
    chunk = [{"page": 4, "text": "exact unique clause"}]
    assert retrieval_scores(chunk, [{"page": 5, "anchor": "exact unique clause"}])["retrieval_recall"] == 0
    assert retrieval_scores(chunk, []) == {"retrieval_precision": None, "retrieval_recall": None}


def test_aggregate_does_not_turn_blocked_cells_into_zero():
    got = aggregate([{"score": None}, {"score": ""}, {"score": 0.8}], ["score"])
    assert got == {"cases": 3, "score": 0.8, "score_n": 1}


def test_bill_field_accuracy_is_only_real_extraction_comparison():
    gold = {
        "hospital": {"value": "Example Clinic", "page": 1, "evidence": "Provider: Example Clinic"},
        "amount": {"value": 1250, "page": 1, "evidence": "Amount: INR 1250.00"},
        "network_type": {"value": None, "page": None, "evidence": None},
    }
    actual = {
        "hospital": {"value": "Example  Clinic", "page": 1, "evidence": "Provider: Example Clinic"},
        "amount": {"value": "1250", "page": 2, "evidence": "Amount: INR 1250.00"},
        "network_type": {"value": None, "page": None, "evidence": None},
    }
    assert bill_field_accuracy(actual, gold) == 2 / 3


def test_generated_bill_names_are_unique_and_present():
    root = Path(__file__).resolve().parents[1]
    names = [x["bill_file"] for x in load_claim_cases()]
    assert len(names) == len(set(names))
    assert all((root / "data" / "synthetic_bills" / name).is_file() for name in names)


def test_dotenv_is_loaded_before_cached_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=evaluation-test-key\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = load_evaluation_settings(env)
    assert settings.groq_api_key == "evaluation-test-key"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from config.settings import get_settings
    get_settings.cache_clear()


class _FakePipeline:
    def __init__(self, fail_retrieval=False):
        self.fail_retrieval = fail_retrieval
        self.retrieve_calls = 0

    def ingest(self, source, filename=None):
        return SimpleNamespace(policy_id=str(filename))

    def retrieve(self, policy_id, query, k=6):
        self.retrieve_calls += 1
        if self.fail_retrieval:
            raise RuntimeError("retrieval failed")
        return [{"id": "gold", "policy_id": policy_id, "text": query, "page": 1, "section": "", "source": policy_id}]


def _settings():
    return SimpleNamespace(groq_api_key="fake-key", groq_model="fake", embedding_model="fake")


def test_agent_uses_router_actual_context_and_writes_raw_records(tmp_path):
    pipeline = _FakePipeline()
    calls = []

    class FakeRouter:
        def run(self, route, **kwargs):
            calls.append((route, kwargs))
            context = {"id": "live", "policy_id": kwargs["policy_id"], "text": "ACTUAL ROUTER CONTEXT", "page": 99, "section": "live", "source": "fake.pdf"}
            return {"answer": "live answer", "sources": [context], "retrieved_contexts": [context]}

    def fake_ragas(question, answer, contexts, reference, llm, embeddings):
        assert contexts == ["ACTUAL ROUTER CONTEXT"]
        return {name: 0.5 for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}

    code = run("agent", tmp_path, pipeline_factory=lambda: pipeline, router_factory=lambda _: FakeRouter(), settings_loader=_settings, ragas_scorer=fake_ragas, ragas_runtime_factory=lambda _: (object(), object()))
    assert code == 0
    assert pipeline.retrieve_calls == 0
    assert calls and all("query" in kwargs for _, kwargs in calls)
    records = [json.loads(line) for line in (tmp_path / "raw_agent_records.jsonl").read_text().splitlines()]
    assert records[0].keys() >= {"question", "answer", "contexts", "reference"}
    assert records[0]["contexts"] == ["ACTUAL ROUTER CONTEXT"]
    assert (tmp_path / "results.csv").is_file()


def test_retrieval_only_never_builds_router_and_failure_is_nonzero(tmp_path):
    pipeline = _FakePipeline(fail_retrieval=True)
    def forbidden_router(_):
        raise AssertionError("retrieval-only instantiated Router")
    code = run("claim-retrieval", tmp_path, pipeline_factory=lambda: pipeline, router_factory=forbidden_router, settings_loader=lambda: SimpleNamespace(groq_api_key=None))
    assert code == 2
    assert "error" in (tmp_path / "results.csv").read_text()


def test_full_claim_scores_context_from_same_router_call(tmp_path):
    pipeline = _FakePipeline()
    cases = load_claim_cases()
    index = 0

    class FakeRouter:
        def run(self, route, **kwargs):
            nonlocal index
            case = cases[index]
            index += 1
            assert route == "Claim Check"
            assert kwargs["bill_source"].name == case["bill_file"]
            context = {"id": "claim-live", "policy_id": kwargs["policy_id"], "text": "EXTRACTION QUERY CONTEXT", "page": 77, "section": "live", "source": "fake.pdf"}
            return {"answer": "live claim cross-check", "sources": [], "retrieved_contexts": [context], "bill_fields": case["expected_bill_fields"], "findings": []}

    code = run("claim-full", tmp_path, pipeline_factory=lambda: pipeline, router_factory=lambda _: FakeRouter(), settings_loader=_settings)
    assert code == 0
    assert pipeline.retrieve_calls == 0
    records = [json.loads(line) for line in (tmp_path / "raw_claim_records.jsonl").read_text().splitlines()]
    assert records[0]["contexts"][0]["text"] == "EXTRACTION QUERY CONTEXT"


def test_nonfinite_ragas_metric_marks_run_incomplete(tmp_path):
    pipeline = _FakePipeline()
    class FakeRouter:
        def run(self, route, **kwargs):
            context = {"id": "live", "policy_id": kwargs["policy_id"], "text": "live", "page": 1, "section": "", "source": "fake"}
            return {"answer": "live", "sources": [], "retrieved_contexts": [context]}
    def bad_ragas(*args):
        return {"faithfulness": float("nan"), "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}
    code = run("agent", tmp_path, pipeline_factory=lambda: pipeline, router_factory=lambda _: FakeRouter(), settings_loader=_settings, ragas_scorer=bad_ragas, ragas_runtime_factory=lambda _: (object(), object()))
    assert code == 2
    assert "incomplete" in (tmp_path / "results.csv").read_text()


def test_small_free_tier_batches_do_not_claim_full_suite(tmp_path):
    import csv
    code = run("agent", tmp_path, case_ids=["qa-lite-room-rent"], settings_loader=lambda: SimpleNamespace(groq_api_key=None))
    assert code == 2
    with (tmp_path / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["id"] == "qa-lite-room-rent"
