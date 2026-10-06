from __future__ import annotations

import hashlib
from io import BytesIO
from dataclasses import asdict

import pytest

from config.settings import Settings
from rag.embeddings import SentenceTransformerEmbeddings
from rag.loader import load_pdf_details
from rag.models import Chunk, Page, PolicyDocument, PolicyLensError
from rag.rag_pipeline import RAGPipeline, expand_clinical_query
from rag.splitter import split_pages
from rag.vector_store import ChromaVectorStore


class FakeModel:
    def encode(self, texts, **kwargs):
        return [[float(len(text)), float("dialysis" in text.lower())] for text in texts]


class MemoryStore:
    def __init__(self):
        self.policies = {}
        self.last_query = ""

    def index(self, policy):
        self.policies[policy.policy_id] = policy

    def retrieve(self, policy_id, query, k):
        self.last_query = query
        if policy_id not in self.policies:
            raise PolicyLensError("unknown")
        return self.policies[policy_id].chunks[:k]

    def get_policy(self, policy_id):
        return self.policies[policy_id]


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), float("dialysis" in text.lower()), 1.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def make_pdf(*page_texts: str) -> bytes:
    from reportlab.pdfgen.canvas import Canvas

    output = BytesIO()
    canvas = Canvas(output)
    for text in page_texts:
        if text:
            canvas.drawString(72, 720, text)
        canvas.showPage()
    canvas.save()
    return output.getvalue()


def test_models_are_asdict_serializable():
    policy = PolicyDocument(
        "a" * 64,
        "policy.pdf",
        [Page(1, "text")],
        [Chunk("id", "a" * 64, "text", 1, "Benefits", "policy.pdf")],
        ["warning"],
    )
    assert asdict(policy)["chunks"][0]["page"] == 1


def test_embeddings_are_lazy_and_injectable():
    calls = []
    embeddings = SentenceTransformerEmbeddings(model_factory=lambda name: calls.append(name) or FakeModel())
    assert calls == []
    assert embeddings.embed_query("renal dialysis") == [14.0, 1.0]
    assert len(calls) == 1


def test_query_expansion_keeps_original_and_adds_synonyms():
    query = "Does phacoemulsification need pre-authorisation?"
    expanded = expand_clinical_query(query)
    assert expanded.startswith(query)
    assert "cataract surgery" in expanded
    dialysis = expand_clinical_query("renal dialysis limits")
    assert dialysis.startswith("renal dialysis limits")
    assert "hemodialysis" in dialysis


def test_loader_preserves_physical_pages_and_warns_for_mixed_scan():
    loaded = load_pdf_details(make_pdf("Policy title", "", "Dialysis benefit"), "mixed.pdf")
    assert [page.page for page in loaded.pages] == [1, 2, 3]
    assert loaded.pages[1].text == ""
    assert "physical page(s) 2" in loaded.warnings[0]
    assert loaded.filename == "mixed.pdf"


def test_loader_rejects_scanned_or_oversized_pdf():
    with pytest.raises(PolicyLensError, match="no extractable text"):
        load_pdf_details(make_pdf(""))
    with pytest.raises(PolicyLensError, match="size limit"):
        load_pdf_details(b"x" * 20, max_bytes=10)


def test_loader_rejects_password_protection_and_malformed_input():
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(BytesIO(make_pdf("Sensitive synthetic policy")))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("test-only-password")
    protected = BytesIO()
    writer.write(protected)
    with pytest.raises(PolicyLensError, match="Encrypted PDF"):
        load_pdf_details(protected.getvalue())
    with pytest.raises(PolicyLensError, match="malformed|parsed"):
        load_pdf_details(b"this is not a PDF")


def test_splitter_is_page_local_and_stable():
    pages = [Page(1, "Benefits\n" + "alpha " * 60), Page(2, "Exclusions\n" + "beta " * 60)]
    first = split_pages(pages, "policy-1", "source.pdf", chunk_size=120, chunk_overlap=20)
    second = split_pages(pages, "policy-1", "source.pdf", chunk_size=120, chunk_overlap=20)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert {chunk.page for chunk in first} == {1, 2}
    assert all("alpha" not in chunk.text for chunk in first if chunk.page == 2)
    assert all(chunk.policy_id == "policy-1" and chunk.source == "source.pdf" for chunk in first)


def test_pipeline_uses_policy_filter_and_needs_no_llm(tmp_path):
    store = MemoryStore()
    settings = Settings(data_dir=tmp_path, chunk_size=1000, chunk_overlap=180)
    pipeline = RAGPipeline(settings=settings, embeddings=SentenceTransformerEmbeddings(model=FakeModel()), vector_store=store)
    policy_id = hashlib.sha256(b"pdf").hexdigest()
    policy = PolicyDocument(
        policy_id,
        "one.pdf",
        [Page(1, "renal dialysis")],
        [Chunk("id", policy_id, "renal dialysis", 1, "", "one.pdf")],
        [],
    )
    store.index(policy)
    result = pipeline.retrieve(policy_id, "renal dialysis", k=1)
    assert result[0].policy_id == policy_id
    assert "kidney dialysis" in store.last_query
    assert pipeline.get_policy(policy_id) is policy


def test_invalid_settings_reject_overlap(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        Settings(data_dir=tmp_path, chunk_size=10, chunk_overlap=10)


def test_chroma_persists_metadata_and_never_mingles_policies(tmp_path):
    embeddings = FakeEmbeddings()
    store = ChromaVectorStore(tmp_path / "vectors", embeddings)
    one = PolicyDocument(
        "policy-one",
        "one.pdf",
        [Page(1, "dialysis benefit")],
        [Chunk("one-1", "policy-one", "dialysis benefit", 1, "Benefits", "one.pdf")],
        [],
    )
    two = PolicyDocument(
        "policy-two",
        "two.pdf",
        [Page(1, "dialysis excluded")],
        [Chunk("two-1", "policy-two", "dialysis excluded", 1, "Exclusions", "two.pdf")],
        [],
    )
    store.index(one)
    store.index(two)
    store.index(one)  # idempotent repeat
    assert {chunk.policy_id for chunk in store.retrieve("policy-one", "dialysis", 6)} == {"policy-one"}

    reopened = ChromaVectorStore(tmp_path / "vectors", embeddings)
    assert reopened.get_policy("policy-one") == one
