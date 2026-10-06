from pathlib import Path

from streamlit.testing.v1 import AppTest

from config.settings import get_settings
from rag.models import Chunk, Page, PolicyDocument

ROOT = Path(__file__).resolve().parents[1]


def test_initial_screen_needs_no_key_or_model(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    app = AppTest.from_file(str(ROOT / "app/ui/streamlit_app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "PolicyLens"
    assert any("Start by uploading" in box.value for box in app.info)


def test_policy_tabs_and_safe_rendering_without_network(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    app = AppTest.from_file(str(ROOT / "app/ui/streamlit_app.py"))
    chunk = Chunk("test:p1", "test", "Waiting period: 24 months.", 1, "Waiting period", "example.pdf")
    app.session_state["policy"] = PolicyDocument("test", "example.pdf", [Page(1, chunk.text)], [chunk], [])
    app.session_state["results"] = {"summary": {"answer": "Example explanation", "sources": [vars_safe(chunk)]}}
    app.run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 6
    assert any("Waiting period: 24 months." in node.value for node in app.text)
    assert next(button for button in app.button if button.label == "Check Claim").disabled


def vars_safe(chunk):
    from dataclasses import asdict
    return asdict(chunk)
