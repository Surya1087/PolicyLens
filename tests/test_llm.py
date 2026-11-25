from __future__ import annotations

import httpx
import pytest

from llm.ollama_client import OllamaClient
from rag.models import PolicyLensError


class FakeResponse:
    def __init__(self, content=None, status_code=200):
        self.payload = {"message": {"content": content}}
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("local model error", request=self.request, response=response)

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_with(*outcomes):
    client = OllamaClient(model="qwen2.5:3b")
    http = FakeHTTPClient(outcomes)
    client._client = http
    return client, http


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_json_response_and_schema_are_validated():
    client, http = client_with(FakeResponse('{"answer":"ok"}'))
    assert client.complete_json("system", "user", SCHEMA) == {"answer": "ok"}
    url, request = http.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert request["json"]["model"] == "qwen2.5:3b"
    assert request["json"]["stream"] is False
    assert request["json"]["format"] == "json"
    assert request["json"]["options"]["num_predict"] == 768
    assert '"answer"' in request["json"]["messages"][0]["content"]


def test_malformed_json_is_rejected():
    client, _ = client_with(FakeResponse("not-json"))
    with pytest.raises(PolicyLensError, match="malformed JSON"):
        client.complete_json("system", "user", SCHEMA)


def test_invalid_response_schema_is_rejected():
    client, _ = client_with(FakeResponse('{"wrong":"field"}'))
    with pytest.raises(PolicyLensError, match="schema validation"):
        client.complete_json("system", "user", SCHEMA)


def test_missing_local_model_has_actionable_error():
    client, _ = client_with(FakeResponse(status_code=404))
    with pytest.raises(PolicyLensError, match=r"ollama pull qwen2.5:3b"):
        client.complete_json("system", "user", SCHEMA)


def test_ollama_not_running_has_actionable_error():
    client, _ = client_with(httpx.ConnectError("refused"))
    with pytest.raises(PolicyLensError, match="Install/start Ollama"):
        client.complete_json("system", "user", SCHEMA)


def test_nested_schema_refs_are_validated():
    schema = {
        "type": "object",
        "$defs": {"Item": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
    }
    client, _ = client_with(FakeResponse('{"item":{}}'))
    with pytest.raises(PolicyLensError, match="schema validation"):
        client.complete_json("system", "user", schema)
