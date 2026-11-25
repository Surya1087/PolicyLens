from __future__ import annotations

import json

import httpx
import pytest

from llm.mistral_client import MistralClient
from rag.models import PolicyLensError


class FakeResponse:
    def __init__(self, content=None, status_code=200, headers=None):
        self.payload = {
            "choices": [{"message": {"content": content}}],
        }
        self.status_code = status_code
        self.headers = headers or {}
        self.request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, headers=self.headers, request=self.request)
            raise httpx.HTTPStatusError("provider error", request=self.request, response=response)

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
    client = MistralClient(api_key="test-key", max_retries=1)
    http = FakeHTTPClient(outcomes)
    client._client = http
    return client, http


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_client_is_lazy_and_missing_key_is_clear():
    client = MistralClient(api_key="")
    assert client._client is None
    with pytest.raises(PolicyLensError, match="Mistral API key is missing"):
        client.complete_json("system", "user", SCHEMA)


def test_malformed_json_error_is_clear():
    client, _ = client_with(FakeResponse("not-json"))
    with pytest.raises(PolicyLensError, match="malformed JSON"):
        client.complete_json("system", "user", SCHEMA)


def test_schema_is_validated():
    client, _ = client_with(FakeResponse('{"wrong":"field"}'))
    with pytest.raises(PolicyLensError, match="schema validation"):
        client.complete_json("system", "user", SCHEMA)


def test_rate_limit_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("llm.mistral_client.time.sleep", lambda _: None)
    client, http = client_with(
        FakeResponse(status_code=429),
        FakeResponse('{"answer":"ok"}'),
    )
    assert client.complete_json("system", "user", SCHEMA) == {"answer": "ok"}
    assert len(http.calls) == 2


def test_exhausted_rate_limit_explains_wait():
    response = FakeResponse(status_code=429, headers={"Retry-After": "30"})
    client, _ = client_with(response, response)
    with pytest.raises(PolicyLensError, match="Wait at least 30 second"):
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


def test_json_schema_and_mistral_endpoint_are_used():
    client, http = client_with(FakeResponse('{"answer":"ok"}'))
    assert client.complete_json("system", "user", SCHEMA) == {"answer": "ok"}
    url, request = http.calls[0]
    assert url == "https://api.mistral.ai/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert json.dumps(SCHEMA) in request["json"]["messages"][0]["content"]
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["json"]["max_tokens"] == 3000


def test_auth_failure_is_not_retried_or_leaked():
    client, http = client_with(FakeResponse(status_code=401))
    with pytest.raises(PolicyLensError, match="authentication failed") as error:
        client.complete_json("system", "user", SCHEMA)
    assert "provider error" not in str(error.value)
    assert len(http.calls) == 1
