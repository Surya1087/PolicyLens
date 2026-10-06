from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.groq_client import GroqClient
from rag.models import PolicyLensError


class Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.last_request = kwargs
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def client_with(*outcomes):
    instance = GroqClient(api_key="test", max_retries=1)
    completions = Completions(outcomes)
    instance._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return instance, completions


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


def test_client_is_lazy_and_missing_key_is_clear():
    client = GroqClient(api_key="")
    assert client._client is None
    with pytest.raises(PolicyLensError, match="API key is missing"):
        client.complete_json("system", "user", SCHEMA)


def test_malformed_json_error_is_clear():
    client, _ = client_with("not-json")
    with pytest.raises(PolicyLensError, match="malformed JSON"):
        client.complete_json("system", "user", SCHEMA)


def test_schema_is_validated():
    client, _ = client_with('{"wrong":"field"}')
    with pytest.raises(PolicyLensError, match="schema validation"):
        client.complete_json("system", "user", SCHEMA)


def test_rate_limit_retries_are_bounded(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

    monkeypatch.setattr("llm.groq_client.time.sleep", lambda _: None)
    client, completions = client_with(RateLimitError(), '{"answer":"ok"}')
    assert client.complete_json("system", "user", SCHEMA) == {"answer": "ok"}
    assert completions.calls == 2


def test_nested_schema_refs_are_validated():
    schema = {
        "type": "object", "$defs": {"Item": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
        "properties": {"item": {"$ref": "#/$defs/Item"}}, "required": ["item"],
    }
    client, _ = client_with('{"item":{}}')
    with pytest.raises(PolicyLensError, match="schema validation"):
        client.complete_json("system", "user", schema)


def test_json_schema_is_sent_to_model_not_only_validated_locally():
    import json
    client, transport = client_with('{"answer":"ok"}')
    client.complete_json("system", "user", SCHEMA)
    assert json.dumps(SCHEMA) in transport.last_request["messages"][0]["content"]
    assert transport.last_request["max_tokens"] == 3000


def test_auth_failure_is_not_retried_or_leaked():
    class AuthenticationError(Exception):
        status_code = 401
    client, transport = client_with(AuthenticationError("private provider payload"))
    with pytest.raises(PolicyLensError, match="authentication failed") as error:
        client.complete_json("system", "user", SCHEMA)
    assert "private provider payload" not in str(error.value)
    assert transport.calls == 1
