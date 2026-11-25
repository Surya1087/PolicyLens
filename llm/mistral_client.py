"""Mistral API client with bounded retries and strict JSON validation."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from config.settings import get_settings
from prompts.prompt import JSON_SCHEMA_INSTRUCTION
from rag.models import PolicyLensError
from .schema import schema_errors


class MistralClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.mistral_api_key
        self.model = model or settings.mistral_model
        self.timeout = max(1.0, min(float(timeout), 120.0))
        self.max_retries = max(0, min(int(max_retries), 4))
        self._client: Any = None

    def _get_client(self) -> httpx.Client:
        if not self.api_key:
            raise PolicyLensError("Mistral API key is missing. Set MISTRAL_API_KEY to use AI analysis.")
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise PolicyLensError("A valid object JSON schema is required.")
        client = self._get_client()
        request = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + JSON_SCHEMA_INSTRUCTION + json.dumps(schema)},
                {"role": "user", "content": user},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    value = json.loads(content or "")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise PolicyLensError("Mistral returned malformed JSON.") from exc
                errors = schema_errors(value, schema, root=schema)
                if errors:
                    raise PolicyLensError("Mistral response failed schema validation: " + "; ".join(errors[:3]))
                return value
            except PolicyLensError:
                raise
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                retry_after = exc.response.headers.get("Retry-After")
                retryable = status == 429 or status in {408, 409, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                if status == 429:
                    suffix = f" Wait at least {retry_after} second(s) before retrying." if retry_after and retry_after.isdigit() else " Check your Mistral console for quota and rate-limit details, then retry later."
                    raise PolicyLensError("Mistral rate limit reached." + suffix) from exc
                if status in {401, 403}:
                    raise PolicyLensError("Mistral authentication failed. Check MISTRAL_API_KEY and account access.") from exc
                if status == 408:
                    raise PolicyLensError("Mistral request timed out.") from exc
                if status in {400, 404, 413}:
                    raise PolicyLensError("Mistral rejected the request. Check the configured model and request-size limits.") from exc
                raise PolicyLensError("Mistral request failed without a usable response.") from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise PolicyLensError("Mistral request timed out.") from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise PolicyLensError("Cannot connect to Mistral. Check internet access and TLS/proxy configuration.") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise PolicyLensError("Mistral returned an invalid response structure.") from exc
        raise PolicyLensError("Mistral request failed.") from last_error
