"""Local Ollama chat client with strict JSON response validation."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

import httpx

from config.settings import get_settings
from prompts.prompt import JSON_SCHEMA_INSTRUCTION
from rag.models import PolicyLensError
from .schema import schema_errors


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 300.0,
    ):
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/") + "/"
        self.timeout = max(1.0, min(float(timeout), 600.0))
        self._client: Any = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise PolicyLensError("A valid object JSON schema is required.")
        request = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system + JSON_SCHEMA_INSTRUCTION + json.dumps(schema)},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._get_client().post(
                urljoin(self.base_url, "api/chat"),
                json=request,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            try:
                value = json.loads(content or "")
            except (TypeError, json.JSONDecodeError) as exc:
                raise PolicyLensError("Ollama returned malformed JSON.") from exc
            errors = schema_errors(value, schema, root=schema)
            if errors:
                raise PolicyLensError("Ollama response failed schema validation: " + "; ".join(errors[:3]))
            return value
        except PolicyLensError:
            raise
        except httpx.ConnectError as exc:
            raise PolicyLensError(
                "Cannot connect to local Ollama. Install/start Ollama, then run `ollama pull "
                f"{self.model}`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise PolicyLensError("Local Ollama timed out. Try again or use a smaller model.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise PolicyLensError(
                    f"Ollama model '{self.model}' is not installed. Run `ollama pull {self.model}`."
                ) from exc
            raise PolicyLensError(
                f"Local Ollama returned HTTP {exc.response.status_code}. Check that the Ollama service is running."
            ) from exc
        except httpx.RequestError as exc:
            raise PolicyLensError("Could not call local Ollama. Check the configured local Ollama address.") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PolicyLensError("Ollama returned an invalid response structure.") from exc
