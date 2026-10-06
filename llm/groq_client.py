"""Small, lazy and defensive Groq JSON client."""

from __future__ import annotations

import json
import time
from typing import Any

from config.settings import get_settings
from rag.models import PolicyLensError
from prompts.prompt import JSON_SCHEMA_INSTRUCTION


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, *, timeout: float = 30.0, max_retries: int = 2):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.groq_model
        self.timeout = max(1.0, min(float(timeout), 120.0))
        self.max_retries = max(0, min(int(max_retries), 4))
        self._client: Any = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise PolicyLensError("Groq API key is missing. Set GROQ_API_KEY to use AI analysis.")
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise PolicyLensError("The groq package is not installed.") from exc
            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise PolicyLensError("A valid object JSON schema is required.")
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=3000,
                    **({"extra_body": {"reasoning_effort": "low"}} if self.model.startswith("openai/gpt-oss-") else {}),
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system + JSON_SCHEMA_INSTRUCTION + json.dumps(schema)},
                        {"role": "user", "content": user},
                    ],
                )
                content = response.choices[0].message.content
                try:
                    value = json.loads(content or "")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise PolicyLensError("Groq returned malformed JSON.") from exc
                errors = _schema_errors(value, schema, root=schema)
                if errors:
                    raise PolicyLensError("Groq response failed schema validation: " + "; ".join(errors[:3]))
                return value
            except PolicyLensError:
                raise
            except Exception as exc:  # SDK exception classes vary by release.
                last_error = exc
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", {}) or {}
                details = body.get("error", body) if isinstance(body, dict) else {}
                code = details.get("code") if isinstance(details, dict) else None
                if code == "json_validate_failed":
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue
                    raise PolicyLensError("Groq could not produce valid JSON for this analysis. No unsupported result is shown. Try a shorter question or retry the analysis.") from exc
                name = type(exc).__name__.lower()
                is_rate_limit = status == 429 or "ratelimit" in name or "rate_limit" in name
                is_retryable = is_rate_limit or status in {408, 409, 500, 502, 503, 504} or "timeout" in name
                if not is_retryable or attempt >= self.max_retries:
                    if is_rate_limit:
                        raise PolicyLensError("Groq rate limit reached; retry later.") from exc
                    if "timeout" in name or status == 408:
                        raise PolicyLensError("Groq request timed out.") from exc
                    if status in {401, 403}:
                        raise PolicyLensError("Groq authentication failed. Check your local GROQ_API_KEY and account access.") from exc
                    if status in {400, 404, 413}:
                        raise PolicyLensError("Groq rejected the request. Check the configured model and request-size limits; try a smaller document.") from exc
                    if "connection" in name:
                        raise PolicyLensError("Cannot connect to Groq. Check internet access and TLS/proxy configuration.") from exc
                    raise PolicyLensError("Groq request failed without a usable response.") from exc
                time.sleep(min(0.25 * (2**attempt), 1.0))
        raise PolicyLensError("Groq request failed.") from last_error


def _schema_errors(value: Any, schema: dict, path: str = "$", root: dict | None = None) -> list[str]:
    """Validate the JSON-schema subset generated by our Pydantic models."""
    root = root or schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return [f"{path} has an unsupported schema reference"]
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, dict) else None
        return _schema_errors(value, target or {}, path, root)
    if "anyOf" in schema:
        if any(not _schema_errors(value, option, path, root) for option in schema["anyOf"]):
            return []
        return [f"{path} does not match any allowed type"]
    expected = schema.get("type")
    checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if expected in checks and not checks[expected](value):
        return [f"{path} must be {expected}"]
    errors: list[str] = []
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key} is not allowed" for key in value if key not in props)
        for key, child in props.items():
            if key in value:
                errors.extend(_schema_errors(value[key], child, f"{path}.{key}", root))
    elif isinstance(value, list) and "items" in schema:
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} has too many items")
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]", root))
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is too long")
        if "pattern" in schema:
            import re
            if re.search(schema["pattern"], value) is None:
                errors.append(f"{path} has an unsupported format")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} has an unsupported value")
    return errors
