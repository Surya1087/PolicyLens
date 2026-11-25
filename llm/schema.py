"""Validation for the JSON Schema subset emitted by our Pydantic response models."""

from __future__ import annotations

import re
from typing import Any


def schema_errors(value: Any, schema: dict, path: str = "$", root: dict | None = None) -> list[str]:
    root = root or schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return [f"{path} has an unsupported schema reference"]
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, dict) else None
        return schema_errors(value, target or {}, path, root)
    if "anyOf" in schema:
        if any(not schema_errors(value, option, path, root) for option in schema["anyOf"]):
            return []
        return [f"{path} does not match any allowed type"]
    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](value):
        return [f"{path} must be {expected}"]
    errors: list[str] = []
    if isinstance(value, dict):
        errors.extend(f"{path}.{key} is required" for key in schema.get("required", []) if key not in value)
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key} is not allowed" for key in value if key not in props)
        for key, child in props.items():
            if key in value:
                errors.extend(schema_errors(value[key], child, f"{path}.{key}", root))
    elif isinstance(value, list) and "items" in schema:
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} has too many items")
        for index, item in enumerate(value):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]", root))
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path} has an unsupported format")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} has an unsupported value")
    return errors
