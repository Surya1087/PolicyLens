"""Load and validate hand-curated evaluation labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "evaluation" / "cases.json"
CLAIMS_PATH = ROOT / "data" / "evaluation" / "claim_labels.json"


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    ids = [str(x.get("id", "")) for x in value]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} has missing/duplicate IDs")
    for case in value:
        for anchor in case.get("anchors", []):
            if not isinstance(anchor.get("page"), int) or not anchor.get("anchor"):
                raise ValueError(f"invalid anchor in {case['id']}")
            if any(not isinstance(page, int) or page < 1 for page in anchor.get("equivalent_pages", [])):
                raise ValueError(f"invalid equivalent page in {case['id']}")
    return value


def load_agent_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return _load(path)


def load_claim_cases(path: Path = CLAIMS_PATH) -> list[dict[str, Any]]:
    return _load(path)
