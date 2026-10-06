from __future__ import annotations

from datetime import date, datetime
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm import GroqClient
from prompts.prompt import BILL_SYSTEM, BILL_TASK
from rag.models import Page, PolicyLensError

MAX_BILL_PAGES = 40
MAX_BILL_CHARS = 120_000
MAX_PAGE_CHARS = 12_000


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    page: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=500)


class TextField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str | None
    evidence: Evidence | None


class DateField(TextField):
    pass


class AmountField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: float | None = Field(default=None, ge=0)
    evidence: Evidence | None


class CurrencyField(TextField):
    pass


class NetworkField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Literal["in_network", "out_of_network"] | None
    evidence: Evidence | None


class BillExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    procedure_or_diagnosis: TextField
    treatment_date: DateField
    amount: AmountField
    currency: CurrencyField
    hospital: TextField
    network_type: NetworkField


class BillExtractor:
    def __init__(self, llm=None):
        self.llm = llm if llm is not None else GroqClient()
        self.last_warnings: list[str] = []

    def extract(self, source: Any, filename: str | None = None) -> dict:
        pages, warnings = _load_pages(source, filename)
        self.last_warnings = list(warnings)
        if not pages or not any(page.text.strip() for page in pages):
            raise PolicyLensError("No readable text was extracted from the bill.")
        if len(pages) > MAX_BILL_PAGES:
            raise PolicyLensError(f"Bill exceeds the {MAX_BILL_PAGES}-page processing limit.")
        total_chars = sum(len(page.text) for page in pages)
        if total_chars > MAX_BILL_CHARS or any(len(page.text) > MAX_PAGE_CHARS for page in pages):
            raise PolicyLensError("Bill text exceeds the bounded extraction limit.")
        payload = "\n\n".join(f'<bill_page number="{p.page}">\n{p.text}\n</bill_page>' for p in pages)
        raw = self.llm.complete_json(BILL_SYSTEM, BILL_TASK + "\n" + payload, BillExtraction.model_json_schema())
        try:
            extraction = BillExtraction.model_validate(raw)
        except ValidationError as exc:
            raise PolicyLensError("Bill extraction did not match the required structure.") from exc
        page_map = {page.page: page.text for page in pages}
        result: dict[str, dict] = {}
        for name, field in extraction.model_dump().items():
            value, evidence = field["value"], field["evidence"]
            if value is None or not _valid_field(name, value, evidence, page_map):
                result[name] = {"value": None, "page": None, "evidence": None}
            else:
                result[name] = {"value": value, "page": evidence["page"], "evidence": evidence["quote"]}
        result["_warnings"] = list(warnings)
        return result


def _valid_field(name: str, value: Any, evidence: dict | None, pages: dict[int, str]) -> bool:
    if not evidence or evidence["page"] not in pages:
        return False
    quote = evidence["quote"]
    if _norm(quote) not in _norm(pages[evidence["page"]]):
        return False
    if name in {"procedure_or_diagnosis", "hospital"}:
        return bool(str(value).strip()) and _norm(str(value)) in _norm(quote)
    if name == "currency":
        currency = str(value).strip().upper()
        return bool(re.fullmatch(r"[A-Z]{3}", currency)) and currency.casefold() in quote.casefold()
    if name == "treatment_date":
        try:
            parsed = date.fromisoformat(str(value))
        except ValueError:
            return False
        candidates = _dates_in(quote)
        return parsed in candidates
    if name == "amount":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            return False
        numbers = [float(token.replace(",", "")) for token in re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?", quote)]
        return any(math.isclose(float(value), number, rel_tol=0, abs_tol=0.005) for number in numbers)
    if name == "network_type":
        normalized = _norm(quote).replace("-", " ")
        if value == "in_network" and re.search(r"\b(?:non network|out of network|not (?:a )?(?:network|in network))\b", normalized):
            return False
        signals = {
            "in_network": ("in network", "network hospital", "cashless network"),
            "out_of_network": ("out of network", "non network", "non-network"),
        }
        return any(signal in normalized for signal in signals[str(value)])
    return False


def _dates_in(text: str) -> set[date]:
    found: set[date] = set()
    for year, month, day in re.findall(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    for day, month, year in re.findall(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text):
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    for token in re.findall(r"\b(?:\d{1,2}[- ](?:[A-Za-z]{3,9})[- ]\d{4}|(?:[A-Za-z]{3,9}) \d{1,2},? \d{4})\b", text):
        for format_string in ("%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                found.add(datetime.strptime(token, format_string).date())
                break
            except ValueError:
                continue
    return found


def _load_pages(source: Any, filename: str | None) -> tuple[list[Page], list[str]]:
    """Delegate all parsing to rag.loader and retain its scan warnings."""
    from rag import loader

    if hasattr(loader, "load_pdf_details"):
        loaded = loader.load_pdf_details(source, filename=filename)
        return list(loaded.pages), list(loaded.warnings)
    if not hasattr(loader, "load_pdf"):
        raise PolicyLensError("rag.loader does not expose a supported PDF loading API.")
    pages = loader.load_pdf(source, filename=filename)
    return list(pages), []


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()
