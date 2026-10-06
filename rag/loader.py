"""Bounded PDF loading with physical page-number provenance."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO

from config.constants import DEFAULT_MAX_PDF_BYTES

from .models import Page, PolicyLensError


@dataclass(frozen=True, slots=True)
class LoadedPDF:
    pages: list[Page]
    warnings: list[str]
    content: bytes
    filename: str


def _read_bounded(source: bytes | bytearray | PathLike[str] | str | BinaryIO, limit: int) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = "uploaded.pdf"
    elif isinstance(source, (str, PathLike)):
        path = Path(source).expanduser()
        try:
            if path.stat().st_size > limit:
                raise PolicyLensError(f"PDF exceeds the {limit // (1024 * 1024)} MB size limit")
            data = path.read_bytes()
        except PolicyLensError:
            raise
        except OSError as exc:
            raise PolicyLensError(f"Unable to read PDF: {exc}") from exc
        name = path.name
    elif hasattr(source, "read"):
        stream = source
        original_position: int | None = None
        try:
            if hasattr(stream, "tell"):
                original_position = stream.tell()
            if hasattr(stream, "seek"):
                stream.seek(0)
            data = stream.read(limit + 1)
            if not isinstance(data, (bytes, bytearray)):
                raise PolicyLensError("PDF file object must be opened in binary mode")
            data = bytes(data)
        except PolicyLensError:
            raise
        except (OSError, ValueError) as exc:
            raise PolicyLensError(f"Unable to read PDF stream: {exc}") from exc
        finally:
            if original_position is not None and hasattr(stream, "seek"):
                try:
                    stream.seek(original_position)
                except (OSError, ValueError):
                    pass
        name = Path(str(getattr(source, "name", "uploaded.pdf"))).name
    else:
        raise PolicyLensError("PDF source must be bytes, a path, or a binary file object")

    if len(data) > limit:
        raise PolicyLensError(f"PDF exceeds the {limit // (1024 * 1024)} MB size limit")
    if not data:
        raise PolicyLensError("PDF is empty")
    return data, name


def load_pdf_details(
    source: bytes | bytearray | PathLike[str] | str | BinaryIO,
    filename: str | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
) -> LoadedPDF:
    """Load a PDF and return pages, warnings, bytes, and its display filename."""

    data, inferred_name = _read_bounded(source, max_bytes)
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - environment configuration
        raise PolicyLensError("PDF support is unavailable; install the pinned pypdf dependency") from exc

    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise PolicyLensError("Encrypted PDF requires a password and cannot be processed") from exc
            if not unlocked:
                raise PolicyLensError("Encrypted PDF requires a password and cannot be processed")
        if not reader.pages:
            raise PolicyLensError("PDF has no pages")
        if len(reader.pages) > 500:
            raise PolicyLensError("PDF exceeds the 500-page processing limit")

        pages: list[Page] = []
        empty_pages: list[int] = []
        for physical_page, pdf_page in enumerate(reader.pages, start=1):
            try:
                text = pdf_page.extract_text() or ""
            except Exception as exc:
                raise PolicyLensError(f"Could not extract text from PDF page {physical_page}") from exc
            text = text.replace("\x00", "").strip()
            if len(text) > 100_000 or sum(len(page.text) for page in pages) + len(text) > 2_000_000:
                raise PolicyLensError("PDF exceeds the extracted-text processing limit")
            if not text:
                empty_pages.append(physical_page)
            pages.append(Page(page=physical_page, text=text))
    except PolicyLensError:
        raise
    except (PdfReadError, EOFError, ValueError, TypeError) as exc:
        raise PolicyLensError("PDF is malformed or unsupported") from exc
    except Exception as exc:
        raise PolicyLensError("PDF could not be parsed") from exc

    if len(empty_pages) == len(pages):
        raise PolicyLensError("PDF contains no extractable text; it may be scanned and require OCR")
    warnings: list[str] = []
    if empty_pages:
        numbers = ", ".join(str(page) for page in empty_pages)
        warnings.append(f"No extractable text on physical page(s) {numbers}; scanned content may be omitted.")

    return LoadedPDF(
        pages=pages,
        warnings=warnings,
        content=data,
        filename=Path(filename).name if filename else inferred_name,
    )


def load_pdf(
    source: bytes | bytearray | PathLike[str] | str | BinaryIO,
    filename: str | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
) -> list[Page]:
    """Return extracted pages. Use ``load_pdf_details`` when warnings are needed."""

    return load_pdf_details(source, filename, max_bytes=max_bytes).pages
