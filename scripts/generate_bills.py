#!/usr/bin/env python3
"""Deterministically generate 12 fictional bills used only for evaluation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "evaluation" / "claim_labels.json"
OUT = ROOT / "data" / "synthetic_bills"


def main() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas
    cases = json.loads(LABELS.read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True)
    for case in cases:
        b = case["bill_document"]; path = OUT / case["bill_file"]
        canvas = Canvas(str(path), pagesize=A4, invariant=1, pageCompression=0)
        canvas.setTitle("SYNTHETIC EVALUATION BILL — NOT A REAL CLAIM")
        lines = ["SYNTHETIC EVALUATION BILL - NOT A REAL CLAIM", f"Invoice: {b['invoice_id']}",
                 f"Provider: {b['provider']}", f"Service date: {b['service_date']}",
                 f"Description: {b['description']}", f"Amount: {b['currency']} {b['amount']:.2f}",
                 "No patient identity or real transaction is represented."]
        y = 790
        for line in lines: canvas.drawString(54, y, line); y -= 28
        canvas.save()
    print(f"generated {len(cases)} deterministic PDFs in {OUT}")

if __name__ == "__main__": main()
