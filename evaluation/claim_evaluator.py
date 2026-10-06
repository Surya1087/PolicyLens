"""Dedicated CLI for the synthetic claim suite."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluator import ROOT, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval-only", "full"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results" / "claims")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--case-id", action="append", help="Run only this case; repeat for a small batch")
    args = parser.parse_args()
    return run("claim-retrieval" if args.mode == "retrieval-only" else "claim-full", args.output, max(0, min(args.retries, 2)), case_ids=args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
