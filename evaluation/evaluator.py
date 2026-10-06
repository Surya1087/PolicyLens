"""Run evaluation against real PolicyLens retrieval and agent outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from evaluation.metrics import aggregate, bill_field_accuracy, citation_scores, ragas_scores, retrieval_scores
from evaluation.test_cases import ROOT, load_agent_cases, load_claim_cases

FIELDS = [
    "retrieval_precision", "retrieval_recall", "citation_precision", "citation_recall",
    "bill_field_accuracy", "no_label_abstention", "faithfulness", "answer_relevancy", "context_precision", "context_recall",
]
RAGAS_FIELDS = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}


def chunk_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)
    return {key: getattr(chunk, key) for key in ("id", "policy_id", "text", "page", "section", "source")}


def load_evaluation_settings(env_file: Path | None = None) -> Any:
    """Load project dotenv before refreshing the cached application settings."""
    load_dotenv(env_file or ROOT / ".env", override=False)
    from config.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


def _write(rows: list[dict[str, Any]], out: Path, raw: list[dict[str, Any]] | None = None, raw_name: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    columns = ["id", "kind", "route", "status", "error", *FIELDS]
    for filename in ("results.csv", "details.csv"):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with (out / "aggregates.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate(rows, FIELDS), handle, indent=2)
    if raw is not None and raw_name:
        with (out / raw_name).open("w", encoding="utf-8") as handle:
            for record in raw:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _policy(pipeline: Any, filename: str, cache: dict[str, str]) -> str:
    if filename not in cache:
        document = pipeline.ingest(ROOT / "data" / "insurance" / filename, filename=filename)
        cache[filename] = document.policy_id
    return cache[filename]


def _blocked(cases: list[dict[str, Any]], out: Path, mode: str) -> int:
    rows = [{
        "id": case["id"], "kind": case["kind"], "route": case["route"],
        "status": "blocked", "error": "GROQ_API_KEY is missing", **{field: None for field in FIELDS},
    } for case in cases]
    raw_name = "raw_claim_records.jsonl" if mode == "claim-full" else "raw_agent_records.jsonl"
    _write(rows, out, [], raw_name)
    return 2


def _actual_contexts(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "retrieved_contexts" not in result:
        raise ValueError("Agent result omitted retrieved_contexts; actual context cannot be evaluated")
    contexts = [chunk_dict(item) for item in result["retrieved_contexts"]]
    if any("text" not in item or "page" not in item for item in contexts):
        raise ValueError("retrieved_contexts must contain chunk dictionaries")
    return contexts


def _default_pipeline_factory() -> Any:
    from rag.pipeline import RAGPipeline
    return RAGPipeline()


def _default_router_factory(pipeline: Any) -> Any:
    from app.agents.router import Router
    return Router(pipeline)


def _default_ragas_runtime(settings: Any) -> tuple[Any, Any]:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    return (
        ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, max_retries=1),
        HuggingFaceEmbeddings(model_name=settings.embedding_model),
    )


def run(
    mode: str,
    out: Path,
    retries: int = 1,
    *,
    case_ids: list[str] | None = None,
    pipeline_factory: Callable[[], Any] | None = None,
    router_factory: Callable[[Any], Any] | None = None,
    settings_loader: Callable[[], Any] = load_evaluation_settings,
    ragas_scorer: Callable[..., dict[str, float]] = ragas_scores,
    ragas_runtime_factory: Callable[[Any], tuple[Any, Any]] = _default_ragas_runtime,
) -> int:
    cases = load_agent_cases() if mode == "agent" else load_claim_cases()
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError("Unknown evaluation case ID(s): " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in case_ids]
    settings = settings_loader()
    if mode in {"agent", "claim-full"} and not settings.groq_api_key:
        return _blocked(cases, out, mode)

    pipeline = (pipeline_factory or _default_pipeline_factory)()
    # Retrieval-only must never instantiate Router or an LLM.
    router = None if mode == "claim-retrieval" else (router_factory or _default_router_factory)(pipeline)
    cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    ragas_runtime = None
    incomplete = False

    for case in cases:  # serial by design to reduce hosted-model rate pressure
        row = {"id": case["id"], "kind": case["kind"], "route": case["route"], "status": "", "error": ""}
        try:
            policy_id = _policy(pipeline, case["policy_file"], cache)
            if mode == "claim-retrieval":
                # This gold query is an isolated retrieval test, never end-to-end extraction.
                from rag.rag_pipeline import claim_retrieval_query
                gold_query = claim_retrieval_query(case["expected_bill_fields"])
                retrieved = [chunk_dict(item) for item in pipeline.retrieve(policy_id, gold_query, k=6)]
                row.update(retrieval_scores(retrieved, case["anchors"]))
                row.update({"citation_precision": None, "citation_recall": None, "status": "success_retrieval_only"})
                raw.append({"id": case["id"], "mode": "gold_extraction_retrieval_only", "query": gold_query, "contexts": retrieved})
                rows.append(row)
                continue

            kwargs: dict[str, Any] = {"policy_id": policy_id}
            if mode == "agent":
                kwargs["query"] = case["query"]
            else:
                kwargs.update({
                    "bill_source": ROOT / "data" / "synthetic_bills" / case["bill_file"],
                    "policy_start_date": case.get("policy_start_date"),
                })
            result = None
            for attempt in range(retries + 1):
                try:
                    result = router.run(case["route"], **kwargs)
                    break
                except Exception:
                    if attempt == retries:
                        raise
                    time.sleep(2 ** attempt)
            assert result is not None
            contexts = _actual_contexts(result)
            citations = [chunk_dict(item) for item in result.get("sources", [])]
            row.update(retrieval_scores(contexts, case["anchors"]))
            row.update(citation_scores(citations, case["anchors"]))
            row["status"] = "success"

            if mode == "claim-full":
                # Evaluate the clause actually quoted, not unrelated text elsewhere in its chunk.
                cited_quotes = [{"page": f["page"], "text": f["citation"]["quote"]}
                                for f in result.get("findings", []) if "citation" in f]
                row.update(citation_scores(cited_quotes, case["anchors"]))
                if not case["anchors"]:
                    row["no_label_abstention"] = float(not result.get("findings"))
                row["bill_field_accuracy"] = bill_field_accuracy(result.get("bill_fields", {}), case["expected_bill_fields"])
                raw.append({"id": case["id"], "bill_file": case["bill_file"], "contexts": contexts, "actual": result})
            else:
                # Preserve generation evidence even if the judge later fails/rate-limits.
                context_texts = [item["text"] for item in contexts]
                raw.append({
                    "id": case["id"], "question": case["query"], "answer": result.get("answer", ""),
                    "contexts": context_texts, "reference": case["reference"],
                    "retrieved_contexts": contexts, "actual": result,
                })
                if ragas_runtime is None:
                    ragas_runtime = ragas_runtime_factory(settings)
                scores = ragas_scorer(case["query"], str(result.get("answer", "")), context_texts, case["reference"], *ragas_runtime)
                valid_scores = {key: value for key, value in scores.items() if key in RAGAS_FIELDS and isinstance(value, (int, float)) and math.isfinite(float(value))}
                row.update(valid_scores)
                missing = RAGAS_FIELDS - valid_scores.keys()
                if missing:
                    row["status"] = "incomplete"
                    row["error"] = "RAGAS missing/non-finite: " + ", ".join(sorted(missing))
                    incomplete = True
                if result.get("scan", {}).get("partial"):
                    row["status"] = "incomplete"
                    row["error"] = "The source agent reported a partial ambiguity review. " + row["error"]
                    incomplete = True
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            incomplete = True
        rows.append(row)

    raw_name = "raw_claim_records.jsonl" if mode == "claim-full" else "raw_agent_records.jsonl" if mode == "agent" else "raw_retrieval_records.jsonl"
    _write(rows, out, raw if raw_name else None, raw_name)
    return 2 if incomplete else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "claim-retrieval", "claim-full"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--case-id", action="append", help="Run only this case; repeat to select a small free-tier batch")
    args = parser.parse_args()
    return run(args.mode, args.output, max(0, min(args.retries, 2)), case_ids=args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
