# Architecture and design decisions

## Separation of concerns

Streamlit owns consent and presentation. The router maps explicit task names to
agent classes rather than asking a model to choose tools. All five agents receive
the same `RAGPipeline` dependency. Agents are independently testable through an
injected LLM with the `complete_json(system, user, schema)` interface.

The PDF loader bounds bytes/pages/text, rejects password protection and textless
documents, and preserves physical page numbers. Chunks never cross a page.
Document IDs are SHA-256 hashes of PDF bytes; chunk IDs combine document hash,
physical page, chunk ordinal and text digest. Filenames are display metadata,
not retrieval keys. Every vector query filters on policy ID, and returned
metadata is checked again before reaching an agent.

Chroma stores normalized local embeddings and text. A per-policy JSON manifest
persists extracted pages, chunks and warnings; replacement is atomic within a
pipeline instance. UI sessions use separate local directories. The application
is intentionally single-user/local, not a multi-tenant database service.

## Grounding

Generated factual items must carry retrieved chunk IDs and exact quotes.
Whitespace/case-normalized quote matching rejects nonexistent passages. The
application never manufactures a printed section number. `sources` contains
cited chunks; `retrieved_contexts` separately records what was actually supplied
to the model, including uncited passages. Evaluation must use the latter to
avoid selection bias in context metrics.

This validation prevents nonexistent citations. It does **not** prove every
free-form explanation logically follows from its quote. That limitation is
shown in the README and addressed experimentally through RAGAS, not hidden.

## Claim safety

1. The bill extractor returns six typed fields with page/quote evidence. Values
   must match that evidence under field-specific rules. It never infers a
   patient's network status or missing policy start date.
2. The shared claim query builder uses the clinical description rather than
   names, amounts or invoice dates. The same embedding store is queried.
   Clinical synonym groups append terms without replacing the original query.
3. The claim model returns only topic enums and citations. There is no free-form
   verdict, recommendation, payout amount or claim-status field in its schema.
4. Application code validates selected quotations and uses fixed factual
   sentence templates. Bill fields appear separately as attributed extractions.
   User-supplied dates produce deterministic elapsed-day facts, never a
   waiting-period eligibility decision.
5. The system output status is always `not_assessed`; applicability is referred
   to the insurer. Exact policy wording remains clearly attributed source text.

Prompts also explicitly prohibit decisions and treat document content as
untrusted data. Restricting the output language is stronger than relying only
on a keyword blacklist or disclaimer.

## Ambiguity analysis

Baseline risk review retrieves multiple limitation facets. Independently,
curated regexes scan all extracted chunks for discretionary wording. Matching
contexts are deduplicated and confirmed in batches of at most six chunks.
Missing, invalid and failed assessments are surfaced as a partial review, not
silently treated as cleared risks. A phrase match alone is not a risk verdict.

## Evaluation integrity

The reference implementation's `evaluation/test_cases.py` contained handwritten
system answers and contexts. PolicyLens retains only independent questions and
reference labels in fixtures. The evaluator invokes the real router and records
actual answers and same-call contexts before judging. Missing hosted access
produces blocked records with blank metrics and a nonzero exit code.

Retrieval-only claim testing supplies labelled bill fields to isolate retrieval.
Full claim testing uses PDF extraction and scores the exact quoted clauses, not
merely a retrieved chunk containing unrelated correct text. The two modes are
labelled and never substituted for one another.
