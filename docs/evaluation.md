# Evaluation

The evaluation records live pipeline retrieval and actual `app.agents.Router` output. It
never substitutes canned answers or contexts. Q&A, executive summary, and risk
analysis use independently written reference answers and verbatim anchors tied
to physical brochure pages. RAGAS 0.2.15 scores successful live generations for
faithfulness, answer relevancy, context precision, and context recall using Groq
as judge and the configured local Hugging Face embeddings (no OpenAI default).

## Commands

```bash
python scripts/generate_bills.py
python -m evaluation.evaluator --mode agent --output evaluation/results/agent
python -m evaluation.claim_evaluator --mode retrieval-only --output evaluation/results/claim-retrieval
python -m evaluation.claim_evaluator --mode full --output evaluation/results/claim-full
pytest -q tests/test_evaluation.py
```

For free-tier quotas, start with one case and a separate output folder:

```bash
python -m evaluation.evaluator --mode agent --case-id qa-lite-room-rent --output evaluation/results/single-qa
python -m evaluation.claim_evaluator --mode full --case-id claim-health-cataract --output evaluation/results/single-claim
```

Repeat `--case-id` to select a small batch. Quotas may require spreading RAGAS
evaluation across several sessions or days. A subset report contains only those
cases; it never presents a subset as a full-suite result. Each command replaces
its output files, so use distinct output folders to retain separate runs.

`claim-retrieval` deliberately uses the hand-labelled bill description, so it
isolates policy retrieval and is **not** an end-to-end bill extraction score.
`claim-full` sends each generated PDF to `Router.run("Claim Check", ...)` and
scores citations emitted by that model separately from retrieved chunks. Gold
retrieval is never presented as a model-generated finding.
Full mode also reports normalized exact-match accuracy across the bill fields
actually extracted by the model; retrieval-only mode leaves that metric empty.

Each output directory contains the primary per-case `results.csv`, a backwards-compatible
`details.csv`, and `aggregates.json`. Agent mode also records
`raw_agent_records.jsonl` with the real question, answer, `retrieved_contexts`,
and independent reference. Full claim mode records `raw_claim_records.jsonl`.
No canned output is written. Empty cells mean not measured or blocked; they
are not zeros. Missing `GROQ_API_KEY`, failed generation, or incomplete RAGAS
metrics produces blocked/error rows and exit status 2. Execution is serial with
at most two retries (`--retries`, default 1) to limit API pressure.

## Metric definitions

Retrieval recall is the fraction of independently labelled page+anchor clauses
found in returned chunks. Retrieval precision is the fraction of returned chunks
containing a labelled anchor on its labelled page. Citation metrics apply the
same definitions to the exact quotations the claim agent actually emitted, rather
than awarding a hit for unrelated gold text elsewhere in a cited chunk. Negative cases
with no relevant brochure clause report these anchor metrics as not applicable,
rather than creating misleading zeros.

The core suite has 15 tasks across ten text-readable reference PDFs (10 Q&A,
2 summaries, 3 risk reviews). The supplied LIC brochure is scanned and is not
sent through RAGAS. An identical premium-break passage repeats on physical pages
17–19 of the ICICI brochure: these manually verified equivalent pages count as
one gold clause, not three separate recall obligations.

`no_label_abstention` records whether the full claim agent emitted no findings for
the two unlabelled/insufficient-evidence cases. It is not a coverage decision or
proof that no relevant policy clause exists. Gold labels intentionally cover only
one target passage per positive claim case, so other genuinely relevant chunks
can lower measured precision. Review raw records rather than treating low
precision as proof of hallucination.

The twelve fixtures are all medical bills or treatment-related service invoices.
They include phacoemulsification, hernia repair, knee arthroplasty, ambulance
transport, dental treatment, home nursing, optical care, emergency evacuation,
adventure-related injury and elective genomic counselling. No luggage, flight-delay
or phone-repair invoices are included in the medical claim suite.

## Limits

The source PDFs are marketing brochures, not full policy contracts. They often
direct readers to policy wording for definitions, complete exclusions, waiting
period rules, and benefit conditions. Evaluation can test whether PolicyLens
finds and cites what the brochures actually say; it cannot establish coverage,
invent sections or terms, or infer a policy start date. The cataract fixture uses
the clinical term “phacoemulsification” to test terminology matching, while its
label remains the brochure's actual cataract waiting-period text. Synthetic
coverage and exclusion examples are evidence-retrieval tests, never claim
decisions.

Agent and full-claim metrics use only the `retrieved_contexts` returned by that
same Router call. They never replace those contexts with cited sources or a
second gold query. Only retrieval-only claim mode runs the hand-labelled gold
query through the SAME `claim_retrieval_query` builder used by the live claim agent,
and that mode does not construct a Router or LLM. No expected clause names are
appended by the evaluator. Actual retrieval contexts are saved to
`raw_retrieval_records.jsonl`. Project `.env` is loaded
before settings are refreshed; secret values are never copied into result files.
