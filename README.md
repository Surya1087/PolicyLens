# PolicyLens

**RAG-Based Insurance Policy Comprehension and Claim Eligibility Cross-Checking System**  
Final-year B.E. Computer Science project · local-first Python/Streamlit application

PolicyLens explains uploaded insurance documents with page-cited evidence. Its claim cross-checker pairs a **digital medical bill** with relevant policy passages without approving, rejecting, or determining coverage for a claim.

## Local build status

The source, UI, 11 reference insurance PDFs, 12 synthetic medical bills, test suite, and evaluation runners are included. See `docs/validation.md` for checks actually performed.

**Live Groq generation and RAGAS scores still require your own free-tier API key.** Missing-key result files explicitly contain `blocked` rows, not invented scores. Passing automated tests is not a guarantee of clinical, legal, or model accuracy. Nothing has been deployed or pushed to GitHub; Git preparation is deferred until you confirm the local run.

## 1. Open locally in VS Code

Open this `policylens` folder as the VS Code workspace. Use **64-bit Python 3.11** (the tested interpreter is 3.11.15); Python 3.13 is not the target for this dependency set.

### macOS / Linux

```bash
cd policylens
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env_example .env
```

If your terminal is already inside `policylens`, omit `cd policylens`. An installed local `.venv` is already available on the build machine; activate it rather than recreating it.

### Windows PowerShell

```powershell
cd policylens
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env_example .env
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`; changing system execution policy is unnecessary.

In VS Code, run **Python: Select Interpreter** and select this folder's `.venv` interpreter.

### Free Groq setup

1. Create a developer account at <https://console.groq.com/> and stay on the **free tier**; do not activate billing. A paid subscription/card is not required for the free developer tier, subject to Groq's current account availability and limits.
2. Generate a key at <https://console.groq.com/keys>.
3. Open your **local `.env`**, and set `GROQ_API_KEY` to that key. Leave `GROQ_MODEL=llama-3.3-70b-versatile` unless Groq documents a supported free-tier Llama replacement.
4. Never paste a key into source code, screenshots, evaluation reports, or chat. Never upload `.env`.

Indexing and retrieval run without a Groq key. Q&A, summaries, risk explanations, bill extraction and claim clause selection need it. A configuration check is not an authentication check; the first request verifies access. The supplied LIC Jeevan Labh PDF is scanned and is intentionally rejected; use a text-readable sample such as Optima Lite for the demonstration.

## 2. Run

```bash
streamlit run app/ui/streamlit_app.py
```

Open <http://localhost:8501>. The server binds to `127.0.0.1`, not your public network interface. If `streamlit` is not on your PATH:

```bash
python -m streamlit run app/ui/streamlit_app.py
```

The first indexing operation downloads the public `sentence-transformers/all-MiniLM-L6-v2` weights. Subsequent embeddings run locally using the model cache. Allow a few GB for the environment and model cache. There is no cloud vector store or paid model fallback.

### Suggested demonstration

1. Select `OptimaLiteBrochure(6)-775316069816.pdf` in the sidebar and click **Index selected sample**.
2. Read the privacy notice and consent before sending text to Groq.
3. Generate an **Executive Summary**. Ask: “What is the room rent limit?”
4. Run **Risk Analysis**; inspect the separate ambiguity scan and any partial-review warnings.
5. In **Check Claim**, choose `health_cataract.pdf`. Optionally provide `2025-06-01` as the fictional policy start date from its test case, then click **Check Claim**.
6. Inspect extracted bill fields, the cited cataract waiting-period passage (physical page 11), and the unverified date interval. This is not a claim verdict.
7. Verify the full page in **Source Pages**. Upload another policy in **Compare Policies** for a cited comparison.

The cataract bill says **phacoemulsification**, not cataract. Clinical query expansion and semantic retrieval bridge this terminology difference. All fixture-to-policy pairings are in `data/evaluation/claim_labels.json`.

## Features

- Digital PDF parsing with physical-page provenance, page-local overlapping chunks, normalized local embeddings, and persistent policy-isolated Chroma retrieval.
- Grounded Q&A; multi-facet summary of benefits, optional covers, exclusions, numbers and claim steps; cited comparison of two policies.
- Risk review of waiting periods, deductibles, exclusions and limitations. A full extracted-text regex scan finds discretionary language, followed by batched LLM confirmation and explanations.
- Strict JSON bill extraction: procedure/diagnosis, date, amount, currency, hospital and explicitly stated network status. Each field must carry matching source-page evidence; unsupported fields become `null`.
- Safe claim cross-checks: the model selects predefined clause topics and verbatim citations. **The application—not generated prose—renders the findings.** Only exact quotations from retrieved passages survive validation; `claim_status` is always `not_assessed`.
- Missing PDF/key, encrypted or malformed PDF, scans, rate limits, malformed model JSON, unsupported citations and partial risk-review failures are handled explicitly.
- Synthetic-fixture evaluation with separate retrieval-only and real-model end-to-end modes.

## Architecture and source layout

```text
policylens/
  app/agents/                # Five agents, shared validation, explicit router
    insurance_agent.py
    summary_agent.py
    risk_agent.py
    claim_agent.py
    comparison_agent.py
    base.py
    router.py
  app/ui/streamlit_app.py     # Consent, upload slots, six tabs, source inspection
  rag/
    loader.py                # Bounded PyPDF input and scan warnings
    splitter.py              # LangChain page-local chunking
    embeddings.py            # Lazy Sentence-Transformers
    vector_store.py          # Local Chroma + durable policy manifests
    ingest.py
    rag_pipeline.py          # Shared ingestion/retrieval, clinical query expansion
    bill_extractor.py        # Strict model extraction + evidence validation
    models.py
  llm/groq_client.py          # JSON schema instructions/validation, bounded retries
  prompts/prompt.py           # Centralized prompts and safety instructions
  config/                    # Environment settings, curated phrases/synonyms
  evaluation/                # Real-output RAGAS and independent claim metrics
  evaluation/results/        # CSVs, aggregates, actual run records
  data/insurance/            # Eleven reference brochures with provenance
  data/synthetic_bills/       # Twelve fictional digital medical documents
  data/evaluation/           # Human-curated references and clause labels
  scripts/generate_bills.py
  tests/
  docs/
  .streamlit/config.toml
  .env_example
  requirements.txt           # Exact direct dependency versions
  requirements-lock.txt      # Resolved dependency snapshot for reproduction
```

Flow: **PDF → page text → chunks → local embeddings → policy-filtered Chroma → agent prompt → Groq JSON → schema/evidence checks → cited UI output**.

Claim flow reuses the **same** retrieval pipeline after bill extraction. It never creates a second embedding system. Every agent has `__init__(pipeline, llm=None)` and `.run(...)`; `Router.run(route, **kwargs)` chooses the agent. See `rag/API.md` for the retrieval contract.

This is a local academic application, not a certified insurer decision engine or a hardened multi-user medical-record service.

## Evaluation and tests

```bash
python -m pytest -q
python scripts/generate_bills.py

# Real local embeddings/retrieval, with gold bill fields; no Groq key needed:
python -m evaluation.claim_evaluator --mode retrieval-only --output evaluation/results/claim-retrieval

# Real answers + actual retrieved contexts, judged by Groq through RAGAS:
python -m evaluation.evaluator --mode agent --output evaluation/results/agent

# Actual PDF bill extraction, retrieval and emitted clause citations:
python -m evaluation.claim_evaluator --mode full --output evaluation/results/claim-full
```

The core evaluation contains independent references for Q&A, summary and risk tasks. `test_cases.py` loads **questions and gold references**, not prewritten system answers. The evaluator invokes the real agents and stores the resulting answers and exact contexts in `raw_agent_records.jsonl`. RAGAS uses Groq plus local embeddings explicitly; there is no implicit paid OpenAI judge. OpenAI-related packages may be installed as RAGAS transitive dependencies but are not called by this project.

Each evaluation directory contains `results.csv`, `details.csv` and `aggregates.json`. Blank cells mean **not measured**, not a score of zero. Missing keys or incomplete runs exit with status `2`. Synthetic bill extraction scores, clause retrieval scores and actual quoted-citation scores are kept separate. Negative/insufficient-evidence cases have no invented gold clause; their precision/recall are not applicable.

This is a small, development-time labelled suite, not an independent clinical validation set. See `docs/evaluation.md` for metric definitions, limitations and how to interpret results. Free-tier rate limits can make a full RAGAS run slow; allow the service's quota to reset rather than enabling billing.

## Privacy, safety and limitations

- **Local-first is not fully offline:** selected policy passages and uploaded bill text are sent to Groq when you request AI analysis. Do not use real health identifiers for demonstrations. Review Groq's data policies before sharing sensitive documents.
- The UI keeps uploaded original bytes in session memory, but extracted policy text and embeddings persist in `.local_data/sessions/`. CLI evaluation uses `.local_data/chroma/`. These files are **not encrypted**. “Clear on-screen session” does not erase disk files. Stop Streamlit and delete `.local_data/` when you want to erase local indexes; delete any exported JSON reports separately.
- API keys load only from environment/local `.env`. No key is written into output or logged. Streamlit usage telemetry, Chroma analytics, RAGAS tracking and LangSmith tracing are disabled by this application.
- Physical page numbers are exact; headings are best-effort extracted strings, not invented section numbers. Source-page text is available for inspection. PDF table extraction can scramble layout, so always verify numeric context in the original PDF.
- Only digital/text PDFs are supported. Entirely scanned documents are rejected; textless pages in a mixed document produce warnings. Text extraction cannot reliably detect every partly scanned page. OCR is intentionally absent.
- PDF inputs have a 25 MB limit, 500-page limit and extracted-text bounds. Bills have additional processing limits. Password-protected PDFs require you to provide an unlocked copy locally.
- Dates in bills must have explicit evidence. Numeric day/month dates use day-first interpretation; confirm these values. Missing policy start dates are never inferred from brochures. User-provided start dates are unverified and do not establish renewal continuity, endorsements or pre-existing-condition history.
- The reference PDFs are **brochures**, not complete contracts. A missing clause does not establish inclusion or exclusion. Exact quotes can contain words such as “covered”; these are labelled document quotations, not verdicts on the submitted bill.
- Valid citations prove a quote exists, not that a model explanation is semantically correct. Claims avoid free-form verdicts by construction, but relevance selection, PDF parsing and other generated explanations can still be wrong. Confirm terms and claim applicability with your insurer.
- Free-tier availability and model IDs can change. This app never silently falls back to a paid provider.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Missing API key | Copy `.env_example` to `.env`, populate `GROQ_API_KEY`, stop and restart Streamlit. |
| Groq authentication/model error | Check the key and model in your Groq console; keep the account on its free tier. |
| Rate limit or timeout | Wait for quota recovery and retry manually. SDK retries are bounded; no infinite retry loop. |
| Model JSON rejected | Retry once; unsupported output is deliberately not displayed. Inspect provider/model compatibility if it recurs. |
| First indexing cannot download model | Check internet/proxy/certificate setup. Do not disable TLS verification. Hugging Face model download requires internet once. |
| “No extractable text” | Use a digital PDF; scans are out of scope. |
| Native library/import error | Recreate a Python 3.11 64-bit venv and install the pinned dependencies; do not mix global packages. |
| Chroma database incompatible after experiments | Stop the application, retain any needed reports, delete the generated `.local_data/` index and re-index the original PDFs. |
| Port 8501 occupied | Use `python -m streamlit run app/ui/streamlit_app.py --server.port 8502`. |

For the build environment's fully resolved versions, install `requirements-lock.txt` instead of `requirements.txt`. That snapshot was resolved and tested on macOS arm64/Python 3.11; other platforms may need platform-specific wheels. They are not claimed tested here.

## Architectural reference and attribution

PolicyLens independently rebuilds its insurance-agent/RAG structure, preserving page provenance and policy isolation and replacing static evaluation answers with real pipeline capture. No finance agent or finance dataset is included. Reference PDF URLs and SHA-256 hashes are in `data/README.md`; third-party brochures remain the property of their issuers. Confirm redistribution permissions before a future public repository release.

**Next milestone:** confirm this local flow works with your key and documents. Only after your confirmation should Git ignore rules, final secret checks and GitHub preparation be performed. There is no deployment or Git automation in this build.
