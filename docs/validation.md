# Local validation record

Date: 2026-10-01. Platform tested: macOS arm64, Python 3.11.15.

## Checks actually performed

| Check | Result |
|---|---|
| Install exact direct dependencies into project `.venv` | Succeeded |
| `uv pip check --python .venv/bin/python` | All installed packages compatible |
| `RUN_EMBEDDING_TESTS=1 .venv/bin/python -m pytest -q` | 49 passed, including real local embeddings and Chroma |
| `python -m compileall -q app config rag llm prompts evaluation scripts tests` | Passed |
| Streamlit AppTest: initial page without API key | No exceptions |
| Streamlit AppTest: actual Optima Lite indexing | 18 chunks; six tabs rendered; no exceptions |
| Local server `GET /_stcore/health` | `ok` |
| All gold anchors checked against original physical PDF pages | Matched across ten text-readable brochures |
| Index and query every supported reference PDF | All ten succeeded; the scanned LIC PDF was correctly rejected |
| Regenerate 12 synthetic medical bills | Byte-for-byte identical |
| RAGAS, LangChain Groq and local embedding integration imports | Succeeded |

The automated suite includes schema transmission/validation, malformed JSON,
authentication/rate-limit handling, PDF encryption and malformed/scanned inputs,
source-page preservation, policy isolation, persistent metadata, invalid citations,
constrained non-decision claim output, unsupported bill values, network negation,
date intervals, ambiguity batches and partial review, actual-context evaluation
capture, blocked/non-finite metric handling, and Streamlit rendering.

## Actual evaluation outputs

- `evaluation/results/claim-retrieval/results.csv`: 12 real local retrieval runs
  using **gold bill extraction**, the shared query builder, actual
  all-MiniLM-L6-v2 embeddings, and Chroma. The latest run recovered 10/10 labelled
  positive target clauses: macro recall **1.00**, macro precision **0.1667** at
  six returned chunks. Two insufficient-evidence cases have no gold target and
  are excluded from these means, with explicit metric sample counts.
- Precision is against one sparsely labelled target clause per positive case;
  other potentially relevant passages are not exhaustively annotated. These
  figures are not an estimate of claim-decision accuracy. An earlier cold run
  recovered 9/10 targets; approximate retrieval and small datasets should not be
  presented as a guarantee of perfect retrieval.
- `evaluation/results/agent/results.csv`: **15 blocked rows** because no
  `GROQ_API_KEY` was configured. No generated answers or RAGAS scores are claimed.
- `evaluation/results/claim-full/results.csv`: **12 blocked rows** for the same
  reason. Actual model bill-extraction and citation precision/recall remain
  **unmeasured**. They are not replaced with gold-field retrieval scores.

Raw actual local contexts are in
`evaluation/results/claim-retrieval/raw_retrieval_records.jsonl`.

## Remaining acceptance steps

1. Add your free-tier key to local `.env` and restart Streamlit.
2. Run the README demonstration with the matching Optima Lite/cataract fixtures.
3. Run both hosted-model evaluation commands; inspect their CSVs and actual
   raw-output records. Resolve provider limits or real model-output failures
   before claiming end-to-end validation.
4. Confirm the application works with your own supported digital documents.

No hosted model request, production security audit, Windows/Linux run, legal
certification, deployment, Git initialization, commit or GitHub push is claimed.
Git preparation is intentionally deferred until the user confirms the local run.
