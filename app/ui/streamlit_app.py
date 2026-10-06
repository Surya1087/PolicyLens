"""Run from the project root: streamlit run app/ui/streamlit_app.py."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.agents.router import Router
from config.settings import get_settings
from rag.models import PolicyLensError
from rag.rag_pipeline import RAGPipeline


def apply_style() -> None:
    """Small, static visual layer; document text is always rendered by Streamlit."""
    st.markdown(
        """
        <style>
        .pl-hero { padding: 1.25rem 1.5rem; border-radius: 18px;
          background: linear-gradient(120deg,#173f5f 0%,#20639b 55%,#3caea3 100%);
          color: white; margin-bottom: 1rem; }
        .pl-hero h1 { color: white; margin: 0; font-size: 2.3rem; }
        .pl-hero p { color: #e5f6f4; font-size: 1.08rem; margin: .35rem 0 0; }
        .pl-card { min-height: 115px; padding: 1rem; border-radius: 14px;
          border: 1px solid #dbe7f3; background: #fff; box-shadow: 0 2px 10px #173f5f12; }
        .pl-card b { color: #173f5f; font-size: 1.05rem; }
        .pl-step { color: #20639b; font-weight: 700; font-size: .85rem; letter-spacing: .04em; }
        [data-testid="stTabs"] button { font-weight: 650; }
        [data-testid="stSidebar"] { border-right: 1px solid #dbe7f3; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        settings = get_settings()
        st.session_state.pipeline = RAGPipeline(
            persist_directory=settings.data_dir / "sessions" / st.session_state.session_id
        )
    return st.session_state.pipeline


def display_result(result: dict, key: str) -> None:
    """Render model/document content as text, never HTML or executable links."""
    for warning in result.get("warnings", []) + result.get("bill_warnings", []):
        st.warning(warning)
    if "bill_fields" in result:
        st.subheader("Extracted bill fields")
        rows = [
            {"Field": name.replace("_", " ").title(), "Value": str(field["value"]) if field["value"] is not None else "Not stated / not verified",
             "Bill page": field.get("page"), "Printed evidence": field.get("evidence") or "—"}
            for name, field in result["bill_fields"].items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Verify these extracted values against your bill. A missing value is not inferred.")
        st.subheader("Clause cross-check — no decision")
    with st.container(border=True):
        st.markdown("#### Your plain-language result")
        st.text(result.get("answer", "No supported answer was returned."))
    if result.get("scope"):
        st.caption(result["scope"])
    if "scan" in result:
        scan = result["scan"]
        st.caption(f"Ambiguity scan: {scan.get('chunks_scanned', 0)} chunks; {scan.get('candidates', 0)} phrase matches.")
        if scan.get("partial"):
            st.warning("The ambiguity review is partial. Unassessed matches are not cleared risks.")
        if result.get("findings"):
            with st.expander("Confirmed ambiguous wording and explanations"):
                for finding in result["findings"]:
                    st.text(f"{finding.get('excerpt', '')}\n{finding.get('why', '')}")
    with st.expander(f"📚 Verified source excerpts ({len(result.get('sources', []))})", expanded=True):
        if not result.get("sources"):
            st.info("No verified citations were produced. Do not infer that a term is absent.")
        for source in result.get("sources", []):
            st.caption(f"{source['source']} · Physical PDF page {source['page']} · {source['section']}")
            st.code(source["id"], language=None)
            st.text(source["text"])
    st.download_button(
        "Download this analysis (JSON)", json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"policylens-{key}.json", mime="application/json", key=f"download-{key}",
    )
    st.caption("Downloads may contain medical information. Keep them private.")


def action_intro(title: str, description: str, note: str | None = None) -> None:
    """Give every analysis the same calm, plain-language starting point."""
    st.subheader(title)
    st.caption(description)
    if note:
        st.info(note)


def analyze(route: str, result_key: str, **kwargs) -> None:
    st.session_state.results.pop(result_key, None)
    try:
        with st.spinner("Retrieving policy clauses and validating the analysis…"):
            result = Router(pipeline()).run(route, **kwargs)
        st.session_state.results[result_key] = result
    except PolicyLensError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Analysis could not finish. Check the local setup and try again. No result from this attempt is shown.")


def ingest(source, filename: str, slot: str) -> None:
    try:
        with st.spinner("Parsing pages and building the local policy index…"):
            document = pipeline().ingest(source, filename=filename)
        st.session_state[slot] = document
        st.session_state.results = {}
        st.success(f"Indexed {document.filename}: {len(document.pages)} pages, {len(document.chunks)} passages.")
    except PolicyLensError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Indexing failed. Check disk space and install the pinned dependencies. The previous active policy is unchanged.")


def main() -> None:
    st.set_page_config(page_title="PolicyLens | Insurance, explained", page_icon="📘", layout="wide")
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("results", {})
    apply_style()
    st.title("PolicyLens")
    st.markdown('<div class="pl-hero"><p>Insurance, explained clearly — अपनी policy को आसानी से समझें</p></div>', unsafe_allow_html=True)
    st.caption("A calm, private place to understand policy wording, check facts, and prepare questions.")
    st.warning("PolicyLens does not approve, reject, or determine coverage for a claim. Brochures may omit binding terms; use your complete policy schedule and wording.")
    settings = get_settings()
    with st.sidebar:
        st.header("👋 Start here")
        st.caption("Choose a digital PDF. It stays in this session's local workspace.")
        uploaded = st.file_uploader("Insurance policy PDF", type=["pdf"], key=f"policy_upload-{st.session_state.session_id}")
        if st.button("Use uploaded policy", disabled=uploaded is None, use_container_width=True, type="primary"):
            ingest(uploaded.getvalue(), uploaded.name, "policy")
        samples = sorted((ROOT / "data" / "insurance").glob("*.pdf"))
        sample = st.selectbox("Or try a reference brochure", samples, format_func=lambda p: p.name, help="Reference brochures are for learning only and may not reflect your cover.")
        if st.button("Use selected brochure", disabled=sample is None, use_container_width=True):
            ingest(sample, sample.name, "policy")
        st.caption("Digital text PDFs only · 25 MB maximum · Scanned images are not supported.")
        with st.expander("Privacy & AI access"):
            st.caption(f"AI access details (advanced) · Model: {settings.groq_model}")
            if settings.groq_api_key:
                st.success("Local API key is configured (not yet verified).")
            else:
                st.info("Add GROQ_API_KEY to your local .env, then restart Streamlit. Local indexing works without a key.")
            consent = st.checkbox("I consent to sending relevant policy text and, for Claim Check, bill text to Groq.", key="consent")
            st.caption("No OCR, paid API, or cloud vector database. Groq generation is an external network request. Avoid real patient identifiers in demonstrations.")
            if st.button("Clear on-screen session", use_container_width=True):
                st.session_state.clear()
                st.rerun()
            st.caption("This clears the screen, not disk. To erase local indexes, stop the app and delete .local_data/. Upload originals are not saved by the UI.")
    policy = st.session_state.get("policy")
    if policy is None:
        st.info("Start by uploading a digital PDF or choosing a sample brochure in the left panel. Nothing else is needed to begin.")
        st.markdown("## Three easy steps")
        one, two, three = st.columns(3)
        with one:
            st.markdown('<div class="pl-card"><div class="pl-step">STEP 1</div><b>Bring your policy</b><br>Upload a PDF or try a sample brochure.</div>', unsafe_allow_html=True)
        with two:
            st.markdown('<div class="pl-card"><div class="pl-step">STEP 2</div><b>Choose what you need</b><br>Ask a question, get a summary, or check a claim.</div>', unsafe_allow_html=True)
        with three:
            st.markdown('<div class="pl-card"><div class="pl-step">STEP 3</div><b>See the evidence</b><br>Read the matching policy pages before you decide.</div>', unsafe_allow_html=True)
        return
    for warning in policy.warnings:
        st.warning(warning)
    st.markdown("## Step 2 · Choose what you need")
    st.caption("You can come back and try another option whenever you like.")
    with st.container(border=True):
        st.markdown(f"**Active policy** · `{policy.filename}`")
        left, middle, right = st.columns(3)
        left.metric("Pages", len(policy.pages))
        middle.metric("Passages ready", len(policy.chunks))
        right.metric("Privacy", "Local index")
    enabled = bool(settings.groq_api_key and consent)
    if not enabled:
        st.info("To generate an answer, add your local Groq key and confirm the privacy consent in the sidebar. You can still inspect source pages.")
    tabs = st.tabs(["🧾 Quick summary", "💬 Ask a question", "🔎 Find things to review", "🏥 Check a bill", "⚖️ Compare policies", "📄 Read source pages"])
    with tabs[0]:
        action_intro("Executive summary", "Get the key benefits, limits, and next steps in a quick overview.", "This is a scoped summary, not an exhaustive contract review.")
        if st.button("Generate summary", disabled=not enabled, type="primary"):
            analyze("Executive Summary", "summary", policy_id=policy.policy_id)
        if "summary" in st.session_state.results:
            display_result(st.session_state.results["summary"], "summary")
    with tabs[1]:
        action_intro("Ask a question", "Ask about benefits, exclusions, numbers, or what to do next.")
        question = st.text_area("Your question", placeholder="For example: Is there a waiting period for hospital treatment?", max_chars=2000, key="question")
        if st.button("Get an answer", disabled=not enabled or not question.strip(), type="primary"):
            analyze("Ask Question", "question", policy_id=policy.policy_id, query=question.strip())
        if "question" in st.session_state.results:
            display_result(st.session_state.results["question"], "question")
    with tabs[2]:
        action_intro("Review possible risks", "Find limitations and discretionary wording that may deserve a closer look.", "This scan highlights wording to discuss; it does not decide whether a risk applies to you.")
        if st.button("Scan for risks", disabled=not enabled, type="primary"):
            analyze("Risk Analysis", "risk", policy_id=policy.policy_id)
        if "risk" in st.session_state.results:
            display_result(st.session_state.results["risk"], "risk")
    with tabs[3]:
        action_intro("Claim Check", "Match a medical bill or treatment report to relevant policy clauses.", "This is a factual cross-check, not a verdict. It does not approve, reject, or determine coverage for your claim.")
        st.caption("Digital bills/treatment reports only. Original quoted terms are evidence, not a decision about your bill.")
        bill = st.file_uploader("Medical bill or treatment report PDF", type=["pdf"], key=f"bill_upload-{st.session_state.session_id}")
        demo_bills = sorted((ROOT / "data" / "synthetic_bills").glob("*.pdf"))
        demo_bill = st.selectbox("Or choose a synthetic demonstration bill", [None, *demo_bills], format_func=lambda p: "Use my upload" if p is None else p.name)
        if demo_bill is not None:
            st.caption("Choose the matching sample policy listed in data/evaluation/claim_labels.json. All sample patients are fictional.")
        has_start = st.checkbox("I know the policy start date (optional, user-supplied)")
        start = st.date_input("Policy start date", value=None, disabled=not has_start)
        start_iso = start.isoformat() if has_start and start else None
        source = bill.getvalue() if bill is not None else (demo_bill.read_bytes() if demo_bill else None)
        signature = (hashlib.sha256(source).hexdigest() if source else None, start_iso, policy.policy_id)
        if signature != st.session_state.get("bill_signature"):
            st.session_state.results.pop("claim", None)
            st.session_state.bill_signature = signature
        if st.button("Check Claim", disabled=not enabled or source is None, type="primary"):
            analyze("Claim Check", "claim", policy_id=policy.policy_id, bill_source=source, policy_start_date=start_iso)
        if "claim" in st.session_state.results:
            display_result(st.session_state.results["claim"], "claim")
    with tabs[4]:
        action_intro("Compare policies", "Place two indexed policies side by side to explore differences.")
        second = st.file_uploader("Second insurance policy PDF", type=["pdf"], key=f"second_policy-{st.session_state.session_id}")
        if st.button("Add comparison policy", disabled=second is None):
            ingest(second.getvalue(), second.name, "policy_b")
        policy_b = st.session_state.get("policy_b")
        if policy_b:
            st.caption(f"Comparison document: {policy_b.filename}")
        if st.button("Compare policies", disabled=not enabled or policy_b is None, type="primary"):
            analyze("Compare Policies", "comparison", policy_id=policy.policy_id, policy_id_b=policy_b.policy_id)
        if "comparison" in st.session_state.results:
            display_result(st.session_state.results["comparison"], "comparison")
    with tabs[5]:
        action_intro("Source pages", "Read the extracted text behind every answer, starting with the physical PDF page.")
        st.caption("Physical PDF pages, starting at 1. Printed page labels can differ. Section headings are best-effort; the page and original excerpt are authoritative.")
        page_number = st.selectbox("Physical page", [page.page for page in policy.pages])
        page = next(page for page in policy.pages if page.page == page_number)
        st.text(page.text or "No extractable text on this page. OCR is not supported.")


if __name__ == "__main__":
    main()
