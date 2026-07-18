import os
import json
from datetime import datetime
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Import our backend modules
import engine
import triage
import context_builder
import draft_machine
from task_logger import log_action, get_action_log
from draft_machine import GeminiError


def _show_gemini_error(exc: Exception) -> None:
    """
    Display a clean, user-facing Streamlit error for a Gemini API failure.
    If ``exc`` is a GeminiError raised by our backends, its ``.friendly``
    message is shown directly.  Any other exception gets a generic message.
    The full raw error is always printed to the console for debugging.
    """
    print(f"[app] Gemini error ({type(exc).__name__}): {exc}")
    if isinstance(exc, GeminiError):
        st.error(exc.friendly)
    else:
        msg = str(exc).lower()
        if "503" in msg or "unavailable" in msg or "overloaded" in msg or "high demand" in msg:
            st.error(
                "⏳ Gemini is temporarily busy (free tier). "
                "Please wait a moment and try again."
            )
        elif "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate" in msg:
            st.error(
                "⚠️ Daily free-tier quota reached for Gemini. "
                "Please wait for it to reset, or upgrade your plan to continue."
            )
        else:
            st.error("Something went wrong. Please try again.")

# Load environment variables
load_dotenv()

# Set up page config
st.set_page_config(
    page_title="The Draft Desk",
    page_icon="✍️",
    layout="wide"
)

# Custom Styling for Dark Theme
st.markdown(
    """
    <style>
    /* Dark Theme background */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161625 !important;
        border-right: 1px solid #2d2d44;
    }
    
    /* Thread boxes */
    .thread-box {
        background-color: #162447;
        border: 1px solid #1f4068;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .thread-sender {
        font-weight: bold;
        color: #00bbee;
        font-size: 0.95em;
    }
    
    .thread-date {
        font-size: 0.8em;
        color: #8888aa;
        float: right;
    }
    
    .thread-body {
        margin-top: 8px;
        white-space: pre-wrap;
        font-size: 0.9em;
        color: #e0e0e0;
    }
    
    /* Draft Display */
    .draft-container {
        background-color: #1f4068;
        border: 2px solid #00bbee;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        font-size: 1.05em;
        line-height: 1.5;
        white-space: pre-wrap;
        color: #ffffff;
    }
    
    /* Status indicators */
    .status-approved {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
        color: #e8f5e9;
        padding: 12px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .status-rejected {
        background-color: #b71c1c;
        border: 1px solid #f44336;
        color: #ffebee;
        padding: 12px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .metadata-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-right: 5px;
        margin-top: 5px;
    }
    .badge-priority-urgent { background-color: #e94560; color: #fff; }
    .badge-priority-needs-reply { background-color: #f0a500; color: #fff; }
    .badge-priority-fyi { background-color: #0f4c75; color: #fff; }
    .badge-priority-ignore { background-color: #4e4e50; color: #fff; }
    .badge-category { background-color: #3282b8; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Monkey-patch context_builder and draft_machine for demo/sample mode ---
# This ensures that if we are using the sample/offline flow, it doesn't fail on Gmail API calls.
def app_get_thread_history(thread_id: str):
    # Retrieve from our current threads in session state
    if "threads" in st.session_state and st.session_state.threads:
        for t in st.session_state.threads:
            if t.get("id") == thread_id:
                history = []
                for msg in t.get("messages", []):
                    history.append({
                        "sender": msg.get("from", "unknown"),
                        "date": msg.get("date", "unknown"),
                        "content": msg.get("body", "")
                    })
                return history
    return []

def app_get_past_replies(limit: int = 3):
    return [
        {
            "subject": "Re: Project proposal",
            "content": "Let's proceed with the phase 1 rollout. I'll review the budget by tomorrow."
        },
        {
            "subject": "Re: Meeting schedule",
            "content": "Thanks for confirming. I'm available at 3 PM as proposed."
        },
        {
            "subject": "Re: Quick Sync",
            "content": "Sounds good. Let's touch base on Thursday afternoon."
        }
    ]

# Apply monkey patching to avoid live external dependencies when showing demo flows
engine.get_thread_history = app_get_thread_history
engine.get_past_replies = app_get_past_replies
context_builder.get_thread_history = app_get_thread_history
context_builder.get_past_replies = app_get_past_replies


@st.cache_resource
def _get_fetch_threads():
    """Cached reference to engine.fetch_threads."""
    return engine.fetch_threads


@st.cache_resource
def _get_send_reply():
    """Cached reference to engine.send_reply."""
    from engine import send_reply
    return send_reply


@st.cache_resource
def _get_calendar_engine():
    """Cached import of calendar_engine module."""
    import calendar_engine
    return calendar_engine


# --- Session State Initialization ---
if "threads" not in st.session_state:
    st.session_state.threads = []
if "triaged" not in st.session_state:
    st.session_state.triaged = []
if "drafts" not in st.session_state:
    st.session_state.drafts = {}
if "approved" not in st.session_state:
    st.session_state.approved = {}
if "rejected" not in st.session_state:
    st.session_state.rejected = set()
if "sent" not in st.session_state:
    st.session_state.sent = set()
if "booked" not in st.session_state:
    st.session_state.booked = {}
if "current_phase" not in st.session_state:
    st.session_state.current_phase = "Inbox & Triage"
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []

# --- Sidebar Configuration & Navigation ---
st.sidebar.title("✍️ The Draft Desk")
st.sidebar.write("Human-in-the-Loop AI Email Ghostwriter")
st.sidebar.write("---")

# Run Full Pipeline button — fetches, triages, and drafts in one shot
if st.sidebar.button(
    "⚡ Run Full Pipeline",
    type="primary",
    use_container_width=True,
    key="btn_run_pipeline",
):
    st.session_state.pipeline_running = True
    st.rerun()
st.sidebar.caption("Fetches, triages, and drafts — stops at Approval Gate.")

# API key — read from environment (local .env) or Streamlit Secrets (Cloud deployment).
# Never entered via the UI; set GEMINI_API_KEY in Streamlit → Settings → Secrets.
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    try:
        _api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        _api_key = None

if _api_key:
    import google.generativeai as legacy_genai
    legacy_genai.configure(api_key=_api_key)
    draft_machine.GEMINI_API_KEY = _api_key
    draft_machine.client = genai.Client(api_key=_api_key)

# Source selector
# Determine the default index from session state so the widget reflects
# the user's last choice after a rerun.
_source_options = ["Sample threads for demo", "Gmail via engine.py"]
_source_default_index = (
    1 if st.session_state.get("source") == "Gmail via engine.py" else 0
)
source_selection = st.sidebar.radio(
    "Data Source Selection:",
    options=_source_options,
    index=_source_default_index,
)
# Persist the selection so run_full_pipeline() and _render_pipeline_execution()
# can read it from session state (they check st.session_state.source).
st.session_state.source = source_selection

st.sidebar.write("---")
st.sidebar.subheader("Navigation Workflow")

# Create buttons for workflow phase navigation
phases = ["Inbox & Triage", "Draft Generation", "Approval Gate", "Export Proof"]
for phase in phases:
    # Highlight current active phase
    is_active = st.session_state.current_phase == phase
    button_label = f"👉 {phase}" if is_active else phase
    if st.sidebar.button(button_label, key=f"nav_{phase}", use_container_width=True):
        st.session_state.current_phase = phase
        st.rerun()

st.sidebar.write("---")
# Quick stats in sidebar
actionable_count = 0
if st.session_state.triaged:
    actionable_count = sum(
        1 for t in st.session_state.triaged 
        if t.get("priority") in ["urgent", "needs reply"]
    )
st.sidebar.metric("Actionable Threads", actionable_count)
st.sidebar.metric("Drafts Generated", len(st.session_state.drafts))
st.sidebar.metric("Approved Drafts", len(st.session_state.approved))

# --- Main App Sections ---


# ==========================================
# Pipeline helper functions
# ==========================================

def load_sample_threads() -> list:
    """Load threads from sample_threads.json. Returns [] on any error."""
    if not os.path.exists("sample_threads.json"):
        return []
    try:
        with open("sample_threads.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[load_sample_threads] ERROR: {exc}")
        return []


def fetch_threads_via_engine() -> list:
    """
    Fetch live threads from Gmail via engine.fetch_threads and normalise
    them into the app's internal format:
        [{id, subject, messages: [{from, date, body}]}]

    Re-raises exceptions so callers can display them in the UI instead of
    silently falling back to sample data.
    """
    fetched = engine.fetch_threads(max_results=10)
    raw = []
    for f in fetched:
        raw.append({
            "id":      f.get("thread_id", ""),
            "subject": f.get("subject", ""),
            "messages": [{
                "from": f.get("sender", ""),
                "date": f.get("date", ""),
                "body": f.get("snippet", ""),
            }],
        })
    return raw


def triage_threads(raw_threads: list) -> list:
    """
    Run triage.triage_inbox on raw_threads and merge the labels back by
    thread id.  Returns the updated thread list (same objects, priority /
    category / reason added in-place).
    """
    if not raw_threads:
        return []

    triage_inputs = []
    for t in raw_threads:
        first_msg = t["messages"][0] if t.get("messages") else {}
        triage_inputs.append({
            "id":      t.get("id", ""),
            "sender":  first_msg.get("from", "unknown"),
            "subject": t.get("subject", ""),
            "snippet": first_msg.get("body", ""),
        })

    triaged_results = triage.triage_inbox(triage_inputs)
    label_by_id = {r["id"]: r for r in triaged_results if r.get("id")}

    for thread_data in raw_threads:
        tid   = thread_data.get("id", "")
        label = label_by_id.get(tid)
        if label is None:
            thread_data.setdefault("priority", "needs reply")
            thread_data.setdefault("category", "other")
            thread_data.setdefault("reason",   "Classification result not returned")
        else:
            thread_data["priority"] = label.get("priority", "needs reply")
            thread_data["category"] = label.get("category", "other")
            thread_data["reason"]   = label.get("reason",   "No reason provided")

    return raw_threads


@st.cache_resource
def _get_draft_reply():
    """Cached reference to draft_machine.draft_reply."""
    return draft_machine.draft_reply


def run_full_pipeline() -> list:
    """
    Run the complete fetch → triage → draft pipeline in one shot.

    Steps
    -----
    1. Read st.session_state.source to decide the thread source.
    2. Fetch threads with load_sample_threads() or fetch_threads_via_engine().
    3. Triage threads with triage_threads().
    4. Reset all downstream session state (drafts, approved, rejected, sent, booked).
    5. For each urgent / needs-reply thread, call draft_reply and store in
       st.session_state.drafts.  Failures are logged but do not abort the loop.
    6. Set current_phase to "Approval Gate".
    7. Return a list of log strings describing every step taken.

    No UI calls are made here — callers are responsible for rendering the log.
    """
    log: list = []

    # ── Step 1: resolve source ───────────────────────────────────────────
    source = st.session_state.get("source", "Sample threads for demo")
    use_gmail = source == "Gmail via engine.py"
    log.append(f"[pipeline] source: {'Gmail via engine.py' if use_gmail else 'sample threads'}")

    # ── Step 2: fetch ────────────────────────────────────────────────────
    try:
        if use_gmail:
            raw_threads = fetch_threads_via_engine()
        else:
            raw_threads = load_sample_threads()
        log.append(f"[pipeline] fetched {len(raw_threads)} thread(s)")
    except Exception as exc:
        log.append(f"[pipeline] FETCH ERROR: {exc}")
        return log   # nothing more we can do

    if not raw_threads:
        log.append("[pipeline] no threads found — aborting pipeline")
        return log

    # ── Step 3: triage ───────────────────────────────────────────────────
    try:
        raw_threads = triage_threads(raw_threads)
        st.session_state.threads = raw_threads
        st.session_state.triaged = raw_threads
        log.append(
            f"[pipeline] triage complete — "
            + ", ".join(
                f"{p}: {sum(1 for t in raw_threads if t.get('priority') == p)}"
                for p in ("urgent", "needs reply", "fyi", "ignore")
            )
        )
    except Exception as exc:
        log.append(f"[pipeline] TRIAGE ERROR: {exc}")
        return log

    # ── Step 4: reset downstream state ───────────────────────────────────
    st.session_state.drafts   = {}
    st.session_state.approved = {}
    st.session_state.rejected = set()
    st.session_state.sent     = set()
    st.session_state.booked   = {}
    log.append("[pipeline] downstream state reset (drafts / approved / rejected / sent / booked)")

    # ── Step 5: draft urgent + needs-reply threads ───────────────────────
    actionable = [
        t for t in raw_threads
        if t.get("priority") in ("urgent", "needs reply")
    ]
    log.append(f"[pipeline] {len(actionable)} actionable thread(s) to draft")

    draft_fn = _get_draft_reply()
    for t in actionable:
        tid       = t.get("id", "")
        subject   = t.get("subject", "(no subject)")
        first_msg = t["messages"][0] if t.get("messages") else {}
        try:
            compat = {
                "thread_id": tid,
                "sender":    first_msg.get("from", ""),
                "subject":   subject,
                "snippet":   first_msg.get("body", ""),
                "date":      first_msg.get("date", ""),
                "priority":  t.get("priority", "needs reply"),
                "category":  t.get("category", "other"),
                "reason":    t.get("reason", ""),
            }
            draft_text = draft_fn(compat)
            st.session_state.drafts[tid] = draft_text
            log.append(f"[pipeline] ✓ drafted: '{subject}' (id={tid})")
        except Exception as exc:
            log.append(f"[pipeline] ✗ draft FAILED for '{subject}' (id={tid}): {exc}")
            # continue to next thread

    # ── Step 6: navigate to Approval Gate ────────────────────────────────
    st.session_state.current_phase = "Approval Gate"
    log.append(
        f"[pipeline] done — {len(st.session_state.drafts)}/{len(actionable)} "
        f"draft(s) generated. Navigating to Approval Gate."
    )

    return log


def _render_pipeline_execution() -> None:
    """
    Execute the full fetch → triage → draft pipeline with live progress UI.

    Runs the same logic as run_full_pipeline() inline so each step can
    update the st.status container before it begins and write a result line
    as soon as it finishes.  No call to run_full_pipeline() is made.

    After the status block closes:
      - pipeline_log is stored in session state
      - current_phase is set to "Approval Gate"
      - pipeline_running is set to False
      - st.rerun() is called so the UI reflects the new phase
    """
    log: list = []

    # ── Resolve source (same logic as run_full_pipeline) ─────────────────
    source = st.session_state.get("source", "Sample threads for demo")
    use_gmail = source == "Gmail via engine.py"
    source_label = "Gmail via engine.py" if use_gmail else "sample threads"
    log.append(f"[pipeline] source: {source_label}")

    with st.status("Running full pipeline…", expanded=True) as status:

        # ── Step 1: Fetch ─────────────────────────────────────────────────
        status.update(label=f"Step 1/3 — Fetching threads ({source_label})…")
        try:
            raw_threads = (
                fetch_threads_via_engine() if use_gmail else load_sample_threads()
            )
            if not raw_threads:
                msg = "No threads found — nothing to process."
                log.append(f"[pipeline] {msg}")
                st.write(f"⚠️ {msg}")
                status.update(label="Pipeline stopped — no threads.", state="error")
                return
            fetch_msg = f"Fetched {len(raw_threads)} thread(s)"
            log.append(f"[pipeline] {fetch_msg}")
            st.write(f"✅ {fetch_msg}")
        except Exception as exc:
            err = f"Fetch failed: {exc}"
            log.append(f"[pipeline] FETCH ERROR: {exc}")
            st.write(f"❌ {err}")
            status.update(label="Pipeline failed at fetch step.", state="error")
            return

        # ── Step 2: Triage ────────────────────────────────────────────────
        status.update(label="Step 2/3 — Triaging threads with Gemini…")
        try:
            raw_threads = triage_threads(raw_threads)
            st.session_state.threads = raw_threads
            st.session_state.triaged = raw_threads

            priority_summary = ", ".join(
                f"{p}: {sum(1 for t in raw_threads if t.get('priority') == p)}"
                for p in ("urgent", "needs reply", "fyi", "ignore")
            )
            triage_msg = f"Triage complete — {priority_summary}"
            log.append(f"[pipeline] {triage_msg}")
            st.write(f"✅ {triage_msg}")
        except Exception as exc:
            err = f"Triage failed: {exc}"
            log.append(f"[pipeline] TRIAGE ERROR: {exc}")
            st.write(f"❌ {err}")
            status.update(label="Pipeline failed at triage step.", state="error")
            return

        # ── Reset downstream state ────────────────────────────────────────
        st.session_state.drafts   = {}
        st.session_state.approved = {}
        st.session_state.rejected = set()
        st.session_state.sent     = set()
        st.session_state.booked   = {}
        log.append("[pipeline] downstream state reset")

        # ── Step 3: Draft loop ────────────────────────────────────────────
        actionable = [
            t for t in raw_threads
            if t.get("priority") in ("urgent", "needs reply")
        ]
        status.update(
            label=f"Step 3/3 — Drafting {len(actionable)} actionable thread(s)…"
        )
        log.append(f"[pipeline] {len(actionable)} actionable thread(s) to draft")

        draft_fn   = _get_draft_reply()
        n_ok       = 0
        n_fail     = 0

        for t in actionable:
            tid       = t.get("id", "")
            subject   = t.get("subject", "(no subject)")
            first_msg = t["messages"][0] if t.get("messages") else {}
            try:
                compat = {
                    "thread_id": tid,
                    "sender":    first_msg.get("from", ""),
                    "subject":   subject,
                    "snippet":   first_msg.get("body", ""),
                    "date":      first_msg.get("date", ""),
                    "priority":  t.get("priority", "needs reply"),
                    "category":  t.get("category", "other"),
                    "reason":    t.get("reason", ""),
                }
                draft_text = draft_fn(compat)
                st.session_state.drafts[tid] = draft_text
                n_ok += 1
                log.append(f"[pipeline] ✓ drafted: '{subject}' (id={tid})")
                st.write(f"✅ Drafted: *{subject}*")
            except Exception as exc:
                n_fail += 1
                log.append(
                    f"[pipeline] ✗ draft FAILED for '{subject}' (id={tid}): {exc}"
                )
                st.write(f"❌ Failed to draft: *{subject}* — {exc}")
                # continue to next thread

        done_msg = (
            f"Done — {n_ok}/{len(actionable)} draft(s) generated"
            + (f", {n_fail} failed" if n_fail else "")
            + ". Proceeding to Approval Gate."
        )
        log.append(f"[pipeline] {done_msg}")
        st.write(f"✅ {done_msg}")
        status.update(label=done_msg, state="complete")

    # ── Outside the status block ──────────────────────────────────────────
    st.session_state.pipeline_log     = log
    st.session_state.current_phase    = "Approval Gate"
    st.session_state.pipeline_running = False
    st.rerun()


# ==========================================
# Phase 1: Inbox & Triage
# ==========================================
if st.session_state.pipeline_running:
    _render_pipeline_execution()
elif st.session_state.current_phase == "Inbox & Triage":
    st.title("📥 Inbox & Triage")
    st.write("Fetch incoming email threads and classify them by priority and category using Gemini.")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        pull_triage = st.button("🔄 Pull & Triage Threads", type="primary", use_container_width=True)
    
    if pull_triage:
        with st.spinner("Fetching and classifying threads..."):
            try:
                raw_threads = []
                if source_selection == "Gmail via engine.py":
                    # Call fetch_threads() from engine.py
                    fetched = engine.fetch_threads(max_results=10)
                    
                    # Convert to required unified format: [{id, subject, messages: [{from, date, body}]}]
                    for f in fetched:
                        raw_threads.append({
                            "id": f.get("thread_id", ""),
                            "subject": f.get("subject", ""),
                            "messages": [
                                {
                                    "from": f.get("sender", ""),
                                    "date": f.get("date", ""),
                                    "body": f.get("snippet", "")
                                }
                            ]
                        })
                else:
                    # Load from sample_threads.json
                    if os.path.exists("sample_threads.json"):
                        with open("sample_threads.json", "r", encoding="utf-8") as f:
                            raw_threads = json.load(f)
                    else:
                        st.error("sample_threads.json not found! Please make sure it was created.")
                
                # We need to triage them using triage_inbox()
                # Include the thread "id" so triage.py can match results back
                # by ID rather than array position (the sort inside triage_inbox
                # would otherwise misalign labels with threads).
                triage_inputs = []
                for t in raw_threads:
                    first_msg = t["messages"][0] if t["messages"] else {}
                    triage_inputs.append({
                        "id":      t.get("id", ""),
                        "sender":  first_msg.get("from", "unknown"),
                        "subject": t.get("subject", ""),
                        "snippet": first_msg.get("body", "")
                    })
                
                # Classify
                triaged_results = triage.triage_inbox(triage_inputs)
                
                # Combine raw_threads with their labels.
                # Match by "id" — never by list position — because triage_inbox
                # sorts results by priority, so positional indexing is wrong.
                label_by_id = {r["id"]: r for r in triaged_results if r.get("id")}
                
                updated_threads = []
                for thread_data in raw_threads:
                    tid = thread_data.get("id", "")
                    label = label_by_id.get(tid)
                    if label is None:
                        # Fallback: no label came back for this thread
                        print(f"[app] WARNING: no triage label for thread id='{tid}' subject='{thread_data.get('subject','')}' — using defaults")
                        thread_data["priority"] = "needs reply"
                        thread_data["category"] = "other"
                        thread_data["reason"]   = "Classification result not returned"
                    else:
                        thread_data["priority"] = label.get("priority", "needs reply")
                        thread_data["category"] = label.get("category", "other")
                        thread_data["reason"]   = label.get("reason", "No reason provided")
                    updated_threads.append(thread_data)
                
                # Save to session state
                st.session_state.threads = updated_threads
                st.session_state.triaged = updated_threads
                st.success(f"Successfully loaded and triaged {len(updated_threads)} email threads!")
                st.rerun()
                
            except Exception as e:
                _show_gemini_error(e)
                
    # Display threads grouped by priority
    if st.session_state.triaged:
        priorities = ["urgent", "needs reply", "fyi", "ignore"]
        
        # Display stat counts
        st.subheader("📬 Triaged Mailbox")
        
        # Group threads
        by_priority = {p: [] for p in priorities}
        for t in st.session_state.triaged:
            p = t.get("priority", "needs reply")
            if p in by_priority:
                by_priority[p].append(t)
            else:
                by_priority["needs reply"].append(t)
                
        # Draw expanders for each priority category
        for p in priorities:
            threads_in_p = by_priority[p]
            count = len(threads_in_p)
            header_text = f"{p.upper()} ({count})"
            
            with st.expander(header_text, expanded=(p in ["urgent", "needs reply"])):
                if not threads_in_p:
                    st.write("*No threads in this category.*")
                else:
                    for t in threads_in_p:
                        # Display individual thread
                        col_hdr, col_badge = st.columns([4, 1])
                        with col_hdr:
                            st.markdown(f"**Subject:** {t.get('subject')}")
                        with col_badge:
                            st.markdown(
                                f'<span class="metadata-badge badge-category">{t.get("category", "other").upper()}</span>',
                                unsafe_allow_html=True
                            )
                        
                        first_msg = t["messages"][0] if t["messages"] else {}
                        st.markdown(f"**From:** {first_msg.get('from')} | **Date:** {first_msg.get('date')}")
                        st.markdown(f"*Reason:* {t.get('reason')}")
                        
                        # Preview content
                        st.markdown(
                            f"""
                            <div class="thread-box" style="border-left: 4px solid #00bbee;">
                                <div class="thread-body">{first_msg.get('body')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.write("---")
    else:
        st.info("Click the **'Pull & Triage Threads'** button above to load your inbox.")

# ==========================================
# Phase 2: Draft Generation
# ==========================================
elif st.session_state.current_phase == "Draft Generation":
    st.title("🤖 Draft Generation")
    st.write("Generate high-quality email replies for actionable threads using the persona rules and context builder.")
    
    if not st.session_state.triaged:
        st.warning("Please go to the 'Inbox & Triage' phase and pull threads first!")
    else:
        # Filter to actionable threads (urgent + needs reply)
        actionable_threads = [
            t for t in st.session_state.triaged 
            if t.get("priority") in ["urgent", "needs reply"]
        ]
        
        if not actionable_threads:
            st.success("🎉 No actionable threads (urgent/needs reply) found! All clear.")
        else:
            st.subheader(f"Actionable Threads requiring Drafts ({len(actionable_threads)})")
            
            for t in actionable_threads:
                tid = t.get("id")
                first_msg = t["messages"][0] if t["messages"] else {}
                
                with st.container():
                    col_det, col_action = st.columns([3, 1])
                    with col_det:
                        st.markdown(f"### {t.get('subject')}")
                        st.markdown(f"**From:** {first_msg.get('from')} | **Priority:** `{t.get('priority').upper()}` | **Category:** `{t.get('category')}`")
                        st.markdown(f"*Snippet:* {first_msg.get('body')[:150]}...")
                    
                    with col_action:
                        # Show current draft status if generated
                        has_draft = tid in st.session_state.drafts
                        has_approved = tid in st.session_state.approved
                        has_rejected = tid in st.session_state.rejected
                        
                        if has_approved:
                            st.success("✅ Approved")
                        elif has_rejected:
                            st.error("❌ Rejected")
                        elif has_draft:
                            st.info("⚡ Draft Ready")
                        else:
                            st.write("*No Draft Generated*")
                            
                        # Generate Draft button
                        if st.button("✨ Generate / Regenerate Reply", key=f"gen_{tid}", use_container_width=True):
                            with st.spinner("Drafting with Gemini..."):
                                try:
                                    # Format the input thread to match draft_machine's expectations
                                    # (thread_id, sender, subject, snippet, date, priority, category, reason)
                                    compat_thread = {
                                        "thread_id": tid,
                                        "sender": first_msg.get("from", ""),
                                        "subject": t.get("subject", ""),
                                        "snippet": first_msg.get("body", ""),
                                        "date": first_msg.get("date", ""),
                                        "priority": t.get("priority", "needs reply"),
                                        "category": t.get("category", "project"),
                                        "reason": t.get("reason", "")
                                    }
                                    
                                    draft_text = draft_machine.draft_reply(compat_thread)
                                    st.session_state.drafts[tid] = draft_text
                                    # Remove from rejected set if regenerated
                                    if tid in st.session_state.rejected:
                                        st.session_state.rejected.remove(tid)
                                    st.success("Draft created!")
                                    st.rerun()
                                except Exception as e:
                                    _show_gemini_error(e)
                                        
                    # Display existing draft
                    if tid in st.session_state.drafts:
                        with st.expander("View Generated Draft", expanded=False):
                            st.markdown(
                                f"""
                                <div class="draft-container">
                                {st.session_state.drafts[tid]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    st.write("---")

# ==========================================
# Phase 3: Approval Gate
# ==========================================
elif st.session_state.current_phase == "Approval Gate":
    st.title("🎛️ Approval Gate")
    st.write("Review, edit, and approve individual draft replies before finalizing.")

    # Pipeline execution log — shown only when the pipeline has been run
    if st.session_state.get("pipeline_log"):
        with st.expander("Pipeline Execution Log", expanded=False):
            for entry in st.session_state.pipeline_log:
                is_failure = "ERROR" in entry or "FAILED" in entry
                prefix = "❌" if is_failure else "✅"
                st.write(f"{prefix} {entry}")
            if st.button("Clear log", key="btn_clear_pipeline_log"):
                st.session_state.pipeline_log = []
                st.rerun()
        st.divider()

    if not st.session_state.drafts:
        st.warning("No drafts have been generated yet! Please go to 'Draft Generation' and create some.")
    else:
        # Load actionable threads that have drafts
        drafted_threads = [
            t for t in st.session_state.triaged
            if t.get("id") in st.session_state.drafts
        ]
        
        if not drafted_threads:
            st.info("No active drafts found.")
        else:
            # Dropdown to select which draft to review
            thread_options = {t["subject"]: t["id"] for t in drafted_threads}
            selected_subject = st.selectbox(
                "Select Draft to Review:",
                options=list(thread_options.keys())
            )
            
            selected_tid = thread_options[selected_subject]
            current_thread = next(t for t in drafted_threads if t["id"] == selected_tid)
            first_msg = current_thread["messages"][0] if current_thread["messages"] else {}
            
            col_left, col_right = st.columns(2)
            
            # Left: Thread History
            with col_left:
                st.subheader("📬 Thread History")
                st.markdown(f"**Subject:** {current_thread.get('subject')}")
                st.markdown(f"**From:** {first_msg.get('from')}")
                st.markdown(
                    f'<span class="metadata-badge badge-priority-{current_thread.get("priority").replace(" ", "-")}">Priority: {current_thread.get("priority").upper()}</span>'
                    f'<span class="metadata-badge badge-category">Category: {current_thread.get("category").upper()}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"*Triage Reason:* {current_thread.get('reason')}")
                st.write("---")
                
                # Messages history
                for m in current_thread.get("messages", []):
                    st.markdown(
                        f"""
                        <div class="thread-box">
                            <span class="thread-sender">{m.get('from')}</span>
                            <span class="thread-date">{m.get('date')}</span>
                            <div class="thread-body">{m.get('body')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            # Right: Draft & Actions
            with col_right:
                st.subheader("🤖 Draft Review")
                
                draft_text = st.session_state.drafts[selected_tid]
                
                # Check status
                is_approved = selected_tid in st.session_state.approved
                is_rejected = selected_tid in st.session_state.rejected
                is_sent = selected_tid in st.session_state.sent

                # Retrieve from approved details if approved
                if is_approved:
                    if is_sent:
                        st.markdown(
                            '<div class="status-approved">📤 Sent</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="status-approved">✅ This draft has been Approved!</div>',
                            unsafe_allow_html=True
                        )
                    approved_draft_body = st.session_state.approved[selected_tid]["draft"]
                    st.markdown(
                        f'<div class="draft-container">{approved_draft_body}</div>',
                        unsafe_allow_html=True
                    )

                    if not is_sent:
                        is_meeting = current_thread.get("category", "").lower() == "meeting"
                        is_booked = selected_tid in st.session_state.booked

                        # Debug: show the raw category and session state keys so mismatches are visible
                        st.caption(
                            f"🔍 Debug — category: `{current_thread.get('category', '(none)')}` | "
                            f"is_meeting: `{is_meeting}` | is_booked: `{is_booked}` | "
                            f"session keys: `{list(st.session_state.keys())}`"
                        )

                        # Calendar engine import check
                        try:
                            cal = _get_calendar_engine()
                            _cal_ok = True
                        except Exception as _cal_err:
                            _cal_ok = False
                            st.error(f"⚠️ calendar_engine import failed: {_cal_err}")

                        # If already booked, show the calendar link in place of the button
                        if is_booked:
                            event_link = st.session_state.booked[selected_tid].get("htmlLink", "")
                            st.success(
                                "📅 Meeting booked!"
                                + (f" [Open in Calendar]({event_link})" if event_link else "")
                            )

                        # Button row: two-column for meeting threads, single for others
                        if is_meeting and not is_booked:
                            _bcol1, _bcol2 = st.columns(2)
                            with _bcol1:
                                _send_clicked = st.button(
                                    "📤 Send Reply",
                                    key=f"send_{selected_tid}",
                                    type="primary",
                                    use_container_width=True,
                                )
                            with _bcol2:
                                _book_clicked = st.button(
                                    "📅 Book Meeting",
                                    key=f"book_{selected_tid}",
                                    use_container_width=True,
                                    disabled=not _cal_ok,
                                )
                        else:
                            _send_clicked = st.button(
                                "📤 Send Reply",
                                key=f"send_{selected_tid}",
                                type="primary",
                                use_container_width=True,
                            )
                            _book_clicked = False

                        # Send handler
                        if _send_clicked:
                            import re as _re
                            raw_from = first_msg.get("from", "")
                            _match = _re.search(r"<(.+?)>", raw_from)
                            recipient = _match.group(1).strip() if _match else raw_from.strip()
                            try:
                                send_fn = _get_send_reply()
                                result = send_fn(
                                    thread_id=selected_tid,
                                    to=recipient,
                                    subject=current_thread.get("subject", ""),
                                    body=approved_draft_body,
                                )
                                st.session_state.sent.add(selected_tid)
                                st.success(f"Reply sent to {recipient}!")
                                if result and result.get("id"):
                                    log_action(
                                        action_type="sent",
                                        thread_subject=current_thread["subject"],
                                        detail=recipient,
                                        action_id=result["id"],
                                    )
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Failed to send: {_e}")

                        # Book Meeting handler
                        if _book_clicked and _cal_ok:
                            with st.spinner("Extracting meeting details…"):
                                try:
                                    parsed = cal.parse_meeting_request(current_thread)
                                except Exception as _e:
                                    parsed = {"parsing_error": str(_e)}

                            # Guard: result must be a dict
                            if not isinstance(parsed, dict):
                                st.error(
                                    f"Could not parse meeting details: unexpected type "
                                    f"{type(parsed).__name__} returned (expected dict). "
                                    f"Raw value: {str(parsed)[:200]}"
                                )
                            elif "parsing_error" in parsed:
                                st.error(parsed['parsing_error'])
                            else:
                                _proposed  = parsed.get("proposed_times", [])
                                _duration  = parsed.get("duration_minutes", 30)
                                _attendees = parsed.get("attendees", [])
                                _topic     = parsed.get("topic") or current_thread.get("subject", "Meeting")

                                st.info(
                                    f"**Topic:** {_topic}  \n"
                                    f"**Proposed times:** {', '.join(_proposed) if _proposed else 'none found'}  \n"
                                    f"**Attendees:** {', '.join(_attendees) if _attendees else 'none found'}  \n"
                                    f"**Duration:** {_duration} min"
                                )

                                if not _proposed:
                                    st.warning("No proposed times found in the thread — cannot book.")
                                else:
                                    _free_slot = None
                                    _avail_error = False
                                    with st.spinner("Checking calendar availability…"):
                                        try:
                                            _free_slot = cal.find_free_slot(_proposed, _duration)
                                        except Exception as _e:
                                            _avail_error = True
                                            st.error(f"Availability check failed: {_e}")

                                    if not _avail_error and _free_slot is None:
                                        st.warning("No free slot found among the proposed times.")
                                    elif _free_slot is not None:
                                        with st.spinner(f"Booking at {_free_slot}…"):
                                            try:
                                                _event = cal.create_event(
                                                    summary=_topic,
                                                    start_time=_free_slot,
                                                    duration_minutes=_duration,
                                                    attendees=_attendees,
                                                    description=approved_draft_body,
                                                )
                                                st.session_state.booked[selected_tid] = _event
                                                _link = _event.get("htmlLink", "")
                                                st.success(
                                                    f"Meeting booked for {_free_slot}!"
                                                    + (f" [Open in Calendar]({_link})" if _link else "")
                                                )
                                                if _event.get("id"):
                                                    log_action(
                                                        action_type="booked",
                                                        thread_subject=current_thread["subject"],
                                                        detail=_topic,
                                                        action_id=_event["id"],
                                                    )
                                                st.rerun()
                                            except Exception as _e:
                                                st.error(f"Failed to create event: {_e}")
                elif is_rejected:
                    st.markdown('<div class="status-rejected">❌ This draft was Rejected. Go to Draft Generation to recreate.</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="opacity: 0.5;" class="draft-container">{draft_text}</div>', unsafe_allow_html=True)
                else:
                    # Editing mode toggle inside session state
                    edit_key = f"editing_mode_{selected_tid}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = False
                        
                    if st.session_state[edit_key]:
                        st.markdown("📝 **Editing Draft**")
                        edited_text = st.text_area(
                            "Modify reply text:",
                            value=draft_text,
                            height=250
                        )
                        
                        btn_s1, btn_s2 = st.columns(2)
                        with btn_s1:
                            if st.button("💾 Save & Approve", type="primary", use_container_width=True):
                                # Save to approved dict
                                st.session_state.approved[selected_tid] = {
                                    "thread_id": selected_tid,
                                    "subject": current_thread.get("subject"),
                                    "sender": first_msg.get("from"),
                                    "draft": edited_text,
                                    "approved_at": datetime.now().isoformat(),
                                    "edited": True
                                }
                                st.session_state.drafts[selected_tid] = edited_text
                                st.session_state[edit_key] = False
                                st.success("Draft approved successfully!")
                                st.rerun()
                        with btn_s2:
                            if st.button("Cancel", use_container_width=True):
                                st.session_state[edit_key] = False
                                st.rerun()
                    else:
                        st.markdown(f'<div class="draft-container">{draft_text}</div>', unsafe_allow_html=True)
                        
                        # Action buttons
                        col_b1, col_b2, col_b3 = st.columns(3)
                        with col_b1:
                            if st.button("👍 APPROVE", use_container_width=True, type="primary"):
                                st.session_state.approved[selected_tid] = {
                                    "thread_id": selected_tid,
                                    "subject": current_thread.get("subject"),
                                    "sender": first_msg.get("from"),
                                    "draft": draft_text,
                                    "approved_at": datetime.now().isoformat(),
                                    "edited": False
                                }
                                st.success("Draft approved!")
                                st.rerun()
                        with col_b2:
                            if st.button("📝 EDIT", use_container_width=True):
                                st.session_state[edit_key] = True
                                st.rerun()
                        with col_b3:
                            if st.button("👎 REJECT", use_container_width=True):
                                st.session_state.rejected.add(selected_tid)
                                # Remove from approved if it was there
                                if selected_tid in st.session_state.approved:
                                    del st.session_state.approved[selected_tid]
                                st.rerun()

# ==========================================
def generate_proof_markdown():
    lines = []
    lines.append("# The Draft Desk – Proof of Work")
    lines.append("")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}*")
    lines.append("")
    for tid, approved_data in st.session_state.approved.items():
        thread = next((t for t in st.session_state.triaged if t.get("id") == tid), {})
        messages = thread.get("messages", [])
        lines.append(f"## Thread: {approved_data.get('subject', 'No Subject')}")
        for msg in messages:
            lines.append(f"> **{msg.get('from', 'unknown')}** ({msg.get('date', 'unknown')}): {msg.get('body', '')}")
        lines.append("")
        lines.append("```")
        lines.append(approved_data.get('draft', ''))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)

def generate_proof_html():
    html = """
    <html>
    <head>
    <style>
    body {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: Arial, sans-serif;
        margin: 20px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    .thread-box {
        border: 2px solid #ff6600;
        padding: 15px;
        border-radius: 8px;
        background-color: #162447;
    }
    .draft-box {
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        background-color: #1f4068;
    }
    blockquote {
        margin: 0;
        padding-left: 10px;
        border-left: 4px solid #ff6600;
        color: #e0e0e0;
    }
    pre {
        background-color: #2d2d44;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    </style>
    </head>
    <body>
    <h1>The Draft Desk – Proof of Work</h1>
    <p><em>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z") + """</em></p>
    """
    for tid, approved_data in st.session_state.approved.items():
        thread = next((t for t in st.session_state.triaged if t.get("id") == tid), {})
        messages = thread.get("messages", [])
        html += "<div class='grid-container'>"
        html += "<div class='thread-box'>"
        for msg in messages:
            html += f"<blockquote><strong>{msg.get('from', 'unknown')}</strong> ({msg.get('date', 'unknown')}): {msg.get('body', '')}</blockquote>"
        html += "</div>"
        html += "<div class='draft-box'>"
        html += f"<pre><code>{approved_data.get('draft', '')}</code></pre>"
        html += "</div>"
        html += "</div>"
    html += "</body></html>"
    return html
# Phase 4: Export Proof
# ==========================================
if st.session_state.current_phase == "Export Proof":
    st.title("📤 Export Proof")
    st.write("Export and save approved draft replies. Approved drafts are recorded in `approved_drafts.json` with timestamps.")
    
    if not st.session_state.approved:
        st.warning("No drafts have been approved yet! Go to 'Approval Gate' to authorize drafts.")
    else:
        st.subheader(f"Approved Drafts ready to be queued ({len(st.session_state.approved)})")
        
        # Side-by-side preview of all approved drafts
        for tid, approved_data in list(st.session_state.approved.items()):
            thread = next((t for t in st.session_state.triaged if t.get("id") == tid), {})
            messages = thread.get("messages", [])
            col_orig, col_draft = st.columns(2)
            with col_orig:
                st.markdown("**📬 Original Thread**")
                for msg in messages:
                    st.markdown(
                        f"""
                        <div class="thread-box" style="border-left: 4px solid #ff6600;">
                            <span class="thread-sender">{msg.get('from', 'unknown')}</span>
                            <span class="thread-date">{msg.get('date', 'unknown')}</span>
                            <div class="thread-body">{msg.get('body', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            with col_draft:
                st.markdown("**🤖 Approved Draft**")
                st.markdown(
                    f"""
                    <div class="draft-container" style="border-color: #4caf50;">
                    {approved_data.get('draft')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Undo approval action
            if st.button("↩️ Revoke Approval", key=f"revoke_{tid}"):
                del st.session_state.approved[tid]
                st.info("Approval revoked.")
                st.rerun()
            st.write("---")
        
        # Download proof buttons
        md_content = generate_proof_markdown()
        html_content = generate_proof_html()
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📄 Download Proof (Markdown)",
                data=md_content,
                file_name="proof_of_work.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                label="🌐 Download Proof (HTML)",
                data=html_content,
                file_name="proof_of_work.html",
                mime="text/html",
                use_container_width=True
            )
                
        # Export button
        if st.button("💾 Export All Approved Drafts to File", type="primary", use_container_width=True):
            try:
                # Load current database list
                if os.path.exists("approved_drafts.json"):
                    with open("approved_drafts.json", "r", encoding="utf-8") as f:
                        all_approved_records = json.load(f)
                else:
                    all_approved_records = []
                
                # Append active session's newly approved drafts (prevent duplicating same thread approvals)
                existing_ids = {rec.get("thread_id") for rec in all_approved_records}
                added_count = 0
                for tid, app_data in st.session_state.approved.items():
                    if tid not in existing_ids:
                        all_approved_records.append(app_data)
                        added_count += 1
                        
                with open("approved_drafts.json", "w", encoding="utf-8") as f:
                    json.dump(all_approved_records, f, indent=4)
                    
                st.success(f"Successfully exported {added_count} new approvals to `approved_drafts.json` (Total records: {len(all_approved_records)})!")
            except Exception as e:
                st.error(f"Error saving to approved_drafts.json: {e}")

        # ── Action Log ────────────────────────────────────────────────────────
        st.write("---")
        st.subheader("Action Log")

        _action_entries = get_action_log()

        if not _action_entries:
            st.info("No actions logged yet.")
        else:
            for _entry in _action_entries:
                _atype = _entry.get("action_type", "")
                _icon  = "📨" if _atype == "sent" else "📅"

                # Parse ISO timestamp → "Jan 01 02:30 PM"
                try:
                    from datetime import timezone as _tz
                    _ts_raw = _entry.get("timestamp", "")
                    _dt = datetime.fromisoformat(_ts_raw)
                    # Convert to local time if the timestamp carries UTC info
                    if _dt.tzinfo is not None:
                        _dt = _dt.astimezone().replace(tzinfo=None)
                    _ts_display = _dt.strftime("%b %d %I:%M %p")
                except Exception:
                    _ts_display = _entry.get("timestamp", "")

                _c1, _c2, _c3, _c4 = st.columns([1, 3, 3, 2])
                with _c1:
                    st.write(f"{_icon} **{_atype.upper()}**")
                with _c2:
                    st.write(f"**{_entry.get('thread_subject', '')}**")
                with _c3:
                    st.write(f"`{_entry.get('detail', '')}`")
                with _c4:
                    st.caption(_ts_display)
