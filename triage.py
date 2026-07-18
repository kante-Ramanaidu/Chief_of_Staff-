import os
import json
import time
import hashlib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Read key from env (local .env) or Streamlit Secrets (Cloud deployment)
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    try:
        _GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        _GEMINI_API_KEY = None

genai.configure(api_key=_GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


# ---------------------------------------------------------------------------
# Error helpers shared across triage functions
# ---------------------------------------------------------------------------

class GeminiError(RuntimeError):
    """
    Raised when a Gemini API call fails after all retries.
    ``friendly`` carries a short, user-facing message.
    The original exception is available via ``__cause__``.
    """
    def __init__(self, friendly: str, original: Exception):
        super().__init__(friendly)
        self.friendly = friendly
        self.__cause__ = original


def _is_transient_triage(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        tag in msg
        for tag in ("503", "500", "unavailable", "overloaded",
                    "high demand", "quota", "rate", "429",
                    "serviceunavailable", "internalservererror")
    )


def _classify_gemini_error(exc: Exception) -> str:
    """Return a short user-facing error string based on the exception text."""
    msg = str(exc).lower()
    if "503" in msg or "unavailable" in msg or "overloaded" in msg or "high demand" in msg:
        return (
            "⏳ Gemini is temporarily busy (free tier). "
            "Please wait a moment and try again."
        )
    if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate" in msg:
        return (
            "⚠️ Daily free-tier quota reached for Gemini. "
            "Please wait for it to reset, or upgrade your plan to continue."
        )
    return "Something went wrong with Gemini. Please try again."


def _make_thread_id(thread: dict, position: int) -> str:
    """
    Return a stable short ID for a thread to use inside the prompt.
    Prefer an existing 'id' field; otherwise derive one from the content.
    The position is appended to guarantee uniqueness across the batch.
    """
    base = thread.get("id") or thread.get("thread_id") or ""
    if not base:
        raw = f"{thread.get('sender','')}{thread.get('subject','')}{position}"
        base = hashlib.md5(raw.encode()).hexdigest()[:8]
    # Keep IDs short and unambiguous inside the prompt
    return f"tid_{position}_{base[:12]}"


def _build_batch_prompt(threads: list, thread_ids: list) -> str:
    """
    Builds a single prompt containing all threads, each labelled with a
    unique thread_id that Gemini must echo back in its response so results
    can be matched by ID — not by array position.
    """
    thread_blocks = []
    for tid, t in zip(thread_ids, threads):
        thread_blocks.append(
            f'thread_id: "{tid}"\n'
            f"  sender: {t['sender']}\n"
            f"  subject: {t['subject']}\n"
            f"  preview: {t['snippet']}"
        )
    joined_threads = "\n\n".join(thread_blocks)

    prompt = f"""
You are an intelligent email assistant helping triage an inbox.
Below is a list of {len(threads)} email threads. Each thread has a unique thread_id.
Classify EACH one and echo back its thread_id in your response so results can be matched correctly.

{joined_threads}

--- CATEGORY DEFINITIONS ---

Use category "meeting" whenever the email contains ANY of these signals, even if the
subject line does not contain the word "meeting":
  • Specific day or time mentioned (e.g. "Monday 11am", "Tuesday 3:30pm", "this afternoon between 2–5pm")
  • Availability language ("I'm free", "works for me", "let's lock in", "does X work for you")
  • Scheduling intent ("let's meet", "set up a call", "quick chat", "sync", "review call",
    "book a time", "catch up", "discuss [topic] on [day]", "hop on a call")
  • Any request to send or receive a calendar invite

Treat these as STRONG signals that override other category signals (support, project, etc.).
A "Design Review Sync", "Q3 Roadmap Sync", "quick chat?", or "Can we talk Tuesday?" should
ALL be classified as "meeting", not "support" or "project".

The distinction to apply:
  • "I'm having trouble with X" → support  (no time/scheduling element)
  • "Can we discuss X on Tuesday at 3pm?" → meeting  (scheduling element present)
  • "Here's the project update for this week" → project  (no scheduling element)
  • "Does Monday 11am work to review the dashboard designs?" → meeting

--- FEW-SHOT EXAMPLES (these are EXAMPLES ONLY — do NOT include them in your output) ---

Example input:
  thread_id: "example_A"
  sender: sofia@designlabs.io
  subject: Q3 Roadmap Sync
  preview: Can we meet Tuesday at 2pm, Wednesday at 10am, or Thursday at 3pm? 45-min call to discuss the Q3 roadmap.
Example output:
  {{"thread_id": "example_A", "priority": "needs reply", "category": "meeting", "reason": "Sender proposes three specific times for a 45-minute roadmap call."}}

Example input:
  thread_id: "example_B"
  sender: boss@company.com
  subject: Re: Performance Review
  preview: Monday at 3pm works perfectly for me. I'll send a calendar invite shortly.
Example output:
  {{"thread_id": "example_B", "priority": "needs reply", "category": "meeting", "reason": "Sender confirms a specific time and plans to send a calendar invite."}}

Example input:
  thread_id: "example_C"
  sender: support@saas-tool.com
  subject: Re: Login issue
  preview: We've identified the bug causing your login failures and are rolling out a fix. No action needed on your end.
Example output:
  {{"thread_id": "example_C", "priority": "fyi", "category": "support", "reason": "Vendor resolving a technical issue; no time or scheduling request present."}}

Example input:
  thread_id: "example_D"
  sender: designer@company.com
  subject: Design Review: Main Dashboard Redesign
  preview: Hi team, the Figma wireframes are done. Are you available either at 11am or 2pm EST next Tuesday for a 30-minute review?
Example output:
  {{"thread_id": "example_D", "priority": "needs reply", "category": "meeting", "reason": "Sender proposes two specific times for a design review session."}}

--- END EXAMPLES ---

Respond with ONLY a valid JSON array (no markdown, no code fences, no extra text).
The array must have exactly {len(threads)} objects — one per thread above, in any order.
Each object MUST include the original thread_id so it can be matched back correctly.
Each object must look like this:
{{"thread_id": "<echo the thread_id exactly>", "priority": "<urgent | needs reply | fyi | ignore>", "category": "<meeting | project | personal | social | spam | followup | newsletter | jobapp | support | finance | other>", "reason": "<one sentence>"}}
"""
    return prompt


def _parse_batch_response(text: str, thread_ids: list) -> dict:
    """
    Parse Gemini's JSON array and return a dict keyed by thread_id.

    Falls back to an 'unknown' entry for any thread_id that is missing
    from the response, and logs a warning for any returned thread_id that
    doesn't match an expected one (which would indicate a hallucination).
    """
    cleaned = text.strip()

    # Strip accidental markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    expected_ids = set(thread_ids)
    results_by_id: dict = {}

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[triage] WARNING: failed to parse Gemini response as JSON:\n{cleaned[:500]}")
        return {}   # caller will fill all gaps with 'unknown'

    for item in parsed:
        tid = item.get("thread_id")
        if tid is None:
            print(f"[triage] WARNING: result item missing thread_id field: {item}")
            continue
        if tid not in expected_ids:
            print(
                f"[triage] WARNING: returned thread_id '{tid}' not in original "
                f"thread list — ignoring to prevent misalignment"
            )
            continue
        results_by_id[tid] = {
            "priority": str(item.get("priority", "unknown")).strip().lower(),
            "category": str(item.get("category", "unknown")).strip().lower(),
            "reason":   str(item.get("reason",   "unknown")).strip(),
        }

    # Warn about any threads Gemini silently dropped
    for tid in thread_ids:
        if tid not in results_by_id:
            print(
                f"[triage] WARNING: no result returned for thread_id '{tid}' "
                f"— will use fallback 'unknown'"
            )

    return results_by_id


def triage_inbox(threads: list) -> list:
    """
    Classifies all threads in a single Gemini API call (batched).
    Results are matched back to threads by thread_id — never by array
    position — so a skipped or reordered item cannot cause misalignment.
    """
    if not threads:
        return []

    # Assign a stable unique ID to each thread for this batch
    thread_ids = [_make_thread_id(t, i) for i, t in enumerate(threads)]

    prompt = _build_batch_prompt(threads, thread_ids)

    # Retry up to 3 times for transient Gemini errors (2 s / 4 s / 8 s backoff).
    # On final failure raise GeminiError with a friendly message so callers
    # can display it in the UI instead of a raw traceback.
    last_exc: Exception = RuntimeError("unknown")
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            break   # success — exit retry loop
        except Exception as exc:
            last_exc = exc
            if _is_transient_triage(exc) and attempt < 2:
                wait = 2 ** (attempt + 1)   # 2 s, 4 s
                print(
                    f"[triage] transient error on attempt {attempt + 1}: "
                    f"{exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"[triage] non-retryable error (attempt {attempt + 1}): "
                    f"{type(exc).__name__}: {exc}"
                )
                raise GeminiError(_classify_gemini_error(exc), exc)
    else:
        # All 3 attempts exhausted
        raise GeminiError(_classify_gemini_error(last_exc), last_exc)

    results_by_id = _parse_batch_response(response.text, thread_ids)

    _fallback = {"priority": "unknown", "category": "unknown", "reason": "no classification returned"}

    triaged = []
    for tid, thread in zip(thread_ids, threads):
        label = results_by_id.get(tid, _fallback)
        if label is _fallback:
            print(
                f"[triage] using fallback for thread_id='{tid}' "
                f"subject='{thread.get('subject', '')}'"
            )
        triaged.append({**thread, **label})

    priority_order = {"urgent": 1, "needs reply": 2, "fyi": 3, "ignore": 4, "unknown": 5}
    triaged.sort(key=lambda x: priority_order.get(x["priority"], 5))
    return triaged


if __name__ == "__main__":
    sample_threads = [
        {"sender": "boss@company.com", "subject": "Urgent Meeting", "snippet": "Let's discuss the quarterly results."},
        {"sender": "colleague@company.com", "subject": "Project Update", "snippet": "Here are the latest developments on the project."},
        {"sender": "manager@company.com", "subject": "Performance Review", "snippet": "Let's schedule a time to discuss your performance."},
    ]

    results = triage_inbox(sample_threads)
    for r in results:
        print(f"[{r['priority']}] {r['subject']} - {r['sender']} ({r['category']}) - {r['reason']}")