"""
Draft Machine Module for Email Reply Generation

Generates email reply drafts using Gemini (gemini-2.5-flash) with context
assembled from context_builder.py. Applies strict drafting rules to ensure
concise, actionable replies.
"""

import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

from google import genai
from google.genai import types
from context_builder import build_reply_context

# Load environment variables
load_dotenv()

# Sample Threads for the Streamlit approval gate or testing
SAMPLE_THREADS = [
    {
        "thread_id": "sample_q3_deck",
        "sender": "teammate@company.com",
        "subject": "Quick question about the Q3 deck",
        "snippet": "Do you have the latest version?",
        "date": "2026-07-08",
        "priority": "needs reply",
        "category": "project",
        "reason": "Colleague needs information to proceed with work",
        "history": [
            {
                "sender": "teammate@company.com",
                "date": "2026-07-08 09:30 AM",
                "content": "Hey, do you have the latest version of the Q3 budget deck? I need to update the slides for the meeting tomorrow."
            }
        ]
    },
    {
        "thread_id": "sample_urgent_meeting",
        "sender": "boss@company.com",
        "subject": "Urgent: Q3 Budget Review",
        "snippet": "Can we meet today at 3 PM to review?",
        "date": "2026-07-09",
        "priority": "urgent",
        "category": "meeting",
        "reason": "Manager requested an urgent review of quarterly results",
        "history": [
            {
                "sender": "boss@company.com",
                "date": "2026-07-09 10:15 AM",
                "content": "Hi there, can we meet today at 3 PM to review the final numbers for Q3? Let me know if you are free."
            }
        ]
    },
    {
        "thread_id": "sample_newsletter_fyi",
        "sender": "newsletter@techcrunch.com",
        "subject": "TechCrunch Daily: AI Revolution",
        "snippet": "The latest updates on AI agents and model performance.",
        "date": "2026-07-10",
        "priority": "fyi",
        "category": "newsletter",
        "reason": "General news subscription",
        "history": [
            {
                "sender": "newsletter@techcrunch.com",
                "date": "2026-07-10 08:00 AM",
                "content": "Welcome to TechCrunch Daily! Today we're covering the rise of the Human in the Loop workflow in AI Agent development. Let us know what you think!"
            }
        ]
    }
]

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file")

# Initialize the client
MODEL_NAME = "gemini-2.5-flash"
client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Error helpers
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


def _is_transient(exc: Exception) -> bool:
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


# Drafting rules to append to the prompt
DRAFTING_RULES = """
DRAFTING RULES - FOLLOW THESE EXACTLY:

1. ONE-ASK RULE: Every email must have exactly ONE clear question OR ONE clear response. Do not ask multiple questions or provide multiple unrelated responses.

2. LENGTH CONTROL: Match the energy of the thread. Maximum 5 sentences. Use numbered points if needed for clarity.

3. NO AI FILLER: Never use phrases like:
   - "I hope this finds you well"
   - "Thank you for reaching out"
   - "I wanted to touch base"
   - "Just circling back"
   - Any other generic AI-sounding openers

4. STRUCTURE: Follow this exact structure:
   - Acknowledge briefly (1 sentence max)
   - Give your response or answer (2-4 sentences)
   - ONE clear next step (1 sentence)

5. TONE: Match the persona and tone from the context above. Be direct, specific, and human.

6. OUTPUT FORMAT: Return ONLY the draft email text. No subject line, no explanations, no markdown formatting, no quotes around the text.
"""


def draft_reply(thread: Dict[str, Any]) -> str:
    """
    Generate a reply draft for an email thread using Gemini.
    
    Args:
        thread: A triaged thread dict containing:
            - thread_id: Gmail thread ID
            - sender: Email sender
            - subject: Email subject
            - snippet: Email snippet
            - date: Email date
            - priority: Priority level (urgent, needs reply, fyi, ignore)
            - category: Email category
            - reason: Reason for classification
    
    Returns:
        str: The generated draft text (no subject line, no explanation)
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your .env file.\n"
            "Create a .env file with: GEMINI_API_KEY=your_key_here"
        )
    
    # Get the base context from context_builder
    base_context = build_reply_context(thread)
    
    # Combine base context with drafting rules
    full_prompt = f"{base_context}\n\n{DRAFTING_RULES}"
    
    # Call Gemini to generate the draft.
    # Retry up to 3 times for transient errors (2 s / 4 s backoff).
    # On final failure raise GeminiError with a friendly message so callers
    # can display it in the UI instead of a raw traceback.
    last_exc: Exception = RuntimeError("unknown")
    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt
            )
            break   # success — exit retry loop
        except Exception as exc:
            last_exc = exc
            if _is_transient(exc) and attempt < 2:
                wait = 2 ** (attempt + 1)   # 2 s, 4 s
                print(
                    f"[draft_machine] transient error on attempt {attempt + 1}: "
                    f"{exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"[draft_machine] non-retryable error (attempt {attempt + 1}): "
                    f"{type(exc).__name__}: {exc}"
                )
                raise GeminiError(_classify_gemini_error(exc), exc)
    else:
        # All 3 attempts exhausted
        raise GeminiError(_classify_gemini_error(last_exc), last_exc)
    
    # Extract and clean the draft text
    draft_text = response.text.strip()
    
    # Remove any markdown formatting or quotes if present
    draft_text = draft_text.strip('"').strip("'")
    if draft_text.startswith("```"):
        # Remove code block markers if present
        lines = draft_text.split('\n')
        draft_text = '\n'.join(lines[1:]) if len(lines) > 1 else draft_text
        if draft_text.endswith("```"):
            draft_text = draft_text[:-3].strip()
    
    return draft_text


def draft_reply_with_metadata(thread: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a reply draft with metadata for tracking and logging.
    
    Args:
        thread: A triaged thread dict (same as draft_reply)
    
    Returns:
        dict: Contains:
            - draft: The generated draft text
            - model: Model name used (gemini-2.5-flash)
            - subject: Thread subject
            - reply_to: Who we're replying to (sender email)
    """
    draft_text = draft_reply(thread)
    
    return {
        "draft": draft_text,
        "model": MODEL_NAME,
        "subject": thread.get("subject", ""),
        "reply_to": thread.get("sender", "")
    }


if __name__ == "__main__":
    # Demo using Q3 Budget Review thread
    sample_thread = {
        "thread_id": "19f46e12b6ee984f",
        "sender": "teammate@company.com",
        "subject": "Quick question about the Q3 deck",
        "snippet": "Do you have the latest version?",
        "date": "2026-07-08",
        "priority": "needs reply",
        "category": "project",
        "reason": "Colleague needs information to proceed with work"
    }
    
    print("=" * 70)
    print("DRAFT MACHINE - EMAIL REPLY GENERATOR")
    print("=" * 70)
    print()
    
    # Check for API key
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found!")
        print()
        print("Please create a .env file in the same directory with:")
        print("GEMINI_API_KEY=your_actual_api_key_here")
        print()
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        exit(1)
    
    print(f"Thread: {sample_thread['subject']}")
    print(f"From: {sample_thread['sender']}")
    print(f"Priority: {sample_thread['priority']}")
    print(f"Category: {sample_thread['category']}")
    print()
    print("-" * 70)
    print("GENERATING DRAFT...")
    print("-" * 70)
    print()
    
    try:
        # Generate draft with metadata
        result = draft_reply_with_metadata(sample_thread)
        
        # Display results
        print("GENERATED DRAFT:")
        print("-" * 70)
        print(result["draft"])
        print("-" * 70)
        print()
        print("METADATA:")
        print(f"  Model: {result['model']}")
        print(f"  Subject: {result['subject']}")
        print(f"  Reply To: {result['reply_to']}")
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR generating draft: {e}")
        print()
        print("Please check:")
        print("1. Your GEMINI_API_KEY is valid")
        print("2. You have internet connectivity")
        print("3. The google-generativeai package is installed (pip install google-generativeai)")
        exit(1)