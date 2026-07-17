"""
Context Builder Module for Email Reply Drafting Agent

Assembles the full prompt context (Layers 1-3) for an email reply drafting
agent, using REAL data pulled from Gmail plus your persona profile:

  Layer 1 - Thread History   -> engine.get_thread_history()
  Layer 2 - Persona / Tone   -> persona.json
  Layer 3 - Past Replies     -> engine.get_past_replies() (few-shot examples)

The actual AI call (sending this context to Gemini to generate a draft)
is intentionally NOT included here — that's next sprint. This module only
builds the prompt string.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from engine import get_thread_history, get_past_replies

PERSONA_PATH = "persona.json"


@dataclass
class EmailContext:
    """Structured context for a single email thread."""
    thread_id: str
    sender: str
    subject: str
    snippet: str
    date: str
    priority: str
    category: str
    reason: str


# Priority-based response guidelines
PRIORITY_GUIDELINES = {
    "urgent": {
        "response_time": "within 2-4 hours",
        "tone": "direct and action-oriented",
        "description": "This email requires immediate attention. Respond quickly with clear action items."
    },
    "needs reply": {
        "response_time": "within 24 hours",
        "tone": "thoughtful and complete",
        "description": "This email expects a response. Take time to craft a thorough reply."
    },
    "fyi": {
        "response_time": "optional",
        "tone": "acknowledgment if needed",
        "description": "This is informational. Reply only if you have valuable input or need to acknowledge receipt."
    },
    "ignore": {
        "response_time": "none",
        "tone": "no response needed",
        "description": "This email can be safely ignored or archived without response."
    }
}

# Category-specific guidelines
CATEGORY_GUIDELINES = {
    "meeting": {"key_elements": ["availability", "agenda", "confirmation"], "tone": "cooperative and clear"},
    "project": {"key_elements": ["status update", "next steps", "blockers"], "tone": "collaborative and informative"},
    "personal": {"key_elements": ["empathy", "personal touch"], "tone": "warm and genuine"},
    "social": {"key_elements": ["enthusiasm", "accept/decline"], "tone": "friendly and engaging"},
    "spam": {"key_elements": ["none"], "tone": "no response"},
    "followup": {"key_elements": ["status", "commitments", "next actions"], "tone": "accountable and transparent"},
    "newsletter": {"key_elements": ["none"], "tone": "no response needed"},
    "jobapp": {"key_elements": ["professionalism", "gratitude", "next steps"], "tone": "professional and enthusiastic"},
    "support": {"key_elements": ["issue details", "urgency"], "tone": "clear and factual"},
    "finance": {"key_elements": ["verification", "deadline"], "tone": "cautious and thorough"},
    "other": {"key_elements": ["context", "clarity"], "tone": "professional and clear"},
}


def load_persona(path: str = PERSONA_PATH) -> Dict[str, Any]:
    """
    Layer 2 - Loads your real tone/persona profile from persona.json.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Safe fallback so the pipeline doesn't crash if persona.json is missing
        return {
            "name": "User",
            "tone": "professional",
            "sentence_length": "medium",
            "greeting_style": "Hi [Name],",
            "sign_off": "Best,",
            "quirks": []
        }


def _format_persona_block(persona: Dict[str, Any]) -> str:
    lines = [
        f"You are drafting this email as: {persona.get('name', 'User')}",
        f"Tone: {persona.get('tone', 'professional')}",
        f"Formality: {persona.get('formality', 'semi-formal')}",
        f"Sentence length: {persona.get('sentence_length', 'medium')}",
        f"Greeting style: {persona.get('greeting_style', 'Hi [Name],')}",
        f"Sign-off: {persona.get('sign_off', 'Best,')}",
    ]
    quirks = persona.get("quirks", [])
    if quirks:
        lines.append("Writing quirks to follow:")
        for q in quirks:
            lines.append(f"  - {q}")
    return "\n".join(lines)


def _format_thread_history_block(history: List[Dict[str, str]]) -> str:
    """
    Layer 1 - Formats the full back-and-forth of a thread into readable text.
    """
    if not history:
        return "(No prior message history available for this thread.)"

    lines = []
    for msg in history:
        lines.append(f"From: {msg.get('sender', 'Unknown')} ({msg.get('date', 'Unknown date')})")
        lines.append(msg.get("content", "").strip())
        lines.append("-" * 40)
    return "\n".join(lines)


def _format_past_replies_block(replies: List[Dict[str, str]]) -> str:
    """
    Layer 3 - Formats 2-3 of your real sent replies as few-shot examples.
    """
    if not replies:
        return "(No past reply examples available.)"

    lines = []
    for i, r in enumerate(replies, 1):
        lines.append(f"Example {i} — Subject: {r.get('subject', '')}")
        lines.append(r.get("content", "").strip())
        lines.append("-" * 40)
    return "\n".join(lines)


def build_reply_context(thread: Dict[str, Any], num_examples: int = 3) -> str:
    """
    Assembles the FULL prompt context for drafting a reply to one thread,
    combining all 3 layers:

        Layer 1: real thread history (fetched live from Gmail)
        Layer 2: your persona/tone profile (persona.json)
        Layer 3: 2-3 of your real past sent replies (fetched live from Gmail)

    Args:
        thread: a triaged thread dict (thread_id, sender, subject, snippet,
                date, priority, category, reason)
        num_examples: how many past sent replies to pull as few-shot examples

    Returns:
        A single formatted prompt string, ready to be sent to an LLM
        in a future sprint.
    """
    email_ctx = EmailContext(
        thread_id=thread.get("thread_id", ""),
        sender=thread.get("sender", ""),
        subject=thread.get("subject", ""),
        snippet=thread.get("snippet", ""),
        date=thread.get("date", ""),
        priority=thread.get("priority", "unknown"),
        category=thread.get("category", "other"),
        reason=thread.get("reason", "")
    )

    priority_guide = PRIORITY_GUIDELINES.get(email_ctx.priority, PRIORITY_GUIDELINES["needs reply"])
    category_guide = CATEGORY_GUIDELINES.get(email_ctx.category, CATEGORY_GUIDELINES["other"])

    persona = load_persona()

    # Layer 1: pull real thread history from Gmail
    thread_history = get_thread_history(email_ctx.thread_id) if email_ctx.thread_id else []

    # Layer 3: pull real past sent replies from Gmail
    past_replies = get_past_replies(limit=num_examples)

    parts = []

    parts.append("=" * 70)
    parts.append("EMAIL REPLY DRAFTING ASSISTANT")
    parts.append("=" * 70)
    parts.append("")

    parts.append("-" * 70)
    parts.append("LAYER 2: YOUR TONE / PERSONA")
    parts.append("-" * 70)
    parts.append(_format_persona_block(persona))
    parts.append("")

    parts.append("-" * 70)
    parts.append("LAYER 3: EXAMPLES OF YOUR REAL PAST REPLIES (write like these)")
    parts.append("-" * 70)
    parts.append(_format_past_replies_block(past_replies))
    parts.append("")

    parts.append("-" * 70)
    parts.append("EMAIL CLASSIFICATION")
    parts.append("-" * 70)
    parts.append(f"Priority: {email_ctx.priority.upper()}")
    parts.append(f"Category: {email_ctx.category}")
    parts.append(f"Reason: {email_ctx.reason}")
    parts.append(f"Recommended tone for this category: {category_guide['tone']}")
    parts.append(f"Key elements to include: {', '.join(category_guide['key_elements'])}")
    parts.append(f"Response urgency: {priority_guide['response_time']}")
    parts.append("")

    parts.append("-" * 70)
    parts.append("LAYER 1: FULL THREAD HISTORY")
    parts.append("-" * 70)
    parts.append(_format_thread_history_block(thread_history))
    parts.append("")

    parts.append("-" * 70)
    parts.append("TASK")
    parts.append("-" * 70)
    parts.append("Draft a reply to the most recent message in this thread.")
    parts.append("Write in the exact tone/style described in LAYER 2, using the")
    parts.append("examples in LAYER 3 as a guide for real voice and phrasing.")
    parts.append("Address the content from LAYER 1 directly.")
    parts.append("=" * 70)

    return "\n".join(parts)


if __name__ == "__main__":
    # Quick manual test using a fake triaged thread + real Gmail data
    sample_thread = {
        "thread_id": "19f46e12b6ee984f",
        "sender": "boss@company.com",
        "subject": "Quick question about the Q3 deck",
        "snippet": "Do you have the latest version?",
        "date": "2026-07-08",
        "priority": "needs reply",
        "category": "project",
        "reason": "Colleague needs information to proceed with work"
    }

    print("NOTE: replace thread_id above with a real one from fetch_threads() to test Layer 1 fully.\n")
    context = build_reply_context(sample_thread)
    print(context)