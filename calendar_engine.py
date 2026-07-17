"""
Calendar engine module for reading and writing Google Calendar events.
Uses the same OAuth credentials as engine.py (credentials.json / token.json).
"""

import os
import re
import json
import socket
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# IPv4 monkey-patch
# Forces the Google API client to prefer IPv4 on networks where IPv6
# connectivity causes slow or failed connections.
# ---------------------------------------------------------------------------
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

# ---------------------------------------------------------------------------
# OAuth configuration — shared with engine.py
# All three scopes must be present in token.json.  If token.json was
# created before the calendar scope was added, delete it and re-auth.
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"

# IST timezone constant — used as the default local timezone for naive datetimes
_IST = timezone(timedelta(hours=5, minutes=30), "IST")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _check_token_has_calendar_scope() -> bool:
    """
    Return True if the stored token.json grants the calendar scope.
    Returns False (does not raise) if the file is missing or unparseable.
    """
    if not os.path.exists(TOKEN_PATH):
        return False
    try:
        with open(TOKEN_PATH, "r") as f:
            data = json.load(f)
        scopes_in_token = data.get("scopes", [])
        return any("calendar" in s for s in scopes_in_token)
    except Exception:
        return False


def _build_calendar_service():
    """
    Return an authenticated Google Calendar v3 service object.

    Shares credentials.json and token.json with engine.py.
    Raises RuntimeError with a clear message if the stored token lacks the
    calendar scope — prompting the user to delete token.json and re-auth.
    """
    # Warn early if the existing token is missing the calendar scope
    if os.path.exists(TOKEN_PATH) and not _check_token_has_calendar_scope():
        raise RuntimeError(
            "token.json does not contain the calendar scope. "
            "Please delete token.json and restart the app to re-authenticate "
            "with all required permissions (gmail.readonly, gmail.send, calendar)."
        )

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Timezone conversion helper
# ---------------------------------------------------------------------------

def _to_utc_rfc3339(dt_str: str, local_tz: timezone = _IST) -> str:
    """
    Convert an ISO-8601 datetime string to a UTC RFC 3339 string (ending in Z).

    Rules:
    - Ends with "Z"        → already UTC, normalise and return.
    - Has explicit offset  → parse with fromisoformat, convert to UTC.
    - Naive (no tz info)   → treat as local_tz, convert to UTC.
    - Date-only string     → treat as midnight in local_tz, convert to UTC.

    Raises ValueError for unparseable input so callers can skip gracefully.
    """
    clean = dt_str.strip()

    if clean.endswith("Z"):
        # Already UTC
        utc_dt = datetime.fromisoformat(clean[:-1]).replace(tzinfo=timezone.utc)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Detect explicit UTC offset (+hh:mm or -hh:mm after the time portion)
    has_explicit_offset = False
    if "T" in clean:
        time_part = clean.split("T", 1)[1]
        # "+" anywhere after T, or a "-" that isn't part of the time digits
        if "+" in time_part or re.search(r"-\d{2}:\d{2}$", time_part):
            has_explicit_offset = True

    if has_explicit_offset:
        aware_dt = datetime.fromisoformat(clean)
        utc_dt = aware_dt.astimezone(timezone.utc)
    else:
        # Naive — may be date-only ("2026-07-15") or datetime ("2026-07-15T14:00:00")
        if "T" not in clean:
            # Date-only: treat as midnight local time
            naive = datetime.fromisoformat(clean + "T00:00:00")
        else:
            naive = datetime.fromisoformat(clean)
        local_dt = naive.replace(tzinfo=local_tz)
        utc_dt = local_dt.astimezone(timezone.utc)
        print(
            f"[_to_utc_rfc3339] naive '{clean}' → "
            f"{local_tz} → UTC {utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Meeting request parser
# ---------------------------------------------------------------------------

# Errors that are worth retrying (transient / rate-limit / server-side)
_TRANSIENT_ERROR_FRAGMENTS = (
    "503", "500", "UNAVAILABLE", "overloaded", "quota", "rate",
    "ServiceUnavailable", "InternalServerError",
)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(fragment.lower() in msg for fragment in _TRANSIENT_ERROR_FRAGMENTS)


def _friendly_gemini_error(exc: Exception) -> str:
    """
    Return a short user-facing message for a Gemini API exception.
    Full exception details are always printed to the console by the caller.
    """
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


def _extract_emails_from_thread(thread: Dict[str, Any]) -> List[str]:
    """
    Pull real email addresses from the thread's message From/To/Cc headers
    so we never rely on Gemini guessing addresses from body text.

    Returns a de-duplicated list of email strings, preserving insertion order.
    """
    seen: dict = {}  # use dict for ordered dedup
    for msg in thread.get("messages", []):
        for field in ("from", "to", "cc"):
            raw = msg.get(field, "")
            if not raw:
                continue
            # Extract all <email> patterns, then fall back to bare addresses
            found = re.findall(r"<([^>]+)>", raw)
            if not found:
                # bare "email@domain" with no angle brackets
                for part in raw.split(","):
                    part = part.strip()
                    if "@" in part:
                        found.append(part)
            for addr in found:
                addr = addr.strip().lower()
                if "@" in addr:
                    seen[addr] = None
    return list(seen.keys())


def parse_meeting_request(thread: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Gemini (gemini-2.5-flash) to extract meeting details from an email thread.

    Returns a dict with keys:
        proposed_times   – list of ISO-8601 datetime strings (local/naive)
        attendees        – list of verified email address strings
        topic            – one-line meeting summary
        duration_minutes – int (default 30)

    On unrecoverable failure returns {"parsing_error": "<description>"}.
    Retries up to 3 times (2 s / 4 s / 8 s) for transient errors only.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"parsing_error": "GEMINI_API_KEY not set in environment"}

    # --- Build transcript ---
    transcript_parts = []
    for msg in thread.get("messages", []):
        header = f"From: {msg.get('from', 'unknown')}\nDate: {msg.get('date', 'unknown')}"
        # Include To/Cc so Gemini sees all participants in context
        if msg.get("to"):
            header += f"\nTo: {msg['to']}"
        if msg.get("cc"):
            header += f"\nCc: {msg['cc']}"
        body = msg.get("body", "").strip()
        transcript_parts.append(f"{header}\n\n{body}")
    transcript = "\n\n---\n\n".join(transcript_parts)

    today = date.today().isoformat()

    system_instruction = (
        "You are a scheduling assistant. "
        "Extract meeting details from the email thread and return ONLY a single "
        "valid JSON object — not a list or array, not wrapped in markdown, "
        "no code fences, no commentary, no text before or after. "
        "The response MUST start with '{' and end with '}'. "
        "The object must have exactly these four keys:\n"
        '  "proposed_times"   – array of ISO-8601 datetime strings '
        '(e.g. ["2026-07-15T14:00:00"]).  Use the date shown in the email headers '
        "to resolve relative day names like 'Tuesday' or 'tomorrow'.\n"
        '  "attendees"        – array of email address strings extracted ONLY from '
        "the From/To/Cc header lines in the transcript, never from the body or "
        "signature. If no email addresses appear in the headers, return [].\n"
        '  "topic"            – string, one-line summary of the meeting purpose.\n'
        '  "duration_minutes" – integer, meeting length in minutes. '
        "Default 30 if not stated.\n"
        "If a field cannot be determined, use [] for arrays, "
        '"" for topic, 30 for duration_minutes. '
        "Do NOT wrap the object in an outer array."
    )

    user_prompt = (
        f"Today's date is {today}. "
        "Resolve any relative day names (e.g. 'tomorrow', 'Tuesday') "
        "relative to this date.\n\n"
        f"Email thread:\n\n{transcript}"
    )

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )
            raw = response.text.strip()

            # Strip markdown code fences if Gemini wraps the JSON anyway
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()

            print(f"[parse_meeting_request] attempt {attempt + 1} raw:\n{raw}")

            parsed = json.loads(raw)
            print(f"[parse_meeting_request] parsed type: {type(parsed).__name__}")

            # Unwrap accidental list wrapping
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    print("[parse_meeting_request] list detected — taking first element")
                    parsed = parsed[0]
                else:
                    return {
                        "parsing_error":
                        f"Gemini returned a list with no dict inside: {raw[:200]}"
                    }

            if not isinstance(parsed, dict):
                return {
                    "parsing_error":
                    f"Unexpected type {type(parsed).__name__} from Gemini: {raw[:200]}"
                }

            # Supplement Gemini's attendee list with real addresses from headers
            header_emails = _extract_emails_from_thread(thread)
            gemini_attendees = [
                a.strip().lower()
                for a in parsed.get("attendees", [])
                if "@" in str(a)
            ]
            # Merge: header emails first (authoritative), then anything Gemini added
            merged_attendees = list({**{e: None for e in header_emails},
                                     **{e: None for e in gemini_attendees}}.keys())

            return {
                "proposed_times":   parsed.get("proposed_times", []),
                "attendees":        merged_attendees,
                "topic":            parsed.get("topic", ""),
                "duration_minutes": int(parsed.get("duration_minutes", 30)),
            }

        except Exception as exc:
            last_exc = exc
            if _is_transient(exc) and attempt < 2:
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                print(
                    f"[parse_meeting_request] transient error on attempt "
                    f"{attempt + 1}: {exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                # Non-transient or final attempt — log raw error, return friendly msg
                print(
                    f"[parse_meeting_request] non-retryable error: "
                    f"{type(exc).__name__}: {exc}"
                )
                return {"parsing_error": _friendly_gemini_error(exc)}

    return {"parsing_error": _friendly_gemini_error(last_exc)}


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def check_availability(
    time_min: str,
    time_max: str,
    local_tz: timezone = _IST,
) -> bool:
    """
    Query the FreeBusy API and return True if the window is free.

    Both timestamps are normalised to UTC via _to_utc_rfc3339 so naive
    IST times are not mistakenly sent as UTC (a 5:30 shift).

    Logs the exact UTC window sent, the raw busy list returned, and any
    exceptions with full tracebacks instead of silently returning False.
    """
    try:
        utc_min = _to_utc_rfc3339(time_min, local_tz)
        utc_max = _to_utc_rfc3339(time_max, local_tz)

        print(f"[check_availability] FreeBusy query: {utc_min} → {utc_max}")

        service = _build_calendar_service()
        result = service.freebusy().query(body={
            "timeMin": utc_min,
            "timeMax": utc_max,
            "items": [{"id": "primary"}],
        }).execute()

        busy_slots = (
            result.get("calendars", {}).get("primary", {}).get("busy", [])
        )
        print(f"[check_availability] busy slots: {busy_slots}")
        is_free = len(busy_slots) == 0
        print(f"[check_availability] → {'FREE' if is_free else 'BUSY'}")
        return is_free

    except RuntimeError as exc:
        # Re-raise scope/auth errors immediately — returning False here would
        # silently mark every slot as busy and hide the real problem.
        raise
    except Exception as exc:
        print(
            f"[check_availability] ERROR — {type(exc).__name__}: {exc}\n"
            + traceback.format_exc()
        )
        return False


def find_free_slot(
    proposed_times: List[str],
    duration_minutes: int = 30,
    local_tz: timezone = _IST,
) -> Optional[str]:
    """
    Return the first proposed start time at which the primary calendar is free,
    or None if no slot is available.

    Naive datetime strings are treated as local_tz (default IST / UTC+5:30)
    and converted to UTC before querying — so "14:00" means 14:00 IST,
    not 14:00 UTC.

    Malformed or date-only strings are skipped with a logged warning.
    """
    print(
        f"[find_free_slot] {len(proposed_times)} slot(s), "
        f"duration={duration_minutes} min, tz={local_tz}"
    )

    for start_str in proposed_times:
        try:
            utc_min = _to_utc_rfc3339(start_str, local_tz)
            utc_start_dt = datetime.fromisoformat(utc_min[:-1]).replace(
                tzinfo=timezone.utc
            )
            utc_end_dt = utc_start_dt + timedelta(minutes=duration_minutes)
            utc_max = utc_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            print(
                f"[find_free_slot] '{start_str}' → UTC window: {utc_min} – {utc_max}"
            )

            # Times are already UTC — pass timezone.utc to skip redundant conversion
            if check_availability(utc_min, utc_max, local_tz=timezone.utc):
                print(f"[find_free_slot] ✓ free slot: '{start_str}' (returning original local string to caller)")
                return start_str

        except RuntimeError:
            # Auth/scope errors — propagate immediately rather than skipping the slot
            raise
        except Exception as exc:
            print(
                f"[find_free_slot] skipping '{start_str}': "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    print("[find_free_slot] no free slot found")
    return None


# ---------------------------------------------------------------------------
# Event creation
# ---------------------------------------------------------------------------

def create_event(
    summary: str,
    start_time: str,
    duration_minutes: int,
    attendees: List[str],
    description: str = "",
    local_tz: timezone = _IST,
) -> Dict[str, Any]:
    """
    Create a Google Calendar event on the primary calendar and email invites
    to all valid attendees.

    start_time is treated as a naive local IST datetime string
    (e.g. "2026-07-18T14:00:00") and passed directly to the Google Calendar
    API with timeZone="Asia/Kolkata".  The API performs the IST→UTC conversion
    internally, so Python never shifts the time and there is no risk of a
    double-conversion.

    Attendees without "@" in their address are silently dropped.

    Returns the full event resource dict from the Calendar API.
    """
    # Guard: never create an event with a blank title.
    safe_summary = summary.strip() if summary and summary.strip() else "Meeting"

    # ── Parse start_time as a local IST naive datetime ───────────────────
    # Strip any trailing Z or offset so fromisoformat can handle it as naive.
    clean_start = start_time.strip()
    if clean_start.endswith("Z"):
        clean_start = clean_start[:-1]
    # Also strip an explicit +HH:MM or -HH:MM offset if present
    clean_start = re.sub(r"[+-]\d{2}:\d{2}$", "", clean_start)

    print(f"[create_event] raw start_time input : '{start_time}'")
    print(f"[create_event] cleaned local IST str: '{clean_start}'")

    local_start_dt = datetime.fromisoformat(clean_start)
    local_end_dt   = local_start_dt + timedelta(minutes=duration_minutes)

    # Format as "YYYY-MM-DDTHH:MM:SS" — no Z, no offset; timeZone field carries the zone.
    start_str = local_start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_str   = local_end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[create_event] local IST window    : {start_str} → {end_str} (Asia/Kolkata)")
    print(f"[create_event] summary             : '{safe_summary}'")

    valid_attendees = [
        {"email": addr.strip()}
        for addr in attendees
        if "@" in str(addr)
    ]

    event_body: Dict[str, Any] = {
        "summary":     safe_summary,
        "description": description,
        "start": {"dateTime": start_str, "timeZone": "Asia/Kolkata"},
        "end":   {"dateTime": end_str,   "timeZone": "Asia/Kolkata"},
    }
    if valid_attendees:
        event_body["attendees"] = valid_attendees

    print(
        f"[create_event] API payload start: {event_body['start']} "
        f"| attendees={[a['email'] for a in valid_attendees]}"
    )

    service = _build_calendar_service()
    created = service.events().insert(
        calendarId="primary",
        body=event_body,
        sendUpdates="all",
    ).execute()

    print(
        f"[create_event] created → id={created.get('id')} "
        f"start={created.get('start')} link={created.get('htmlLink')}"
    )
    return created
