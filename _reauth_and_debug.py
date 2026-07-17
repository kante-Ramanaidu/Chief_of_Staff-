"""
Run this script once from a terminal:
    python _reauth_and_debug.py

It will:
  1. Delete token.json so the next OAuth flow requests all three scopes.
  2. Open a browser window — sign in and GRANT calendar access when prompted.
  3. Save the new token.json with all three scopes.
  4. Immediately run the FreeBusy check for the three proposed times and
     print the exact timeMin/timeMax sent and the raw API response.

Delete this file after you're done.
"""

import os
import sys
import json
import traceback
from datetime import datetime, timedelta, timezone

TOKEN_PATH = "token.json"

# ── Step 1: force re-auth ────────────────────────────────────────────────────
if os.path.exists(TOKEN_PATH):
    os.remove(TOKEN_PATH)
    print(f"[setup] Deleted {TOKEN_PATH} — will re-authenticate now.\n")

# Import after deleting token so _build_calendar_service does a fresh flow
sys.path.insert(0, ".")
from calendar_engine import _build_calendar_service, _to_utc_rfc3339, _IST

print("[setup] Starting OAuth flow — a browser window will open.")
print("[setup] Make sure you tick Google Calendar when granting permissions.\n")
try:
    service = _build_calendar_service()
    print("[setup] ✓ Authenticated. Checking scopes in new token.json:")
    with open(TOKEN_PATH) as f:
        tok = json.load(f)
    print("         scopes:", tok.get("scopes", tok.get("scope", "NOT FOUND")))
    print()
except Exception as e:
    print(f"[setup] AUTH FAILED: {e}")
    sys.exit(1)

# ── Step 2: FreeBusy diagnostic ──────────────────────────────────────────────
PROPOSED_TIMES = [
    "2026-07-20T11:00:00",
    "2026-07-21T15:30:00",
    "2026-07-17T09:00:00",
]
DURATION_MINUTES = 30

print("=" * 65)
print("FreeBusy diagnostic")
print("=" * 65)

# Also print calendar's own timezone so we know how Google stores events
try:
    cal_tz = service.settings().get(setting="timezone").execute().get("value", "unknown")
    print(f"Your Google Calendar timezone: {cal_tz}\n")
except Exception:
    cal_tz = "unknown"
    print("Could not fetch calendar timezone.\n")

for start_str in PROPOSED_TIMES:
    print(f"── Input: {start_str}  (treating as IST / UTC+5:30)")

    # Convert
    utc_min = _to_utc_rfc3339(start_str, _IST)
    utc_start_dt = datetime.fromisoformat(utc_min[:-1]).replace(tzinfo=timezone.utc)
    utc_end_dt   = utc_start_dt + timedelta(minutes=DURATION_MINUTES)
    utc_max      = utc_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"   timeMin → {utc_min}")
    print(f"   timeMax → {utc_max}")

    try:
        raw = service.freebusy().query(body={
            "timeMin": utc_min,
            "timeMax": utc_max,
            "items": [{"id": "primary"}],
        }).execute()

        primary    = raw.get("calendars", {}).get("primary", {})
        busy_slots = primary.get("busy",   [])
        errors     = primary.get("errors", [])

        print(f"   raw busy  : {busy_slots}")
        print(f"   raw errors: {errors}")

        if errors:
            print(f"   check_availability → False  (API returned errors)")
        elif busy_slots:
            print(f"   check_availability → False  (calendar busy: {busy_slots})")
        else:
            print(f"   check_availability → True   ✓ FREE")

    except Exception as exc:
        print(f"   API call FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print()

print("=" * 65)
