"""
task_logger.py
--------------
Lightweight action log backed by action_log.json in the project root.

Records every sent email and booked calendar event so the app can display
a history of what the assistant has done.

Log file location: action_log.json  (same directory as this module)
"""

import json
import os
from datetime import datetime, timezone

# Resolve the log file path relative to this module, not the cwd.
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_log.json")

_VALID_ACTION_TYPES = {"sent", "booked"}


def log_action(
    action_type: str,
    thread_subject: str,
    detail: str,
    action_id: str,
) -> dict:
    """Append one record to action_log.json and return the record.

    Parameters
    ----------
    action_type   : "sent"   – an email was sent (detail = recipient address)
                    "booked" – a meeting was created (detail = meeting title)
    thread_subject: subject line of the originating email thread
    detail        : recipient email  (action_type == "sent")
                    meeting title    (action_type == "booked")
    action_id     : Gmail message_id or Google Calendar event_id
    """
    if action_type not in _VALID_ACTION_TYPES:
        raise ValueError(
            f"action_type must be one of {_VALID_ACTION_TYPES!r}, got {action_type!r}"
        )

    record = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "action_type":    action_type,
        "thread_subject": thread_subject,
        "detail":         detail,
        "id":             action_id,
    }

    entries = get_action_log()
    entries.append(record)

    with open(_LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    return record


def get_action_log() -> list:
    """Return the full list of logged actions.

    Returns [] if action_log.json does not exist, is empty, or is malformed.
    """
    if not os.path.exists(_LOG_PATH):
        return []

    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def clear_log() -> None:
    """Overwrite action_log.json with an empty list."""
    with open(_LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump([], fh, indent=2)
