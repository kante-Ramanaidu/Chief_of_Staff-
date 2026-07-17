"""
Engine module for fetching and processing Gmail threads.
Standalone version — uses the Gmail API directly via OAuth, no MCP/Cline needed.
"""

import os
import base64
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from triage import triage_inbox

# If modifying scopes, delete token.json and re-authenticate.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly",
          "https://www.googleapis.com/auth/gmail.send",
           "https://www.googleapis.com/auth/calendar",
          
          ]

CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
GMAIL_USER_ID = "me"


def _get_gmail_service():
    """
    Handles OAuth login and returns an authenticated Gmail API service object.
    - First run: opens a browser window for you to log in and consent.
    - Later runs: reuses token.json (refreshing it automatically if expired).
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the token for next time so we don't need to log in again
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


 
 
def send_reply(thread_id: str, to: str, subject: str, body: str,
               message_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Send a reply email within an existing Gmail thread.

    Args:
        thread_id: The Gmail thread ID to reply within.
        to: Recipient email address.
        subject: Email subject (will be prefixed with "Re: " if missing).
        body: Plain-text email body.
        message_id: Optional Message-ID of the message being replied to,
            used to set In-Reply-To and References threading headers.

    Returns:
        Dict with message_id, thread_id, and status "sent".
    """
    service = _get_gmail_service()

    # Prepend "Re: " to the subject if not already present
    if not subject.startswith("Re: "):
        subject = "Re: " + subject

    # Build the MIME message
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    # Set threading headers when replying to a specific message
    if message_id:
        message["In-Reply-To"] = message_id
        message["References"] = message_id

    # Base64url-encode the raw message
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    # Include threadId in the send body so the reply stays in the thread
    send_body = {
        "raw": raw,
        "threadId": thread_id,
    }

    sent = service.users().messages().send(
        userId=GMAIL_USER_ID,
        body=send_body
    ).execute()

    return {
        "message_id": sent.get("id"),
        "thread_id": sent.get("threadId", thread_id),
        "status": "sent",
    }


def fetch_threads(max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch the last N inbox threads using the Gmail API directly.

    Returns:
        List of dictionaries, each containing:
            - thread_id
            - sender
            - subject
            - snippet
            - date
    """
    service = _get_gmail_service()

    # Step 1: get the list of thread IDs in the inbox
    response = service.users().threads().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max_results
    ).execute()

    thread_stubs = response.get("threads", [])
    threads = []

    # Step 2: for each thread, fetch its most recent message to pull details
    for stub in thread_stubs:
        thread_id = stub["id"]
        thread_data = service.users().threads().get(
            userId="me",
            id=thread_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        messages = thread_data.get("messages", [])
        if not messages:
            continue

        # Use the most recent message in the thread for header info
        last_message = messages[-1]
        headers = last_message.get("payload", {}).get("headers", [])
        header_map = {h["name"]: h["value"] for h in headers}

        threads.append({
            "thread_id": thread_id,
            "sender": header_map.get("From", ""),
            "subject": header_map.get("Subject", ""),
            "snippet": last_message.get("snippet", ""),
            "date": header_map.get("Date", "")
        })

    return threads


def get_thread_history(thread_id: str) -> List[Dict[str, str]]:
    """
    Fetch the FULL back-and-forth of one thread (Layer 1 context) —
    every message in order, not just the latest one.

    Returns:
        List of dicts, each with: sender, date, content
    """
    service = _get_gmail_service()

    thread_data = service.users().threads().get(
        userId="me",
        id=thread_id,
        format="full"
    ).execute()

    messages = thread_data.get("messages", [])
    history = []

    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        header_map = {h["name"]: h["value"] for h in headers}

        # Try to pull plain text body; fall back to snippet if body is missing/HTML-only
        body_text = _extract_plain_text_body(msg.get("payload", {}))
        if not body_text:
            body_text = msg.get("snippet", "")

        history.append({
            "sender": header_map.get("From", ""),
            "date": header_map.get("Date", ""),
            "content": body_text.strip()
        })

    return history


def _extract_plain_text_body(payload: Dict[str, Any]) -> str:
    """
    Recursively searches a Gmail message payload for the plain-text body
    and decodes it from base64url.
    """
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        result = _extract_plain_text_body(part)
        if result:
            return result

    return ""


def get_past_replies(limit: int = 3) -> List[Dict[str, str]]:
    """
    Fetch your most recent SENT emails (Layer 3 — few-shot examples of
    your real writing voice), for the model to mimic.

    Returns:
        List of dicts, each with: subject, content
    """
    service = _get_gmail_service()

    response = service.users().messages().list(
        userId="me",
        labelIds=["SENT"],
        maxResults=limit
    ).execute()

    message_stubs = response.get("messages", [])
    replies = []

    for stub in message_stubs:
        msg = service.users().messages().get(
            userId="me",
            id=stub["id"],
            format="full"
        ).execute()

        headers = msg.get("payload", {}).get("headers", [])
        header_map = {h["name"]: h["value"] for h in headers}

        body_text = _extract_plain_text_body(msg.get("payload", {}))
        if not body_text:
            body_text = msg.get("snippet", "")

        replies.append({
            "subject": header_map.get("Subject", ""),
            "content": body_text.strip()
        })

    return replies


def format_digest(results: List[Dict[str, Any]]) -> None:
    """
    Print a clean, readable digest of triaged threads to the terminal.

    Format:
        [PRIORITY] Sender — Subject
           Reason: <reason>
    """
    print("\n" + "=" * 60)
    print("INBOX DIGEST")
    print("=" * 60 + "\n")

    for r in results:
        priority_label = r.get("priority", "unknown").upper()
        print(f"[{priority_label}] {r['sender']} — {r['subject']}")
        print(f"   Reason: {r['reason']}\n")


if __name__ == "__main__":
    threads = fetch_threads(20)
    print(f"Fetched {len(threads)} threads. Triaging...\n")

    results = triage_inbox(threads)

    format_digest(results)