"""Backboard API client for TruthLens chat assistant."""

import os
from typing import Optional

import requests

BASE_URL = "https://app.backboard.io/api"


def _get_headers() -> dict:
    api_key = os.environ.get("BACKBOARD_API_KEY", "").strip()
    return {"X-API-Key": api_key}


def is_configured() -> bool:
    """Return True if Backboard API key is set."""
    return bool(os.environ.get("BACKBOARD_API_KEY", "").strip())


FACT_CHECK_SYSTEM_PROMPT = (
    "You are a fact-checker. For each claim you receive, use your knowledge to determine "
    "whether it is supported, refuted, or unknown. Respond with EXACTLY these three lines (nothing else):\n"
    "VERDICT: Supported OR Refuted OR Unknown\n"
    "EVIDENCE: One short sentence summarizing what you found.\n"
    "SOURCES: Optional URLs or 'none'."
)


def create_assistant(
    name: str = "TruthLens Support",
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Create a Backboard assistant. Returns assistant_id or None on failure.
    """
    default_prompt = (
        "You are a helpful assistant for TruthLens, an app that helps users "
        "spot misinformation, scams, and manipulation in text. You can explain "
        "how fact-checking works, what the metrics mean (e.g. confidence, risk levels), "
        "and how to use the Analyzer and Trainer. Be concise and friendly."
    )
    headers = _get_headers()
    if not headers.get("X-API-Key"):
        return None
    try:
        response = requests.post(
            f"{BASE_URL}/assistants",
            json={
                "name": name,
                "system_prompt": system_prompt or default_prompt,
            },
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("assistant_id")
    except (requests.RequestException, KeyError):
        return None


def create_fact_check_assistant() -> Optional[str]:
    """Create a Backboard assistant for fact-checking claims. Returns assistant_id or None."""
    return create_assistant(
        name="TruthLens Fact Checker",
        system_prompt=FACT_CHECK_SYSTEM_PROMPT,
    )


def create_thread(assistant_id: str) -> Optional[str]:
    """Create a thread for the given assistant. Returns thread_id or None."""
    headers = _get_headers()
    if not headers.get("X-API-Key"):
        return None
    try:
        response = requests.post(
            f"{BASE_URL}/assistants/{assistant_id}/threads",
            json={},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("thread_id")
    except (requests.RequestException, KeyError):
        return None


def send_message(
    thread_id: str,
    content: str,
    stream: bool = False,
) -> Optional[str]:
    """
    Send a message to a thread and return the assistant's reply content.
    Returns None on failure.
    """
    headers = _get_headers()
    if not headers.get("X-API-Key"):
        return None
    try:
        response = requests.post(
            f"{BASE_URL}/threads/{thread_id}/messages",
            headers=headers,
            data={"content": content, "stream": "true" if stream else "false"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("content") if isinstance(data, dict) else None
    except (requests.RequestException, KeyError):
        return None
