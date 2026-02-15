"""Backboard API client for TruthLens chat assistant."""

import os
from typing import Any, Dict, List, Optional

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


SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a fact-check synthesizer. You receive claim verification results from web search. "
    "Output a concise 1-2 sentence fact-check summary and 3-5 bullet-point reasons. "
    "Use EXACTLY this format (nothing else):\n"
    "SUMMARY: Your one or two sentence summary here.\n"
    "REASONS:\n"
    "- First reason\n"
    "- Second reason\n"
    "- (up to 5 reasons)"
)


def create_synthesizer_assistant() -> Optional[str]:
    """Create a Backboard assistant for synthesizing fact-check results. Returns assistant_id or None."""
    return create_assistant(
        name="TruthLens Synthesizer",
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
    )


def _parse_synthesis_response(content: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse synthesizer reply into { fact_check_summary, top_reasons }."""
    if not content or not content.strip():
        return None
    summary = ""
    reasons: List[str] = []
    in_reasons = False
    for line in content.strip().splitlines():
        line_stripped = line.strip()
        if line_stripped.upper().startswith("SUMMARY:"):
            summary = line_stripped.split(":", 1)[1].strip()
            in_reasons = False
        elif line_stripped.upper().startswith("REASONS:"):
            in_reasons = True
        elif in_reasons and line_stripped.startswith("-"):
            reason = line_stripped.lstrip("-").strip()
            if reason:
                reasons.append(reason)
    if not summary:
        return None
    return {"fact_check_summary": summary, "top_reasons": reasons[:5]}


def synthesize_fact_check(claims_with_verdicts_and_evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Send web verification results to Backboard and get a fact-check summary and reasons.
    One assistant, one thread, one message. Returns { fact_check_summary, top_reasons } or None.
    """
    headers = _get_headers()
    if not headers.get("X-API-Key"):
        return None
    assistant_id = create_synthesizer_assistant()
    if not assistant_id:
        return None
    thread_id = create_thread(assistant_id)
    if not thread_id:
        return None
    # Build a condensed message for the assistant
    parts = []
    for item in claims_with_verdicts_and_evidence:
        claim = item.get("claim", "")
        verdict = item.get("verdict", "")
        evidence = item.get("evidence", [])
        evidence_preview = (evidence[0][:200] + "..." if evidence and len(evidence[0]) > 200 else (evidence[0] if evidence else ""))
        parts.append(f"Claim: {claim}\nVerdict: {verdict}\nEvidence: {evidence_preview}")
    message = "Claim verification results from web search:\n\n" + "\n\n".join(parts)
    content = send_message(thread_id, message, stream=False)
    return _parse_synthesis_response(content)


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
