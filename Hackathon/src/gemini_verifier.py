"""Gemini-powered claim verifier with Google Search grounding."""

import os
import re
from typing import List, Optional

from .rag_verifier import VerdictResult


def _get_gemini_client(api_key: Optional[str] = None):
    """
    Create a Gemini client. Uses provided key, then env var.
    Returns None if no key or SDK unavailable.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def _build_prompt(claim: str) -> str:
    """Build a fact-checking prompt for Gemini."""
    return (
        "You are a fact-checker. Evaluate the following claim using web search.\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Respond in EXACTLY this format (three lines, nothing else):\n"
        "VERDICT: Supported OR Refuted OR Unknown\n"
        "EVIDENCE: One-sentence summary of what you found.\n"
        "SOURCES: Comma-separated URLs (or 'none' if no sources).\n"
    )


def _parse_verdict(text: str) -> tuple[str, str, List[str]]:
    """
    Parse Gemini response into (verdict, evidence_summary, source_urls).
    Robust to minor formatting variations.
    """
    verdict = "Unknown"
    evidence = ""
    sources: List[str] = []

    for line in text.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "support" in raw:
                verdict = "Supported"
            elif "refut" in raw or "false" in raw:
                verdict = "Refuted"
            else:
                verdict = "Unknown"
        elif upper.startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()
        elif upper.startswith("SOURCE"):
            raw_sources = line.split(":", 1)[1].strip()
            if raw_sources.lower() != "none":
                sources = [
                    s.strip() for s in re.split(r"[,\s]+", raw_sources)
                    if s.strip().startswith("http")
                ]

    return verdict, evidence, sources


class GeminiVerifier:
    """Verify claims using Gemini + Google Search grounding."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None

    def _ensure_client(self) -> bool:
        """Lazily initialize the Gemini client. Returns True if ready."""
        if self._client is not None:
            return True
        self._client = _get_gemini_client(self.api_key)
        return self._client is not None

    def verify_claim(self, claim: str) -> VerdictResult:
        """
        Verify a single claim via Gemini with Google Search grounding.
        Returns VerdictResult compatible with the existing pipeline.
        """
        if not self._ensure_client():
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=["Gemini API key not configured"],
                similarity=0.0,
            )

        try:
            from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

            search_tool = Tool(google_search=GoogleSearch())
            prompt = _build_prompt(claim)

            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[search_tool],
                    temperature=0.1,
                ),
            )

            raw_text = response.text or ""
            verdict, evidence_summary, source_urls = _parse_verdict(raw_text)

            # Build evidence list: summary + source URLs
            evidence_parts: List[str] = []
            if evidence_summary:
                evidence_parts.append(evidence_summary)
            for url in source_urls:
                evidence_parts.append(f"Source: {url}")

            # Extract grounding metadata sources if available
            try:
                candidates = response.candidates or []
                if candidates:
                    gm = getattr(candidates[0], "grounding_metadata", None)
                    if gm:
                        chunks = getattr(gm, "grounding_chunks", None) or []
                        for chunk in chunks:
                            web = getattr(chunk, "web", None)
                            if web:
                                uri = getattr(web, "uri", "")
                                title = getattr(web, "title", "")
                                if uri and uri not in " ".join(evidence_parts):
                                    label = f"Source: {title} — {uri}" if title else f"Source: {uri}"
                                    evidence_parts.append(label)
            except Exception:
                pass  # grounding metadata is optional

            if not evidence_parts:
                evidence_parts.append(raw_text[:500] if raw_text else "No evidence returned")

            # Confidence proxy: Supported/Refuted = 0.85, Unknown = 0.3
            sim = 0.85 if verdict != "Unknown" else 0.3

            return VerdictResult(
                claim=claim,
                verdict=verdict,
                evidence=evidence_parts,
                similarity=sim,
            )

        except Exception as exc:
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=[f"Gemini error: {exc}"],
                similarity=0.0,
            )

    def verify_claims(self, claims: List[str]) -> List[VerdictResult]:
        """Verify multiple claims sequentially."""
        return [self.verify_claim(c) for c in claims]
