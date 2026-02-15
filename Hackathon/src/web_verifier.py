"""Web-based claim verifier using DuckDuckGo search (no API key required)."""

import re
from typing import List

from .rag_verifier import VerdictResult

# Same contradiction keywords as RAG for consistent verdicts
CONTRADICTION_KEYWORDS = [
    "false", "debunked", "hoax", "not true", "myth", "misleading",
    "incorrect", "untrue", "fabricated", "disproven", "fake",
    "no evidence", "lacks evidence", "unfounded",
]


def _keyword_similarity(claim: str, text: str) -> float:
    """Word overlap similarity 0-1."""
    cw = set(re.findall(r"\b\w+\b", claim.lower()))
    kw = set(re.findall(r"\b\w+\b", text.lower()))
    if not cw:
        return 0.0
    return len(cw & kw) / len(cw)


def _has_contradiction(text: str) -> bool:
    """Check if text contains contradiction keywords."""
    lower = text.lower()
    return any(kw in lower for kw in CONTRADICTION_KEYWORDS)


def _has_matching_entities(claim: str, text: str) -> bool:
    """Do numbers or capitalized terms appear in both?"""
    c_nums = set(re.findall(r"\d+", claim))
    k_nums = set(re.findall(r"\d+", text))
    if c_nums and (c_nums & k_nums):
        return True
    c_caps = set(re.findall(r"\b[A-Z][a-z]+\b", claim))
    k_caps = set(re.findall(r"\b[A-Z][a-z]+\b", text))
    return bool(c_caps & k_caps) or bool(c_nums & k_nums)


def _search_duckduckgo(query: str, max_results: int = 8, retries: int = 2) -> List[dict]:
    """
    Run DuckDuckGo text search over the internet. Returns list of dicts with 'title', 'href', 'body'.
    No API key required. Retries on failure.
    """
    from ddgs import DDGS

    for attempt in range(max(1, retries)):
        try:
            # Explicit DuckDuckGo backend; .text() returns iterable of dicts
            results = list(DDGS().text(query, max_results=max_results, backend="duckduckgo"))
            return results if results else []
        except Exception:
            if attempt == retries - 1:
                return []
    return []


class WebVerifier:
    """
    Verify claims using DuckDuckGo web search over the internet.
    No API key required. Uses the same heuristic verdict logic as the offline RAG.
    """

    def __init__(self, max_results_per_claim: int = 8):
        self.max_results = max_results_per_claim

    def verify_claim(self, claim: str) -> VerdictResult:
        """Search the web for the claim and assign Supported/Refuted/Unknown from snippets."""
        if not claim or not claim.strip():
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=[],
                similarity=0.0,
            )

        results = _search_duckduckgo(claim.strip(), max_results=self.max_results)

        if not results:
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=["No search results found."],
                similarity=0.0,
            )

        # Build evidence list: snippet text + URL for each result
        evidence: List[str] = []
        for r in results:
            title = r.get("title") or ""
            body = r.get("body") or ""
            href = r.get("href") or ""
            snippet = f"{title}. {body}".strip()
            if href:
                evidence.append(f"{snippet}\nSource: {href}")
            else:
                evidence.append(snippet)

        # Score each result and take best
        best_sim = 0.0
        best_has_contradiction = False
        best_has_entity = False

        for r in results:
            body = (r.get("title") or "") + " " + (r.get("body") or "")
            if not body.strip():
                continue
            sim = _keyword_similarity(claim, body)
            if sim > best_sim:
                best_sim = sim
                best_has_contradiction = _has_contradiction(body)
                best_has_entity = _has_matching_entities(claim, body)

        # Verdict logic (aligned with RAG)
        strong_sim = best_sim > 0.35
        moderate_sim = best_sim > 0.20

        if strong_sim and best_has_contradiction:
            verdict = "Refuted"
        elif moderate_sim and best_has_contradiction and best_has_entity:
            verdict = "Refuted"
        elif strong_sim and best_has_entity:
            verdict = "Supported"
        elif moderate_sim and best_has_entity and not best_has_contradiction:
            verdict = "Supported"
        else:
            verdict = "Unknown"

        # Confidence proxy
        sim_display = 0.85 if verdict != "Unknown" else max(0.3, best_sim)

        return VerdictResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence,
            similarity=sim_display,
        )

    def verify_claims(self, claims: List[str]) -> List[VerdictResult]:
        """Verify multiple claims sequentially."""
        return [self.verify_claim(c) for c in claims]
