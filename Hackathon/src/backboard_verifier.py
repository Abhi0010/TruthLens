"""Backboard-powered claim verifier for the TruthLens pipeline."""

from typing import List, Optional, Tuple

from .backboard_client import (
    create_fact_check_assistant,
    create_thread,
    is_configured as backboard_configured,
    send_message,
)
from .rag_verifier import VerdictResult


def _parse_response(content: Optional[str]) -> Tuple[str, str]:
    """
    Parse Backboard fact-check response into (verdict, evidence_text).
    Expects lines like VERDICT: Supported/Refuted/Unknown, EVIDENCE: ...
    """
    verdict = "Unknown"
    evidence = "No evidence returned."
    if not content or not content.strip():
        return verdict, evidence

    for line in content.strip().splitlines():
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
            evidence = line.split(":", 1)[1].strip() or evidence

    return verdict, evidence


class BackboardVerifier:
    """Verify claims using Backboard API (assistant + thread + message per claim)."""

    def __init__(self):
        self._assistant_id: Optional[str] = None

    def _ensure_assistant(self) -> bool:
        """Create fact-check assistant once. Returns True if ready."""
        if self._assistant_id is not None:
            return True
        if not backboard_configured():
            return False
        self._assistant_id = create_fact_check_assistant()
        return self._assistant_id is not None

    def verify_claim(self, claim: str) -> VerdictResult:
        """Verify a single claim via Backboard. Returns VerdictResult."""
        if not claim or not claim.strip():
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=[],
                similarity=0.0,
            )

        if not self._ensure_assistant():
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=["Backboard not configured or assistant creation failed."],
                similarity=0.0,
            )

        thread_id = create_thread(self._assistant_id)
        if not thread_id:
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=["Failed to create Backboard thread."],
                similarity=0.0,
            )

        prompt = f"Fact-check this claim: \"{claim.strip()}\""
        content = send_message(thread_id, prompt, stream=False)
        verdict, evidence_text = _parse_response(content)

        similarity = 0.85 if verdict != "Unknown" else 0.3
        evidence_list = [evidence_text] if evidence_text else []

        return VerdictResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence_list,
            similarity=similarity,
        )

    def verify_claims(self, claims: List[str]) -> List[VerdictResult]:
        """Verify multiple claims sequentially."""
        return [self.verify_claim(c) for c in claims]
