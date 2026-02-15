"""Fact-check metrics: fact-checker only — correct/incorrect and confidence in the response."""

from dataclasses import dataclass
from typing import List


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if over max_len."""
    s = (s or "").strip()
    return (s[: max_len - 3] + "...") if len(s) > max_len else s


@dataclass
class ClaimVerdict:
    """Verdict for a single claim."""

    claim: str
    verdict: str  # "Supported", "Refuted", "Misclassification", "Unknown"
    evidence: List[str]
    similarity: float


def compute_fact_check_metrics(
    claim_verdicts: List[ClaimVerdict],
) -> tuple[int, int, float, List[str], str]:
    """
    Fact-checker only: correct count, incorrect count, confidence, reasons, and a single summary.
    Returns: (correct_count, incorrect_count, response_confidence, top_reasons, fact_check_summary)
    - Correct = Supported claims. Incorrect = Refuted claims.
    - response_confidence = how confident the fact checker is in its response (0-1), from claims only.
    - fact_check_summary = one-line summary of what the fact checker thinks of the text.
    """
    correct_count = sum(1 for c in claim_verdicts if c.verdict == "Supported")
    incorrect_count = sum(1 for c in claim_verdicts if c.verdict == "Refuted")
    total_claims = len(claim_verdicts)
    verified = sum(1 for c in claim_verdicts if c.verdict not in ("Unknown", "Misclassification"))

    # Confidence in fact-check response: from claim verification only
    response_confidence = verified / total_claims if total_claims > 0 else 0.0
    response_confidence = max(0.2, min(0.95, response_confidence))

    reasons: List[str] = []
    if claim_verdicts:
        unknown = sum(1 for c in claim_verdicts if c.verdict == "Unknown")
        misclassification = sum(1 for c in claim_verdicts if c.verdict == "Misclassification")
        if correct_count > 0:
            correct_claims = [c.claim for c in claim_verdicts if c.verdict == "Supported"]
            claim_text = "; ".join(f'"{_truncate(c, 80)}"' for c in correct_claims)
            reasons.append(f"{correct_count} claim(s) correct (supported by evidence): {claim_text}")
        if incorrect_count > 0:
            incorrect_claims = [c.claim for c in claim_verdicts if c.verdict == "Refuted"]
            claim_text = "; ".join(f'"{_truncate(c, 80)}"' for c in incorrect_claims)
            reasons.append(f"{incorrect_count} claim(s) not supported by evidence: {claim_text}")
        if misclassification > 0:
            misc_claims = [c.claim for c in claim_verdicts if c.verdict == "Misclassification"]
            claim_text = "; ".join(f'"{_truncate(c, 80)}"' for c in misc_claims)
            reasons.append(f"{misclassification} claim(s) misclassified (off-topic/wrong category): {claim_text}")
        if unknown == total_claims and total_claims > 0:
            reasons.append("Claims not in knowledge base (unverifiable)")
    if not reasons:
        reasons.append("No claims to verify")

    # Single result only: Correct, Incorrect, Mixed, or Unverifiable
    if total_claims == 0 or verified == 0:
        fact_check_summary = "Unverifiable"
    elif correct_count > 0 and incorrect_count == 0:
        fact_check_summary = "Correct"
    elif incorrect_count > 0 and correct_count == 0:
        fact_check_summary = "Incorrect"
    else:
        fact_check_summary = "Mixed"

    return correct_count, incorrect_count, response_confidence, reasons[:5], fact_check_summary
