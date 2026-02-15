"""Main orchestration pipeline for TruthLens Suite."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ai_text_detector import AIDetectionResult, detect_ai_generated
from .claim_extraction import extract_claims
from .misinformation_detector import MisinformationResult, detect_misinformation
from .preprocessing import clean_text
from .rag_verifier import RAGVerifier, VerdictResult
from .scoring import ClaimVerdict, compute_fact_check_metrics
from .social_engineering_detector import RiskLevel, SocialEngineeringResult, detect_social_engineering
from .utils import is_empty_input
from .web_verifier import WebVerifier


@dataclass
class PipelineResult:
    """Unified result from the TruthLens pipeline."""

    correct_count: int = 0
    incorrect_count: int = 0
    response_confidence: float = 0.0
    top_reasons: List[str] = field(default_factory=list)
    fact_check_summary: str = ""  # single-line summary of what the fact checker thinks
    claims: List[ClaimVerdict] = field(default_factory=list)
    misinformation: MisinformationResult = field(default_factory=lambda: MisinformationResult(0.0, []))
    social_engineering: SocialEngineeringResult = field(default_factory=lambda: SocialEngineeringResult(
        risk_level=RiskLevel.LOW,
        red_flags=[],
        safer_rewrite_suggestion="",
    ))
    ai_detection: AIDetectionResult = field(default_factory=lambda: AIDetectionResult(0.0, []))
    evidence_passages: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    verification_mode: str = "offline"  # "offline" or "web"


def run_pipeline(
    text: str,
    content_type: str = "Normal news",
    rag_verifier: Optional[RAGVerifier] = None,
) -> PipelineResult:
    """
    Run the full TruthLens pipeline on input text.
    Uses Web Verify (DuckDuckGo) by default for claim verification (normal news and misinformation).
    Falls back to offline RAG if web verification fails.
    """
    result = PipelineResult()

    if is_empty_input(text):
        result.top_reasons = ["No input provided"]
        return result

    text = clean_text(text)
    result.raw_text = text

    # 1. Claim extraction
    claims_raw = extract_claims(text)

    # 2. Claim verification — use internet (DuckDuckGo) first; fallback to RAG only if web fails
    verdicts: List[VerdictResult] = []
    for attempt in range(2):  # retry once before falling back to RAG
        try:
            web_verifier = WebVerifier(max_results_per_claim=8)
            verdicts = web_verifier.verify_claims(claims_raw) if claims_raw else []
            result.verification_mode = "web"
            break
        except Exception:
            if attempt == 1:
                verifier = rag_verifier or RAGVerifier()
                verdicts = verifier.verify_claims(claims_raw) if claims_raw else []
                result.verification_mode = "offline"
            continue

    # Convert to ClaimVerdict for scoring
    result.claims = [
        ClaimVerdict(
            claim=v.claim,
            verdict=v.verdict,
            evidence=v.evidence,
            similarity=v.similarity,
        )
        for v in verdicts
    ]

    # Collect evidence passages
    for v in verdicts:
        for i, ev in enumerate(v.evidence):
            result.evidence_passages.append({
                "claim": v.claim,
                "passage": ev,
                "similarity": v.similarity,
                "verdict": v.verdict,
            })

    # 3. Misinformation detector
    result.misinformation = detect_misinformation(text)

    # 4. Social engineering detector
    result.social_engineering = detect_social_engineering(text)

    # 5. AI-generated detector
    result.ai_detection = detect_ai_generated(text)

    # 6. Fact-check metrics (fact-checker only): correct, incorrect, confidence, summary
    result.correct_count, result.incorrect_count, result.response_confidence, result.top_reasons, result.fact_check_summary = compute_fact_check_metrics(
        claim_verdicts=result.claims,
    )

    return result
