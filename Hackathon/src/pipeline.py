"""Main orchestration pipeline for Clarion."""

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

from .backboard_client import is_configured as backboard_configured, synthesize_fact_check
from .backboard_verifier import BackboardVerifier


@dataclass
class PipelineResult:
    """Unified result from the Clarion pipeline."""

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
    verification_mode: str = "offline"  # "offline", "web", "backboard", or "web+backboard"
    citations: List[str] = field(default_factory=list)  # Aggregated URLs from evidence


def run_pipeline(
    text: str,
    content_type: str = "Normal news",
    rag_verifier: Optional[RAGVerifier] = None,
) -> PipelineResult:
    """
    Run the full Clarion pipeline on input text.
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

    is_phishing = content_type == "scam_phishing"
    is_fact_check_only = content_type == "fact_check"
    is_normal_news = content_type == "normal_news"

    def _extract_citations(verdicts_list: List[VerdictResult]) -> List[str]:
        """Extract unique URLs from evidence for citations."""
        import re
        seen: set = set()
        urls: List[str] = []
        for v in verdicts_list:
            for ev in (v.evidence or []):
                # Match URLs in evidence (e.g. "Source: https://...")
                for match in re.finditer(r"https?://[^\s\)\]\"\']+", ev):
                    u = match.group(0).rstrip(".,;:)")
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
        return urls[:20]  # Limit to 20 citations

    # 2. Claim verification
    # Fact checker: Backboard only
    # Normal news: DuckDuckGo first → Backboard synthesis (finalize)
    # Scam/phishing: Backboard first, fallback DuckDuckGo/RAG
    # Other: DuckDuckGo first, fallback Backboard/RAG
    verdicts: List[VerdictResult] = []
    if claims_raw:
        if is_fact_check_only and backboard_configured() and BackboardVerifier is not None:
            try:
                backboard_verifier = BackboardVerifier()
                verdicts = backboard_verifier.verify_claims(claims_raw)
                result.verification_mode = "backboard"
                result.citations = _extract_citations(verdicts)
            except Exception:
                pass
        elif is_normal_news:
            # Normal news: DuckDuckGo first, then Backboard synthesis
            for attempt in range(2):
                try:
                    web_verifier = WebVerifier(max_results_per_claim=8)
                    verdicts = web_verifier.verify_claims(claims_raw)
                    result.verification_mode = "web"  # Will become "web+backboard" after synthesis
                    result.citations = _extract_citations(verdicts)
                    break
                except Exception:
                    if attempt == 1:
                        verdicts = []
                    continue
        elif is_phishing and backboard_configured() and BackboardVerifier is not None:
            try:
                backboard_verifier = BackboardVerifier()
                verdicts = backboard_verifier.verify_claims(claims_raw)
                result.verification_mode = "backboard"
                result.citations = _extract_citations(verdicts)
            except Exception:
                pass
        if not verdicts and not is_fact_check_only:
            # Fact checker uses Backboard only; skip DuckDuckGo fallback
            for attempt in range(2):
                try:
                    web_verifier = WebVerifier(max_results_per_claim=8)
                    verdicts = web_verifier.verify_claims(claims_raw)
                    result.verification_mode = "web"
                    result.citations = _extract_citations(verdicts)
                    break
                except Exception:
                    if attempt == 1:
                        verdicts = []
                    continue
        if not verdicts and backboard_configured() and BackboardVerifier is not None and not is_fact_check_only:
            try:
                backboard_verifier = BackboardVerifier()
                verdicts = backboard_verifier.verify_claims(claims_raw)
                result.verification_mode = "backboard"
                result.citations = _extract_citations(verdicts)
            except Exception:
                pass
        if not verdicts:
            try:
                verifier = rag_verifier or RAGVerifier()
                verdicts = verifier.verify_claims(claims_raw)
                result.verification_mode = "offline"
                result.citations = _extract_citations(verdicts)
            except Exception:
                pass

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

    # 2b. For normal_news when web succeeded: Backboard synthesis to finalize (summary + reasons + citations)
    synthesis_result: Optional[Dict[str, Any]] = None
    if (
        verdicts
        and result.verification_mode == "web"
        and is_normal_news
        and backboard_configured()
    ):
        try:
            payload = [
                {"claim": v.claim, "verdict": v.verdict, "evidence": v.evidence}
                for v in verdicts
            ]
            synthesis_result = synthesize_fact_check(payload)
            if synthesis_result:
                result.verification_mode = "web+backboard"
                syn_cites = synthesis_result.get("citations", [])
                existing = set(result.citations)
                for c in syn_cites:
                    if c and c not in existing:
                        result.citations.append(c)
        except Exception:
            pass

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
    # Overwrite summary with Backboard synthesis when available; keep top_reasons from
    # compute_fact_check_metrics so they always include the actual claims.
    if synthesis_result and synthesis_result.get("fact_check_summary"):
        result.fact_check_summary = synthesis_result["fact_check_summary"]

    return result
