"""Main orchestration pipeline for Clarion."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .ai_text_detector import AIDetectionResult, detect_ai_generated
from .claim_extraction import extract_claims
from .preprocessing import clean_text
from .rag_verifier import RAGVerifier, VerdictResult
from .scoring import ClaimVerdict, compute_fact_check_metrics
from .utils import is_empty_input
from .web_verifier import WebVerifier

from .backboard_client import is_configured as backboard_configured, synthesize_fact_check
from .backboard_verifier import BackboardVerifier
from .phishing_verifier import verify_claims as phishing_verify_claims


class RiskLevel(str, Enum):
    """Risk level derived from verifier verdicts (Backboard/BERT)."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class SocialEngineeringResult:
    """Safe/unsafe result derived from Backboard or BERT verdicts."""

    risk_level: RiskLevel
    red_flags: List[str]
    safer_rewrite_suggestion: str


def _social_engineering_from_verdicts(
    verdicts: List[VerdictResult], verification_mode: str
) -> SocialEngineeringResult:
    """
    Derive safe/unsafe (risk level, red flags) from verifier verdicts.
    Phishing mode (local_model): Supported = unsafe, Refuted = safe.
    Fact-check modes (backboard, web, web+backboard, offline): Refuted = unsafe, Supported = safe.
    """
    is_phishing_mode = verification_mode == "local_model"

    if not verdicts:
        return SocialEngineeringResult(
            risk_level=RiskLevel.LOW,
            red_flags=["No claims to verify; no AI verdict available."],
            safer_rewrite_suggestion="No content was verified by Backboard or BERT.",
        )

    unsafe_verdict = "Supported" if is_phishing_mode else "Refuted"
    safe_verdict = "Refuted" if is_phishing_mode else "Supported"

    unsafe_count = sum(1 for v in verdicts if v.verdict == unsafe_verdict)
    safe_count = sum(1 for v in verdicts if v.verdict == safe_verdict)
    unknown_count = len(verdicts) - unsafe_count - safe_count

    red_flags: List[str] = []
    for v in verdicts:
        if v.verdict == unsafe_verdict and v.evidence:
            red_flags.append(v.evidence[0][:200] if v.evidence[0] else f"Claim flagged: {v.claim[:80]}...")
        elif v.verdict == "Unknown" and v.evidence:
            red_flags.append(f"Unclear: {v.evidence[0][:150]}..." if len(v.evidence[0]) > 150 else v.evidence[0])

    if not red_flags and unsafe_count > 0:
        red_flags = [f"{unsafe_count} claim(s) flagged as unsafe by {verification_mode}"]
    if not red_flags and unknown_count > 0:
        red_flags = [f"{unknown_count} claim(s) could not be verified"]

    if unsafe_count >= 1:
        level = RiskLevel.HIGH if unsafe_count >= len(verdicts) // 2 + 1 else RiskLevel.MEDIUM
    elif unknown_count > 0:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    if level == RiskLevel.LOW:
        suggestion = "Content appears safe based on verifier analysis. No rewrite needed."
    elif level == RiskLevel.HIGH:
        suggestion = "Safer approach: Treat this content with caution. Verify through official channels before taking action. Do not share credentials or send money via links in messages."
    else:
        suggestion = "Safer approach: Some claims could not be fully verified. Cross-check with trusted sources before relying on this information."

    return SocialEngineeringResult(
        risk_level=level,
        red_flags=red_flags if red_flags else ["No obvious risks detected by verifier."],
        safer_rewrite_suggestion=suggestion,
    )


@dataclass
class MisinformationResult:
    """Misinformation risk derived from Backboard or BERT verdicts."""

    risk_score: float  # 0-1
    reasons: List[str]


def _misinformation_from_verdicts(
    verdicts: List[VerdictResult], verification_mode: str
) -> MisinformationResult:
    """
    Derive misinformation risk from verifier verdicts.
    Same semantics as social_engineering: unsafe verdicts = misinformation/deceptive content.
    """
    is_phishing_mode = verification_mode == "local_model"
    unsafe_verdict = "Supported" if is_phishing_mode else "Refuted"

    if not verdicts:
        return MisinformationResult(
            risk_score=0.0,
            reasons=["No claims to verify; no AI verdict available."],
        )

    unsafe_count = sum(1 for v in verdicts if v.verdict == unsafe_verdict)
    total = len(verdicts)
    risk_score = min(1.0, unsafe_count / total if total > 0 else 0.0)

    reasons: List[str] = []
    for v in verdicts:
        if v.verdict == unsafe_verdict and v.evidence:
            reasons.append(v.evidence[0][:180] + ("..." if len(v.evidence[0]) > 180 else ""))
    if not reasons and unsafe_count > 0:
        reasons = [f"{unsafe_count} claim(s) flagged by {verification_mode}"]
    if not reasons:
        reasons = ["No misinformation signals from verifier analysis."]

    return MisinformationResult(risk_score=risk_score, reasons=reasons[:5])


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
    verification_mode: str = "offline"  # "offline", "web", "backboard", "web+backboard", "local_model"
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

    # 2. Claim verification — all sections use BERT/Backboard/DuckDuckGo (RAG fallback)
    # Fact Check: Backboard → DuckDuckGo → RAG
    # Normal News: DuckDuckGo → Backboard → RAG (Backboard synthesis applied if web succeeded)
    # Scam/Phishing: BERT (message + URL phishing) → Backboard → DuckDuckGo → RAG
    verdicts: List[VerdictResult] = []
    # Use extracted claims or full text so we always try verification
    claims_to_verify = claims_raw if claims_raw else [result.raw_text[:2000]] if result.raw_text else []
    if claims_to_verify:
        def _try_backboard() -> bool:
            if not backboard_configured() or BackboardVerifier is None:
                return False
            try:
                nonlocal verdicts
                bv = BackboardVerifier()
                verdicts = bv.verify_claims(claims_to_verify)
                result.verification_mode = "backboard"
                result.citations = _extract_citations(verdicts)
                return bool(verdicts)
            except Exception:
                return False

        def _try_web() -> bool:
            for attempt in range(2):
                try:
                    nonlocal verdicts
                    wv = WebVerifier(max_results_per_claim=8)
                    verdicts = wv.verify_claims(claims_to_verify)
                    result.verification_mode = "web"
                    result.citations = _extract_citations(verdicts)
                    return bool(verdicts)
                except Exception:
                    if attempt == 1:
                        verdicts = []
            return False

        def _try_rag() -> bool:
            try:
                nonlocal verdicts
                verifier = rag_verifier or RAGVerifier()
                verdicts = verifier.verify_claims(claims_to_verify)
                result.verification_mode = "offline"
                result.citations = _extract_citations(verdicts)
                return bool(verdicts)
            except Exception:
                return False

        # Primary by content type
        if is_phishing:
            try:
                # Prefer full text for phishing (better context); fallback to claims
                phish_input = [result.raw_text[:2000]] if result.raw_text else claims_to_verify
                verdicts, phish_mode = phishing_verify_claims(phish_input)
                if verdicts:
                    result.verification_mode = phish_mode
                    result.citations = _extract_citations(verdicts)
            except Exception:
                pass

        if not verdicts and is_fact_check_only:
            _try_backboard() or _try_web() or _try_rag()
        elif not verdicts and is_normal_news:
            _try_web() or _try_backboard() or _try_rag()
        elif not verdicts:
            _try_backboard() or _try_web() or _try_rag()

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

    # 3. Misinformation risk — derived from Backboard or BERT verdicts
    result.misinformation = _misinformation_from_verdicts(verdicts, result.verification_mode)

    # 4. Social engineering (safe/unsafe) — derived from Backboard or BERT verdicts
    result.social_engineering = _social_engineering_from_verdicts(verdicts, result.verification_mode)

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
