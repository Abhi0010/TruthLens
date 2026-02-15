"""Phishing claim verifier using BERT model (message + URL phishing)."""

from typing import List, Tuple

from .local_model import is_available, predict_phishing
from .preprocessing import extract_urls
from .rag_verifier import VerdictResult


def verify_claims(claims: List[str]) -> Tuple[List[VerdictResult], str]:
    """
    Verify each claim with BERT model. Also extracts URLs from the combined text
    and classifies each URL with BERT (URL phishing). Supported = phishing/scam;
    Refuted = legitimate; Unknown = unclear or error.
    Returns (results, verification_mode) where mode is "local_model" or "offline".
    """
    if not is_available():
        return [
            VerdictResult(
                claim=c,
                verdict="Unknown",
                evidence=["BERT phishing model not available."],
                similarity=0.0,
            )
            for c in claims
        ], "offline"

    results: List[VerdictResult] = []

    # 1. Classify each message/claim with BERT
    for claim in claims:
        verdict, confidence = predict_phishing(claim)
        evidence = [f"BERT: {verdict} (confidence: {confidence:.2f})"]
        results.append(
            VerdictResult(
                claim=claim,
                verdict=verdict,
                evidence=evidence,
                similarity=float(confidence),
            )
        )

    # 2. Extract URLs from combined text and classify each URL with BERT (URL phishing)
    combined_text = " ".join(claims)
    urls = extract_urls(combined_text)
    seen_urls: set = set()
    for url in urls:
        url_norm = url.strip().rstrip(".,;:)")
        if not url_norm or url_norm in seen_urls:
            continue
        seen_urls.add(url_norm)
        verdict, confidence = predict_phishing(url_norm)
        evidence = [f"URL phishing: {verdict} (confidence: {confidence:.2f})"]
        results.append(
            VerdictResult(
                claim=f"URL: {url_norm}",
                verdict=verdict,
                evidence=evidence,
                similarity=float(confidence),
            )
        )

    return results, "local_model"
