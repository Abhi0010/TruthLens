"""Phishing claim verifier using BERT model (message + URL phishing)."""

from typing import List, Tuple

from .local_model import is_available, predict_phishing_batch
from .preprocessing import extract_urls
from .rag_verifier import VerdictResult


def verify_claims(claims: List[str]) -> Tuple[List[VerdictResult], str]:
    """
    Verify each claim with BERT model. Also extracts URLs from the combined text
    and classifies each URL with BERT (URL phishing). Supported = phishing/scam;
    Refuted = legitimate; Unknown = unclear or error.
    Uses a single batched BERT run for all claims + URLs for speed.
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

    # 1. Collect all texts to classify: claims first, then unique URLs
    texts_to_classify: List[str] = list(claims)
    combined_text = " ".join(claims)
    urls = extract_urls(combined_text)
    seen_urls: set = set()
    url_list: List[str] = []
    for url in urls:
        url_norm = url.strip().rstrip(".,;:)")
        if not url_norm or url_norm in seen_urls:
            continue
        seen_urls.add(url_norm)
        url_list.append(url_norm)
    texts_to_classify.extend(url_list)

    if not texts_to_classify:
        return [], "local_model"

    # 2. Single batched BERT run for all claims + URLs
    batch_results = predict_phishing_batch(texts_to_classify)

    # 3. Build results: claims then URL verdicts
    results: List[VerdictResult] = []
    n_claims = len(claims)
    for i, (verdict, confidence) in enumerate(batch_results):
        if i < n_claims:
            claim = claims[i]
            evidence = [f"BERT: {verdict} (confidence: {confidence:.2f})"]
            results.append(
                VerdictResult(
                    claim=claim,
                    verdict=verdict,
                    evidence=evidence,
                    similarity=float(confidence),
                )
            )
        else:
            url_norm = url_list[i - n_claims]
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
