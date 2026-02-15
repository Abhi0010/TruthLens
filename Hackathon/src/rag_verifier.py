"""Local RAG verifier: chunking, TF-IDF retrieval, verdict.
Uses TF-IDF for similarity (no BERT). BERT is used only for the phishing section."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import get_kb_path


# Contradiction keywords in KB chunks
CONTRADICTION_KEYWORDS = [
    "false", "debunked", "hoax", "not true", "myth", "misleading",
    "incorrect", "untrue", "fabricated", "disproven", "fake",
    "no evidence", "lacks evidence", "unfounded",
]


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks that always start and end at sentence boundaries.
    Avoids chunks that appear "cut at the beginning" when displayed.
    """
    if not text:
        return []
    text = text.strip()
    # Split into sentences (handle . ! ?) and paragraphs (double newlines)
    sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        return [text[:chunk_size]] if text else []

    chunks = []
    current = []
    current_len = 0

    for i, sent in enumerate(sentences):
        sent_len = len(sent) + 1  # +1 for space
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last few sentences for context
            overlap_len = 0
            overlap_sents = []
            for j in range(len(current) - 1, -1, -1):
                overlap_sents.insert(0, current[j])
                overlap_len += len(current[j]) + 1
                if overlap_len >= overlap:
                    break
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c]


def _load_kb(path: Optional[Path] = None) -> str:
    """Load knowledge base content."""
    path = path or get_kb_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _build_tfidf_index(chunks: List[str]) -> tuple:
    """
    Build TF-IDF index for chunks. Returns (vectorizer, chunk_matrix).
    Lightweight, no BERT - used only for RAG fallback (Fact Check / Normal News).
    """
    if not chunks:
        return None, None
    try:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        chunk_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, chunk_matrix
    except Exception:
        return None, None


def _has_contradiction(chunk: str) -> bool:
    """Check if chunk contains contradiction keywords near content."""
    lower = chunk.lower()
    return any(kw in lower for kw in CONTRADICTION_KEYWORDS)


def _has_matching_entities(claim: str, chunk: str) -> bool:
    """Simple check: do numbers or capitalized terms appear in both?"""
    c_nums = set(re.findall(r"\d+", claim))
    k_nums = set(re.findall(r"\d+", chunk))
    if c_nums and (c_nums & k_nums):
        return True
    c_caps = set(re.findall(r"\b[A-Z][a-z]+\b", claim))
    k_caps = set(re.findall(r"\b[A-Z][a-z]+\b", chunk))
    return bool(c_caps & k_caps) or bool(c_nums & k_nums)


@dataclass
class VerdictResult:
    """Result for a single claim verification (information checker output)."""

    claim: str
    verdict: str  # "Supported", "Refuted", "Misclassification", "Unknown"
    evidence: List[str]
    similarity: float


class RAGVerifier:
    """Local RAG verifier with TF-IDF (no BERT). BERT is used only for phishing section."""

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or get_kb_path()
        self.chunks: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._chunk_matrix = None  # sparse matrix from TfidfVectorizer
        self._indexed = False

    def _build_index(self) -> None:
        """Load KB, chunk, and build TF-IDF index for similarity search."""
        if self._indexed:
            return
        text = _load_kb(self.kb_path)
        self.chunks = _chunk_text(text)
        if not self.chunks:
            self._indexed = True
            return
        self._vectorizer, self._chunk_matrix = _build_tfidf_index(self.chunks)
        self._indexed = True

    def check_information(self, claim: str, top_k: int = 5) -> VerdictResult:
        """
        Information checker: verify claim against KB and return verdict.
        Supported: high similarity + matching entities, no contradiction
        Refuted: high similarity + contradiction keywords
        Misclassification: relevant-looking chunks but no entity match (wrong category/off-topic)
        Unknown: else (no good evidence)
        """
        return self.verify_claim(claim, top_k)

    def verify_claim(self, claim: str, top_k: int = 5) -> VerdictResult:
        """
        Retrieve top-k chunks for claim, compute verdict.
        Supported: high similarity + matching entities, no contradiction
        Refuted: high similarity + contradiction keywords
        Misclassification: moderate/strong similarity but no entity match (off-topic)
        Unknown: else
        """
        self._build_index()
        if not self.chunks or not self._vectorizer or self._chunk_matrix is None:
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=[],
                similarity=0.0,
            )

        try:
            q_vec = self._vectorizer.transform([claim])
            sims = cosine_similarity(q_vec, self._chunk_matrix).ravel()
            scores = [(i, float(sims[i])) for i in range(len(sims))]
        except Exception:
            return VerdictResult(
                claim=claim,
                verdict="Unknown",
                evidence=[],
                similarity=0.0,
            )

        scores.sort(key=lambda x: -x[1])
        top = scores[:top_k]

        evidence = [self.chunks[i] for i, _ in top]
        best_sim = top[0][1] if top else 0.0

        # Verdict logic — tiered thresholds; Misclassification = information checker
        has_entity_match = any(_has_matching_entities(claim, self.chunks[i]) for i, _ in top)
        has_contradiction = any(_has_contradiction(self.chunks[i]) for i, _ in top)

        # Strong match: high similarity + entity overlap
        strong_sim = best_sim > 0.35
        moderate_sim = best_sim > 0.20

        if strong_sim and has_contradiction:
            verdict = "Refuted"
        elif moderate_sim and has_contradiction and has_entity_match:
            verdict = "Refuted"
        elif strong_sim and has_entity_match:
            verdict = "Supported"
        elif moderate_sim and has_entity_match and not has_contradiction:
            verdict = "Supported"
        elif (strong_sim or moderate_sim) and not has_entity_match:
            # Relevant-looking chunks but no entity overlap → misclassification (wrong category)
            verdict = "Misclassification"
        else:
            verdict = "Unknown"

        return VerdictResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence,
            similarity=best_sim,
        )

    def verify_claims(self, claims: List[str], top_k: int = 5) -> List[VerdictResult]:
        """Verify multiple claims."""
        return [self.verify_claim(c, top_k) for c in claims]

    def check_claims(self, claims: List[str], top_k: int = 5) -> List[VerdictResult]:
        """Information checker: verify multiple claims (alias for verify_claims)."""
        return self.verify_claims(claims, top_k)
