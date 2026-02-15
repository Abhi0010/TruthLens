"""AI-generated text detector using heuristics (optional model)."""

import re
from dataclasses import dataclass
from typing import List

from .preprocessing import split_sentences


@dataclass
class AIDetectionResult:
    """Result of AI-generated text analysis."""

    ai_likelihood: float  # 0-1
    indicators: List[str]


def _unique_word_ratio(text: str) -> float:
    """Unique words / total words. AI text often has lower ratio (repetition)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 1.0
    return len(set(words)) / len(words)


def _sentence_length_variance(sentences: List[str]) -> float:
    """Variance of sentence lengths. Human text often varies more."""
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return var ** 0.5  # std dev


def _avg_sentence_length(sentences: List[str]) -> float:
    """Average words per sentence."""
    if not sentences:
        return 0.0
    total = sum(len(s.split()) for s in sentences)
    return total / len(sentences)


def _has_generic_phrases(text: str) -> int:
    """Count common AI-like generic phrases."""
    phrases = [
        r"\bit's important to note\b",
        r"\bin conclusion\b",
        r"\bhowever, it is (worth noting|important)\b",
        r"\badditionally\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bcomprehensive(ly)?\b",
        r"\bdelve (into|deeper)\b",
        r"\bnavigate (the|these)\b",
        r"\blandscape\b",
        r"\bnuanced\b",
        r"\bholistic\b",
        r"\bleverage\b",
        r"\bparadigm\b",
    ]
    count = sum(1 for p in phrases if re.search(p, text, re.I))
    return count


def _paragraph_structure(text: str) -> float:
    """AI often uses uniform short paragraphs."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return 0.5
    lengths = [len(p.split()) for p in paras]
    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    # Low variance = more uniform = more AI-like
    std = var ** 0.5
    return max(0, 1 - std / 50)  # normalize


def detect_ai_generated(text: str) -> AIDetectionResult:
    """
    Estimate AI-generated likelihood using heuristics.
    - Unique word ratio (repetition)
    - Sentence length variance
    - Generic AI-like phrases
    - Paragraph uniformity
    """
    if not text or not str(text).strip():
        return AIDetectionResult(ai_likelihood=0.0, indicators=[])

    text = str(text).strip()
    sentences = split_sentences(text)
    indicators: List[str] = []

    # Unique word ratio
    uwr = _unique_word_ratio(text)
    if uwr < 0.5:
        indicators.append("Low lexical diversity (repetitive vocabulary)")
    elif uwr < 0.65:
        indicators.append("Moderate lexical diversity")

    # Sentence length variance
    var = _sentence_length_variance(sentences)
    if var < 5 and len(sentences) >= 3:
        indicators.append("Uniform sentence lengths")
    elif var > 15:
        indicators.append("Variable sentence structure (more human-like)")

    # Generic phrases
    gen = _has_generic_phrases(text)
    if gen >= 3:
        indicators.append(f"Multiple generic AI-style phrases ({gen})")
    elif gen >= 1:
        indicators.append("Some generic phrasing")

    # Paragraph uniformity
    para_score = _paragraph_structure(text)
    if para_score > 0.6:
        indicators.append("Uniform paragraph structure")

    # Average sentence length (very long = possible AI)
    avg_len = _avg_sentence_length(sentences)
    if avg_len > 25 and len(sentences) >= 2:
        indicators.append("Long, complex sentences")

    # Compute composite score
    score = 0.0
    score += (1 - uwr) * 0.25  # low diversity -> higher AI score
    score += (1 - min(1, var / 15)) * 0.2  # low variance -> higher
    score += min(0.25, gen * 0.08)
    score += para_score * 0.15
    score += min(0.15, (avg_len - 15) / 50) if avg_len > 15 else 0

    score = min(1.0, max(0.0, score))

    if not indicators:
        indicators.append("No strong AI-generation indicators")

    return AIDetectionResult(ai_likelihood=score, indicators=indicators)
