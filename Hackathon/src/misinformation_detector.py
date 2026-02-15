"""Lightweight misinformation detector using rule-based signals."""

import re
from dataclasses import dataclass
from typing import List, Tuple

from .preprocessing import clean_text


@dataclass
class MisinformationResult:
    """Result of misinformation analysis."""

    risk_score: float  # 0-1
    reasons: List[str]


# Sensational/urgent language patterns
SENSATIONAL_PATTERNS = [
    r"\b(breaking|urgent|alert|shocking|exclusive|revealed)\b",
    r"\b(they don't want you to know|they're hiding|cover.?up)\b",
    r"\b(share this|forward this|tell everyone|spread the word)\b",
    r"\b(must see|you won't believe|mind.?blowing)\b",
    r"\b(conspiracy|mainstream media|fake news)\b",
    r"\b(100% (true|guaranteed|proven))\b",
    r"\b(doctors hate|big pharma|big tech)\b",
]

# Emotionally charged words
EMOTIONAL_WORDS = [
    "danger", "scandal", "outrage", "horror", "terrifying", "devastating",
    "exposed", "secret", "hidden", "truth", "lies", "corruption",
    "crisis", "emergency", "panic", "fear", "warning",
]

# Compile patterns
SENSATIONAL_RE = [re.compile(p, re.IGNORECASE) for p in SENSATIONAL_PATTERNS]
EMOTIONAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in EMOTIONAL_WORDS) + r")\b",
    re.IGNORECASE,
)


def _all_caps_ratio(text: str) -> float:
    """Ratio of uppercase letters to total letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _excessive_punctuation(text: str) -> float:
    """Score for excessive punctuation (!!! ??? etc)."""
    exclam = len(re.findall(r"!+", text))
    quest = len(re.findall(r"\?+", text))
    return min(1.0, (exclam + quest) / 3)


def _sensational_matches(text: str) -> int:
    """Count sensational pattern matches."""
    return sum(1 for p in SENSATIONAL_RE if p.search(text))


def _emotional_matches(text: str) -> int:
    """Count emotionally charged word matches."""
    return len(EMOTIONAL_RE.findall(text))


def detect_misinformation(text: str) -> MisinformationResult:
    """
    Use rule-based signals to estimate misinformation risk 0-1.
    Returns risk score and list of reasons.
    """
    if not text:
        return MisinformationResult(risk_score=0.0, reasons=[])

    text = clean_text(text)
    reasons: List[str] = []

    # All-caps ratio
    caps = _all_caps_ratio(text)
    if caps > 0.3:
        reasons.append(f"High proportion of ALL CAPS ({caps:.0%})")
    elif caps > 0.15:
        reasons.append(f"Moderate use of caps ({caps:.0%})")

    # Excessive punctuation
    punct = _excessive_punctuation(text)
    if punct > 0.3:
        reasons.append("Excessive punctuation (!!! ???)")

    # Sensational language
    sens = _sensational_matches(text)
    if sens >= 2:
        reasons.append(f"Sensational/urgent language ({sens} patterns)")
    elif sens == 1:
        reasons.append("Some sensational phrasing")

    # Emotional words
    emot = _emotional_matches(text)
    if emot >= 3:
        reasons.append(f"Multiple emotionally charged words ({emot})")
    elif emot >= 1:
        reasons.append("Emotionally charged vocabulary")

    # Share/forward prompts
    if re.search(r"\b(share|forward|tell everyone|spread)\b", text, re.I):
        reasons.append("Encourages viral sharing")

    # "They don't want you to know"
    if re.search(r"don't want you to know|they're hiding|cover.?up", text, re.I):
        reasons.append("Conspiracy-style framing")

    # Compute composite score (0-1)
    score = 0.0
    score += min(0.25, caps * 0.5)
    score += min(0.15, punct * 0.5)
    score += min(0.25, sens * 0.12)
    score += min(0.2, emot * 0.06)
    score += 0.1 if "share" in text.lower() or "forward" in text.lower() else 0
    score += 0.15 if "don't want you to know" in text.lower() else 0

    score = min(1.0, score)

    if not reasons:
        reasons.append("No strong misinformation signals detected")

    return MisinformationResult(risk_score=score, reasons=reasons)
