"""Social engineering and scam/phishing detector."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from .preprocessing import clean_text, extract_urls


class RiskLevel(str, Enum):
    """Risk level for social engineering."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class SocialEngineeringResult:
    """Result of social engineering analysis."""

    risk_level: RiskLevel
    red_flags: List[str]
    safer_rewrite_suggestion: str


# Urgency patterns
URGENCY_PATTERNS = [
    r"\b(act now|act immediately|within 24 hours|within 48 hours)\b",
    r"\b(urgent|asap|right away|don't wait)\b",
    r"\b(limited time|expires soon|last chance)\b",
    r"\b(account (will be|has been) (suspended|closed|locked))\b",
    r"\b(verify (now|immediately)|confirm (now|immediately))\b",
]

# Authority impersonation
AUTHORITY_PATTERNS = [
    r"\b(IRS|Internal Revenue Service|tax (authority|office))\b",
    r"\b(bank|credit union|financial institution)\b",
    r"\b(police|FBI|law enforcement|government)\b",
    r"\b(tech support|Microsoft|Apple|Amazon)\b",
    r"\b(Social Security|SSA|Medicare)\b",
]

# Credential requests
CREDENTIAL_PATTERNS = [
    r"\b(password|login|credentials|account (details|info))\b",
    r"\b(verify your (identity|account|email))\b",
    r"\b(click (here|below) to (log in|verify|confirm))\b",
    r"\b(enter your (password|PIN|SSN))\b",
]

# Money/gift card requests
MONEY_PATTERNS = [
    r"\b(wire (transfer|money)|send (money|cash))\b",
    r"\b(gift card|iTunes|Amazon|Google Play)\b",
    r"\b(pay (now|immediately)|payment (required|due))\b",
    r"\b(bitcoin|crypto| cryptocurrency)\b",
    r"\b(prize|winner|you've won|claim your)\b",
]

# Compile
URGENCY_RE = [re.compile(p, re.IGNORECASE) for p in URGENCY_PATTERNS]
AUTHORITY_RE = [re.compile(p, re.IGNORECASE) for p in AUTHORITY_PATTERNS]
CREDENTIAL_RE = [re.compile(p, re.IGNORECASE) for p in CREDENTIAL_PATTERNS]
MONEY_RE = [re.compile(p, re.IGNORECASE) for p in MONEY_PATTERNS]


def _count_matches(text: str, patterns: list) -> int:
    """Count how many patterns match."""
    return sum(1 for p in patterns if p.search(text))


def detect_social_engineering(text: str) -> SocialEngineeringResult:
    """
    Identify phishing/scam signals.
    Returns risk level, red flags, and a safer rewrite suggestion.
    """
    if not text:
        return SocialEngineeringResult(
            risk_level=RiskLevel.LOW,
            red_flags=[],
            safer_rewrite_suggestion="No content to analyze.",
        )

    text = clean_text(text)
    red_flags: List[str] = []

    # Urgency
    u = _count_matches(text, URGENCY_RE)
    if u >= 2:
        red_flags.append("Strong urgency pressure (act now, limited time)")
    elif u == 1:
        red_flags.append("Urgency language detected")

    # Authority impersonation
    a = _count_matches(text, AUTHORITY_RE)
    if a >= 1:
        red_flags.append("Possible authority impersonation (bank, IRS, etc.)")

    # Credential requests
    c = _count_matches(text, CREDENTIAL_RE)
    if c >= 2:
        red_flags.append("Multiple credential/verification requests")
    elif c == 1:
        red_flags.append("Request for login/verification")

    # Money/gift card
    m = _count_matches(text, MONEY_RE)
    if m >= 1:
        red_flags.append("Money/gift card/crypto request")

    # Suspicious links
    urls = extract_urls(text)
    if urls:
        red_flags.append(f"Suspicious links present ({len(urls)} URL(s))")

    # Generic greeting (common in phishing)
    if re.search(r"\b(dear (customer|user|valued))\b", text, re.I):
        red_flags.append("Generic greeting (common in phishing)")

    # Determine risk level
    score = len(red_flags)
    if score >= 4:
        level = RiskLevel.HIGH
    elif score >= 2:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    # Safer rewrite suggestion
    suggestion = _generate_safer_rewrite(text, red_flags)

    return SocialEngineeringResult(
        risk_level=level,
        red_flags=red_flags if red_flags else ["No obvious scam signals detected"],
        safer_rewrite_suggestion=suggestion,
    )


def _generate_safer_rewrite(original: str, red_flags: List[str]) -> str:
    """Generate a safer rewrite suggestion based on red flags."""
    if not red_flags or red_flags[0] == "No obvious scam signals detected":
        return "Original text appears low-risk. No rewrite needed."

    tips = []
    if any("urgency" in f.lower() for f in red_flags):
        tips.append("Remove urgency language—legitimate organizations don't pressure you to act immediately.")
    if any("authority" in f.lower() for f in red_flags):
        tips.append("Verify sender through official channels—don't trust contact info in the message.")
    if any("credential" in f.lower() or "login" in f.lower() for f in red_flags):
        tips.append("Never enter passwords via links in messages—go directly to the official site.")
    if any("money" in f.lower() or "gift" in f.lower() for f in red_flags):
        tips.append("Legitimate organizations rarely ask for gift cards or wire transfers.")
    if any("link" in f.lower() for f in red_flags):
        tips.append("Hover over links to check URLs before clicking—or avoid clicking entirely.")

    return "Safer approach: " + " ".join(tips)
