"""Trainer quiz rounds: Misinformation, Normal news, Phishing & crypto."""

from typing import List, Dict, Any


def _rounds_misinformation() -> List[Dict[str, Any]]:
    """Rounds for the Misinformation quiz."""
    return [
        {
            "id": 1,
            "message": "BREAKING: The vaccine contains microchips! BIG PHARMA doesn't want you to know. SHARE THIS NOW!!! 100% proven. Doctors HATE this one secret.",
            "highlights": ["BREAKING", "microchips!", "BIG PHARMA", "SHARE THIS NOW!!!", "100% proven", "Doctors HATE"],
            "correct_tactic": "Misinformation / conspiracy",
            "wrong_tactics": ["Legitimate health warning", "Satire", "Normal news"],
            "explanation": "Conspiracy language ('microchips', 'BIG PHARMA doesn't want you to know'), fake urgency ('SHARE THIS NOW', '100% proven'), and conflict framing ('Doctors HATE') are hallmarks of health misinformation.",
            "what_to_do": "Don't share. Check official health sources (CDC, WHO). Real health info doesn't rely on 'secrets' or urgency.",
        },
        {
            "id": 2,
            "message": "BREAKING: They don't want you to know the truth! SHARE THIS NOW!!! 100% proven. Mainstream media is hiding what really happened.",
            "highlights": ["SHARE THIS NOW!!!", "100% proven", "Mainstream media is hiding"],
            "correct_tactic": "Emotional manipulation / urgency",
            "wrong_tactics": ["Legitimate news framing", "Neutral fact statement", "Expert citation"],
            "explanation": "Phrases like 'SHARE THIS NOW' and '100% proven' create urgency and bypass critical thinking. Claiming 'mainstream media is hiding' the truth is a common disinformation tactic.",
            "what_to_do": "Pause before sharing. Check multiple sources. Be skeptical of 'they don't want you to know' claims.",
        },
        {
            "id": 3,
            "message": "Scientists CONFIRM this one trick cures everything. Doctors hate it! Click before it's banned.",
            "highlights": ["CONFIRM", "doctors hate it", "before it's banned"],
            "correct_tactic": "False authority / scarcity",
            "wrong_tactics": ["Peer-reviewed summary", "Balanced reporting", "Transparent sourcing"],
            "explanation": "Fake urgency ('before it's banned'), false conflict ('doctors hate it'), and oversold claims ('cures everything') are hallmarks of health misinformation.",
            "what_to_do": "Real science doesn't need 'tricks' or bans. Look for peer-reviewed studies and official health sources.",
        },
        {
            "id": 4,
            "message": "Viral post: Everyone at [Company] is getting laid off next week. My cousin works there. Spread the word.",
            "highlights": ["Viral post", "My cousin works there", "Spread the word"],
            "correct_tactic": "Unverified rumor / anonymity",
            "wrong_tactics": ["Official company statement", "Named source with role", "Verified announcement"],
            "explanation": "Rumors framed as 'viral' with unnamed sources ('my cousin') encourage sharing before verification. Real layoffs are announced officially.",
            "what_to_do": "Don't spread unverified layoff rumors. Wait for official announcements or named, credible sources.",
        },
    ]


def _rounds_normal_news() -> List[Dict[str, Any]]:
    """Rounds for the Normal news quiz (spotting trustworthy content)."""
    return [
        {
            "id": 1,
            "message": "The Federal Reserve announced a 0.25% interest rate hike today. Markets reacted with modest gains. Economists had expected the move.",
            "highlights": ["Federal Reserve", "0.25%", "Markets reacted", "Economists had expected"],
            "correct_tactic": "Neutral fact statement",
            "wrong_tactics": ["Emotional manipulation", "False urgency", "Unverified rumor"],
            "explanation": "Specific numbers, named institutions, and balanced framing ('economists had expected') indicate standard news reporting.",
            "what_to_do": "This is how reliable news reads: specific, sourced, and not pushing you to act or share.",
        },
        {
            "id": 2,
            "message": "According to the health ministry, vaccination rates rose 5% in the last quarter. The data was published in the weekly bulletin.",
            "highlights": ["According to the health ministry", "5%", "published in the weekly bulletin"],
            "correct_tactic": "Transparent sourcing",
            "wrong_tactics": ["Anonymous insider", "Urgency to share", "Doctors hate this"],
            "explanation": "Attribution to an official body and a named publication (weekly bulletin) are signs of transparent, verifiable reporting.",
            "what_to_do": "When you see 'according to [named source]' and a clear publication, you can verify the claim.",
        },
        {
            "id": 3,
            "message": "The company reported revenue of $2.1B, in line with analyst estimates. CEO Jane Smith commented on the earnings call.",
            "highlights": ["$2.1B", "in line with analyst estimates", "CEO Jane Smith", "earnings call"],
            "correct_tactic": "Named source with role",
            "wrong_tactics": ["Viral post", "Someone said", "Leaked document"],
            "explanation": "Specific figures, comparison to estimates, and a named executive with a clear channel (earnings call) indicate credible business reporting.",
            "what_to_do": "Financial and corporate news that names people and events is easier to verify and less likely to be manipulation.",
        },
    ]


def _rounds_phishing_crypto() -> List[Dict[str, Any]]:
    """Rounds for the Phishing & crypto quiz."""
    return [
        {
            "id": 1,
            "message": "Dear Customer, Your bank account has been suspended. Act within 24 hours to verify your identity or lose access. Click here: https://fake-bank-secure.com/verify.",
            "highlights": ["suspended", "Act within 24 hours", "lose access", "Click here", "fake-bank-secure.com"],
            "correct_tactic": "Urgency / fake authority",
            "wrong_tactics": ["Legitimate security notice", "Neutral reminder", "Official communication"],
            "explanation": "Banks don't suspend accounts by email with a random link. Urgent deadlines and 'or lose access' are classic phishing pressure tactics.",
            "what_to_do": "Never click links in such emails. Log in via the official app or type the bank URL yourself. Call the number on your card if unsure.",
        },
        {
            "id": 2,
            "message": "URGENT! You've won 5 Bitcoin! Send 0.1 BTC to this address to verify and claim. Act now!!!",
            "highlights": ["URGENT!", "won 5 Bitcoin", "Send 0.1 BTC", "to verify and claim", "Act now!!!"],
            "correct_tactic": "Crypto scam / advance fee",
            "wrong_tactics": ["Legitimate prize", "Official lottery", "Free giveaway"],
            "explanation": "Real prizes never require you to send crypto 'to verify.' This is an advance-fee scam: you send 0.1 BTC and get nothing.",
            "what_to_do": "Never send crypto to 'claim' a prize. Legitimate giveaways don't ask for payment. Report and ignore.",
        },
        {
            "id": 3,
            "message": "Hi, I'm from IT Support. We need to fix your account now. Please reply with your password and 2FA code so we can resolve the issue.",
            "highlights": ["IT Support", "fix your account now", "reply with your password", "2FA code"],
            "correct_tactic": "Credential harvesting",
            "wrong_tactics": ["Official IT process", "Security update", "Password reset"],
            "explanation": "Real IT or security teams never ask for your password or 2FA code by email or chat. Anyone who does is trying to steal your account.",
            "what_to_do": "Never share passwords or 2FA codes. Contact IT through a known channel (e.g. company portal) if you get such a request.",
        },
    ]


def get_trainer_rounds(quiz_type: str) -> List[Dict[str, Any]]:
    """Return the list of rounds for the given quiz type."""
    if quiz_type == "misinformation":
        return _rounds_misinformation()
    if quiz_type == "normal_news":
        return _rounds_normal_news()
    if quiz_type == "phishing_crypto":
        return _rounds_phishing_crypto()
    return _rounds_misinformation()
