"""Trainer quiz rounds: Misinformation, Normal news, Phishing & crypto.
Each round presents content the user might see and asks: What should you do?
Options are actions (what to do), not tactic labels."""

from typing import List, Dict, Any


def _rounds_misinformation() -> List[Dict[str, Any]]:
    """Rounds for the Fact Check quiz — what should you do when you see this?"""
    return [
        {
            "id": 1,
            "message": "New study finds link between processed meat and heart disease. Experts say the results are concerning. Share if you want your family to be safe.",
            "correct_tactic": "Look up the study (who funded it, sample size, peer review) and what health authorities say before sharing or acting.",
            "wrong_tactics": [
                "Share it so your family can avoid the risk.",
                "Forward it with 'Worth looking into' so people can decide.",
                "Comment that you'll wait for more studies before changing habits.",
            ],
            "explanation": "Vague 'new study' and 'experts say' with a share prompt is a common pattern: the study may be preliminary, cherry-picked, or misrepresented. Sharing spreads the claim before anyone checks. The right move is to not share and verify—who did the study, was it peer-reviewed, what do official health bodies say—then act on that.",
            "what_to_do": "Don't share health claims that don't name the study or experts. Check the source and what official health authorities say.",
        },
        {
            "id": 2,
            "message": "The health minister said exactly this in 2019: 'We can't afford to fix the hospitals.' No wonder people are angry. Pass it on.",
            "correct_tactic": "Find the full quote and context (when, where, what was asked) before you believe or pass it on.",
            "wrong_tactics": [
                "Share it so people see what was really said.",
                "Forward it and add 'Need to verify but if true this is bad.'",
                "Search for a fact-check, then share the fact-check only.",
            ],
            "explanation": "Quote-mining—using a real quote stripped of context—is a standard way to mislead. 'Pass it on' pushes sharing before verification. The right move is to not share and look up the full context (full quote, occasion, question asked). Then you can share with context if it still matters.",
            "what_to_do": "Out-of-context quotes are easy to weaponize. Find the full quote and context before sharing.",
        },
        {
            "id": 3,
            "message": "Cases of norovirus have risen 300% in the last week. Officials are not telling you the full story. Here's what's really going on.",
            "correct_tactic": "Check official data (baseline, how 'case' is defined, what agencies say) before you share or draw conclusions.",
            "wrong_tactics": [
                "Share it so people see the real numbers.",
                "Forward only the statistic and ask others to verify.",
                "Comment that 300% sounds alarming and tag a fact-checker.",
            ],
            "explanation": "Big percentage jumps without baseline or definition (e.g. 300% from 10 to 40) can be misleading. 'Officials are not telling you' invites distrust of actual data. The right move is to not share and check official sources—what's the denominator, how are cases defined, what's the trend—then share that if you want.",
            "what_to_do": "Verify stats with official sources. Percentages without context or 'what they're not telling you' framing are red flags.",
        },
        {
            "id": 4,
            "message": "According to sources close to the matter, the central bank is about to announce an emergency rate cut. We're hearing this from multiple people. More soon.",
            "correct_tactic": "Wait for the central bank's official statement or named reporting; don't share or act on the rumor.",
            "wrong_tactics": [
                "Share it so people are prepared for the announcement.",
                "Forward it only to people who work in that field.",
                "Save it and share only after the announcement is confirmed.",
            ],
            "explanation": "Anonymous 'sources close to the matter' plus 'we're hearing from multiple people' is designed to sound like insider news while avoiding accountability. Many such 'scoops' never materialize. The right move is to not share or act until there's an official announcement or reporting with named sources.",
            "what_to_do": "Don't spread unconfirmed 'insider' claims. Wait for official statements or named, on-the-record reporting.",
        },
        {
            "id": 5,
            "message": "This happened to my friend's sister. She was overcharged at the hospital and when she complained they threatened to send the debt to collections. She tried to get help but no one listened. We need to get this out there so it doesn't happen to anyone else.",
            "correct_tactic": "Check whether this has been reported by journalists or verified before sharing it as fact.",
            "wrong_tactics": [
                "Share it so the story gets attention and things might change.",
                "Forward it with a trigger warning so people can decide.",
                "Comment that you're sorry this happened and tag relevant organizations.",
            ],
            "explanation": "Emotional, anecdotal stories can be true, exaggerated, or fabricated. 'We need to get this out there' pushes sharing before verification. Real harms get reported by journalists who verify and give the other side a chance. The right move is to not share as fact—look for verified reporting or official channels, then share that if you want to help.",
            "what_to_do": "Anecdotes aren't evidence. Before amplifying, look for verified reporting or official responses.",
        },
    ]


def _rounds_normal_news() -> List[Dict[str, Any]]:
    """Rounds for the Normal news quiz — what do you do first when you see this? (Action-oriented, reliable content.)"""
    return [
        {
            "id": 1,
            "message": "The Federal Reserve announced a 0.25% interest rate hike today. Markets reacted with modest gains. Economists had widely expected the move.",
            "correct_tactic": "Check who said it and where it's from (e.g. which outlet, when)—then read and use or share if the source is one you trust.",
            "wrong_tactics": [
                "Share it right away so others see it.",
                "Skip to the comments to see what others think first.",
                "Search for whether this story is real or fake before doing anything else.",
            ],
            "explanation": "When you see news, a good first step is to note the source and when it's from. Here you already have a named institution (Federal Reserve) and specific numbers. Checking the outlet and date is a habit that works for any story—then you can read and share with confidence.",
            "what_to_do": "When you see news, first check who said it and where it's from. Then you can read, use, or share it.",
        },
        {
            "id": 2,
            "message": "According to the health ministry, vaccination rates rose 5% in the last quarter. The data was published in the weekly epidemiological bulletin.",
            "correct_tactic": "Note where the claim comes from (health ministry, bulletin)—then you can read it and, if you want, verify using that publication.",
            "wrong_tactics": [
                "Forward it to friends first, then read the details.",
                "Look for another article that says the same before you read this one.",
                "Decide if the story is real or fake before checking the source.",
            ],
            "explanation": "A good first step is to see where the info comes from. Here it's already stated (health ministry, weekly epidemiological bulletin). Once you note that, you can read and optionally check the bulletin yourself—no need to hunt for a second source before engaging.",
            "what_to_do": "When you see a claim, first note who said it and where it was published. Then read and use it; verify from that source if you want.",
        },
        {
            "id": 3,
            "message": "The company reported revenue of $2.1B, in line with analyst estimates. CEO Jane Smith commented on the earnings call.",
            "correct_tactic": "See who is quoted and where the info came from (e.g. earnings call)—then read and use or share if it fits your needs.",
            "wrong_tactics": [
                "Share the headline first, then read the full article.",
                "Wait for someone else to confirm the numbers before you read it.",
                "Ignore it until you can prove the company really said this.",
            ],
            "explanation": "When you see business news, a good first step is to note who said what and where (here: CEO, earnings call). That gives you a clear, checkable source. Then you can read and use the story without waiting for another outlet to confirm.",
            "what_to_do": "When you see business news, first note who is quoted and where the info came from. Then read and use it.",
        },
        {
            "id": 4,
            "message": "A spokesperson for the agency confirmed the policy change. The full report is available on the agency's official website.",
            "correct_tactic": "Note who confirmed it (spokesperson) and that the full report is linked—then read the summary and open the report if you want more detail.",
            "wrong_tactics": [
                "Share the headline and link without reading the summary first.",
                "Try to find out if the policy change is true before looking at the source.",
                "Read the full report before you read the summary.",
            ],
            "explanation": "A good first step is to see who said it and whether you can get more (here: spokesperson + official report link). Note the source and the link, read the summary, and open the report if you need details—that order works for any similar story.",
            "what_to_do": "When you see a policy or official update, first note who said it and where the full report is. Then read the summary and the report if you need more.",
        },
        {
            "id": 5,
            "message": "According to the latest survey, 62% of respondents said they support the measure. The margin of error is 3%.",
            "correct_tactic": "Check where the data comes from and whether limits are stated (e.g. margin of error)—then read and use or share it.",
            "wrong_tactics": [
                "Share the 62% figure first, then look up the source.",
                "Search for a fact-check to see if the survey is real before reading.",
                "Ignore the margin of error and treat the number as exact.",
            ],
            "explanation": "When you see survey or poll results, a good first step is to note the source and whether they state limitations (like margin of error). Here they do—so you can read and use the story, and keep the margin in mind when you share or discuss.",
            "what_to_do": "When you see survey or poll data, first check where it's from and whether they state the margin of error or other limits. Then read and use it.",
        },
    ]


def _rounds_phishing_crypto() -> List[Dict[str, Any]]:
    """Rounds for the Phishing & crypto quiz — what should you do when you see this?"""
    return [
        {
            "id": 1,
            "message": "Dear Customer, Your account has been suspended. You must verify your identity within 24 hours or lose access permanently. Click here to verify: https://secure-bank-verify.com/login.",
            "correct_tactic": "Don't click the link; log in via the bank's official app or type the URL yourself.",
            "wrong_tactics": [
                "Click the link to verify before you lose access.",
                "Forward the email to the bank to ask if it's real.",
                "Reply with your date of birth to prove your identity.",
            ],
            "explanation": "Banks don't suspend accounts by email with a generic link. Clicking can steal your credentials or install malware. Forwarding to the bank doesn't protect you if you already clicked. The safe move is to never click—log in via the official app or type the bank's URL yourself, and call the number on your card if unsure.",
            "what_to_do": "Never click links in account emails. Use the official app or type the URL yourself. Call the number on your card if unsure.",
        },
        {
            "id": 2,
            "message": "CONGRATULATIONS! You've been selected to receive 5 Bitcoin. To claim, send 0.1 BTC to the address below for verification. Offer expires in 48 hours.",
            "correct_tactic": "Don't send any crypto; ignore and report the message.",
            "wrong_tactics": [
                "Send the 0.1 BTC to claim the prize.",
                "Reply to ask for an official claim form.",
                "Forward it to a crypto expert to verify.",
            ],
            "explanation": "Real prizes never require you to send crypto or money 'to verify' or 'claim.' This is an advance-fee scam: you send 0.1 BTC and get nothing. Replying or forwarding doesn't protect you—the only safe move is to ignore and report. Legitimate giveaways don't ask for payment.",
            "what_to_do": "Never send crypto or money to 'claim' a prize. Ignore and report. Legitimate giveaways don't ask for payment.",
        },
        {
            "id": 3,
            "message": "Hi, this is IT Support. We're seeing suspicious activity on your account. To secure it, please reply with your current password and the 2FA code you just received. We'll fix it from our end.",
            "correct_tactic": "Don't reply with password or 2FA; contact IT through a known channel (e.g. company portal).",
            "wrong_tactics": [
                "Reply with your password so they can fix the issue.",
                "Forward the message to IT to verify it's real.",
                "Reply asking for a ticket number first.",
            ],
            "explanation": "Real IT or security never asks for your password or 2FA code by email, chat, or phone. Anyone who does is trying to take over your account. Forwarding to IT doesn't undo the risk if you already sent credentials. The safe move is to not reply with any secrets and contact IT through a known channel (e.g. company portal or the number on the official website).",
            "what_to_do": "Never share passwords or 2FA codes with anyone. Contact IT or support through a known channel if you're worried.",
        },
        {
            "id": 4,
            "message": "Your package is at the depot. A delivery fee of $2.99 is required. Pay within 12 hours or the package will be returned to sender. Click here to pay.",
            "correct_tactic": "Don't click the link; check tracking on the carrier's official website (typed yourself).",
            "wrong_tactics": [
                "Click and pay the fee so your package isn't returned.",
                "Call the number in the email to verify.",
                "Forward the email to the carrier's support to ask if it's real.",
            ],
            "explanation": "Real carriers don't demand payment via a random link with a short deadline. Clicking can steal payment details or install malware. Calling a number in the email can connect you to scammers. The safe move is to ignore the email and check tracking on the carrier's official site (URL typed yourself).",
            "what_to_do": "Check tracking on the carrier's official website (typed yourself). Never pay through links in unsolicited emails or texts.",
        },
        {
            "id": 5,
            "message": "You've won our exclusive giveaway! To release your prize we need a small processing fee ($50). This is standard policy. Send payment within 24 hours to secure your winnings.",
            "correct_tactic": "Don't pay; ignore and report. Legitimate prizes don't require a fee.",
            "wrong_tactics": [
                "Pay the fee to release your winnings.",
                "Reply to ask for an official invoice.",
                "Forward to a friend to get a second opinion.",
            ],
            "explanation": "Real giveaways don't charge a 'processing fee' or 'release' fee. Once you pay, there is no prize—it's an advance-fee scam. Asking for an invoice or a second opinion doesn't make it safe. The only safe move is to not pay, ignore, and report.",
            "what_to_do": "Never pay money or send crypto to 'release' or 'claim' a prize. Ignore and report. Legitimate offers don't require you to pay first.",
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
