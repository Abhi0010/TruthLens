"""Clarion - Streamlit webapp for misinformation & manipulation analysis."""

import sys
from pathlib import Path

# Ensure src is on path and load .env before any code reads env vars
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import random
import threading
import time

import streamlit as st

from src.pipeline import run_pipeline
from src.rag_verifier import RAGVerifier
from src.report_generator import generate_html_report
from src.trainer_rounds import get_trainer_rounds
from src.url_fetcher import fetch_and_extract
from src.document_upload import extract_text_from_file
from src.backboard_client import is_configured as backboard_configured, summarize_document


@st.cache_resource
def get_rag_verifier() -> RAGVerifier:
    """Cached RAG verifier - builds index once, reuses across sessions."""
    return RAGVerifier()


@st.cache_data(ttl=300)
def run_analysis(text: str, content_type: str, _v: int = 5) -> dict:
    """Cached pipeline - same input returns instantly."""
    verifier = get_rag_verifier()
    result = run_pipeline(text, content_type, rag_verifier=verifier)
    return {
        "correct_count": result.correct_count,
        "incorrect_count": result.incorrect_count,
        "response_confidence": result.response_confidence,
        "top_reasons": result.top_reasons,
        "fact_check_summary": result.fact_check_summary,
        "claims": [
            {"claim": c.claim, "verdict": c.verdict, "evidence": c.evidence, "similarity": c.similarity}
            for c in result.claims
        ],
        "misinformation": {"risk_score": result.misinformation.risk_score, "reasons": result.misinformation.reasons},
        "social_engineering": {
            "risk_level": result.social_engineering.risk_level.value,
            "red_flags": result.social_engineering.red_flags,
            "safer_rewrite_suggestion": result.social_engineering.safer_rewrite_suggestion,
        },
        "ai_detection": {"ai_likelihood": result.ai_detection.ai_likelihood, "indicators": result.ai_detection.indicators},
        "evidence_passages": result.evidence_passages,
        "verification_mode": result.verification_mode,
        "citations": result.citations,
    }


def _run_analysis_in_thread(text: str, content_type: str) -> dict:
    """Run analysis in a thread so Streamlit's 'Running' UI shows join() not run_analysis()."""
    result_holder = {}

    def _run():
        result_holder["result"] = run_analysis(text, content_type)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()
    return result_holder["result"]


# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Clarion",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Single source of truth for modes: Analyzer section + Trainer quiz type
# Add or reorder entries here; home cards, sidebar, and Analyzer/Trainer stay in sync
SECTIONS = [
    {
        "id": "fact_check",
        "title": "📊 Fact Check",
        "description": "See how many claims are correct vs incorrect and how confident we are in the response.",
        "sample_label": "Fact Check",
        "quiz_type": "misinformation",
    },
    {
        "id": "scam_phishing",
        "title": "🛡️ Scam & Phishing",
        "description": "Detect social engineering, urgency tactics, and red flags in emails and messages.",
        "sample_label": "Phishing scams",
        "quiz_type": "phishing_crypto",
    },
    {
        "id": "normal_news",
        "title": "📰 Normal News",
        "description": "Extract factual claims and check them against web sources or our knowledge base.",
        "sample_label": "Normal news",
        "quiz_type": "normal_news",
    },
]
SECTION_CONTENT = {s["id"]: {"title": s["title"], "description": s["description"]} for s in SECTIONS}
QUICK_LOAD_TO_QUIZ = {s["sample_label"]: s["quiz_type"] for s in SECTIONS}
QUICK_LOAD_TO_SECTION = {s["sample_label"]: s["id"] for s in SECTIONS}

# Session state
if "show_analyze" not in st.session_state:
    st.session_state.show_analyze = False
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "fact_check"  # default
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_input_hash" not in st.session_state:
    st.session_state.last_input_hash = None
if "suggested_quiz_type" not in st.session_state:
    st.session_state.suggested_quiz_type = "misinformation"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "analyzer"
# Trainer quiz state
if "quiz_initialized" not in st.session_state:
    st.session_state.quiz_initialized = False
if "quiz_type" not in st.session_state:
    st.session_state.quiz_type = "misinformation"
if "rounds" not in st.session_state:
    st.session_state.rounds = []
if "current_round" not in st.session_state:
    st.session_state.current_round = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "reveal" not in st.session_state:
    st.session_state.reveal = False
if "history" not in st.session_state:
    st.session_state.history = []
if "source_url" not in st.session_state:
    st.session_state.source_url = ""
if "source_label" not in st.session_state:
    st.session_state.source_label = ""

# Corpus: 10 samples per category. Each entry is (text, type_label).
# Fact Check: clear misinformation (conspiracy, viral) vs verifiable factual claims (named sources).
# Phishing: strong red flags (urgency, credentials, fake links, too-good-to-be-true).
# Normal news: neutral, factual, verifiable claims with institutions/dates.
SAMPLE_INPUTS: dict[str, list[tuple[str, str]]] = {
    "Fact Check": [
        (
            "The Earth orbits the Sun. This was established by astronomers and is taught in science curricula worldwide.",
            "True",
        ),
        (
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure at sea level.",
            "True",
        ),
        (
            "Paris is the capital of France. The French government and international references confirm this.",
            "True",
        ),
        (
            "Light travels at approximately 299,792 kilometers per second in a vacuum. This is a fundamental constant in physics.",
            "True",
        ),
        (
            "The Pacific Ocean is the largest ocean on Earth by surface area. Geographic and oceanographic sources confirm this.",
            "True",
        ),
        (
            "The Great Wall of China is visible from the Moon. NASA and astronauts have confirmed this.",
            "Fake",
        ),
        (
            "Humans only use 10% of their brains. The rest is unused potential that we could tap into.",
            "Fake",
        ),
        (
            "Bats are blind. They rely entirely on echolocation and cannot see.",
            "Fake",
        ),
        (
            "Christopher Columbus proved the Earth is round. Before his voyage, everyone believed the world was flat.",
            "Fake",
        ),
        (
            "Vitamin C cures the common cold. Major health authorities recommend megadoses for cold prevention and cure.",
            "Fake",
        ),
    ],
    "Phishing scams": [
        (
            "URGENT - Your bank account has been suspended. You must verify your identity within 24 hours or lose access permanently. "
            "Click here now: https://secure-bank-verify.com/confirm. Do not delay. We have already flagged your account.",
            "Email",
        ),
        (
            "Microsoft Security Alert: We detected someone trying to sign in to your account from an unknown device. "
            "Call 1-800-XXX-XXXX within 2 hours to avoid permanent lockout. Do not use the website - call only. Your account will be disabled if you ignore this.",
            "Tech support scam",
        ),
        (
            "Your package is held at customs. Pay a release fee of $2.99 now: [click here]. "
            "Parcel will be destroyed in 48 hours. We tried to deliver but you were not home. Act immediately.",
            "Delivery scam",
        ),
        (
            "CONGRATULATIONS! You have been chosen for a $500 Walmart gift card. Click to claim - offer expires in 2 hours. "
            "Enter your card number to verify you are a real person. No purchase necessary. Limited to first 100 respondents.",
            "Prize scam",
        ),
        (
            "Your Netflix payment failed. We will cancel your subscription in 24 hours. Update your payment method here: [link]. "
            "Do not reply to this email. If you do not update now, you will lose access to all content.",
            "Subscription phishing",
        ),
        (
            "I am a senior official and need to move $18.5 million out of my country. You will receive 30% ($5.55M) for helping. "
            "Reply with your full name, bank name, account number, and routing number. This is confidential. Time is very limited.",
            "Advance-fee",
        ),
        (
            "IRS Final Notice: You have an unpaid tax balance of $3,247. Pay immediately via this secure portal to avoid arrest and asset seizure. "
            "Do not contact your accountant. You have 72 hours. Click here to pay now.",
            "IRS scam",
        ),
        (
            "Your Apple ID was used to sign in from Moscow, Russia. If this wasn't you, tap here to secure your account now. "
            "We will lock your account in 24 hours unless you verify. Do not ignore this message.",
            "Account alert phishing",
        ),
        (
            "HR Department - URGENT: We need your direct deposit information for the next payroll. Reply to this email with your full bank routing number and account number. "
            "Payroll runs in 48 hours. Failure to respond will delay your paycheck. Reply ASAP.",
            "HR phishing",
        ),
        (
            "[Name] invited you to view a document: 'Q4 Budget.xlsx'. Open with one click: bit.ly/xxxxx. "
            "You may need to sign in with your Google or Microsoft account to view. Do not share this link.",
            "Document phishing",
        ),
        # Safe / legitimate messages (for demo: system should label Safe)
        (
            "Your order #8842 has shipped. Track your delivery at example.com/orders using the link in your account. "
            "Estimated delivery: Thursday. No action needed. Questions? Contact us through the Help Center.",
            "Safe – order confirmation",
        ),
        (
            "Reminder: Team standup is Tuesday at 10:00 AM. Agenda and Zoom link are in your calendar invite. "
            "Please join from your usual workspace. No reply required.",
            "Safe – meeting reminder",
        ),
        (
            "We've received your support request. Ticket #9012 is in progress. Our team typically responds within 24 business hours. "
            "You can check status anytime at support.example.com. We will not ask for your password by email.",
            "Safe – support acknowledgment",
        ),
        (
            "Your statement is ready. Log in at your bank's official website or app (type the URL yourself) to view it. "
            "We never ask for your password, PIN, or full SSN by email or phone. If you did not request this, ignore this message.",
            "Safe – bank notice",
        ),
        (
            "Thanks for subscribing to our newsletter. You'll get the next issue on Monday. Unsubscribe link is in the footer. "
            "We're at 123 Main St, City, State. No payment or personal details requested.",
            "Safe – newsletter",
        ),
    ],
    "Normal news": [
        (
            "The Federal Reserve announced a 0.25 percentage point increase in the federal funds rate today. "
            "Markets reacted with modest gains. The decision was widely expected by economists surveyed by Reuters.",
            "Normal news",
        ),
        (
            "The city council voted 7-2 to approve a $2 million budget for park renovations. Construction is set to begin in spring 2025. "
            "The project includes new playground equipment and improved accessibility, according to the parks department.",
            "Local news",
        ),
        (
            "A study published in Circulation found that walking 30 minutes a day was linked to lower blood pressure in adults. "
            "Researchers at Johns Hopkins followed 400 participants over six months. The study was funded by the National Institutes of Health.",
            "Health news",
        ),
        (
            "Tesla reported quarterly revenue of $25.5 billion, above analyst expectations. The stock rose 5% in after-hours trading. "
            "The company cited strong demand in China and Europe in its earnings release.",
            "Business news",
        ),
        (
            "The Riverside School District will extend winter break by two days due to forecasted severe weather. "
            "Parents will be notified via the district's alert system. School buses will not run on January 6 and 7.",
            "School update",
        ),
        (
            "A new species of tree frog was discovered in the Amazon rainforest. Scientists from the University of São Paulo say it is highly sensitive to pollution. "
            "The finding was published in the journal Nature. Conservation groups are calling for protected status.",
            "Science news",
        ),
        (
            "The mayor announced a plan to add 100 affordable housing units by 2026. "
            "Funding will come from federal grants and a small property tax adjustment, according to a statement from the mayor's office.",
            "Policy news",
        ),
        (
            "The National Weather Service issued a winter storm warning for the region from Friday through Sunday. "
            "Accumulations of 6 to 12 inches are expected. Residents are advised to avoid travel.",
            "Weather news",
        ),
        (
            "Highway 101 will have lane closures between exits 42 and 45 this weekend for bridge repair. "
            "The state Department of Transportation said work is expected to finish by Monday morning. Drivers should use alternate routes.",
            "Traffic update",
        ),
        (
            "The Metropolitan Museum's new exhibit on ancient Rome opens Saturday. Tickets are available online. "
            "The exhibit includes more than 200 artifacts on loan from museums in Italy, France, and the UK.",
            "Culture news",
        ),
    ],
}

# All styles (home + analyze) — dynamic, polished, glassmorphism
st.markdown("""
<style>
    /* ----- Animations ----- */
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes heroGlow {
        0%, 100% { box-shadow: 0 8px 40px rgba(45, 90, 135, 0.4), 0 0 60px rgba(30, 58, 95, 0.2); }
        50% { box-shadow: 0 12px 48px rgba(45, 90, 135, 0.5), 0 0 80px rgba(30, 58, 95, 0.25); }
    }
    @keyframes softPulse {
        0%, 100% { opacity: 0.95; }
        50% { opacity: 1; }
    }
    @keyframes cardShine {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes progressPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ----- Hero: glassmorphism + richer gradient ----- */
    .hero {
        text-align: center;
        padding: 3.5rem 2rem;
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.9) 0%, rgba(59, 130, 246, 0.85) 50%, rgba(30, 64, 175, 0.9) 100%);
        background-size: 200% 200%;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.15);
        animation: gradientShift 8s ease infinite, heroGlow 4s ease-in-out infinite, fadeInUp 0.8s ease-out;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    /* ----- Home: feature strip + section intro + footer ----- */
    .feature-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem 2rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 2rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        animation: fadeInUp 0.6s ease-out 0.4s both;
    }
    .feature-strip span {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .feature-strip span.dot { color: rgba(59, 130, 246, 0.8); font-weight: bold; }
    .section-title-home {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.35rem;
        animation: fadeInUp 0.5s ease-out 0.5s both;
    }
    .section-intro-home {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.5s ease-out 0.55s both;
    }
    .home-footer {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        animation: fadeInUp 0.5s ease-out 0.7s both;
    }
    .home-footer strong { color: #94a3b8; }

    /* ----- Analyzer page: what we check + steps ----- */
    .analyzer-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem 1.5rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .analyzer-strip span { color: #94a3b8; font-size: 0.88rem; font-weight: 500; }
    .analyzer-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem 1.5rem;
        margin-bottom: 1.25rem;
        padding: 1rem 1.25rem;
        background: rgba(30, 41, 59, 0.35);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .analyzer-steps .step { display: flex; align-items: center; gap: 0.5rem; color: #cbd5e1; font-size: 0.9rem; }
    .analyzer-steps .step-num { background: rgba(59, 130, 246, 0.35); color: #93c5fd; width: 1.5rem; height: 1.5rem; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; }
    .analyzer-tip-box {
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        background: rgba(59, 130, 246, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .trainer-intro-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem 1.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .trainer-intro-strip span { color: #94a3b8; font-size: 0.85rem; }

    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(105deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%);
        background-size: 200% 100%;
        animation: shimmer 6s ease-in-out infinite;
        pointer-events: none;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 24px rgba(0,0,0,0.25);
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    .hero p {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.95;
        animation: fadeInUp 0.6s ease-out 0.35s both;
    }

    /* ----- Homepage cards: glassmorphism + dynamic hover ----- */
    body:has(#home-page) [data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton > button:not([data-testid="baseButton-primary"]) {
        width: 100%;
        background: rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: #e2e8f0 !important;
        padding: 1.5rem !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        text-align: left !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        font-size: 1rem !important;
        height: auto !important;
        min-height: 120px !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    body:has(#home-page) [data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) .stButton > button:not([data-testid="baseButton-primary"]) {
        animation: fadeInUp 0.6s ease-out 0.15s both;
    }
    body:has(#home-page) [data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) .stButton > button:not([data-testid="baseButton-primary"]) {
        animation: fadeInUp 0.6s ease-out 0.3s both;
    }
    body:has(#home-page) [data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(3) .stButton > button:not([data-testid="baseButton-primary"]) {
        animation: fadeInUp 0.6s ease-out 0.45s both;
    }
    body:has(#home-page) [data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton > button:not([data-testid="baseButton-primary"]):hover {
        transform: translateY(-10px) scale(1.02) !important;
        box-shadow: 0 20px 48px rgba(59, 130, 246, 0.25), 0 0 0 1px rgba(255,255,255,0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        background: rgba(255,255,255,0.12) !important;
    }

    /* ----- Trust cards: glassmorphism + staggered entrance ----- */
    .trust-card {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.85) 0%, rgba(45, 90, 135, 0.9) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.5rem 1.75rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: fadeInUp 0.5s ease-out;
    }
    .trust-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(255,255,255,0.2);
    }
    .trust-card:nth-child(1) { animation-delay: 0.1s; }
    .trust-card:nth-child(2) { animation-delay: 0.2s; }

    .metric-big { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; }
    .verdict-supported { color: #4ade80; font-weight: bold; text-shadow: 0 0 20px rgba(74, 222, 128, 0.3); }
    .verdict-refuted { color: #f87171; font-weight: bold; }
    .verdict-unknown { color: #fbbf24; font-weight: bold; }
    .risk-high { color: #f87171; font-weight: 600; }
    .risk-medium { color: #fbbf24; font-weight: 600; }
    .risk-low { color: #4ade80; font-weight: 600; }
    .web-badge { display: inline-block; background: linear-gradient(135deg,#166534,#22c55e); color: white; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 8px rgba(34,197,94,0.3); }
    .offline-badge { display: inline-block; background: linear-gradient(135deg,#854d0e,#ca8a04); color: white; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; }
    .backboard-badge { display: inline-block; background: linear-gradient(135deg,#1e40af,#3b82f6); color: white; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 8px rgba(59,130,246,0.3); }
    .local-model-badge { display: inline-block; background: linear-gradient(135deg,#7c3aed,#a78bfa); color: white; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 8px rgba(167,139,250,0.3); }

    /* ----- Trainer: glassmorphism message card ----- */
    .trainer-message-card {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.9) 0%, rgba(45, 90, 135, 0.95) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeInUp 0.4s ease-out;
        transition: all 0.3s ease;
    }
    .trainer-message-card:hover {
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.25);
        transform: translateY(-2px);
    }
    .trainer-message-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9; margin-bottom: 0.5rem; font-weight: 600; }
    .trainer-highlight { background: #fca5a5; color: #1e293b; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .trainer-metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
        animation: fadeInUp 0.4s ease-out;
    }
    .trainer-metric-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }

    /* ----- Segmented control: pill-style ----- */
    [data-testid="stSegmentedControl"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 16px !important;
        padding: 4px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSegmentedControl"] button[kind="secondary"] {
        background: rgba(59, 130, 246, 0.4) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* ----- Text area: polished input ----- */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(30, 41, 59, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.4) !important;
        border-color: rgba(59, 130, 246, 0.6) !important;
    }

    /* ----- Primary Analyze button ----- */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(59, 130, 246, 0.5) !important;
    }

    /* ----- Tabs: modern pill style ----- */
    [data-testid="stTabs"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.25) !important;
        color: #93c5fd !important;
    }

    /* ----- Progress bar: animated ----- */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #3b82f6 100%) !important;
        background-size: 200% 100% !important;
        animation: progressPulse 2s ease-in-out infinite !important;
        border-radius: 8px !important;
    }

    /* ----- Sidebar: glassmorphism + decorative design ----- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.95) 50%, rgba(15, 23, 42, 0.98) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse 120% 80% at 20% 20%, rgba(59, 130, 246, 0.06), transparent 50%),
                    radial-gradient(ellipse 80% 60% at 80% 90%, rgba(30, 58, 95, 0.08), transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        position: relative;
        z-index: 1;
    }
    .sidebar-brand-block {
        padding: 1.25rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.6) 0%, rgba(59, 130, 246, 0.15) 100%);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        border-radius: 0 0 20px 0;
    }
    .sidebar-brand-block .brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sidebar-brand-block .brand-tagline {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .sidebar-feature-pill {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 0.75rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-feature-pill span {
        font-size: 0.78rem;
        color: #94a3b8;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .sidebar-section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin-bottom: 0.75rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    /* ----- Status / expander: polished ----- */
    [data-testid="stStatus"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }

    /* ----- Success / Error / Info: polished feedback ----- */
    [data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        animation: fadeInUp 0.4s ease-out !important;
    }
    div[data-baseweb="notification"][kind="positive"] {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.15) 100%) !important;
        border-color: rgba(34, 197, 94, 0.4) !important;
    }
    div[data-baseweb="notification"][kind="negative"] {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.15) 100%) !important;
        border-color: rgba(239, 68, 68, 0.4) !important;
    }
    div[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.15) 100%) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    /* ----- Metrics: dynamic cards ----- */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }

    /* ----- Global: smooth buttons, inputs, scroll ----- */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 35%, #0f172a 70%, #0c1222 100%);
        background-size: 100% 200%;
        background-attachment: fixed;
    }
    /* Subtle mesh overlay for depth */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59, 130, 246, 0.08), transparent),
                    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(30, 58, 95, 0.1), transparent);
        pointer-events: none;
        z-index: 0;
    }
</style>
""", unsafe_allow_html=True)


def _init_quiz(quiz_type: str):
    """Initialize or reset quiz state for the given quiz type."""
    rounds_list = get_trainer_rounds(quiz_type)
    st.session_state.quiz_initialized = True
    st.session_state.quiz_type = quiz_type
    st.session_state.rounds = rounds_list
    st.session_state.current_round = 0
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.selected_option = None
    st.session_state.reveal = False
    st.session_state.history = []


def _message_with_highlights(message: str, highlights: list) -> str:
    """Return HTML with highlighted phrases (escaped for safety)."""
    def escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if not highlights:
        return escape(message)
    text = message
    for i, phrase in enumerate(highlights):
        if phrase in text:
            text = text.replace(phrase, f"\x00H{i}\x00", 1)
    escaped = escape(text)
    for i, phrase in enumerate(highlights):
        escaped = escaped.replace(f"\x00H{i}\x00", f'<span class="trainer-highlight">{escape(phrase)}</span>')
    return escaped


def _render_trainer():
    """Render the Trainer tab: quiz flow, feedback, Review Mistakes, and friend progress."""
    # Use selected_section as source of truth so switching section in sidebar (including while in Trainer) works
    _sec_id = st.session_state.get("selected_section", "fact_check")
    quiz_type = next((s["quiz_type"] for s in SECTIONS if s["id"] == _sec_id), "misinformation")
    if not st.session_state.quiz_initialized or st.session_state.quiz_type != quiz_type:
        _init_quiz(quiz_type)
        # No st.rerun() here so section switch from sidebar is not lost; we render with new state this run

    rounds = st.session_state.rounds
    current = st.session_state.current_round
    n = len(rounds)
    quiz_labels = {"misinformation": "Misinformation", "normal_news": "Normal news", "phishing_crypto": "Phishing & crypto"}
    quiz_icons = {"misinformation": "🇷🇺", "normal_news": "📰", "phishing_crypto": "🛡️"}

    st.title("🎯 Trainer Quiz")
    st.caption(f"Practice for: {quiz_icons.get(quiz_type, '')} {quiz_labels.get(quiz_type, quiz_type)} (from sidebar).")
    st.markdown(
        """
        <div class="trainer-intro-strip">
            <span>📩 Read the message</span>
            <span>•</span>
            <span>🎯 Pick the best response</span>
            <span>•</span>
            <span>📖 Get feedback & review mistakes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quiz complete: show results first (before header so we don't render Round 5/4 or progress > 1)
    if current >= n:
        pct = (st.session_state.score / n * 100) if n else 0
        st.success(f"Quiz complete! Score: {st.session_state.score}/{n} ({pct:.0f}%)")
        if st.button("Restart Quiz", key="trainer_restart"):
            _init_quiz(quiz_type)
        return

    # During quiz: single Restart option
    if st.button("Restart Quiz", key="trainer_restart"):
        _init_quiz(quiz_type)

    # Header: Round x/N, Score, Streak, progress bar (only when still in quiz)
    progress_pct = min(1.0, (current + 1) / n) if n else 0
    st.markdown(
        f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div class="trainer-metric-card" style="flex: 1; min-width: 120px;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Round</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{current + 1} <span style="opacity: 0.6;">/</span> {n}</div>
            </div>
            <div class="trainer-metric-card" style="flex: 1; min-width: 100px;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Score</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #4ade80;">{st.session_state.score}</div>
            </div>
            <div class="trainer-metric-card" style="flex: 1; min-width: 100px;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Streak</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24;">{st.session_state.streak} 🔥</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(progress_pct)

    round_data = rounds[current]
    message = round_data["message"]
    correct_tactic = round_data["correct_tactic"]
    wrong_tactics = list(round_data.get("wrong_tactics", []))
    all_tactics = [correct_tactic] + wrong_tactics
    shuffle_key = f"trainer_shuffle_{current}"
    if shuffle_key not in st.session_state:
        idx = list(range(len(all_tactics)))
        random.shuffle(idx)
        st.session_state[shuffle_key] = idx
    ordered = [all_tactics[i] for i in st.session_state[shuffle_key]]

    # Two-column: message card (left), tactic choices (right)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown(
            '<div class="trainer-message-card">'
            '<div class="trainer-message-label">Message</div>'
            f'{_message_with_highlights(message, [])}'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_right:
        st.caption("What should you do?")
        selected = None
        if not st.session_state.reveal:
            # Tactic options as buttons (like the design)
            for i, tactic in enumerate(ordered):
                if st.button(tactic, key=f"trainer_btn_{current}_{i}", use_container_width=True):
                    selected = tactic
                    break
        else:
            st.caption(f"You chose: **{st.session_state.selected_option}**")
        if selected is not None and not st.session_state.reveal:
            st.session_state.selected_option = selected
            st.session_state.reveal = True
            correct = selected == correct_tactic
            st.session_state.history.append({
                "round_id": round_data["id"],
                "correct": correct,
                "tactic": selected,
                "correct_tactic": correct_tactic,
                "timestamp": time.time(),
            })
            if correct:
                st.session_state.score += 1
                st.session_state.streak += 1
            else:
                st.session_state.streak = 0

    if st.session_state.reveal:
        correct = st.session_state.selected_option == correct_tactic
        if correct:
            st.success("Correct!")
        else:
            st.error("Incorrect.")
        st.info(f"**Explanation:** {round_data.get('explanation', '')}")
        st.caption(f"**What to do:** {round_data.get('what_to_do', '')}")
        is_last_round = (current + 1) >= n
        btn_label = "See results" if is_last_round else "Next"
        if st.button(btn_label, key=f"trainer_next_{current}"):
            st.session_state.current_round += 1
            st.session_state.selected_option = None
            st.session_state.reveal = False

    st.divider()
    with st.expander("Review Mistakes"):
        mistakes = [h for h in st.session_state.history if not h["correct"]]
        if not mistakes:
            st.caption("No mistakes yet.")
        else:
            for m in mistakes:
                st.write(f"Round {m['round_id']}: You chose **{m['tactic']}**. Correct: **{m['correct_tactic']}**.")


# Sidebar
with st.sidebar:
    if st.session_state.show_analyze:
        st.markdown(
            '<div class="sidebar-brand-block">'
            '<div class="brand-title">📋 Try Samples</div>'
            '<div class="brand-tagline">Click to load sample text for each mode</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        for sec in SECTIONS:
            label = sec["sample_label"]
            if st.button(label, use_container_width=True, key=f"sample_{hash(label)}"):
                st.session_state.input_text = ""
                st.session_state.last_result = None
                st.session_state.suggested_quiz_type = sec["quiz_type"]
                st.session_state.selected_section = sec["id"]
                # Switch to Analyzer when changing section (so Trainer → section change shows Analyzer first)
                st.session_state.active_tab = "analyzer"
                st.session_state.main_section = "🔍 Analyzer"
        st.divider()
        if st.button("← Back to Home", use_container_width=True, key="back_home"):
            st.session_state.show_analyze = False
            st.rerun()
        if st.button("🗑️ Clear results", use_container_width=True, key="clear_results"):
            st.session_state.last_result = None
            if "cached_report_hash" in st.session_state:
                del st.session_state["cached_report_hash"]
                del st.session_state["cached_report_html"]
    else:
        # Homepage sidebar — richer design
        st.markdown(
            '<div class="sidebar-brand-block">'
            '<div class="brand-title">🔍 Clarion</div>'
            '<div class="brand-tagline">Trust what you read</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-feature-pill">'
            '<span>🔬 Extract claims</span>'
            '<span>•</span>'
            '<span>🌐 Web verify</span>'
            '<span>•</span>'
            '<span>🛡️ Scam detect</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-label">Get started</div>', unsafe_allow_html=True)
        st.caption("Choose a mode below to analyze text, fact-check claims, or practice spotting scams.")
        for sec in SECTIONS:
            if st.button(sec["title"], use_container_width=True, key=f"sidebar_{sec['id']}"):
                st.session_state.show_analyze = True
                st.session_state.selected_section = sec["id"]
                st.session_state.suggested_quiz_type = sec["quiz_type"]
                st.rerun()
    st.divider()
    st.caption("Clarion • Siren's Call Track")

if not st.session_state.show_analyze:
    # Homepage
    st.markdown('<div id="home-page"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>🔍 Clarion</h1>
        <p>Trust what you read. Spot misinformation, scams, and manipulation in seconds.</p>
    </div>
    <div class="feature-strip">
        <span>🔬 Extract claims</span>
        <span class="dot">•</span>
        <span>🌐 Verify with web & knowledge base</span>
        <span class="dot">•</span>
        <span>🛡️ Detect scams & AI-generated text</span>
    </div>
    <div class="section-title-home">What we do</div>
    <div class="section-intro-home">Pick a mode to analyze pasted text or practice with our quiz.</div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(SECTIONS))
    for i, sec in enumerate(SECTIONS):
        with cols[i]:
            btn_text = f"{sec['title']}\n\n{sec['description']}"
            if st.button(btn_text, key=f"card_{sec['id']}", use_container_width=True):
                st.session_state.show_analyze = True
                st.session_state.selected_section = sec["id"]
                st.session_state.suggested_quiz_type = sec["quiz_type"]
                st.rerun()
    st.markdown("""
    <div class="home-footer">
        <strong>Siren's Call Track</strong> • Hackathon — Built to help you verify information and stay safe online.
    </div>
    """, unsafe_allow_html=True)
else:
    # Analyzer / Trainer section switcher (persists across reruns so clicking a section always works)
    SECTION_OPTIONS = ["🔍 Analyzer", "🎯 Trainer"]
    
    if "main_section" not in st.session_state:
        st.session_state.main_section = (
            SECTION_OPTIONS[0]
            if st.session_state.active_tab == "analyzer"
            else SECTION_OPTIONS[1]
        )
    choice = st.segmented_control(
    "Section",
    SECTION_OPTIONS,
    key="main_section",
    label_visibility="collapsed",
    )
    st.session_state.active_tab = "analyzer" if choice == "🔍 Analyzer" else "trainer"

    if choice == "🔍 Analyzer":
        section = SECTION_CONTENT.get(st.session_state.selected_section, SECTION_CONTENT["fact_check"])
        st.title(section["title"])
        st.markdown(section["description"])
        st.markdown("""
        <div class="analyzer-strip">
            <span>📋 Extract claims</span>
            <span>•</span>
            <span>🌐 Verify with web & knowledge base</span>
            <span>•</span>
            <span>🛡️ Scam & AI detection</span>
        </div>
        """, unsafe_allow_html=True)
        # Try a sample: only show the sample that matches the current mode; randomize from corpus
        _section_to_sample = {
            "fact_check": ("Fact Check", "Load Fact Check sample"),
            "scam_phishing": ("Phishing scams", "Load Phishing sample"),
            "normal_news": ("Normal news", "Load Normal news sample"),
        }
        _sample_key, _sample_label = _section_to_sample.get(st.session_state.selected_section, ("Fact Check", "Load Fact Check sample"))
        with st.expander("💡 Try a sample", expanded=False):
            st.caption("Load a random sample into the box below to run a quick analysis.")
            if st.button(_sample_label, key="sample_analyzer_current", use_container_width=True):
                chosen = random.choice(SAMPLE_INPUTS[_sample_key])
                st.session_state.input_text = chosen[0]
                st.session_state.source_url = ""
                st.session_state.source_label = ""
        # URL section for Normal News only: fetch URL, extract text, run same pipeline
        _show_url_section = st.session_state.selected_section == "normal_news"
        if _show_url_section:
            with st.expander("🔗 Or analyze from URL", expanded=False):
                st.caption("Enter a news or article URL to fetch and fact-check the same way as pasted text.")
                url_input = st.text_input(
                    "Article URL",
                    placeholder="https://example.com/article...",
                    key="analyzer_url_input",
                    label_visibility="collapsed",
                )
                fetch_analyze_clicked = st.button("Fetch & Analyze", key="fetch_analyze_url")
                if fetch_analyze_clicked and url_input and url_input.strip():
                    url_to_fetch = url_input.strip()
                    if not url_to_fetch.startswith(("http://", "https://")):
                        url_to_fetch = "https://" + url_to_fetch
                    with st.spinner("Fetching URL and extracting text..."):
                        try:
                            text, title = fetch_and_extract(url_to_fetch)
                            if not text or len(text) < 30:
                                st.error("Could not extract enough text from this URL. Try pasting the article text instead.")
                            else:
                                st.session_state.input_text = text
                                st.session_state.source_url = url_to_fetch
                                st.session_state.source_label = title or url_to_fetch
                                content_type = st.session_state.get("selected_section", "fact_check")
                                with st.status("Analyzing...", expanded=True) as status:
                                    st.write("Extracting claims...")
                                    if content_type == "fact_check":
                                        st.write("Verifying claims with Backboard...")
                                    elif content_type == "normal_news":
                                        st.write("Searching the web (DuckDuckGo) for evidence...")
                                        st.write("Sending results to Backboard for synthesis...")
                                    st.write("Checking for manipulation & AI signals...")
                                    with st.spinner("Analyzing..."):
                                        result_dict = _run_analysis_in_thread(text, content_type)
                                    st.write("Computing fact-check metrics...")
                                    status.update(label="Done!", state="complete")
                                result_dict["source_url"] = st.session_state.get("source_url", "")
                                result_dict["source_label"] = st.session_state.get("source_label", "")
                                result_dict["input_text"] = text
                                st.session_state.last_result = result_dict
                                st.session_state.last_input_hash = hash(text)
                                if "cached_report_hash" in st.session_state:
                                    del st.session_state["cached_report_hash"]
                                    del st.session_state["cached_report_html"]
                                st.success(f"Fetched: **{title[:80]}{'...' if len(title) > 80 else ''}**")
                        except Exception as e:
                            st.error(f"Could not fetch URL: {e}")
        # Document upload for Fact Check only: PDF/DOCX → extract text, summarize (Backboard), fact-check
        if st.session_state.selected_section == "fact_check":
            with st.expander("📄 Or upload a document", expanded=False):
                st.caption("Upload a PDF or DOCX to summarize and fact-check. We extract text, summarize it, and flag claims that may be incorrect.")
                doc_file = st.file_uploader(
                    "Choose a file",
                    type=["pdf", "docx"],
                    key="fact_check_doc_upload",
                    label_visibility="collapsed",
                )
                doc_analyze_clicked = st.button("Extract & Analyze", key="analyze_doc_btn")
                if doc_analyze_clicked and doc_file:
                    with st.spinner("Extracting text from document..."):
                        try:
                            text, doc_name = extract_text_from_file(doc_file)
                            st.session_state.input_text = text
                            st.session_state.source_url = ""
                            st.session_state.source_label = doc_name
                            content_type = "fact_check"
                            doc_summary = None
                            with st.status("Analyzing...", expanded=True) as status:
                                st.write("Extracting claims...")
                                st.write("Verifying claims with Backboard...")
                                st.write("Checking for manipulation & AI signals...")
                                with st.spinner("Analyzing..."):
                                    result_dict = _run_analysis_in_thread(text, content_type)
                                if backboard_configured():
                                    st.write("Summarizing document...")
                                    doc_summary = summarize_document(text)
                                st.write("Computing fact-check metrics...")
                                status.update(label="Done!", state="complete")
                            result_dict["source_url"] = ""
                            result_dict["source_label"] = doc_name
                            result_dict["input_text"] = text
                            if doc_summary:
                                result_dict["document_summary"] = doc_summary
                            st.session_state.last_result = result_dict
                            st.session_state.last_input_hash = hash(text)
                            if "cached_report_hash" in st.session_state:
                                del st.session_state["cached_report_hash"]
                                del st.session_state["cached_report_html"]
                            st.success(f"Analyzed **{doc_name}**")
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Could not process document: {e}")
        user_input = st.text_area(
        "Paste text to analyze",
        height=200,
        max_chars=50000,
        placeholder="Paste a tweet, news article, email, or chat message here...",
        key="input_text",
    )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        if analyze_clicked:
            if not user_input or not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # Pasted text: clear URL source so result is not attributed to a previous URL
                st.session_state.source_url = ""
                st.session_state.source_label = ""
                input_hash = hash(user_input.strip())
                content_type = st.session_state.get("selected_section", "fact_check")
                with st.status("Analyzing...", expanded=True) as status:
                    st.write("Extracting claims...")
                    if content_type == "fact_check":
                        st.write("Verifying claims with Backboard...")
                    elif content_type == "normal_news":
                        st.write("Searching the web (DuckDuckGo) for evidence...")
                        st.write("Sending results to Backboard for synthesis...")
                    else:
                        st.write("Checking with BERT (message + URL phishing)...")
                    st.write("Checking for manipulation & AI signals...")
                    with st.spinner("Analyzing..."):
                        result_dict = _run_analysis_in_thread(user_input.strip(), content_type)
                    st.write("Computing fact-check metrics...")
                    status.update(label="Done!", state="complete")
                result_dict["source_url"] = st.session_state.get("source_url", "")
                result_dict["source_label"] = st.session_state.get("source_label", "")
                result_dict["input_text"] = user_input.strip()
                st.session_state.last_result = result_dict
                st.session_state.last_input_hash = input_hash
                if "cached_report_hash" in st.session_state:
                    del st.session_state["cached_report_hash"]
                    del st.session_state["cached_report_html"]
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            source_label = result.get("source_label", "") or st.session_state.get("source_label", "")
            source_url = result.get("source_url", "") or st.session_state.get("source_url", "")
            if source_label or source_url:
                if source_url and source_url.startswith("http"):
                    st.caption(f"**Analyzed from:** [{source_label or source_url}]({source_url})")
                else:
                    st.caption(f"**Analyzed from:** {source_label}")
            vmode = result.get("verification_mode", "offline")
            if vmode == "backboard":
                mode_badge = '<span class="backboard-badge">BACKBOARD</span>'
            elif vmode == "web+backboard":
                mode_badge = '<span class="web-badge">DUCKDUCKGO</span> <span class="backboard-badge">BACKBOARD</span>'
            elif vmode == "web":
                mode_badge = '<span class="web-badge">WEB</span>'
            elif vmode == "local_model":
                mode_badge = '<span class="local-model-badge">BERT</span>'
            else:
                mode_badge = '<span class="offline-badge">OFFLINE</span>'
            col1, col2 = st.columns(2)
            with col1:
                content_type = st.session_state.get("selected_section", "fact_check")
                if content_type == "scam_phishing":
                    risk = result.get("social_engineering", {}).get("risk_level", "Low")
                    summary = "Scam" if risk in ("Medium", "High") else "Safe"
                    card_title = "Scam check"
                else:
                    # Show short verdict: Correct / Incorrect / Mixed / No claims to verify
                    correct = result.get("correct_count", 0)
                    incorrect = result.get("incorrect_count", 0)
                    total = correct + incorrect
                    if total == 0:
                        summary = "No claims to verify"
                    elif correct > 0 and incorrect == 0:
                        summary = "Correct"
                    elif incorrect > 0 and correct == 0:
                        summary = "Incorrect"
                    else:
                        summary = "Mixed"
                    card_title = "Fact check"
                _src_map = {"backboard": "Using Backboard", "web+backboard": "DuckDuckGo → Backboard", "web": "Using internet (DuckDuckGo)", "local_model": "BERT (message + URL phishing)"}
                source_note = _src_map.get(vmode) or "No internet — local KB only (limited)"
                st.markdown(
                    f'<div class="trust-card"><div>{card_title} {mode_badge}</div>'
                    f'<div class="metric-big" style="font-size: 1.5rem;">{summary}</div>'
                    f'<div style="font-size: 0.75rem; opacity: 0.9; margin-top: 0.25rem;">{source_note}</div></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="trust-card"><div>Confidence in response</div><div class="metric-big">{result["response_confidence"]*100:.0f}%</div></div>',
                    unsafe_allow_html=True,
                )
            tab1, tab2, tab3 = st.tabs([
                "📋 Quick Check", "📚 Evidence", "🛡️ Manipulation & Scam",
            ])
            with tab1:
                st.subheader("Quick Check Summary")
                if result.get("document_summary"):
                    st.write("**Document summary:**", result["document_summary"])
                if content_type == "scam_phishing":
                    st.write(f"**Scam check:** {summary}")
                else:
                    st.write(f"**Fact check:** {result.get('fact_check_summary', 'No claims to verify')}")
                st.write(f"**Confidence in response:** {result['response_confidence']*100:.0f}%")
                _src_text = {"backboard": "Using Backboard", "web+backboard": "DuckDuckGo web search → Backboard synthesis", "web": "Using internet (DuckDuckGo web search)", "local_model": "BERT (message + URL phishing)"}
                st.write(f"**Fact checker source:** {_src_text.get(vmode) or 'No internet — local knowledge base only (results may be limited)'}")
                if content_type != "scam_phishing":
                    st.write("**Top reasons:**")
                    # Build reasons from claims so each line includes the claim text
                    claims_list = result.get("claims", [])
                    if claims_list:
                        for c in claims_list:
                            label = "Correct (supported by evidence)" if c["verdict"] == "Supported" else "Not supported by evidence" if c["verdict"] == "Refuted" else "Unclear" if c["verdict"] == "Unknown" else c["verdict"]
                            claim_preview = c["claim"][:100] + "..." if len(c["claim"]) > 100 else c["claim"]
                            st.write(f'- **{label}:** {claim_preview}')
                    else:
                        for r in result["top_reasons"]:
                            st.write(f"- {r}")
                citations = result.get("citations", [])
                if citations:
                    st.write("**Citations:**")
                    for url in citations[:15]:
                        st.markdown(f"- [{url}]({url})")
                st.write("**Fact checker risk:**", f"{result['misinformation']['risk_score']*100:.0f}%")
                st.write("**Social engineering risk:**", result["social_engineering"]["risk_level"])
            with tab2:
                st.subheader("Evidence")
                if result["claims"]:
                    for c in result["claims"]:
                        verdict_class = f"verdict-{c['verdict'].lower()}"
                        st.markdown(f"**Claim:** {c['claim']}")
                        # Softer labels: Assessment (not Verdict), Not supported (not Refuted), Unclear (not Unknown)
                        assessment_label = (
                            "Correct" if c["verdict"] == "Supported"
                            else "Not supported" if c["verdict"] == "Refuted"
                            else "Unclear" if c["verdict"] == "Unknown"
                            else c["verdict"]
                        )
                        st.markdown(
                            f"**Assessment:** <span class='{verdict_class}'>{assessment_label}</span> (confidence in this claim: {c['similarity']*100:.0f}%)",
                            unsafe_allow_html=True,
                        )
                        if c["evidence"]:
                            with st.expander("Evidence & sources"):
                                for e in c["evidence"][:5]:
                                    if e.startswith("Source:") and ("http://" in e or "https://" in e):
                                        parts = e.split("http", 1)
                                        label = parts[0].replace("Source:", "").strip()
                                        url = "http" + parts[1].strip()
                                        st.markdown(f"🔗 {label} — [{url}]({url})" if label else f"🔗 [{url}]({url})")
                                    else:
                                        st.write(e if len(e) <= 800 else e[:800] + "...")
                        st.divider()
                else:
                    st.info("No claims extracted from this text.")
            with tab3:
                st.subheader("Manipulation & Scam Analysis")
                risk_class = f"risk-{result['social_engineering']['risk_level'].lower()}"
                st.markdown(f"**Risk level:** <span class='{risk_class}'>{result['social_engineering']['risk_level']}</span>", unsafe_allow_html=True)
                st.write("**Red flags:**")
                for f in result["social_engineering"]["red_flags"]:
                    st.write(f"- {f}")
                st.write("**Safer approach:**")
                st.info(result["social_engineering"]["safer_rewrite_suggestion"])
            st.divider()
            # Cache report HTML by result so we don't regenerate on every rerun
            _rh = st.session_state.get("last_input_hash")
            if (
                st.session_state.get("cached_report_hash") == _rh
                and st.session_state.get("cached_report_html")
            ):
                report_html = st.session_state["cached_report_html"]
            else:
                report_html = generate_html_report(
                    result,
                    source_url=result.get("source_url", ""),
                    source_label=result.get("source_label", ""),
                    input_text=result.get("input_text", ""),
                )
                st.session_state["cached_report_hash"] = _rh
                st.session_state["cached_report_html"] = report_html
            st.download_button(
                "Download report (HTML)",
                data=report_html,
                file_name="truthlens_report.html",
                mime="text/html",
                use_container_width=True,
                key="download_report",
            )
            st.caption("Open the file in a browser and use Print → Save as PDF to get a PDF.")
        else:
            _tip = (
                '👆 Paste text above (or use <strong>Or analyze from URL</strong> for Normal news), '
                'then click <strong>Analyze</strong>. Results show verdicts, evidence, and scam/AI signals.'
            )
            if st.session_state.selected_section == "fact_check":
                _tip = (
                    '👆 Paste text above, or <strong>Or upload a document</strong> (PDF/DOCX) to summarize and fact-check. '
                    'Then click <strong>Analyze</strong>. Results show document summary, verdicts, evidence, and scam/AI signals.'
                )
            elif _show_url_section:
                _tip = (
                    '👆 Paste text above, or use <strong>Or analyze from URL</strong> to fetch an article and analyze it. '
                    'Then click <strong>Analyze</strong>. Results show verdicts, evidence, and scam/AI signals.'
                )
            st.markdown(
                f'<div class="analyzer-tip-box">{_tip}</div>',
                unsafe_allow_html=True,
            )

    else:
        _render_trainer()
