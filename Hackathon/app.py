"""TruthLens Suite - Streamlit webapp for misinformation & manipulation analysis."""

import sys
from pathlib import Path

# Ensure src is on path and load .env before any code reads env vars
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import random
import time

import streamlit as st

from src.pipeline import run_pipeline
from src.rag_verifier import RAGVerifier
from src.trainer_rounds import get_trainer_rounds


@st.cache_resource
def get_rag_verifier() -> RAGVerifier:
    """Cached RAG verifier - builds index once, reuses across sessions."""
    return RAGVerifier()


@st.cache_data(ttl=300)
def run_analysis(text: str, content_type: str, _v: int = 4) -> dict:
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
    }


# Page config - must be first Streamlit command
st.set_page_config(
    page_title="TruthLens Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Section-specific titles and descriptions
SECTION_CONTENT = {
    "normal_news": {
        "title": "🔍 Normal News",
        "description": "Extract factual claims and verify them against web sources or our knowledge base.",
    },
    "scam_phishing": {
        "title": "🛡️ Scam & Phishing Analysis",
        "description": "Detect social engineering, urgency tactics, and red flags in emails and messages.",
    },
    "fact_check": {
        "title": "📊 Fact Check",
        "description": "See how many claims are correct vs incorrect and how confident we are in the response.",
    },
}

# Quick Load label -> suggested quiz type for Trainer
QUICK_LOAD_TO_QUIZ = {
    "Fact checker": "misinformation",
    "Phishing scams": "phishing_crypto",
    "Normal news": "normal_news",
}
# Quick Load label -> section key (so Analyzer title/description match the loaded sample)
QUICK_LOAD_TO_SECTION = {
    "Fact checker": "fact_check",
    "Phishing scams": "scam_phishing",
    "Normal news": "normal_news",  

}

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

SAMPLE_INPUTS = {
    "Fact checker": (
        "BREAKING: They don't want you to know the truth! SHARE THIS NOW!!! "
        "100% proven. Mainstream media is hiding what really happened. "
        "Tell everyone before it's too late.",
        "Tweet",
    ),
    "Phishing scams": (
        "Dear Customer, Your bank account has been suspended. Act within 24 hours "
        "to verify your identity or lose access. Click here: https://fake-bank-secure.com/verify. "
        "URGENT! You've won 5 Bitcoin! Send 0.1 BTC to verify. Act now!!!",
        "Email",
    ),
    "Normal news": (
        "The Federal Reserve announced a 0.25% interest rate hike today. "
        "Markets reacted with modest gains. Economists had expected the move.",
        "Normal news",
    ),
}

# All styles (home + analyze)
st.markdown("""
<style>
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 3rem;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .hero h1 { font-size: 3rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.25rem; opacity: 0.95; }
    /* Card-style buttons on homepage only - exclude primary (Analyze) button */
    section.main .stHorizontalBlock div[data-testid="column"] .stButton > button:not([data-testid="baseButton-primary"]) {
        width: 100%;
        background: white !important;
        color: #1f2937 !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid #e5e7eb !important;
        text-align: left !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        font-size: 1rem !important;
        height: auto !important;
        min-height: 120px !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }
    section.main .stHorizontalBlock div[data-testid="column"] .stButton > button:not([data-testid="baseButton-primary"]):hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
    }
    .trust-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .metric-big { font-size: 2.5rem; font-weight: 700; }
    .verdict-supported { color: #22c55e; font-weight: bold; }
    .verdict-refuted { color: #ef4444; font-weight: bold; }
    .verdict-unknown { color: #f59e0b; font-weight: bold; }
    .risk-high { color: #ef4444; }
    .risk-medium { color: #f59e0b; }
    .risk-low { color: #22c55e; }
    .web-badge { display: inline-block; background: #166534; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin-left: 6px; }
    .offline-badge { display: inline-block; background: #854d0e; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin-left: 6px; }
    .backboard-badge { display: inline-block; background: #1e40af; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin-left: 6px; }
    .trainer-message-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
        color: white; border: 1px solid #334155;
    }
    .trainer-message-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9; margin-bottom: 0.5rem; }
    .trainer-highlight { background: #fca5a5; color: #1e293b; padding: 0 2px; border-radius: 2px; }
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
    quiz_type = st.session_state.suggested_quiz_type
    if not st.session_state.quiz_initialized or st.session_state.quiz_type != quiz_type:
        _init_quiz(quiz_type)
        st.rerun()

    rounds = st.session_state.rounds
    current = st.session_state.current_round
    n = len(rounds)
    quiz_labels = {"misinformation": "Misinformation", "normal_news": "Normal news", "phishing_crypto": "Phishing & crypto"}
    quiz_icons = {"misinformation": "🇷🇺", "normal_news": "📰", "phishing_crypto": "🛡️"}

    st.title("🎯 Trainer Quiz")
    st.caption(f"Practice for: {quiz_icons.get(quiz_type, '')} {quiz_labels.get(quiz_type, quiz_type)} (from Quick Load).")

    # Quiz complete: show results first (before header so we don't render Round 5/4 or progress > 1)
    if current >= n:
        pct = (st.session_state.score / n * 100) if n else 0
        st.success(f"Quiz complete! Score: {st.session_state.score}/{n} ({pct:.0f}%)")
        if st.button("Restart Quiz", key="trainer_restart"):
            _init_quiz(quiz_type)
            st.rerun()
        return

    # During quiz: single Restart option
    if st.button("Restart Quiz", key="trainer_restart"):
        _init_quiz(quiz_type)
        st.rerun()

    # Header: Round x/N, Score, Streak, progress bar (only when still in quiz)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.metric("Round", f"{current + 1} / {n}")
    with c2:
        st.metric("Score", st.session_state.score)
    with c3:
        st.metric("Streak", f"{st.session_state.streak} 🔥")
    st.progress(min(1.0, (current + 1) / n) if n else 0)

    round_data = rounds[current]
    message = round_data["message"]
    highlights = round_data.get("highlights", [])
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
            f'{_message_with_highlights(message, highlights)}'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_right:
        st.caption("Choose the main tactic:")
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
            st.rerun()

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
            st.rerun()

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
        st.header("Quick Load")
        for label, (sample_text, _) in SAMPLE_INPUTS.items():
            if st.button(label, use_container_width=True, key=f"sample_{hash(label)}"):
                st.session_state["input_text"] = sample_text
                st.session_state.last_result = None
                st.session_state.suggested_quiz_type = QUICK_LOAD_TO_QUIZ.get(label, "misinformation")
                st.session_state.selected_section = QUICK_LOAD_TO_SECTION.get(label, "fact_check")
                st.rerun()
        st.divider()
        if st.button("← Back to Home", use_container_width=True, key="back_home"):
            st.session_state.show_analyze = False
            st.rerun()
        if st.button("🗑️ Clear results", use_container_width=True, key="clear_results"):
            st.session_state.last_result = None
            st.rerun()
    st.divider()
    st.caption("TruthLens Suite • Siren's Call Track")

if not st.session_state.show_analyze:
    # Homepage
    st.markdown("""
    <div class="hero">
        <h1>🔍 TruthLens Suite</h1>
        <p>Trust what you read. Spot misinformation, scams, and manipulation in seconds.</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("What we do")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📰 **Normal News**\n\nExtract factual claims and check them against web sources or our knowledge base.", key="card1", use_container_width=True):
            st.session_state.show_analyze = True
            st.session_state.selected_section = "normal_news"
            st.rerun()
    with col2:
        if st.button("🛡️ **Scam & Phishing**\n\nDetect social engineering, urgency tactics, and red flags in emails and messages.", key="card2", use_container_width=True):
            st.session_state.show_analyze = True
            st.session_state.selected_section = "scam_phishing"
            st.rerun()
    with col3:
        if st.button("📊 **Fact Check**\n\nSee how many claims are correct vs incorrect and how confident we are in the response.", key="card3", use_container_width=True):
            st.session_state.show_analyze = True
            st.session_state.selected_section = "fact_check"
            st.rerun()
    st.caption("Siren's Call Track • Hackathon")
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
        user_input = st.text_area(
        "Paste text to analyze",
        value=st.session_state.get("input_text", ""),
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
                input_hash = hash(user_input.strip())
                with st.status("Analyzing...", expanded=True) as status:
                    st.write("Extracting claims...")
                    st.write("Searching the web (DuckDuckGo) for evidence...")
                    st.write("Checking for manipulation & AI signals...")
                    content_type = st.session_state.get("selected_section", "fact_check")
                    result_dict = run_analysis(user_input.strip(), content_type)
                    st.write("Computing fact-check metrics...")
                    status.update(label="Done!", state="complete")
                st.session_state.last_result = result_dict
                st.session_state.last_input_hash = input_hash
                st.rerun()
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            vmode = result.get("verification_mode", "offline")
            if vmode == "backboard":
                mode_badge = '<span class="backboard-badge">BACKBOARD</span>'
            elif vmode == "web":
                mode_badge = '<span class="web-badge">WEB</span>'
            else:
                mode_badge = '<span class="offline-badge">OFFLINE</span>'
            col1, col2 = st.columns(2)
            with col1:
                summary = result.get("fact_check_summary", "No claims to verify")
                source_note = "Using Backboard" if vmode == "backboard" else "Using internet (DuckDuckGo)" if vmode == "web" else "No internet — local KB only (limited)"
                st.markdown(
                    f'<div class="trust-card"><div>Fact check {mode_badge}</div>'
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
                st.write(f"**Fact check:** {result.get('fact_check_summary', 'No claims to verify')}")
                st.write(f"**Confidence in response:** {result['response_confidence']*100:.0f}%")
                st.write(f"**Fact checker source:** {'Using Backboard' if vmode == 'backboard' else 'Using internet (DuckDuckGo web search)' if vmode == 'web' else 'No internet — local knowledge base only (results may be limited)'}")
                st.write("**Top reasons:**")
                for r in result["top_reasons"]:
                    st.write(f"- {r}")
                st.write("**Fact checker risk:**", f"{result['misinformation']['risk_score']*100:.0f}%")
                st.write("**Social engineering risk:**", result["social_engineering"]["risk_level"])
            with tab2:
                st.subheader("Evidence")
                if result["claims"]:
                    for c in result["claims"]:
                        verdict_class = f"verdict-{c['verdict'].lower()}"
                        st.markdown(f"**Claim:** {c['claim']}")
                        verdict_label = "Correct" if c["verdict"] == "Supported" else "Incorrect" if c["verdict"] == "Refuted" else c["verdict"]
                        st.markdown(
                            f"**Verdict:** <span class='{verdict_class}'>{verdict_label}</span> (confidence in this claim: {c['similarity']*100:.0f}%)",
                            unsafe_allow_html=True,
                        )
                        if c["evidence"]:
                            with st.expander("Evidence & Sources"):
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
        else:
            st.info("👆 Paste text above and click **Analyze** to get started. Use the sidebar to load sample inputs.")

    else:
        _render_trainer()
