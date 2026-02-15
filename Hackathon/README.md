# Clarion

A hackathon-ready webapp for the **Siren's Call** track: analyzing text for misinformation, social engineering, and deceptive AI.

## Features

- **Web Verify (default)** – Fact-check claims using DuckDuckGo web search. No API key. Used by default for normal news and misinformation; falls back to local RAG if web fails.
- **Claim extraction** – Rule-based extraction of 1–6 atomic claims (no LLM)
- **Local RAG verification** – Offline fallback when web verification is unavailable
- **Misinformation detector** – Sensational language, all-caps, urgency signals
- **Social engineering detector** – Phishing/scam patterns (urgency, authority, credentials)
- **AI-generated detector** – Heuristics for AI-like text (repetition, generic phrases)
- **Fact-check metrics** – Correct count, incorrect count, and confidence in the response
- **Backboard** – Optional claim verification via Backboard API when `BACKBOARD_API_KEY` is set; falls back to DuckDuckGo then local RAG.

## Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true
```

Or simply `streamlit run app.py` (you may be prompted for email—leave blank to skip).

The app will open in your browser at `http://localhost:8501`.

**Note:** The first run may take 1–2 minutes while the embedding model (`all-MiniLM-L6-v2`) downloads. Subsequent runs are faster.

### Optional: API keys (.env)

Create a `.env` file in the `Hackathon` folder (copy from `.env.example`) and add any keys you use:

```bash
cp .env.example .env
# Edit .env and set:
# BACKBOARD_API_KEY=your_backboard_key   # for claim verification in Analyzer
# GEMINI_API_KEY=your_gemini_key         # for Gemini-based verification
```

The app loads `.env` at startup, so **Backboard** (claim verification) and **Gemini** features will use these keys. Do not commit `.env`.

## Web Verification (default, no toggle)

Claims are verified against the **live web** using **DuckDuckGo search** (no API key). This is always on for normal news and misinformation—there is no toggle. If web verification fails (e.g. no network), the app falls back to the local knowledge base.

The Evidence tab shows search snippets and clickable source URLs when web verification was used.

## How Fact-Check Metrics Work

The fact checker reports:

1. **Correct** – Number of claims supported by evidence (verdict: Supported)
2. **Incorrect** – Number of claims refuted by evidence (verdict: Refuted)
3. **Confidence in response** – How confident the system is in the overall response (0–100%), blending claim verification (40%) with signal detector results (60%)
4. **Summary** – Top reasons (e.g. X claim(s) correct, Y claim(s) incorrect, misinformation/social-engineering signals)

## Sample Inputs

Use the sidebar buttons to load examples:

| Type | Description |
|------|-------------|
| Misinformation | Sensational claims, "they don't want you to know", share prompts |
| Phishing scams | Urgency, bank/crypto impersonation, credential or Bitcoin request |
| Normal news | Factual, low-risk content |
| AI-like text | Generic phrases, uniform structure |

## Project Structure

```
├── app.py                 # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── pipeline.py        # Main orchestration
    ├── preprocessing.py   # Text cleaning
    ├── claim_extraction.py
    ├── misinformation_detector.py
    ├── social_engineering_detector.py
    ├── ai_text_detector.py
    ├── web_verifier.py    # DuckDuckGo web search (default)
    ├── rag_verifier.py    # Local RAG (fallback)
    ├── backboard_client.py   # Backboard API (fact-check assistant)
    ├── backboard_verifier.py  # Claim verification via Backboard
    ├── scoring.py
    ├── utils.py
    └── kb/
        └── seed_kb.md     # Knowledge base
```

## Limitations

- **Web Verify** requires internet (DuckDuckGo); falls back to local KB offline.
- **Claim extraction** is rule-based; complex or implicit claims may be missed.
- **AI detection** is heuristic-only; no perplexity model by default.
- **Language** – Optimized for English; other languages may have limited support.

## Future Work

- Add optional perplexity-based AI detection
- Support for URLs (fetch + analyze)
- Multi-language claim extraction
- Batch analysis mode
