"""Generate a self-contained HTML report from TruthLens analysis (print-friendly for Save as PDF)."""

import html


MAX_INPUT_DISPLAY = 15000  # truncate very long input in report


def generate_html_report(
    result_dict: dict,
    source_url: str = "",
    source_label: str = "",
    input_text: str = "",
) -> str:
    """
    Build a single self-contained HTML report. Styled for on-screen view and print (Save as PDF).
    Includes source (URL or document) and the analyzed input text. Does not include AI detection.
    """
    r = result_dict
    source_url = source_url or r.get("source_url", "")
    source_label = source_label or r.get("source_label", "")
    input_text = input_text or r.get("input_text", "")

    source_line = ""
    if source_label:
        if source_url and source_url.startswith("http"):
            source_line = f'<p class="meta-line"><strong>Source:</strong> <a href="{html.escape(source_url)}">{html.escape(source_label)}</a></p>'
        else:
            source_line = f'<p class="meta-line"><strong>Source:</strong> {html.escape(source_label)}</p>'
    elif source_url and source_url.startswith("http"):
        source_line = f'<p class="meta-line"><strong>Source URL:</strong> <a href="{html.escape(source_url)}">{html.escape(source_url)}</a></p>'

    summary = r.get("fact_check_summary") or "No claims to verify"
    confidence = r.get("response_confidence", 0) * 100
    vmode = r.get("verification_mode", "offline")
    mode_label = "Backboard" if vmode == "backboard" else "Internet (DuckDuckGo)" if vmode == "web" else "Local knowledge base"

    claims_html = ""
    for c in r.get("claims") or []:
        claim = html.escape(c.get("claim", ""))
        verdict = c.get("verdict", "Unknown")
        verdict_class = "verdict-supported" if verdict == "Supported" else "verdict-refuted" if verdict == "Refuted" else "verdict-unknown"
        sim = c.get("similarity", 0) * 100
        claims_html += f"""
        <section class="claim-block">
            <p class="claim-text"><strong>Claim:</strong> {claim}</p>
            <p class="claim-verdict"><strong>Verdict:</strong> <span class="{verdict_class}">{verdict}</span> (confidence: {sim:.0f}%)</p>
            <ul class="evidence-list">
        """
        for e in (c.get("evidence") or [])[:5]:
            esc = html.escape(e[:2000] + ("..." if len(e) > 2000 else ""))
            claims_html += f"<li>{esc}</li>"
        claims_html += "</ul></section>"

    reasons_html = "".join(f"<li>{html.escape(x)}</li>" for x in r.get("top_reasons") or [])

    mis = r.get("misinformation") or {}
    mis_score = mis.get("risk_score", 0) * 100
    mis_reasons = "".join(f"<li>{html.escape(x)}</li>" for x in mis.get("reasons") or [])

    soc = r.get("social_engineering") or {}
    risk_level = soc.get("risk_level", "Low")
    risk_class = "risk-low" if risk_level == "Low" else "risk-medium" if risk_level == "Medium" else "risk-high"
    red_flags = "".join(f"<li>{html.escape(x)}</li>" for x in soc.get("red_flags") or [])
    safer = html.escape((soc.get("safer_rewrite_suggestion") or "").strip())

    # Input / analyzed content for the report (source URL or document + text analyzed)
    input_display = (input_text or "").strip()
    if len(input_display) > MAX_INPUT_DISPLAY:
        input_display = input_display[:MAX_INPUT_DISPLAY] + "\n\n[... truncated for report ...]"
    input_html = ""
    if source_line or input_display:
        input_html = '<section class="section-block"><h2>Input / Analyzed content</h2>'
        if source_line:
            input_html += f'<div class="meta-block">{source_line}'
        if input_display:
            input_html += f'<p class="meta-line"><strong>Text analyzed:</strong></p><pre class="input-text">{html.escape(input_display)}</pre>'
        if source_line:
            input_html += "</div>"
        input_html += "</section>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TruthLens Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  color: #1f2937;
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
  background: #fff;
}}
.report-header {{
  border-bottom: 3px solid #1e3a5f;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}}
.report-header h1 {{
  margin: 0;
  font-size: 1.75rem;
  font-weight: 700;
  color: #1e3a5f;
}}
.meta-block {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
}}
.meta-line {{ margin: 0.25rem 0; font-size: 0.95rem; }}
.meta-line:first-child {{ margin-top: 0; }}
h2 {{
  font-size: 1.15rem;
  font-weight: 600;
  color: #1e3a5f;
  margin: 1.5rem 0 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid #e2e8f0;
}}
ul {{ margin: 0.5rem 0; padding-left: 1.5rem; }}
li {{ margin: 0.35rem 0; }}
.claim-block {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  break-inside: avoid;
}}
.claim-text {{ margin: 0 0 0.5rem; }}
.claim-verdict {{ margin: 0 0 0.5rem; font-size: 0.95rem; }}
.evidence-list {{ margin: 0.5rem 0 0; padding-left: 1.25rem; font-size: 0.9rem; }}
.verdict-supported {{ color: #059669; font-weight: 600; }}
.verdict-refuted {{ color: #dc2626; font-weight: 600; }}
.verdict-unknown {{ color: #d97706; font-weight: 600; }}
.risk-low {{ color: #059669; }}
.risk-medium {{ color: #d97706; }}
.risk-high {{ color: #dc2626; }}
.section-block {{
  margin-bottom: 1.25rem;
}}
.report-footer {{
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e2e8f0;
  font-size: 0.8rem;
  color: #64748b;
}}
.input-text {{
  white-space: pre-wrap;
  word-break: break-word;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 1rem;
  font-size: 0.9rem;
  max-height: 40em;
  overflow: auto;
  margin: 0.5rem 0 0;
}}
a {{ color: #1e40af; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

@media print {{
  body {{ padding: 1rem; font-size: 11pt; }}
  .report-header {{ border-bottom-color: #1e3a5f; }}
  .meta-block, .claim-block {{ border-color: #cbd5e1; }}
  h2 {{ break-after: avoid; }}
  .claim-block {{ break-inside: avoid; page-break-inside: avoid; }}
  a {{ color: #1e40af; }}
}}
</style>
</head>
<body>
<header class="report-header">
  <h1>TruthLens Report</h1>
</header>

{input_html}

<div class="meta-block">
<p class="meta-line"><strong>Fact check summary:</strong> {html.escape(summary)}</p>
<p class="meta-line"><strong>Confidence in response:</strong> {confidence:.0f}%</p>
<p class="meta-line"><strong>Verification source:</strong> {html.escape(mode_label)}</p>
</div>

<section class="section-block">
<h2>Top reasons</h2>
<ul>{reasons_html}</ul>
</section>

<section class="section-block">
<h2>Claims and evidence</h2>
{claims_html if claims_html else "<p>No claims extracted.</p>"}
</section>

<section class="section-block">
<h2>Misinformation risk</h2>
<p><strong>Risk score:</strong> {mis_score:.0f}%</p>
<ul>{mis_reasons}</ul>
</section>

<section class="section-block">
<h2>Social engineering</h2>
<p><strong>Risk level:</strong> <span class="{risk_class}">{html.escape(risk_level)}</span></p>
<ul>{red_flags}</ul>
{f'<p><strong>Safer approach:</strong> {safer}</p>' if safer else ''}
</section>

<footer class="report-footer">
Generated by TruthLens Suite — Siren's Call Track
</footer>
</body>
</html>
"""
