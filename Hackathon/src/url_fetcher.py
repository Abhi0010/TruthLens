"""Fetch a URL and extract main article/text for fact-checking or news analysis."""

import re
from typing import Tuple

import requests
from bs4 import BeautifulSoup

# Timeout and size limits to avoid hanging or huge pages
FETCH_TIMEOUT = 15
MAX_CONTENT_LENGTH = 2_000_000  # ~2MB
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_and_extract(url: str) -> Tuple[str, str]:
    """
    Fetch a URL and extract main text and title.

    Returns:
        (extracted_text, title_or_url)
    Raises:
        ValueError: on invalid URL or non-200 response
        requests.RequestException: on network errors
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("URL is empty")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(
        url,
        headers=headers,
        timeout=FETCH_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_CONTENT_LENGTH:
        raise ValueError(f"Page too large (>{MAX_CONTENT_LENGTH // 1_000_000}MB)")

    raw = response.content
    if len(raw) > MAX_CONTENT_LENGTH:
        raise ValueError(f"Page too large (>{MAX_CONTENT_LENGTH // 1_000_000}MB)")

    # Try to decode; fallback to latin-1 to avoid crashes
    try:
        html = raw.decode("utf-8", errors="replace")
    except Exception:
        html = raw.decode("latin-1", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "form"]):
        tag.decompose()

    # Prefer article/main content
    main = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"article|content|post", re.I))
    if main:
        body = main
    else:
        body = soup.find("body") or soup

    text = body.get_text(separator=" ", strip=True) if body else ""
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) < 50:
        # Fallback: all paragraphs
        paras = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paras if p.get_text(strip=True))
        text = re.sub(r"\s+", " ", text).strip()

    # Title: og:title, then <title>
    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        title = url

    return (text[:50000], title)  # Cap text length for pipeline
