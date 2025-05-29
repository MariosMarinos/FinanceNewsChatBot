# annotated_news_pipeline.py

"""
Annotated news analysis pipeline
================================

This module provides a complete pipeline from raw Yahoo Finance news lookups
through to clean article fetching and abstractive summarisation, orchestrated
via a LangGraph graph.

Layers:
1. **Data utilities**: fetch and clean news metadata & article bodies.
2. **Summariser**: BART-large-CNN sliding-window + optional second-pass.
3. **LangGraph pipeline**: nodes for scraping & summarising, exposed via
   `run_pipeline(ticker, limit)`.
"""

from __future__ import annotations
import json
import re
import os
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, TypedDict

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import torch
from langgraph.graph import START, END, StateGraph

# ── SECTION 1 · DATA UTILITIES ────────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}

def _best_news_url(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract the first usable URL from a news item dict.

    Prefers URLs on finance.yahoo.com for consistent scraping, otherwise
    falls back to any http(s) link.
    """
    # Consider both top-level and nested 'content' keys
    layers = [item, item.get("content", {})]
    # (i) Look for Yahoo Finance URLs first
    for layer in layers:
        for key in ("clickThroughUrl", "canonicalUrl", "link", "url", "previewUrl"):
            val = layer.get(key)
            # Direct string URL check
            if isinstance(val, str) and "finance.yahoo.com" in val:
                return val
            # Nested dict with 'url'
            if isinstance(val, dict) and "finance.yahoo.com" in val.get("url", ""):
                return val["url"]
    # (ii) Fallback to any http/https URL found
    for layer in layers:
        for key in ("clickThroughUrl", "canonicalUrl", "link", "url", "previewUrl"):
            val = layer.get(key)
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                return val
            if isinstance(val, dict) and val.get("url", "").startswith(("http://", "https://")):
                return val["url"]
    # No valid URL found
    return None


def get_latest_news(ticker: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Fetch up to `limit` recent news items for `ticker` via yfinance.

    Returns a deduplicated list of dicts with 'title', 'url', and 'timestamp'.
    """
    try:
        # yfinance >= 0.2: use .news attribute
        raw = yf.Ticker(ticker).news[:limit]
    except AttributeError:
        # Legacy yfinance API
        raw = yf.Ticker(ticker).get_news(count=limit)

    seen_urls: set[str] = set()
    results: List[Dict[str, str]] = []

    for item in raw:
        url = _best_news_url(item)
        # Skip missing or duplicate URLs
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Extract title from possible locations
        title = (
            item.get("title")
            or item.get("content", {}).get("title")
            or "(untitled)"
        )

        # Normalize publish time to ISO string
        ts: Optional[str] = None
        if item.get("providerPublishTime"):
            ts = datetime.utcfromtimestamp(item["providerPublishTime"]).isoformat()
        elif (iso := item.get("content", {}).get("pubDate")):
            ts = iso

        results.append({"title": title, "url": url, "timestamp": ts})

    return results

# ── SECTION 2 · PROMO & BOILERPLATE STRIPPING ─────────────────────────────────

# Regex patterns to filter out promotional or footer text
_PROMO_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"Don[’']?t miss", re.I),
    re.compile(r"See the \d+ stocks", re.I),
    # ... add additional patterns as needed ...
]
_BREAK_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"was originally published", re.I),
    re.compile(r"More From", re.I),
    # ... etc ...
]

def _clean_paragraphs(paragraphs: List[str]) -> str:
    """
    Remove promotional lines and stop at break markers.

    - Skips any paragraph matching a promo pattern.
    - Stops (breaks) on the first break pattern.
    - Trims trailing promo paragraphs after the break.
    """
    keep: List[str] = []

    for para in paragraphs:
        # Skip promotional text
        if any(pat.search(para) for pat in _PROMO_PATTERNS):
            continue
        # Stop at section markers
        if any(brk.search(para) for brk in _BREAK_PATTERNS):
            break
        keep.append(para)

    # Remove trailing promos, if any
    while keep and any(pat.search(keep[-1]) for pat in _PROMO_PATTERNS):
        keep.pop()

    # Rejoin with spacing
    return "\n\n".join(keep)


def safe_body(url: str, timeout: int = 5) -> str:
    """
    Fetch article HTML and return clean, promo-free text.

    Performs a static GET (no JS), parses <p> tags, and cleans text.
    """
    try:
        html = requests.get(url, headers=HEADERS, timeout=timeout).text
    except Exception:
        return ""

    soup = BeautifulSoup(html, "lxml")
    container = (
        soup.select_one("div.caas-body")
        or soup.select_one('div[data-test-locator="content"]')
        or soup.select_one("article")
        or soup
    )
    paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    return _clean_paragraphs(paras)


def get_full_article_text(url: str) -> str:
    """
    Public wrapper returning the cleaned article body (or empty string).
    """
    return safe_body(url)

# ── SECTION 3 · SUMMARISER ───────────────────────────────────────────────────

# Load model & tokenizer once at import
MODEL_NAME = os.getenv("FINE_TUNED_MODEL_PATH", "facebook/bart-large-cnn")
print(f"Using summarisation model: {MODEL_NAME}...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 1024
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(tokenizer.vocab_size, mean_resizing=False)
device = 0 if torch.cuda.is_available() else -1
summariser = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)


def chunk_by_tokens(text: str, max_len: int = 1024, stride: int = 200) -> Generator[str, None, None]:
    """
    Yield overlapping chunks of up to `max_len` tokens for sliding-window summarisation.
    """
    ids = tokenizer(text, truncation=False)["input_ids"]
    step = max_len - stride
    for start in range(0, len(ids), step):
        yield tokenizer.decode(ids[start:start+max_len], skip_special_tokens=True)


def auto_chunk_and_summarise(
    text: str,
    *,
    model_window: int = 1024,
    stride: int = 200,
    chunk_max: int = 70,
    final_max: int = 150,
) -> str:
    """
    Two-pass summarisation for arbitrarily long text:
    1. Map: summarise each sliding-window chunk.
    2. Reduce (if needed): summarise the merged chunk summaries.
    """
    def tlen(s: str) -> int:
        return len(tokenizer(s, truncation=False)["input_ids"])

    if tlen(text) <= model_window:
        return summariser(text, max_length=final_max, min_length=40, truncation=True)[0]["summary_text"].strip()

    # First pass: chunk summarisation
    chunks = list(chunk_by_tokens(text, model_window, stride))
    chunk_summaries = []
    for c in chunks:
        chunk_summaries.append(
            summariser(c, max_length=chunk_max, min_length=15, truncation=True)[0]["summary_text"].strip()
        )
    merged = "\n\n".join(chunk_summaries)

    # Second pass if output still too long
    if len(chunk_summaries) > 1 and tlen(merged) > final_max:
        merged = summariser(
            merged,
            max_length=final_max,
            min_length=int(chunk_max * 1.3),
            truncation=True,
        )[0]["summary_text"].strip()

    return merged

# ── SECTION 4 · LANGGRAPH PIPELINE ───────────────────────────────────────────

class _State(TypedDict, total=False):
    ticker: str
    article_count: int
    articles: List[Dict[str, str]]
    result: List[Dict[str, str]]

# Node 1: scrape & clean
def _scrape_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch `article_count` news items and clean their bodies.
    Populates state['articles'] with dicts {url, body}.
    """
    limit = state.get("article_count", 5)
    raw_items = get_latest_news(state["ticker"], limit=limit)
    cleaned: List[Dict[str, str]] = []
    for art in raw_items:
        body = get_full_article_text(art["url"])
        if body and not body.startswith("ERROR") and len(body.split()) >= 100:
            cleaned.append({"url": art["url"], "body": body})
    state["articles"] = cleaned
    return state

# Node 2: summarise
def _summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarise each cleaned article body.
    Populates state['result'] with dicts {url, full_text, summary}.
    """
    summaries: List[Dict[str, str]] = []
    for art in state["articles"]:
        summaries.append({
            "url": art["url"],
            "full_text": art["body"],
            "summary": auto_chunk_and_summarise(art["body"]),
        })
    state["result"] = summaries
    return state

# Build and compile the graph
graph = StateGraph(_State)
graph.add_node("scrape", _scrape_node)
graph.add_node("summarise", _summary_node)
graph.add_edge(START, "scrape")
graph.add_edge("scrape", "summarise")
graph.add_edge("summarise", END)
app = graph.compile()

# Convenience runner

def run_pipeline(ticker: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Execute the pipeline: fetch up to `limit` articles, clean, summarise.

    Returns a list of {url, full_text, summary} dicts.
    """
    final_state = app.invoke({"ticker": ticker, "article_count": limit})
    return final_state["result"]
