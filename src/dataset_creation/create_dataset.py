import html
import json
import pathlib
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

###############################################################################
# CONFIGURATION
###############################################################################
HEADERS = {"User-Agent": "Mozilla/5.0"}
_URL_RE = re.compile(r"^https?://", re.I)
_MIN_WORDS = 80               # minimum words in article body to keep
_OVERSAMPLE_FACTOR = 4        # fetch N * factor raw hits → filter down
_REQUEST_DELAY = 0.4          # polite pause between page‑scrapes (seconds)

PROMO_PATTERNS = [
    re.compile(r"Don[’']?t miss (?:this )?second chance", re.I),
    re.compile(r"See the \d+\s+stocks",                   re.I),
    re.compile(r"Stock Advisor",                          re.I),
    re.compile(r"join (?:our )?(?:newsletter|service|Stock Advisor)", re.I),
    re.compile(r"subscribe.*?(?:today|now)",              re.I),
    re.compile(r"if you invested \$?\d+[,0-9]*.*?you[’']?d have \$",  re.I),
    re.compile(r"The Motley Fool",                        re.I),
    re.compile(r"Investor'?s Business Daily",             re.I),
    re.compile(r"was originally published by",            re.I),
    re.compile(r"positions in (?:and )?recommends",       re.I),
    re.compile(r"disclosure policy",                      re.I),
    re.compile(r"Disclosure:",                            re.I),
    re.compile(r"returns as of \w+ \d{1,2},? \d{4}",      re.I),
    re.compile(r"More From",                              re.I),
    re.compile(r"Related:(?:.*)$",                        re.I),
    re.compile(r"Don[’']?t miss .*?chance",               re.I),
    re.compile(r"Do ?n[’']?t miss .*?chance",             re.I),
    re.compile(r".+!\*$"),                                # Stock-Advisor footnote
]

BREAK_PATTERNS = [
    re.compile(r"was originally published",  re.I),
    re.compile(r"Before (?:you buy|invest)", re.I),
    re.compile(r"Related",                   re.I),
]

CONTINUE_PATTERNS = [
    re.compile(r"\bstory continue(s|d)?\b", re.I),
    re.compile(r"\bcontinue reading\b", re.I),
    re.compile(r"\bread more\b", re.I),
]

###############################################################################
# YAHOO HELPERS
###############################################################################

def yahoo_search_news(ticker: str, count: int = 100) -> List[Dict]:
    """Call Yahoo's public /finance/search endpoint and return the `news` list."""
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": ticker, "newsCount": count, "quotesCount": 0}
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json().get("news", [])


###############################################################################
# SCRAPING UTILITIES
###############################################################################

def _clean_paragraphs(paragraphs: List[str]) -> str:
    """Strip promo lines and cut off at break markers."""
    keep: List[str] = []
    for p in paragraphs:
        if any(pat.search(p) for pat in PROMO_PATTERNS):
            continue
        if any(pat.search(p) for pat in BREAK_PATTERNS):
            break
        keep.append(p)
    while keep and any(pat.search(keep[-1]) for pat in PROMO_PATTERNS):
        keep.pop()
    return "\n\n".join(keep)


def _find_continuation_url(soup, base_url):
    # 1. Look for anchor links/buttons that look like "Continue reading" or "Story continues"
    candidates = soup.find_all("a", href=True)
    for a in candidates:
        txt = a.get_text(" ", strip=True)
        if "continue reading" in txt.lower() or "story continues" in txt.lower() or "read more" in txt.lower():
            href = a["href"]
            if href.startswith("http"):
                return href
            # Handle relative links
            from urllib.parse import urljoin
            return urljoin(base_url, href)
    # 2. Try canonical or og:url as last resort (not always continuation, but sometimes)
    for rel in ["canonical", "og:url"]:
        link = soup.find("link", rel=rel)
        if link and link.get("href") and link["href"] != base_url:
            return link["href"]
        meta = soup.find("meta", property=rel)
        if meta and meta.get("content") and meta["content"] != base_url:
            return meta["content"]
    return None

def fetch_article(url: str, timeout: int = 10) -> Dict[str, Optional[str]]:
    """
    • Fetch Yahoo article
    • If <script type='application/ld+json'> is present, use its articleBody
    • Otherwise fall back to <p> extraction (+ optional continuation fetch)
    • Return {'summary', 'body', 'used_continuation'}
    """
    if not _URL_RE.match(url):
        return {"summary": None, "body": "", "used_continuation": False}

    html_raw = requests.get(url, headers=HEADERS, timeout=timeout).text
    soup     = BeautifulSoup(html_raw, "lxml")

    # ① publisher summary from meta tags ----------------------------------
    summary = None
    for attr in ("name", "property"):
        for key in ("description", "og:description", "twitter:description"):
            tag = soup.find("meta", attrs={attr: key})
            if tag and tag.get("content"):
                summary = tag["content"].strip()
                break
        if summary:
            break

    # ② try JSON-LD → always contains the FULL article -------------------
    ld = soup.find("script", type="application/ld+json")
    if ld and ld.string:
        try:
            data = json.loads(ld.string)
            if isinstance(data, dict) and "articleBody" in data:
                body_text = html.unescape(data["articleBody"])
                body = _clean_paragraphs(body_text.split("\\n"))
                return {"summary": summary, "body": body, "used_continuation": False}
        except json.JSONDecodeError:
            pass  # fall back to normal <p> scraping

    # ③ classic <p> scraping (plus continuation logic) -------------------
    container = (
        soup.select_one("div.caas-body")
        or soup.select_one('div[data-test-locator="content"]')
        or soup.select_one("article")
        or soup
    )
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    body = _clean_paragraphs(paragraphs)
    used_cont = False

    text_flat = " ".join(paragraphs).lower()
    if any(pat.search(text_flat) for pat in CONTINUE_PATTERNS):
        cont_url = _find_continuation_url(soup, url)
        if cont_url and cont_url != url:
            try:
                cont_html = requests.get(cont_url, headers=HEADERS, timeout=timeout).text
                cont_soup = BeautifulSoup(cont_html, "lxml")
                cont_paras = [
                    p.get_text(" ", strip=True)
                    for p in cont_soup.select_one("div.caas-body, article").find_all("p")
                ]
                cont_body = _clean_paragraphs(cont_paras)
                if len(cont_body) > len(body):
                    body += "\\n\\n" + cont_body
                    used_cont = True
            except Exception as e:
                print("Continuation fetch error:", e)

    return {"summary": summary, "body": body, "used_continuation": used_cont}

###############################################################################
# CORE HARVESTING LOGIC
###############################################################################

def harvest_ticker(
    ticker: str,
    max_articles: int = 40,
) -> List[Dict]:
    """Return up to `max_articles` high‑quality examples for one ticker."""
    rows, seen_urls = [], set()

    raw = yahoo_search_news(ticker, count=max_articles * _OVERSAMPLE_FACTOR)

    for item in raw:
        if len(rows) >= max_articles:
            break

        url = item.get("link") or item.get("url") or ""
        if not _URL_RE.match(url) or url in seen_urls:
            continue
        seen_urls.add(url)

        # scrape page → summary + body
        scraped = fetch_article(url)
        summary = scraped["summary"] or item.get("summary") or item.get("abstract")
        body    = scraped["body"]
        used_co = scraped['used_continuation']

        if not summary or len(body.split()) < _MIN_WORDS:
            continue  # need ground‑truth & minimum body length

        ts_raw = item.get("providerPublishTime")
        if isinstance(ts_raw, (int, float)):
            ts_iso = datetime.fromtimestamp(ts_raw, tz=timezone.utc).isoformat()
        else:
            ts_iso = None

        rows.append({
            "ticker": ticker,
            "title": item.get("title", "(untitled)"),
            "url": url,
            "timestamp": ts_iso,
            "body": body,
            "summary": summary,
            "used_continuation": used_co
        })
        time.sleep(_REQUEST_DELAY)

    return rows

###############################################################################
# CREATE THE DATASET
###############################################################################

def build_dataset(
    tickers: List[str],
    per_ticker: int = 40,
    out_path: str = "yahoo_finance_summaries.json",
) -> None:
    """Collect `per_ticker` examples for each ticker and write JSON to `out_path`."""
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for tkr in tickers:
            print(f"⇢  Harvesting {tkr} …")
            examples = harvest_ticker(tkr, per_ticker)
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n✅  Done. Wrote {total} examples → {out_path}")



def create_train_eval_set(
    input_path: str = "data/yahoo_finance_summaries.jsonl",
    train_path: str = "data/train.json",
    eval_path:  str = "data/eval.json",
    test_size: float = 0.25,
    seed: int = 42,
) -> None:
    # 1️⃣ load your full dataset
    df = pd.read_json(input_path, lines=True)

    # 2️⃣ one-time, reproducible split
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # 3️⃣ save them back to disk
    train_df.to_json(train_path, orient="records", lines=True)
    eval_df.to_json(eval_path, orient="records", lines=True)




###############################################################################
if __name__ == "__main__":
    # List of tickers to harvest
    tickers = [
        "AAPL","MSFT","AMZN","GOOG","GOOGL","META","TSLA","NVDA",
        "BRK-B","JPM","JNJ","V","UNH","HD","PG","MA","BAC","PFE",
        "XOM","DIS","CVX","VZ","ADBE","T","CMCSA","KO","NKE","ORCL",
        "ABT","INTC","CRM","MCD","IBM","CSCO","WMT","LLY","TXN","MRK",
        "SBUX","QCOM","PEP","AXP","ADP","SPGI","BKNG","CAT","MMM","HON",
        "MRNA","AMAT"
    ]

    raw_path = pathlib.Path("data/yahoo_finance_summaries.json")
    print(raw_path)

    if raw_path.exists():
        print(f"Found {raw_path}, skipping harvest.")
    else:
        print(f"{raw_path} not found, starting harvest.")
        build_dataset(tickers, per_ticker=100, out_path=str(raw_path))

    print("Splitting into train/eval...")
    create_train_eval_set(
        input_path=str(raw_path),
        train_path="data/train_test.json",
        eval_path="data/eval_test.json",
        test_size=0.25,
        seed=42,
    )