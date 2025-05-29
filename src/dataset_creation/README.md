# Dataset Creation Module

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   cd dataset_creation
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. And all you have to do is run: 
   ```
   python3 create_dataset.py

## What This Script Does

1. **Ticker-based News Harvesting**  
   - Uses Yahoo Finance’s public search API (`yahoo_search_news`) to retrieve recent news items for each ticker symbol in the hard-coded list.  
   - Oversamples by a factor (e.g. 4×) to account for later filtering, then loops until it has up to `per_ticker` valid articles per ticker.

2. **Article Fetching & Cleaning**  
   - **`fetch_article(url)`** downloads the article page and tries two extraction strategies:  
     1. **JSON-LD**: many sites embed a `<script type="application/ld+json">` with a full `"articleBody"`.  
     2. **HTML `<p>` tags**: if JSON-LD isn’t available, falls back to scraping paragraphs from the DOM.  
   - **Promo-line stripping**: paragraphs matching common “subscribe,” “related,” or “disclosure” patterns are removed (`PROMO_PATTERNS`).   
   - **Continuation logic**: if the page shows a “Continue reading” link, it follows and appends that content (so you don’t miss the end of long articles).

3. **Quality Filtering**  
   - Discards any item with fewer than `_MIN_WORDS` words in the cleaned body.  
   - Ensures there’s a non-empty summary (from JSON-LD, meta tags, or the Yahoo API “abstract”).

4. **Output: JSON Lines File**  
   - Writes each harvested example as one JSON object per line, containing:  
     ```json
     {
       "ticker": "...",
       "title": "...",
       "url": "...",
       "timestamp": "...",    // ISO-formatted publish time
       "body": "...",         // clean article text
       "summary": "...",      // publisher or API summary
       "used_continuation": false
     }
     ```
   - This `.json` file becomes the raw dataset.

5. **Automatic Train/Eval Split**  
   - Once harvesting is done (or skipped if the file already exists), the script reads the JSONL into a Pandas DataFrame.  
   - Uses `sklearn.model_selection.train_test_split` with a fixed seed for reproducibility.  
   - Writes out two new JSONL files (`train.json` & `eval.json`) to feed directly into your Hugging Face tokenization & fine-tuning pipeline.

---