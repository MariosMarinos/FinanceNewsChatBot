"""evaluation.py
=================================================
A **minimal, fully‑annotated** helper for evaluating an abstractive‑summarisation
model (BART‑Large‑CNN by default) on a JSON‑Lines dataset.  The script exposes a
single public API – `evaluate_file` – which:

1. Loads every row from a *JSONL* file that stores
   * `body`   – the full article text to summarise, and
   * `summary` – the human‑written reference abstract.
2. Generates a model summary for the *entire* article while respecting the
   1 024‑token context of BART‑large via a sliding‑window, map‑reduce strategy.
3. Computes corpus‑level ROUGE‑1/2/L and BERTScore metrics.
4. Returns a pair `(pred_df, metrics)` so that downstream notebooks can inspect
   per‑row predictions **and** aggregate scores.

Designed to be light on dependencies, boilerplate, and hidden state – drop it
into any project and call

```python
from mini_summarisation_eval_commented import evaluate_file
pred_df, metrics = evaluate_file("evaluation.jsonl", device=0)
```

All functions are documented with **docstrings** and annotated with **inline
comments** for quick comprehension.
"""

from __future__ import annotations  # ↩ allow forward type references (Py<3.11)

# ──────────────────────────── Standard library ─────────────────────────────
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import evaluate  # 🤗 Evaluate – ROUGE & BERTScore

# ─────────────────────── Third‑party scientific stack ──────────────────────
import pandas as pd  # DataFrame I/O and manipulation
import torch  # Detect CUDA / place model on GPU
from transformers import (
    BartForConditionalGeneration,  # Model checkpoint loader
    BartTokenizer,  # Tokeniser for BART
    pipeline,  # High‑level HF task wrapper
)

# =============================================================================
# 1.  Model helper – build a *summarization* pipeline
# =============================================================================

def build_summariser(
    model_name: str = "facebook/bart-large-cnn",
    device: int | str | None = None,
    *,  # force keyword‑only for the tuning knobs below
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.1,
):
    """Create a HuggingFace **summarization** pipeline.

    Parameters
    ----------
    model_name
        Seq2Seq checkpoint with ≤1 024‑token context; defaults to BART‑Large‑CNN.
    device
        `0`  → first CUDA GPU · `-1` → CPU · `None` (default) → autodetect.
    repetition_penalty, no_repeat_ngram_size, length_penalty
        Decoding‑time knobs that reduce verbatim repetition and favour slightly
        longer abstracts.

    Returns
    -------
    tuple[pipeline, BartTokenizer]
        * The ready‑to‑call HF pipeline.
        * The matching tokenizer (needed for token‑window chunking).
    """

    # 1.  Tokeniser – enforces the 1 024 token window of BART‑large
    tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1024

    # 2.  Model weights – downloaded / cached by transformers
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # 3.  Device autotune – pick GPU if available and caller didn’t override
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # 4.  Wrap everything in a task‑specific pipeline for convenience
    summariser = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
    )

    return summariser, tokenizer

# =============================================================================
# 2.  Token‑safe summarisation – map‑reduce over 1 k+ token articles
# =============================================================================

def _chunk(
    article: str,
    tok: BartTokenizer,
    *,
    window: int = 1024,
    overlap: int = 200,
):
    # 1️⃣  Get raw content token IDs only
    ids = tok(article, truncation=False, add_special_tokens=False)["input_ids"]
    step = window - overlap

    for i, start in enumerate(range(0, len(ids), step)):
        # 2️⃣  Slice content IDs
        slice_ids = ids[start : start + window]

        # 3️⃣  Decode back to text
        txt = tok.decode(slice_ids, skip_special_tokens=True)

        # 4️⃣  Re-encode content only
        re_ids = tok(txt, truncation=False, add_special_tokens=False)["input_ids"]

        # 5️⃣  Print lengths—these *should* match exactly
        print(f"[DEBUG] window {i}:")
        print(f"  content IDs slice      → {len(slice_ids)} tokens")
        print(f"  re-tokenized content   → {len(re_ids)} tokens")
        if len(re_ids) != len(slice_ids):
            print("  ⚠️  MISMATCH in content token counts!")

        yield txt


def summarise(
    article: str,
    summariser,       # HF pipeline
    tok: BartTokenizer,
    *,
    window: int = 1024,
    overlap: int = 200,
    chunk_len: int = 60,
    final_len: int = 120,
) -> str:
    """"Map-reduce summarisation with verbose token‐count debug info."""

  # ─── EARLY BAILOUT FOR SHORT INPUTS ─────────────────────────────────────────
    raw_ids = tok(article, truncation=False)["input_ids"]
    print(f"Article is {len(raw_ids)} tokens long; window={window}")
    if len(raw_ids) <= window:
        # Single pass at the 'final' length: no map / no reduce.
        print("→ fits in one window; doing one final summarisation and returning\n")
        return summariser(
            article,
            max_length=final_len,
            min_length=max(40, final_len // 2),
            do_sample=False,
            truncation=True,
        )[0]["summary_text"].strip()

    # ─── OTHERWISE: TRUE MAP-REDUCE FLOW ────────────────────────────────────────
    # 1. Create sliding-window chunks
    chunks = list(_chunk(article, tok, window=window, overlap=overlap))
    print(f"→ {len(chunks)} chunks generated")

    # 2. MAP: summarise each chunk
    blurbs = []
    for i, chunk in enumerate(chunks):
        ids = tok(chunk, truncation=False)["input_ids"]
        print(f"[MAP] chunk {i}: {len(ids)} tokens → generating summary")
        try:
            out = summariser(
                chunk,
                max_length=chunk_len,
                min_length=15,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"].strip()
        except Exception as e:
            print(f"⚠️  Exception in MAP chunk {i}: {e}")
            raise
        blurbs.append(out)

    # 3. If for some reason only one chunk remains (shouldn’t happen here),
    #    you could either return it directly or fall through to REDUCE.
    if len(blurbs) == 1:
        return blurbs[0]

    # 4. REDUCE: fuse the blurbs
    bullet_prompt = (
        f"Summarise the following bullet-point section summaries into ONE coherent "
        f"paragraph (≤{final_len} tokens):\n\n" +
        "\n".join(f"• {b}" for b in blurbs)
    )
    bp_ids = tok(bullet_prompt, truncation=False)["input_ids"]
    print(f"[REDUCE] bullet_prompt is {len(bp_ids)} tokens")

    try:
        final = summariser(
            bullet_prompt,
            max_length=final_len,
            min_length=max(40, final_len // 2),
            do_sample=False,
            truncation=True,
        )[0]["summary_text"].strip()
    except Exception as e:
        print(f"⚠️  Exception in REDUCE prompt: {e}")
        raise

    return final

# =============================================================================
# 3.  Metrics – ROUGE‑1/2/L and (baseline‑rescaled) BERTScore
# =============================================================================

def compute_metrics(
    preds: List[str],
    refs: List[str],
    *,
    use_baseline: bool = True,
) -> dict:
    """Return corpus‑level ROUGE and BERTScore averages.

    Parameters
    ----------
    preds / refs
        Must be equal length – i‑th hypothesis compared against i‑th reference.
    use_baseline
        If *True*, BERTScore is baseline‑rescaled (scores can be negative).

    Returns
    -------
    dict with two keys:
        * ``rouge`` – whatever 🤗 *evaluate*'s ROUGE loader returns
        * ``bertscore`` – dict with mean precision/recall/F1 floats
    """

    # Compute ROUGE‑1/2/L (precision, recall, F1) – micro‑averaged by *evaluate*
    rouge = evaluate.load("rouge").compute(predictions=preds, references=refs)

    # Compute BERTScore and average over the dataset manually (macro‑average)
    bs = evaluate.load("bertscore").compute(
        predictions=preds,
        references=refs,
        lang="en",
        rescale_with_baseline=use_baseline,
    )

    return {
        "rouge": rouge,
        "bertscore": {
            "precision": mean(bs["precision"]),
            "recall": mean(bs["recall"]),
            "f1": mean(bs["f1"]),
        },
    }

# =============================================================================
# 4.  Top‑level public API – one function to rule them all
# =============================================================================

def evaluate_file(
    path: str | Path,
    *,
    device: int | str | None = None,
    use_baseline: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Run the full evaluation loop on *path* and return `(pred_df, metrics)`.

    Parameters
    ----------
    path
        Path to a JSON‑Lines file (*one JSON object per line*) with `body` and
        `summary` keys.
    device
        `0/1/…` → that CUDA GPU · `-1` → CPU · `None` → autodetect.
    use_baseline
        Whether to baseline‑rescale BERTScore.

    Returns
    -------
    pred_df : pandas.DataFrame
        Original rows plus a new `prediction` column holding model summaries.
    metrics : dict
        Output of :func:`compute_metrics` – corpus ROUGE & BERTScore.
    """

    # 1.  Load dataset (assumes JSON‑Lines / one article per line)
    df = pd.read_json(path, lines=True)

    # 2.  Build model once – reused for every row # WE NEED TO CHANGE IT HERE SO WE CAN PASS WHICH MODEL TO RUN
    summariser, tok = build_summariser(device=device)

    # 3.  Generate summaries – pandas apply keeps things tidy
    df = df.copy()  # avoid SettingWithCopyWarning
    df["prediction"] = df["body"].apply(lambda txt: summarise(txt, summariser, tok))

    # 4.  Aggregate metrics
    metrics = compute_metrics(
        preds=df["prediction"].tolist(),
        refs=df["summary"].tolist(),
        use_baseline=use_baseline,
    )

    return df, metrics

# =============================================================================
# 5.  CLI demo – only executed when you `python mini_summarisation_eval…`.
#     Keeps the file usable as both a module *and* a tiny script.
# =============================================================================
if __name__ == "__main__":
    # ─── Quick & dirty smoke test ────────────────────────────────────────────
    pred_df, metrics = evaluate_file(
        path="data/eval.json", 
        device=0,                 # set to –1 for CPU
        use_baseline=True,
    )