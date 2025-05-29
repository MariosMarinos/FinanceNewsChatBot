# Evaluation Helper

A minimal, fully-annotated helper for evaluating an abstractive-summarisation model (BART-Large-CNN by default) on a JSON-Lines dataset. Drop it into any project and call:

```python
from evaluation import evaluate_file
pred_df, metrics = evaluate_file("evaluation.jsonl", device=0)
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install pandas torch transformers evaluate rouge_score bertscore
   ```

## What This Script Does

1. **Build Summariser Pipeline**  
   - `build_summariser()` loads the BART-Large-CNN checkpoint and its tokenizer.  
   - Auto-detects GPU (if available) or falls back to CPU.  
   - Applies decoding safeguards (repetition penalty, n-gram blocking, length penalty) to reduce verbatim repeats and encourage coherent abstracts.

2. **Token-Safe Summarisation**  
   - Articles longer than 1 024 tokens are split into overlapping windows (default window = 1024, overlap = 200) via the `_chunk()` generator.  
   - Each chunk is summarised (the “map” step), then the partial summaries (“blurbs”) are fused into a single final abstract (the “reduce” step) by re-feeding them through the pipeline.

3. **Metric Computation**  
   - `compute_metrics()` computes corpus-level **ROUGE-1/2/L** and **BERTScore** (precision, recall, F1).  
   - ROUGE is micro-averaged by 🤗 Evaluate; BERTScore is macro-averaged over all examples.

4. **Public API: `evaluate_file()`**  
   - Loads your JSON-Lines file (one article per line, with `body` and `summary` fields).  
   - Runs the summarisation pipeline over each article (handling both short and long inputs).  
   - Returns a pandas DataFrame with columns `body`, `summary`, and `prediction`, plus an aggregate metrics dictionary.

5. **CLI Demo**  
   - If the script is invoked directly (`python evaluation.py`), it runs a quick smoke test on the path of evaluation set, printing both per-row predictions and overall scores.

## Why These Steps Matter

- **Sliding-window approach** ensures you never truncate important context in very long articles while still using a fixed-size model.  
- **Dual metrics** (ROUGE + BERTScore) give you both surface-level and semantic-level evaluation.  
- **Single-function API** makes it trivial to plug into notebooks or pipelines without extra boilerplate.  
- **Lightweight dependencies** keep this helper focused and easy to integrate.
