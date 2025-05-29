# Fine-Tune Module

This directory contains everything needed to fine-tune the `facebook/bart-large-cnn` model using LoRA adapters.

## 1. Setup Environment

1. **Enter the directory**:
   ```bash
   cd fine_tune
   ```
2. Create and activate a virtual env:
```
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4.(Optional) If you encounter errors or want to ensure the latest versions:
```
pip install --upgrade accelerate transformers bitsandbytes
```

## 2. Configure Hyperparameters
Unfortunately, this has to be manually inside the scripts because I didnt have the time to implement config.yaml

## 3. Run training script

```
python train.py
```


## What This Script Does

1. **Dataset Loading & Splitting**  
   - Reads your newline-delimited JSON file (`train.json`), which  contain two fields per line:  
     ```json
     { "body": "...", "summary": "..." }
     ```  
   - Keeps only those two columns, renaming `body → article` and `summary → summary`.  
   - Splits into train/validation according to a fixed fraction (default 25 %), using a hard-coded seed for reproducibility.

2. **Tokenization**  
   - Uses the Hugging Face `AutoTokenizer` for your chosen Seq2Seq model (default `facebook/bart-large-cnn` in 8-bit).  
   - Truncates long articles at `max_src` tokens (default 1 024) and summaries at `max_tgt` tokens (default 128).  
   - Packages inputs into the standard `input_ids`, `attention_mask` and `labels` (for teacher-forcing).

3. **Metric Computation**  
   - During evaluation, automatically computes **ROUGE-1/2/L** (stemmed) and **BERTScore-F1** to capture both n-gram overlap and semantic fidelity.  
   - Converts any `-100` label padding back to the tokenizer’s pad token before decoding.

4. **LoRA Model Builder**  
   - Loads your base Seq2Seq model in **8-bit** via `BitsAndBytesConfig`, which slashes VRAM without a significant performance hit.  
   - Wraps the frozen base with a small PEFT-LoRA adapter (rank `r`, α, dropout, bias settings), so only a few million parameters are trained.

5. **TrainingArguments Helper**  
   - Encapsulates all the “tiny-data” best practices:  
     - FP16 mixed precision  
     - Gradient accumulation to hit an effective batch of ~16  
     - Cosine or default scheduler with warm-up  
     - Logging every few steps, evaluation & checkpointing each epoch  
     - Early stopping on the best **BERTScore-F1**  
     - Beam search inference during evaluation  

6. **Grid Runner**  
   - Sweeps a small grid of LoRA capacities (`r`) and training-loop settings (`batch_size`, `lr`, `epochs`).  
   - For each combination, creates its own `expXX_rXX_bsXX_lrX.X` folder under your output root:  
     - Saves the **adapter** (+ tokenizer)  
     - Dumps per-epoch logs (`history.csv`)  
     - Writes a summary `metrics.json` with final ROUGE-L & BERTScoreF1  
   - Prints a final map of tags → ROUGE-L so you can immediately spot the best run.

---

### Why These Steps Matter

- **8-bit + LoRA** keeps your training on limited-memory GPUs (e.g. T4s) both **fast** and **stable**, without sacrificing much accuracy.  
- **Dual metrics** (ROUGE + BERTScore) ensure you’re optimizing both surface overlap and deeper meaning.  
- **Grid search** is automated, reproducible, and self-contained—no manual re-runs or lost checkpoints.  
- **Modular helpers** let you import any step (data loading, model building, etc.) in a notebook for quick experiments.
