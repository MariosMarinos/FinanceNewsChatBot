"""train.py
=================================================
End‑to‑end pipeline for LoRA fine‑tuning Bart-large-cnn summarization model

Highlights
----------
* **Dataset loader / splitter** – reads dataset (train.json) JSON with
  `body` + `summary`, keeps only those two columns.
* **Tokenisation** – converts text → model inputs and
  attaches `labels` for teacher forcing.
* **Model builder** – wraps an 8‑bit Seq2Seq base model
  with PEFT‑LoRA so we only train a few million params.
* **TrainingArguments helper** – hides all the boilerplate
  for tiny‑data defaults.
* **Grid runner** – sweeps 3×3 hyper‑parameter combos and
  persists *every* artefact (model, tokenizer, hyper‑params)
  under an output folder`.

Improvments
----------
* ** The current implementation doesn't take into account that the 
articles might overpass the 1024 tokens per article. Is something that I 
wanted to handle, but due to time constraints that wasn't feasible.

Run this file directly or import its helpers from a notebook.

"""

import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------------------------
# 1. Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(json_path: str, val_pct: float = 0.25, seed: int = 2025) -> Tuple[Dataset, Dataset]:
    """Load the finance‑news JSON and return *train* / *val* splits.

    Parameters
    ----------
    json_path : str
        Path to the newline‑delimited JSON file. Each line must have
        columns `body` (article) and `summary`.
    val_pct : float, default ``0.25``
        Fraction of the full set to reserve for validation.
    seed : int, default ``2025``
        Random seed for deterministic shuffling / splitting.
    Returns
    -------
    train_ds, val_ds : datasets.Dataset
        Two HF Datasets ready for tokenisation.
    """
    df = pd.read_json(json_path, lines=True)

    # Keep only the text we need and rename to generic column names
    ds_full = (
        Dataset.from_pandas(df)
        .rename_columns({"body": "article", "summary": "summary"})
        .remove_columns([c for c in df.columns if c not in {"body", "summary"}])
    )

    split = ds_full.train_test_split(test_size=val_pct, seed=seed)
    return split["train"], split["test"]

# ---------------------------------------------------------------------------
# 2. Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(
    train_ds: Dataset,
    val_ds: Dataset,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    max_src: int = 1024,
    max_tgt: int = 128,
) -> Tuple[AutoTokenizer, Dataset, Dataset]:
    """Tokenise *both* splits and attach `labels`.

    Parameters
    ----------
    train_ds, val_ds : Dataset
        Raw text splits from :pyfunc:`load_dataset`.
    model_name : str
        Any seq‑to‑seq model on the Hub; defaults to DistilBART‑CNN.
    max_src : int
        Truncation length for the **input** article.
    max_tgt : int
        Truncation length for the **target** summary.

    Returns
    -------
    tok : transformers.AutoTokenizer
    train_tok, val_tok : Dataset
        The tokenised splits (columns: *input_ids*, *attention_mask*, *labels*).
    """

    tok = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        # Encode source
        model_inputs = tok(batch["article"], max_length=max_src, truncation=True)
        # Encode target (summary)
        with tok.as_target_tokenizer():
            labels = tok(batch["summary"], max_length=max_tgt, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)
    return tok, train_tok, val_tok



# ---------------------------------------------------------------------------
# 3. Metric
# ---------------------------------------------------------------------------

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    """Compute ROUGE‑1/2/L and BERTScore‑F1 (×100, 2 d.p.)."""
    preds, labels = eval_pred
    # Replace ignore‑idx ‑100 with PAD so decode works
    labels = np.where(labels != -100, labels, tok.pad_token_id)

    decoded_preds  = tok.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = tok.batch_decode(labels,  skip_special_tokens=True)

    rouge_dict = rouge.compute(
        predictions=[p.strip() for p in decoded_preds],
        references=[l.strip() for l in decoded_labels],
        use_stemmer=True,
    )

    bs = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="en",
        rescale_with_baseline=True,
    )
    bert_f1 = float(np.mean(bs["f1"]))

    return {
        "rouge1": round(rouge_dict["rouge1"] * 100, 2),
        "rouge2": round(rouge_dict["rouge2"] * 100, 2),
        "rougeL": round(rouge_dict["rougeL"] * 100, 2),
        "bertscore_f1": round(bert_f1 * 100, 2),
    }

# ---------------------------------------------------------------------------
# 4. LoRA model builder
# ---------------------------------------------------------------------------

def build_lora_model(
    base_name: str = "facebook/bart-large-cnn",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
):
    """Create an 8‑bit base model wrapped with LoRA adapters.

    Only the low‑rank adapter weights are trainable, so GPU memory
    stays minimal.

    Parameters mirror :class:`peft.LoraConfig`.
    """
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_name,
        quantization_config=quant_cfg,
        device_map="auto",
        gradient_checkpointing=True,  # saves ~3 GB
    )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------------------
# 5. TrainingArguments helper
# ---------------------------------------------------------------------------

def make_args(
    output_dir: str,
    epochs: int = 4,
    batch_size: int = 4,
    lr: float = 2e-4,
    gen_max_len: int = 128,
) -> Seq2SeqTrainingArguments:
    """Return HF :class:`Seq2SeqTrainingArguments` with sane tiny‑data defaults."""

    return Seq2SeqTrainingArguments(
        # where checkpoints + artefacts go
        output_dir=output_dir,

        # optimisation loop
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),  # eff. batch ≈16
        learning_rate=lr,
        warmup_ratio=0.1,            # 10 % linear warm‑up
        weight_decay=0.01,
        fp16=True,

        # evaluation / saving cadence
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1", # early stop on semantics instead of Rouge

        # logging
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",           

        # generation parameters for eval
        predict_with_generate=True,
        generation_max_length=gen_max_len,
        generation_num_beams=1,
    )

# ---------------------------------------------------------------------------
# 6. Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(json_path: str, lora_kw: Dict, train_kw: Dict, tag: str) -> float:
    """Train a single LoRA + Trainer combo and persist artefacts.

    • All artefacts go under */kaggle/working/<tag>/*
    • Returns the validation ROUGE‑L so the grid search can pick winners.
    """
    print(f"\n▶️  Running {tag}")
    out_dir = Path(f"/kaggle/working/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_raw, val_raw = load_dataset(json_path)
    global tok  # exposed for compute_metrics
    tok, train_ds, val_ds = tokenize_dataset(train_raw, val_raw)

    # Model + Trainer
    model = build_lora_model(**lora_kw)
    args = make_args(output_dir=str(out_dir), **train_kw)
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)
    # Setup callbacks: early stop if metric_for_best_model is set
    callbacks = []
    if args.metric_for_best_model:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
        
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks = callbacks
    )
    
    train_output = trainer.train()
    val_bs = train_output.metrics.get("eval_bertscore_f1", float("nan"))
    val_rl = train_output.metrics.get("eval_rougeL",      float("nan"))

    # ---------------- persist artefacts ----------------
    model.save_pretrained(out_dir / "model")
    tok.save_pretrained(out_dir / "tokenizer")

    # • per‑epoch history
    pd.DataFrame(trainer.state.log_history).to_csv(out_dir / "history.csv", index=False)

    # • summary json
    with open(out_dir / "metrics.json", "w") as fp:
        json.dump({
            "lora": lora_kw,
            "trainer": train_kw,
            "val_rougeL": val_rl,
            "val_bertscore_f1": val_bs,
        }, fp, indent=2)

    print(f"🏁 {tag} done – BERT‑F1 {val_bs:.2f} • ROUGE-L {val_rl:.2f}")
    return val_bs


# ---------------------------------------------------------------------------
# 7. Default tiny grid
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    JSON_PATH = "src/dataset_creation/data/train.json" 

    # LoRA capacity sweep (three reasonable sizes)
    lora_grid: List[Dict] = [
        {"r": 8,  "lora_alpha": 16},
        {"r": 16, "lora_alpha": 32}
    ]

    # Training-loop sweep (batch & LR)
    train_grid: List[Dict] = [
        {"batch_size": 4, "lr": 1e-4, "epochs": 5}, 
        {"batch_size": 4, "lr": 2e-4, "epochs": 4},  
        {"batch_size": 8, "lr": 2e-4, "epochs": 3}, 
    ]


    results: Dict[str, float] = {}
    ### Create the corresponding experiments folders.
    for i, (l, t) in enumerate(product(lora_grid, train_grid), start=1):
        tag = f"exp{i:02d}_r{l['r']}_bs{t['batch_size']}_lr{t['lr']}"
        rougeL = run_experiment(JSON_PATH, l, t, tag)
        results[tag] = rougeL