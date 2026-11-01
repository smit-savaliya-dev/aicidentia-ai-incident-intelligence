"""Fine-tune DistilBERT on MIT risk domains with the repository's folds (designed to run on Google Colab).

This file is the single source of the training code: notebooks/colab_distilbert_finetune.ipynb embeds it
verbatim, and tests/test_phase6.py smoke-tests it locally on a handful of incidents.

Protocol (fixed in advance; nothing is tuned on a test fold):
* model distilbert-base-uncased, max length 128 tokens, 4 epochs, batch 16, AdamW lr 5e-5, weight decay 0.01,
  linear schedule with 10% warm-up, cross-entropy weighted by 'balanced' inverse class frequency of the
  training part (the counterpart of class_weight="balanced" in the scikit-learn models);
* the SAME 5 stratified folds as the local models (column `fold` of the input CSV) and the same time split
  (column `is_test_time_split`: train <= 2023, test >= 2024);
* seed = 42 + fold for fold runs and 42 for the time-split run; fp32; deterministic algorithms where possible.

Input CSV columns: incident_id, text, year, label, fold, is_test_time_split.
Output CSV columns: incident_id, setting ('cv' | 'time'), fold, y_true, y_pred, proba_<label> for every label.
"""

import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
BASE_SEED = 42


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and ask for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """n / (k * count) per class (scikit-learn's 'balanced'); absent classes get weight 0."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = np.divide(len(labels), n_classes * counts, out=np.zeros(n_classes), where=counts > 0)
    return torch.tensor(weights, dtype=torch.float32)


def encode(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Tokenise to fixed-length tensors."""
    return tokenizer(
        texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )


def train_and_predict(
    train_texts: list[str],
    train_y: np.ndarray,
    test_texts: list[str],
    n_classes: int,
    seed: int,
    device: str,
    epochs: int = EPOCHS,
) -> np.ndarray:
    """Fine-tune on the training part and return softmax probabilities for the test texts."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes).to(
        device
    )
    enc = encode(tokenizer, train_texts)
    labels = torch.tensor(train_y, dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=balanced_class_weights(train_y, n_classes).to(device)
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = epochs * int(np.ceil(len(train_texts) / BATCH_SIZE))
    scheduler = get_linear_schedule_with_warmup(optimiser, int(WARMUP_FRACTION * steps), steps)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(train_texts), generator=generator)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            batch = {k: v[idx].to(device) for k, v in enc.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            total += float(loss) * len(idx)
        print(f"  epoch {epoch + 1}/{epochs}: training loss {total / len(order):.4f}", flush=True)
    model.eval()
    test_enc = encode(tokenizer, test_texts)
    probs = []
    with torch.no_grad():
        for start in range(0, len(test_texts), 64):
            batch = {k: v[start : start + 64].to(device) for k, v in test_enc.items()}
            probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)


def run(input_csv: str, output_csv: str, smoke: bool = False) -> pd.DataFrame:
    """Run all five folds plus the time split and write the predictions CSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}", flush=True)
    df = pd.read_csv(input_csv)
    classes = sorted(df["label"].unique())
    code = {c: i for i, c in enumerate(classes)}
    if smoke:  # tiny end-to-end check: a few incidents per class, one epoch
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=8, replace=False, random_state=0)
            .reset_index(drop=True)
        )
    y_all = df["label"].map(code).to_numpy()
    epochs = 1 if smoke else EPOCHS
    parts = []

    def collect(setting: str, test: np.ndarray, probs: np.ndarray, fold: int) -> None:
        part = pd.DataFrame(
            {
                "incident_id": df.loc[test, "incident_id"].to_numpy(),
                "setting": setting,
                "fold": fold,
                "y_true": [classes[i] for i in y_all[test]],
                "y_pred": [classes[i] for i in probs.argmax(axis=1)],
            }
        )
        for k, name in enumerate(classes):
            part[f"proba_{name}"] = probs[:, k]
        parts.append(part)

    for fold in sorted(df["fold"].unique()):
        started = time.time()
        test = (df["fold"] == fold).to_numpy()
        print(f"fold {fold}: train {int((~test).sum())}, test {int(test.sum())}", flush=True)
        probs = train_and_predict(
            df.loc[~test, "text"].tolist(),
            y_all[~test],
            df.loc[test, "text"].tolist(),
            len(classes),
            BASE_SEED + int(fold),
            device,
            epochs,
        )
        collect("cv", test, probs, int(fold))
        print(f"  done in {time.time() - started:.0f}s", flush=True)
        if smoke:
            break
    test = (df["is_test_time_split"] == 1).to_numpy()
    if test.any() and (~test).any():
        print(f"time split: train {int((~test).sum())}, test {int(test.sum())}", flush=True)
        probs = train_and_predict(
            df.loc[~test, "text"].tolist(),
            y_all[~test],
            df.loc[test, "text"].tolist(),
            len(classes),
            BASE_SEED,
            device,
            epochs,
        )
        collect("time", test, probs, -1)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(output_csv, index=False)
    print(f"wrote {output_csv} ({len(out)} rows)", flush=True)
    return out
