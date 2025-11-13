
Track C — NLP Text Classification • Dataset Setup
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

Two datasets via `datasets` library.

## C1) AG News (primary)
```python
!pip -q install datasets
from datasets import load_dataset
ds = load_dataset("ag_news")
print(ds)
```
Train with default config:
```bash
python src/train.py --config configs/nlp_agnews.yaml
python src/evaluate.py --config configs/nlp_agnews.yaml --ckpt outputs/best.pt
```
Macro-F1 early stopping; metrics to `outputs/metrics.json` / `outputs/eval.json`.

## C2) IMDb (secondary, subsample)
```python
!pip -q install datasets
from datasets import load_dataset
raw = load_dataset("imdb")
# Subsample for speed
train = raw["train"].shuffle(seed=42).select(range(10000))
test  = raw["test"].shuffle(seed=42).select(range(5000))
# Convert to CSV for the CSV mode if preferred
train.to_csv("data/train.csv", index=False)
test.to_csv("data/val.csv", index=False)
```
Switch config to CSV mode in `configs/nlp_agnews.yaml`:
```yaml
data:
  dataset: csv
  train_csv: data/train.csv
  val_csv: data/val.csv
  text_col: text
  label_col: label
```

## C3) Your own CSV
CSV with columns `text` and `label` (strings or 0..C-1). Labels as strings are auto-mapped to IDs. Then:
```bash
python src/train.py --config configs/nlp_agnews.yaml
```

Tips: adjust `max_len`, `emb_dim`, `hidden_dim`, `dropout`. Use bidirectional LSTM for a stronger baseline.
