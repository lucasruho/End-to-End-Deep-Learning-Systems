
Track C — NLP Text Classification
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

## C•Intermediate — Yahoo Answers Topics (10-class)
**Why:** longer texts, 10 classes, topic skew.
**Preset:** `configs/nlp_yahoo.yaml`.

**Download**
```python
!pip -q install datasets
from datasets import load_dataset
ds = load_dataset("yahoo_answers_topics")
print(ds)
```
Run with:
```bash
python src/train.py --config configs/nlp_yahoo.yaml
python src/evaluate.py --config configs/nlp_yahoo.yaml --ckpt outputs/best.pt
```

## C•Difficult — Yelp Review Full (5-class)
**Why:** fine-grained sentiment at scale.
**Preset:** `configs/nlp_yelp_full.yaml`.
Optionally subsample to 50k train / 10k test for Colab.
```python
from datasets import load_dataset
raw = load_dataset("yelp_review_full")
train = raw["train"].shuffle(seed=42).select(range(50000))
test  = raw["test"].shuffle(seed=42).select(range(10000))
train.to_csv("data/train.csv", index=False); test.to_csv("data/val.csv", index=False)
# switch config to CSV mode if desired
```
