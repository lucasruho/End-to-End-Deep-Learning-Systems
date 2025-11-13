from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import tokenize

SPECIALS = {"<pad>":0, "<unk>":1}

@dataclass
class TextExample:
    tokens: List[int]
    label: int
    length: int

class Vocab:
    def __init__(self, max_size:int=30000, min_freq:int=2):
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>":0, "<unk>":1}
        self.max_size = max_size
        self.min_freq = min_freq

    def build(self, token_lists: List[List[str]]):
        cnt = Counter()
        for toks in token_lists:
            cnt.update(toks)
        for tok, c in cnt.most_common():
            if c < self.min_freq: break
            if tok in SPECIALS: continue
            if len(self.itos) >= self.max_size: break
            self.stoi.setdefault(tok, len(self.itos))
            self.itos.append(tok)

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [ self.stoi.get(t, unk) for t in toks ]

    @property
    def pad_idx(self): return self.stoi["<pad>"]

class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocab, max_len:int):
        self.vocab = vocab
        self.max_len = max_len
        self.examples: List[TextExample] = []
        for t, y in zip(texts, labels):
            toks = tokenize(t)
            ids  = vocab.encode(toks)[:max_len]
            self.examples.append(TextExample(tokens=ids, label=int(y), length=len(ids)))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx): return self.examples[idx]

def collate_pad(batch: List[TextExample], pad_idx: int):
    lens = torch.tensor([ex.length for ex in batch], dtype=torch.long)
    maxlen = int(lens.max().item()) if len(lens)>0 else 1
    toks = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
    for i, ex in enumerate(batch):
        if ex.length>0:
            toks[i, :ex.length] = torch.tensor(ex.tokens, dtype=torch.long)
    return toks, lens, labels

def _load_ag_news():
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    train_texts = [ex["text"] for ex in ds["train"]]
    train_labels = [int(ex["label"]) for ex in ds["train"]]
    test_texts = [ex["text"] for ex in ds["test"]]
    test_labels = [int(ex["label"]) for ex in ds["test"]]
    label_names = ["World","Sports","Business","Sci/Tech"]
    return (train_texts, train_labels), (test_texts, test_labels), label_names

def _load_csv(path: str, text_col: str, label_col: str, delimiter: str):
    import csv
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            texts.append(row[text_col])
            labels.append(row[label_col])
    return texts, labels

def _labels_to_ids(labels: List[str]):
    # map strings to 0..C-1 if needed
    uniq = sorted(set(labels))
    to_id = {s:i for i,s in enumerate(uniq)}
    y = [ to_id[l] for l in labels ]
    return y, uniq

def build_loaders(cfg):
    mode = cfg["data"]["dataset"]
    max_vocab = int(cfg["data"]["max_vocab"])
    min_freq  = int(cfg["data"]["min_freq"])
    max_len   = int(cfg["data"]["max_len"])
    bs        = int(cfg["data"]["batch_size"])
    nw        = int(cfg["data"]["num_workers"])

    if mode == "ag_news":
        (tr_texts, tr_labels), (te_texts, te_labels), label_names = _load_ag_news()
        # split small val from train
        val_size = max(2000, int(0.1*len(tr_texts)))
        tr_texts_, va_texts = tr_texts[:-val_size], tr_texts[-val_size:]
        tr_labels_, va_labels = tr_labels[:-val_size], tr_labels[-val_size:]
    elif mode == "csv":
        d = cfg["data"]
        tr_texts, tr_labels = _load_csv(d["train_csv"], d["text_col"], d["label_col"], d["delimiter"]) if d["train_csv"] else ([],[])
        va_texts, va_labels = _load_csv(d["val_csv"], d["text_col"], d["label_col"], d["delimiter"]) if d["val_csv"] else ([],[])
        te_texts, te_labels = _load_csv(d["test_csv"], d["text_col"], d["label_col"], d["delimiter"]) if d["test_csv"] else ([],[])
        # convert labels to ids if not numeric
        try:
            tr_labels = [int(y) for y in tr_labels]
            va_labels = [int(y) for y in va_labels]
            te_labels = [int(y) for y in te_labels] if te_texts else []
            label_names = [str(i) for i in sorted(set(tr_labels+va_labels+te_labels))]
        except:
            tr_labels, uniq = _labels_to_ids(tr_labels)
            va_labels = [uniq.index(y) for y in va_labels]
            te_labels = [uniq.index(y) for y in te_labels] if te_texts else []
            label_names = uniq
        tr_texts_, va_texts, tr_labels_, va_labels = tr_texts, va_texts, tr_labels, va_labels
    else:
        raise ValueError("unknown dataset" )

    # Build vocab from train only
    tokenized = [ tokenize(t) for t in tr_texts_ ]
    vocab = Vocab(max_size=max_vocab, min_freq=min_freq); vocab.build(tokenized)

    train_ds = SimpleTextDataset(tr_texts_, tr_labels_, vocab, max_len)
    val_ds   = SimpleTextDataset(va_texts,   va_labels,   vocab, max_len)
    test_ds  = SimpleTextDataset(te_texts,   te_labels,   vocab, max_len) if mode=="ag_news" or cfg["data"]["test_csv"] else None

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, collate_fn=lambda b: collate_pad(b, vocab.pad_idx))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, collate_fn=lambda b: collate_pad(b, vocab.pad_idx))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, collate_fn=lambda b: collate_pad(b, vocab.pad_idx)) if test_ds else None

    num_classes = len(set(tr_labels_ + va_labels + (te_labels if (mode=="ag_news" or cfg["data"]["test_csv"]) else [])))
    return train_loader, val_loader, test_loader, vocab, num_classes, label_names
