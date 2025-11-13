"""
Lightweight sanity check for the NLP starter.

This loads a tiny batch from the configured dataset, runs a single forward/backward
pass through the LSTM classifier, and writes metrics to `outputs/smoke_metrics.json`.
Use it inside the Colab/Kaggle notebook (or locally) before spending time on a full train run.
"""

from pathlib import Path
import torch
import torch.nn as nn

from utils import load_yaml, set_seed, get_device, save_json
from data import build_loaders
from model import LSTMClassifier


def run_smoke(cfg_path: str = "configs/nlp_agnews.yaml") -> Path:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, _, _, vocab, num_classes, label_names = build_loaders(cfg)
    model = LSTMClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=cfg["model"]["dropout"],
        num_classes=num_classes,
        pad_idx=vocab.pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg["train"]["lr"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )

    model.train()
    batch = next(iter(train_loader))
    toks, lens, y = (t.to(device) for t in batch)

    optimizer.zero_grad(set_to_none=True)
    logits = model(toks, lens)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_metrics.json"
    save_json(
        {
            "loss": float(loss.item()),
            "batch_size": int(toks.size(0)),
            "seq_len": int(toks.size(1)),
            "num_classes": num_classes,
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    path = run_smoke()
    print(f"Smoke check succeeded. Metrics written to {path}.")
