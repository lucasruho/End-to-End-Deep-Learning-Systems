import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from utils import load_yaml, get_device
from data import build_loaders
from model import LSTMClassifier

@torch.no_grad()
def main(cfg_path, ckpt_path):
    cfg = load_yaml(cfg_path)
    device = get_device()

    _, val_loader, test_loader, vocab, num_classes, label_names = build_loaders(cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = LSTMClassifier(
        vocab_size=len(ckpt.get("vocab", vocab.itos)),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=cfg["model"]["dropout"],
        num_classes=num_classes,
        pad_idx=vocab.pad_idx
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    crit = nn.CrossEntropyLoss()
    def _eval(loader):
        acc = MulticlassAccuracy(num_classes=num_classes).to(device)
        f1m = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        loss_sum, n = 0.0, 0
        for toks, lens, y in tqdm(loader, desc="eval", leave=False):
            toks, lens, y = toks.to(device), lens.to(device), y.to(device)
            logits = model(toks, lens)
            loss = crit(logits, y)
            loss_sum += loss.item() * y.size(0); n += y.size(0)
            acc.update(logits, y); f1m.update(logits, y)
        return loss_sum / n, acc.compute().item(), f1m.compute().item()

    val_loss, val_acc, val_f1 = _eval(val_loader)
    report = {"val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1}

    if test_loader is not None:
        test_loss, test_acc, test_f1 = _eval(test_loader)
        report.update({"test_loss": test_loss, "test_acc": test_acc, "test_f1_macro": test_f1})

    out = Path(cfg["output_dir"]) / "eval.json"
    with open(out, "w") as f: json.dump(report, f, indent=2)
    print("Saved", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/nlp_agnews.yaml")
    p.add_argument("--ckpt", type=str, default="outputs/best.pt")
    a = p.parse_args()
    main(a.config, a.ckpt)
