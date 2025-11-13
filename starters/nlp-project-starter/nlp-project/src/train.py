import argparse, csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from utils import set_seed, get_device, load_yaml, save_json
from data import build_loaders
from model import LSTMClassifier

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    loss_sum, n = 0.0, 0
    for toks, lens, y in tqdm(loader, desc="train", leave=False):
        toks, lens, y = toks.to(device), lens.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(toks, lens)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0); n += y.size(0)
    return loss_sum / n

@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes):
    model.eval()
    acc = MulticlassAccuracy(num_classes=num_classes).to(device)
    f1m = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    loss_sum, n = 0.0, 0
    for toks, lens, y in tqdm(loader, desc="val", leave=False):
        toks, lens, y = toks.to(device), lens.to(device), y.to(device)
        logits = model(toks, lens)
        loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0); n += y.size(0)
        acc.update(logits, y); f1m.update(logits, y)
    return loss_sum / n, acc.compute().item(), f1m.compute().item()

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "log.csv").write_text("epoch,train_loss,val_loss,val_acc,val_f1_macro\n")

    train_loader, val_loader, test_loader, vocab, num_classes, label_names = build_loaders(cfg)
    model = LSTMClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=cfg["model"]["dropout"],
        num_classes=num_classes,
        pad_idx=vocab.pad_idx
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"].lower() == "sgd":
        opt = SGD(params, lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])
    else:
        opt = AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    if cfg["train"]["scheduler"] == "cosine":
        sch = CosineAnnealingLR(opt, T_max=cfg["train"]["t_max"])
    else:
        sch = None

    crit = nn.CrossEntropyLoss()

    best_f1, best_path = 0.0, out / "best.pt"
    patience, waited = cfg["early_stopping"]["patience"], 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, crit, device, num_classes)
        if sch: sch.step()

        with open(out / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}"])

        if val_f1 > best_f1:
            best_f1, waited = val_f1, 0
            torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos, "label_names": label_names}, best_path)
        else:
            waited += 1
            if waited > patience:
                break

    save_json({"best_val_f1_macro": best_f1, "label_names": label_names}, out / "metrics.json")
    print(f"Done. Best val F1-macro: {best_f1:.4f}. Checkpoint: {best_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/nlp_agnews.yaml")
    args = ap.parse_args()
    main(args.config)
