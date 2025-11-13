import argparse, csv, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils import set_seed, get_device, load_yaml, save_json, StandardScaler, metrics_regression
from data import build_loaders
from models import MLPRegressor, LSTMRegressor

def train_one_epoch(model, loader, crit, opt, device, is_timeseries=False):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device).squeeze(-1)
        opt.zero_grad(set_to_none=True)
        preds = model(xb) if not is_timeseries else model(xb)
        loss = crit(preds, yb)
        loss.backward(); opt.step()
        loss_sum += loss.item() * yb.size(0); n += yb.size(0)
    return loss_sum / max(n,1)

@torch.no_grad()
def evaluate(model, loader, crit, device, is_timeseries=False):
    model.eval()
    loss_sum, n = 0.0, 0
    preds_all, y_all = [], []
    for xb, yb in tqdm(loader, desc="val", leave=False):
        xb, yb = xb.to(device), yb.to(device).squeeze(-1)
        out = model(xb) if not is_timeseries else model(xb)
        loss = crit(out, yb)
        loss_sum += loss.item() * yb.size(0); n += yb.size(0)
        preds_all.append(out.detach().cpu().numpy()); y_all.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(y_all) if y_all else np.array([])
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    return loss_sum / max(n,1), y_true, y_pred

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "log.csv").write_text("epoch,train_loss,val_loss,val_mse,val_mae,val_r2\n")

    train_loader, val_loader, test_loader, in_dim, meta = build_loaders(cfg)
    is_ts = cfg["task"] == "regression_timeseries"

    if cfg["model"]["type"] == "mlp":
        model = MLPRegressor(in_dim, tuple(cfg["model"]["hidden_dims"]), cfg["model"]["dropout"], cfg["model"]["activation"]).to(device)
    else:
        model = LSTMRegressor(in_dim, cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], bool(cfg["model"]["bidirectional"]), cfg["model"]["dropout"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"].lower() == "sgd":
        opt = SGD(params, lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])
    else:
        opt = AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = CosineAnnealingLR(opt, T_max=cfg["train"]["t_max"]) if cfg["train"]["scheduler"] == "cosine" else None

    crit = nn.MSELoss()
    best_mse, best_path = float("inf"), out / "best.pt"
    patience, waited = cfg["early_stopping"]["patience"], 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device, is_ts)
        val_loss, y_true, y_pred = evaluate(model, val_loader, crit, device, is_ts)
        if sch: sch.step()

        m = metrics_regression(y_true, y_pred)
        with open(out / "log.csv", "a", newline="") as f:
            import csv as _csv
            _csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{m['mse']:.6f}", f"{m['mae']:.6f}", f"{m['r2']:.6f}"])

        if m["mse"] < best_mse - cfg["early_stopping"]["min_delta"]:
            best_mse, waited = m["mse"], 0
            torch.save({"state_dict": model.state_dict(), "meta": meta}, best_path)
        else:
            waited += 1
            if waited > patience:
                break

    save_json({"best_val_mse": best_mse, "meta": meta}, out / "metrics.json")
    print(f"Done. Best val MSE: {best_mse:.6f}. Checkpoint: {best_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/reg_tabular_mlp.yaml")
    args = ap.parse_args()
    main(args.config)
