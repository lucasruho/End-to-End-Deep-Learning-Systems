import argparse, json, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm
from utils import load_yaml, get_device, StandardScaler, metrics_regression, save_json
from data import build_loaders
from models import MLPRegressor, LSTMRegressor

@torch.no_grad()
def main(cfg_path, ckpt_path):
    cfg = load_yaml(cfg_path)
    device = get_device()
    _, val_loader, test_loader, in_dim, meta = build_loaders(cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    if cfg["model"]["type"] == "mlp":
        model = MLPRegressor(in_dim, tuple(cfg["model"]["hidden_dims"]), cfg["model"]["dropout"], cfg["model"]["activation"]).to(device)
    else:
        model = LSTMRegressor(in_dim, cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], bool(cfg["model"]["bidirectional"]), cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    crit = nn.MSELoss()

    def _eval(loader):
        loss_sum, n = 0.0, 0
        y_true, y_pred = [], []
        for xb, yb in tqdm(loader, desc="eval", leave=False):
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            out = model(xb)
            loss = crit(out, yb)
            loss_sum += loss.item() * yb.size(0); n += yb.size(0)
            y_true.append(yb.cpu().numpy()); y_pred.append(out.cpu().numpy())
        y_true = np.concatenate(y_true) if y_true else np.array([])
        y_pred = np.concatenate(y_pred) if y_pred else np.array([])
        m = metrics_regression(y_true, y_pred)
        return loss_sum / max(n,1), m

    val_loss, val_m = _eval(val_loader)
    report = {"val": {"loss": val_loss, **val_m}}
    if test_loader is not None:
        test_loss, test_m = _eval(test_loader)
        report["test"] = {"loss": test_loss, **test_m}

    out = Path(cfg["output_dir"]) / "eval.json"; save_json(report, out)
    print("Saved", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/reg_tabular_mlp.yaml")
    p.add_argument("--ckpt", type=str, default="outputs/best.pt")
    a = p.parse_args()
    main(a.config, a.ckpt)
