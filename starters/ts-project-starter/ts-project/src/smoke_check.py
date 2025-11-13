"""
Lightweight sanity check for the regression starter.

Loads the configured dataset (tabular or time-series), runs a single forward/backward
pass, and writes a tiny metrics JSON to `outputs/smoke_metrics.json`. If the dataset
files are missing, it raises a helpful error so you can add them before full training.
"""

from pathlib import Path
import torch
import torch.nn as nn

from utils import load_yaml, set_seed, get_device, save_json
from data import build_loaders
from models import MLPRegressor, LSTMRegressor


def _check_data_exists(cfg):
    task = cfg["task"]
    data_cfg = cfg["data"]
    if task == "regression_tabular":
        csv_path = Path(data_cfg["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected CSV at {csv_path}. Create a small CSV with a numeric 'target' column "
                "before running the smoke test or full training."
            )
    elif task == "regression_timeseries":
        csv_path = Path(data_cfg["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected time-series CSV at {csv_path}. Add your data (timestamp + target + features) "
                "before running the smoke test or training."
            )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_smoke(cfg_path: str = "configs/reg_tabular_mlp.yaml") -> Path:
    cfg = load_yaml(cfg_path)
    _check_data_exists(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, _, _, in_dim, meta = build_loaders(cfg)
    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is empty. Check your CSV paths and splits before continuing.")

    task = cfg["task"]
    if cfg["model"]["type"] == "mlp":
        model = MLPRegressor(
            in_dim,
            tuple(cfg["model"]["hidden_dims"]),
            cfg["model"]["dropout"],
            cfg["model"]["activation"],
        ).to(device)
    else:
        model = LSTMRegressor(
            in_dim,
            cfg["model"]["hidden_dim"],
            cfg["model"]["num_layers"],
            bool(cfg["model"]["bidirectional"]),
            cfg["model"]["dropout"],
        ).to(device)

    criterion = nn.MSELoss()
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
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device).squeeze(-1)

    optimizer.zero_grad(set_to_none=True)
    preds = model(xb)
    loss = criterion(preds, yb)
    loss.backward()
    optimizer.step()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_metrics.json"
    save_json(
        {
            "loss": float(loss.item()),
            "batch_size": int(xb.size(0)),
            "input_shape": list(xb.shape),
            "task": task,
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    path = run_smoke()
    print(f"Smoke check succeeded. Metrics written to {path}.")
