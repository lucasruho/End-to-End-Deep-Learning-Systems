import argparse, pandas as pd, numpy as np, torch
from pathlib import Path
from utils import StandardScaler
from models import MLPRegressor, LSTMRegressor
import json

@torch.no_grad()
def main(cfg_path, ckpt_path, csv_path, feature_cols=None, time_series=False, seq_len=48):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    scaler = StandardScaler.from_dict(meta["scaler"]) if meta.get("scaler") else None
    feat_cols = feature_cols or meta.get("feature_cols")

    df = pd.read_csv(csv_path)
    if not time_series:
        X = df[feat_cols].to_numpy(dtype=np.float32)
        Xs = scaler.transform(X) if scaler else X
        model = MLPRegressor(len(feat_cols), (256,128), 0.1, "relu")
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        yhat = model(torch.from_numpy(Xs)).numpy().reshape(-1).tolist()
    else:
        feats = df[feat_cols].to_numpy(dtype=np.float32)
        feats = scaler.transform(feats) if scaler else feats
        xs = []
        for t in range(len(feats) - seq_len + 1):
            xs.append(feats[t:t+seq_len])
        if not xs:
            print("Not enough rows for the given seq_len")
            return
        Xs = np.stack(xs)
        model = LSTMRegressor(len(feat_cols), 128, 1, False, 0.1)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        yhat = model(torch.from_numpy(Xs)).numpy().reshape(-1).tolist()

    out_path = Path("outputs/preds.json"); out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pred": yhat}, open(out_path, "w"), indent=2)
    print("Saved", out_path)

if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/reg_tabular_mlp.yaml")
    p.add_argument("--ckpt",   type=str, default="outputs/best.pt")
    p.add_argument("--csv",    type=str, required=True)
    p.add_argument("--features", type=str, default="")  # comma-separated
    p.add_argument("--timeseries", action="store_true")
    p.add_argument("--seq_len", type=int, default=48)
    a = p.parse_args()
    feats = a.features.split(",") if a.features else None
    main(a.config, a.ckpt, a.csv, feats, a.timeseries, a.seq_len)
