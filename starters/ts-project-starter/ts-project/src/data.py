from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import StandardScaler

def _select_numeric(df, exclude: List[str]):
    cols = [c for c in df.columns if c not in exclude]
    numerics = df[cols].select_dtypes(include=['number']).columns.tolist()
    return numerics

class TabularCSVDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str, feature_cols=None, standardize=True, scaler: Optional[StandardScaler]=None):
        df = pd.read_csv(csv_path)
        if feature_cols is None:
            feature_cols = _select_numeric(df, exclude=[target_col])
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.feature_cols = feature_cols
        self.y = torch.from_numpy(y)
        if standardize:
            self.scaler = scaler or StandardScaler(); 
            if scaler is None: self.scaler.fit(x)
            x = self.scaler.transform(x)
        else:
            self.scaler = None
        self.x = torch.from_numpy(x)

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

class TimeseriesCSVDataset(Dataset):
    def __init__(self, csv_path: str, time_col: str, target_col: str, feature_cols=None, seq_len=48, horizon=1, standardize=True, scaler: Optional[StandardScaler]=None):
        df = pd.read_csv(csv_path)
        if time_col not in df.columns:
            raise ValueError(f"time_col {time_col} not in CSV")
        df = df.sort_values(time_col)
        if feature_cols is None:
            feature_cols = _select_numeric(df, exclude=[target_col])
            if time_col in feature_cols: feature_cols.remove(time_col)
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        tgt   = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

        if standardize:
            self.scaler = scaler or StandardScaler()
            if scaler is None: self.scaler.fit(feats)
            feats = self.scaler.transform(feats)
        else:
            self.scaler = None

        xs, ys = [], []
        for t in range(len(feats) - seq_len - horizon + 1):
            xs.append(feats[t:t+seq_len])
            ys.append(tgt[t+seq_len + horizon - 1])
        self.x = torch.from_numpy(np.stack(xs)) if xs else torch.empty(0)
        self.y = torch.from_numpy(np.stack(ys)) if ys else torch.empty(0)
        self.feature_cols = feature_cols

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

def split_tabular_dataset(ds, val_ratio: float, test_ratio: float, shuffle: bool, seed: int):
    n = len(ds); idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed); rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test+n_val]
    train_idx= idx[n_test+n_val:]
    return torch.utils.data.Subset(ds, train_idx), torch.utils.data.Subset(ds, val_idx), torch.utils.data.Subset(ds, test_idx)

def build_loaders(cfg):
    from utils import StandardScaler
    task = cfg["task"]
    bs = int(cfg["train"]["batch_size"])
    nw = 0

    if task == "regression_tabular":
        dcfg = cfg["data"]
        base = TabularCSVDataset(dcfg["csv_path"], dcfg["target_col"], dcfg.get("feature_cols"), dcfg["standardize"])
        train_ds, val_ds, test_ds = split_tabular_dataset(base, dcfg["val_split"], dcfg["test_split"], dcfg["shuffle"], seed=42)
        scaler = base.scaler
        feature_cols = base.feature_cols
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw) if len(test_ds)>0 else None
        meta = {"scaler": scaler.to_dict() if scaler else None, "feature_cols": feature_cols}
        in_dim = len(feature_cols)
        return train_loader, val_loader, test_loader, in_dim, meta

    elif task == "regression_timeseries":
        dcfg = cfg["data"]
        base = TimeseriesCSVDataset(dcfg["csv_path"], dcfg["time_col"], dcfg["target_col"], dcfg.get("feature_cols"),
                                    dcfg["seq_len"], dcfg["horizon"], dcfg["standardize"])
        # chronological split: last parts for val/test
        n = len(base)
        n_test = int(n * dcfg["test_split"])
        n_val  = int(n * dcfg["val_split"])
        train_ds = torch.utils.data.Subset(base, range(0, n - n_val - n_test))
        val_ds   = torch.utils.data.Subset(base, range(n - n_val - n_test, n - n_test))
        test_ds  = torch.utils.data.Subset(base, range(n - n_test, n)) if n_test>0 else None
        scaler = base.scaler
        feature_cols = base.feature_cols
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw) if test_ds else None
        meta = {"scaler": scaler.to_dict() if scaler else None, "feature_cols": feature_cols,
                "seq_len": dcfg["seq_len"]}
        in_dim = len(feature_cols)
        return train_loader, val_loader, test_loader, in_dim, meta

    else:
        raise ValueError("unknown task")
