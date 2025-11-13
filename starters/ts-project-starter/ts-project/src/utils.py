import json, random
from pathlib import Path
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std if self.mean is not None else x
    def inverse(self, x: np.ndarray):
        return x * self.std + self.mean if self.mean is not None else x
    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
    @staticmethod
    def from_dict(d):
        s = StandardScaler()
        s.mean = np.array(d["mean"], dtype=np.float32)
        s.std  = np.array(d["std"], dtype=np.float32)
        return s

def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred)**2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # R^2 with protection against zero variance
    var = np.var(y_true)
    r2 = float(1.0 - mse / var) if var > 1e-12 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}
