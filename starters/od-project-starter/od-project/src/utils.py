from pathlib import Path
import yaml, json, random, numpy as np, torch

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
