import json, random, re
from pathlib import Path
import numpy as np
import torch
import yaml

TOKEN_RE = re.compile(r"\w+")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())
