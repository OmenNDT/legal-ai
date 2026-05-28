import json
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except ImportError:
        pass

def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f, indent = 2, ensure_ascii = False, default = str)

def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)
