import json
import pickle
from pathlib import Path
from typing import Any

# Tiện ích đọc ghi JSON
class JsonIO:
    @staticmethod
    def read(path: Path):
        with open(path, "r", encoding = "utf-8") as f:
            return json.load(f)

    @staticmethod
    def write(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(data, f, ensure_ascii = False, indent = 2)

# Tiện ích đọc ghi pickle (cache nhị phân)
class PickleIO:
    @staticmethod
    def read(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def write(path: Path, obj: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
