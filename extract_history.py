"""
Reconstruct <model>_history.json from training log when trainer.save()
only persisted phase-1 history (the AdvancedTrainer phase-2 history was lost).

Parses lines like:
  ========== Model: EFFICIENTNETB4 ==========
  [EFFICIENTNETB4] Phase 1 — Head Warm-up — 10 epochs
  [01/10] Train 0.7891/0.5934  Val 0.4567/0.7812  38.7s *
  [EFFICIENTNETB4] LLRD Fine-tune — 15 epochs
  [01/15] Train 0.9784/0.5796  Val 0.8169/0.7987  37.0s *

Output: writes <name>_history.json with concatenated train/val acc/loss.

Usage:
  python3 extract_history.py /path/to/train.log
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

from config.GetPath import paths

EPOCH_RE = re.compile(r"\[(\d+)/(\d+)\]\s+Train\s+([\d.]+)/([\d.]+)\s+Val\s+([\d.]+)/([\d.]+)")
MODEL_RE = re.compile(r"=+\s*Model:\s*(\w+)\s*=+")

KEY_MAP = {
    "EFFICIENTNETB4": "efficientnetb4",
    "RESNET50":       "resnet50",
    "VIT":            "vit",
}


def parse(log_text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    current = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # split log into lines after normalising \r (tqdm carriage returns)
    lines = log_text.replace("\r", "\n").splitlines()
    for line in lines:
        m_model = MODEL_RE.search(line)
        if m_model:
            if current and any(history[k] for k in history):
                out[current] = {k: v[:] for k, v in history.items()}
            current = KEY_MAP.get(m_model.group(1).upper())
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            continue
        m_ep = EPOCH_RE.search(line)
        if m_ep and current:
            tl, ta, vl, va = (float(m_ep.group(i)) for i in (3, 4, 5, 6))
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
    if current and any(history[k] for k in history):
        out[current] = history
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: extract_history.py <training_log>")
        return 1
    log_path = Path(sys.argv[1])
    text = log_path.read_text(errors = "replace")
    histories = parse(text)
    if not histories:
        print("No epochs parsed.")
        return 1
    for key, h in histories.items():
        n = len(h["train_acc"])
        out_path = paths.models / f"{key}_history.json"
        with open(out_path, "w") as f:
            json.dump(h, f)
        final_train = h["train_acc"][-1] if h["train_acc"] else 0
        final_val   = h["val_acc"][-1]   if h["val_acc"]   else 0
        print(f"[{key}] {n} epochs · final train_acc={final_train:.4f} · val_acc={final_val:.4f} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
