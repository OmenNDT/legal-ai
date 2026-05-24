import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)

from config.GetPath import paths
from utils.config import Config
from utils.transforms import SolarTransforms

class Evaluator:

    def __init__(self, device = Config.DEVICE, class_names: list[str] = Config.CLASS_NAMES):
        self.device = device
        self.class_names = class_names

    def predict(self, model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc = "Predicting", leave = False):
                outputs = model(inputs.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average = "macro", zero_division = 0),
            "recall": recall_score(y_true, y_pred, average = "macro", zero_division = 0),
            "f1": f1_score(y_true, y_pred, average = "macro", zero_division = 0)
        }

    def full_report(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        labels = list(range(len(self.class_names)))
        cm = confusion_matrix(y_true, y_pred, labels = labels).tolist()
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class = "ovr", average = "macro", labels = labels)
        except ValueError:
            auc = None
        per_class_f1 = np.asarray(f1_score(y_true, y_pred, labels = labels, average = None, zero_division = 0)).tolist()
        per_class_precision = np.asarray(precision_score(y_true, y_pred, labels = labels, average = None, zero_division = 0)).tolist()
        per_class_recall = np.asarray(recall_score(y_true, y_pred, labels = labels, average = None, zero_division = 0)).tolist()
        return {
            **self.compute_metrics(y_true, y_pred),
            "auc": float(auc) if auc is not None else None,
            "confusion_matrix": cm,
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "class_names": list(self.class_names),
            "n_samples": int(len(y_true))
        }

    def benchmark_inference_ms(self, model: torch.nn.Module, img_size: int = 224, n_warmup: int = 5, n_runs: int = 30) -> float:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device = self.device)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        return (elapsed / n_runs) * 1000.0

    def print_report(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, split: str = "Val") -> dict:
        m = self.compute_metrics(y_true, y_pred)
        print(
            f"\n[{model_name.upper()}] {split} — "
            f"Accuracy:{m['accuracy']:.4f} | "
            f"Precision:{m['precision']:.4f} | "
            f"Recall:{m['recall']:.4f} | "
            f"F1:{m['f1']:.4f}"
        )
        present = sorted(set(y_true) | set(y_pred))
        target_names = [self.class_names[i] for i in present] if max(present) < len(self.class_names) else None
        print(classification_report(y_true, y_pred, labels=present, target_names=target_names, zero_division=0))
        return m

class TTAEvaluator:
    
    def __init__(self, data_dir = None, test_indices: list[int] | None = None, device = Config.DEVICE, class_names: list[str] = Config.CLASS_NAMES, batch_size: int = Config.BATCH_SIZE, img_size: int = Config.IMG_SIZE):
        self.data_dir = data_dir or paths.data
        self.test_indices = test_indices or []
        self.device = device
        self.class_names = class_names
        self.batch_size = batch_size
        self.tta_tfs = SolarTransforms.tta_transforms(img_size)

    def predict(self, model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs: np.ndarray | None = None
        y_true: np.ndarray | None = None

        for i, tf in enumerate(self.tta_tfs):
            print(f"TTA pass {i+1}/{len(self.tta_tfs)}")
            subset = Subset(datasets.ImageFolder(str(self.data_dir), transform = tf), self.test_indices)
            loader = DataLoader(subset, batch_size = self.batch_size, shuffle = False, num_workers = 2)
            probs_i, labels_i = [], []
            with torch.no_grad():
                for inputs, labels in loader:
                    out = model(inputs.to(self.device))
                    probs_i.extend(torch.softmax(out, 1).cpu().numpy())
                    if i == 0:
                        labels_i.extend(labels.numpy())
            probs_arr = np.array(probs_i)
            all_probs = probs_arr if all_probs is None else all_probs + probs_arr
            if i == 0:
                y_true = np.array(labels_i)
        assert all_probs is not None and y_true is not None
        y_pred = np.argmax(all_probs / len(self.tta_tfs), axis = 1)
        return y_true, y_pred
