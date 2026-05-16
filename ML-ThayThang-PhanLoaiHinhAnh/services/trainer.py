import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.GetPath import paths
from utils.config import Config

class Trainer:
    
    # 2-Phase train with Early Stopping and class-weighted loss

    def __init__(self, model: nn.Module, model_name: str, loaders: dict, class_counts: list[int], device = Config.DEVICE, models_dir = None, patience: int = 5):
        self.model = model
        self.model_name = model_name
        self.loaders = loaders
        self.device = device
        self.models_dir = models_dir or paths.models
        self.patience = patience
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        w = torch.tensor(1.0 / np.array(class_counts, dtype = float), dtype = torch.float32)
        w = (w / w.sum() * len(class_counts)).to(device)
        self.criterion = nn.CrossEntropyLoss(weight = w)

    def _run_epoch(self, phase: str, optimizer) -> tuple[float, float]:
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()
        running_loss = correct = total = 0

        with torch.set_grad_enabled(is_train):
            for inputs, labels in tqdm(self.loaders[phase], desc = f"{phase}", leave = False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if is_train:
                    optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()
                running_loss = running_loss + loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct = correct + (preds == labels).sum().item()
                total = total + labels.size(0)

        return running_loss / total, correct / total

    def _fit(self, num_epochs: int, optimizer, scheduler, phase_label: str) -> float:
        best_val_acc = 0.0
        best_wts = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        print(f"\n{'='*60}")
        print(f"[{self.model_name.upper()}] {phase_label} — {num_epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            tl, ta = self._run_epoch("train", optimizer)
            vl, va = self._run_epoch("val", optimizer)
            scheduler.step()

            for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"], [tl, ta, vl, va]):
                self.history[k].append(v)

            star = " *" if va > best_val_acc else ""
            print(
                f"[{epoch:02d}/{num_epochs}] "
                f"Train {tl:.4f}/{ta:.4f}  Val {vl:.4f}/{va:.4f}  "
                f"{time.time()-t0:.1f}s{star}"
            )

            if va > best_val_acc:
                best_val_acc = va
                best_wts = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve = no_improve + 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(best_wts)
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        return best_val_acc

    def train(self, num_epochs: int, lr: float, phase_label: str = "Training") -> float:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = lr, weight_decay = 1e-4
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min = lr * 0.01)
        return self._fit(num_epochs, optimizer, scheduler, phase_label)

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Unfreeze all — Trainable params: {trainable:,}")

    def save(self, suffix: str = "") -> str:
        path = self.models_dir / f"{self.model_name}{suffix}_best.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved -> {path}")
        return str(path)

    def plot_history(self) -> None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        for ax, metric in zip(axes, ["loss", "acc"]):
            ax.plot(epochs, self.history[f"train_{metric}"], "b-o", label = "Train", ms = 4)
            ax.plot(epochs, self.history[f"val_{metric}"], "r-o", label = "Val", ms = 4)
            ax.set_title(f"{self.model_name.upper()} — {metric.capitalize()}")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha = 0.3)
        plt.tight_layout()
        plt.show()

class AdvancedTrainer(Trainer):

    # Advanced Training: Label Smoothing + Mixup + Layer-wise LR Decay (LLRD).

    HEAD_NAMES = ("classifier", "fc", "heads")

    def __init__(self, model: nn.Module, model_name: str, loaders: dict, class_counts: list[int], device = Config.DEVICE, models_dir = None, patience: int = 5, label_smoothing: float = 0.1, mixup_alpha: float = 0.2):
        super().__init__(model, model_name, loaders, class_counts, device, models_dir, patience)
        self.mixup_alpha = mixup_alpha
        w = torch.tensor(1.0 / np.array(class_counts, dtype = float), dtype = torch.float32)
        w = (w / w.sum() * len(class_counts)).to(device)
        self.criterion = nn.CrossEntropyLoss(weight = w, label_smoothing = label_smoothing)
        print(f"Label Smoothing = {label_smoothing} | Mixup Alpha = {mixup_alpha}")

    def _mixup_batch(self, inputs, labels):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx = torch.randperm(inputs.size(0)).to(inputs.device)
        return lam * inputs + (1 - lam) * inputs[idx], labels, labels[idx], lam

    def _run_epoch(self, phase: str, optimizer) -> tuple[float, float]:
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()
        running_loss = correct = total = 0

        with torch.set_grad_enabled(is_train):
            for inputs, labels in tqdm(self.loaders[phase], desc = f"{phase}", leave = False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if is_train and self.mixup_alpha > 0:
                    inputs, la, lb, lam = self._mixup_batch(inputs, labels)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = lam * self.criterion(outputs, la) + (1 - lam) * self.criterion(outputs, lb)
                    loss.backward()
                    optimizer.step()
                    ref_labels = la
                else:
                    if is_train:
                        optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    ref_labels = labels

                running_loss = running_loss + loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct = correct + (preds == ref_labels).sum().item()
                total = total + inputs.size(0)

        return running_loss / total, correct / total

    def train_with_llrd(self, num_epochs: int, base_lr: float, decay_factor: float = 0.1) -> float:
        backbone_p, head_p = [], []
        for name, p in self.model.named_parameters():
            (head_p if any(h in name for h in self.HEAD_NAMES) else backbone_p).append(p)
        optimizer = optim.AdamW([
            {"params": backbone_p, "lr": base_lr * decay_factor},
            {"params": head_p, "lr": base_lr}
        ], weight_decay = 1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min = base_lr * 0.001)
        print(f"\n{'='*60}")
        print(f"[{self.model_name.upper()}] LLRD + Mixup + Label Smooth")
        print(f"Backbone LR: {base_lr*decay_factor:.2e} | Head LR: {base_lr:.2e}")
        print(f"{'='*60}")
        return self._fit(num_epochs, optimizer, scheduler, "LLRD Fine-tune")