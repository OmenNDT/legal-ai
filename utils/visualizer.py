import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from utils.transforms import SolarTransforms
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class EDAVisualizer:

    COLORS = ["#4CAF50", "#FF9800", "#2196F3"]

    def __init__(self, data_dir: Path, class_names: list[str]):
        self.data_dir = data_dir
        self.class_names = class_names

    def _count_per_class(self) -> dict[str, int]:
        return {
            cls: sum(
                1 for p in (self.data_dir / cls).iterdir()
                if p.suffix.lower() in IMG_EXTS and not p.stem.startswith("aug_")
            )
            for cls in self.class_names
        }

    def plot_distribution(self) -> None:
        counts = self._count_per_class()
        total = sum(counts.values())

        print("Data distribution:")
        for cls, cnt in counts.items():
            print(f"{cls:<8}: {cnt:>5} ({cnt/total*100:.1f}%)")
        print(f"{'TOTAL':<8}: {total:>5}")

        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(counts.keys(), counts.values(), color = self.COLORS)
        axes[0].set_title("Number of images per class")
        axes[0].set_ylabel("Number of images")
        for i, cnt in enumerate(counts.values()):
            axes[0].text(i, cnt + 10, str(cnt), ha = "center", fontweight = "bold")

        axes[1].pie(
            counts.values(), labels=counts.keys(),
            colors=self.COLORS, autopct="%.1f%%", startangle=90,
        )
        axes[1].set_title("Proportion of distribution")
        plt.tight_layout()
        plt.show()

    def show_samples(self, n_per_class: int = 4) -> None:
        _, axes = plt.subplots(len(self.class_names), n_per_class, figsize = (n_per_class * 3.5, len(self.class_names) * 3))
        for row, cls in enumerate(self.class_names):
            cls_dir = self.data_dir / cls
            files = random.sample(list(cls_dir.iterdir()), n_per_class)
            for col, fpath in enumerate(files):
                img = Image.open(fpath).resize((224, 224))
                axes[row, col].imshow(img)
                axes[row, col].set_title(cls, fontsize = 9)
                axes[row, col].axis("off")
        plt.suptitle("Image Sample per class", fontsize = 14, fontweight = "bold")
        plt.tight_layout()
        plt.show()

class AugmentationVisualizer:

    # Compare original versus augmented images side by side

    def __init__(self, tf: SolarTransforms, class_names: list[str]):
        self.tf = tf
        self.class_names = class_names

    def show(self, full_eval_ds, full_train_ds, n: int = 8) -> None:
        indices = random.sample(range(len(full_eval_ds)), n)
        _, axes = plt.subplots(2, n, figsize = (n * 2, 5))
        for col, idx in enumerate(indices):
            for row, ds in enumerate([full_eval_ds, full_train_ds]):
                t, lbl = ds[idx]
                img = self.tf.inverse(t).permute(1, 2, 0).clamp(0, 1).numpy()
                axes[row, col].imshow(img)
                axes[row, col].set_title(self.class_names[lbl], fontsize = 8)
                axes[row, col].axis("off")
        axes[0, 0].set_ylabel("Original", fontsize = 10)
        axes[1, 0].set_ylabel("Augmented", fontsize = 10)
        plt.suptitle("Augmentation preview", fontsize = 13, fontweight = "bold")
        plt.tight_layout()
        plt.show()

class PredictionVisualizer:

    # Show some correct and wrong predictions with confidence scores
    # Green title for correct, red for wrong

    def __init__(self, class_names: list[str], device, tf: SolarTransforms):
        self.class_names = class_names
        self.device = device
        self.tf = tf

    def _to_img(self, t):
        return self.tf.inverse(t).permute(1, 2, 0).clamp(0, 1).numpy()

    def show(self, model, loader, n_correct: int = 8, n_wrong: int = 8) -> None:
        
        model.eval()
        correct_items, wrong_items = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs.to(self.device))
                confs, preds = torch.max(torch.softmax(outputs, 1), 1)
                for img, lbl, pred, conf in zip(inputs, labels, preds.cpu(), confs.cpu()):
                    item = (img, lbl.item(), pred.item(), conf.item())
                    if pred == lbl and len(correct_items) < n_correct:
                        correct_items.append(item)
                    elif pred != lbl and len(wrong_items) < n_wrong:
                        wrong_items.append(item)
                if len(correct_items) >= n_correct and len(wrong_items) >= n_wrong:
                    break

        for title, items in [("Correct", correct_items), ("Wrong", wrong_items)]:
            if not items:
                continue
            cols = min(len(items), 8)
            rows = (len(items) + cols - 1) // cols
            _, axes = plt.subplots(rows, cols, figsize = (cols * 2.2, rows * 2.8))
            axes = np.array(axes).flatten()
            for ax, (img, true, pred, conf) in zip(axes, items):
                ax.imshow(self._to_img(img))
                color = "green" if pred == true else "red"
                ax.set_title(
                    f"T:{self.class_names[true]}\nP:{self.class_names[pred]} {conf:.0%}",
                    fontsize = 8, color = color
                )
                ax.axis("off")
            for ax in axes[len(items):]:
                ax.axis("off")
            plt.suptitle(f"{title} predictions", fontsize = 13, fontweight = "bold")
            plt.tight_layout()
            plt.show()

class ComparisonVisualizer:

    # Compare multiple models' metrics side by side in a bar chart

    def plot(self, df_cmp) -> None:
        _, ax = plt.subplots(figsize = (10, 5))
        x = np.arange(len(df_cmp))
        w = 0.2
        cols = ["Accuracy", "Precision", "Recall", "F1"]
        clrs = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

        for i, (col, clr) in enumerate(zip(cols, clrs)):
            bars = ax.bar(x + i * w, df_cmp[col].astype(float), w, label = col, color = clr, alpha = 0.85)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}",
                    ha = "center", va = "bottom", fontsize = 7
                )

        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(df_cmp["Model"])
        ax.set_ylim(0.7, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparing — Test Set")
        ax.legend(loc = "lower right")
        ax.grid(axis = "y", alpha = 0.3)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_name: str, y_true, y_pred, class_names: list[str], split: str = "Val") -> None:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis = 1, keepdims = True)
        _, axes = plt.subplots(1, 2, figsize = (12, 4))
        for ax, data, fmt, title in zip(axes, [cm, cm_norm], ["d", ".2f"], ["Count", "Normalized"]):
            sns.heatmap(data, annot = True, fmt = fmt, cmap = "Blues", ax = ax, xticklabels = class_names, yticklabels = class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name.upper()} — {title} ({split})")
        plt.tight_layout()
        plt.show()