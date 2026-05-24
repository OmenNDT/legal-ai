import random
import numpy as np
import torch
from pathlib import Path

class Config:

    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 3
    CLASS_NAMES = ["Clean", "Dusty", "Snow"]
    NUM_EPOCHS_PHASE1 = 10
    NUM_EPOCHS_PHASE2 = 10
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-4
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    LLRD_BASE_LR = 5e-5
    LLRD_DECAY = 0.1
    LLRD_EPOCHS = 15

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_class_counts(data_dir: Path) -> list[int]:
        return [
            len(list((data_dir / cls).iterdir()))
            for cls in Config.CLASS_NAMES
        ]

    @classmethod
    def seed_everything(cls) -> None:
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)

    @classmethod
    def summary(cls) -> None:
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"Classes: {cls.CLASS_NAMES}")
        print(f"Img size: {cls.IMG_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
