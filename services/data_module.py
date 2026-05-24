from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
import logging

from config.GetPath import paths
from utils.config import Config
from utils.transforms import SolarTransforms

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_CLASSES = set(Config.CLASS_NAMES)

class SolarPanelDataModule:

    # 70% train / 15% val / 15% test (stratified, original images only).
    # Train split uses on-the-fly augmentation via SolarTransforms.train.
    # Val/test always use eval transform (no augmentation).

    def __init__(self, data_dir: Path | None = None, img_size: int = 224, batch_size: int = 32, seed: int = 42):
        self.data_dir = data_dir or paths.data
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.tf = SolarTransforms(img_size)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.class_names: list[str] = []
        self.train_paths_by_class: dict[str, list[Path]] = {}
        logger.info("SolarPanelDataModule initialized | data_dir = %s | img_size = %d | batch_size = %d | seed = %d", self.data_dir, img_size, batch_size, seed)

    def setup(self) -> "SolarPanelDataModule":
        logger.info("--- setup() START ---")

        eval_ds = self._load_originals(self.tf.eval)
        train_ds = self._load_originals(self.tf.train)
        logger.info("Total original images: %d | Classes: %s", len(eval_ds), eval_ds.classes)

        train_idx, val_idx, test_idx = self._split(eval_ds.targets)
        logger.info("Split | train = %d | val = %d | test = %d", len(train_idx), len(val_idx), len(test_idx))

        self.class_names = eval_ds.classes
        self.train_paths_by_class = self._collect_train_paths(eval_ds, train_idx)

        self.train_dataset = Subset(train_ds, list(train_idx))
        self.val_dataset = Subset(eval_ds, list(val_idx))
        self.test_dataset = Subset(eval_ds, list(test_idx))

        self._print_stats(len(train_idx), len(val_idx), len(test_idx))
        logger.info("--- setup() DONE ---")
        return self

    def get_loaders(self, num_workers: int = 2) -> dict:
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise RuntimeError("Call setup() or setup_after_augment() before get_loaders()")
        logger.info("Creating DataLoaders | batch_size=%d | num_workers=%d", self.batch_size, num_workers)
        loaders = {
            "train": DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True,  num_workers = num_workers, pin_memory = True),
            "val": DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = num_workers, pin_memory = True),
            "test": DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)
        }
        logger.info("DataLoaders ready | train_batches = %d | val_batches = %d | test_batches = %d", len(loaders["train"]), len(loaders["val"]), len(loaders["test"]))
        return loaders

    def _load_originals(self, transform) -> datasets.ImageFolder:
        class FilteredImageFolder(datasets.ImageFolder):
            def find_classes(self, directory: str):  # type: ignore[override]
                classes = sorted(d.name for d in Path(directory).iterdir() if d.is_dir() and d.name in ALLOWED_CLASSES)
                if not classes:
                    raise FileNotFoundError(f"No allowed class dirs found under {directory}")
                class_to_idx = {cls: i for i, cls in enumerate(classes)}
                return classes, class_to_idx
        return FilteredImageFolder(
            str(self.data_dir),
            transform = transform,
            is_valid_file = lambda p: Path(p).suffix.lower() in IMG_EXTS
        )

    def _split(self, targets: list[int]) -> tuple[list[int], list[int], list[int]]:
        logger.debug("Splitting %d samples | seed = %d | 70/15/15", len(targets), self.seed)
        train_idx, temp_idx = train_test_split(
            range(len(targets)), test_size = 0.30,
            stratify = targets, random_state = self.seed
        )
        temp_targets = [targets[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size = 0.50,
            stratify = temp_targets, random_state = self.seed
        )
        return list(train_idx), list(val_idx), list(test_idx)

    def _collect_train_paths(self, ds: datasets.ImageFolder, train_idx: list[int]) -> dict[str, list[Path]]:
        logger.debug("Collecting train paths for %d samples...", len(train_idx))
        result: dict[str, list[Path]] = {cls: [] for cls in ds.classes}
        for i in train_idx:
            path, label = ds.samples[i]
            result[ds.classes[label]].append(Path(path))
        return result

    def _print_stats(self, n_train: int, n_val: int, n_test: int) -> None:
        logger.info("Dataset summary:")
        logger.info("Classes: %s", self.class_names)
        logger.info("Train: %d", n_train)
        logger.info("Validation: %d", n_val)
        logger.info("Test: %d", n_test)
