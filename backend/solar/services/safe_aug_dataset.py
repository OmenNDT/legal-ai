"""
SafeAugDataset (OOP) — wraps a base PIL dataset and applies augmentations one
step at a time. After each step the AugmentationValidator compares the partial
result against the original; rejected steps are skipped (state stays on the
last accepted candidate). Final image goes through Resize + Normalize.
Classes:
    AugStep — a single augmentation step (name + callable + probability).
    AugStepPipeline — builds the canonical Solar train pipeline as discrete steps.
    AugStats — accumulates accept/reject counts per step.
    SafeAugDataset — Dataset facade that ties everything together.
"""
from __future__ import annotations
import random
from typing import Callable
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.aug_validator import AugmentationValidator
from utils.transforms import SolarTransforms

class AugStep:

    def __init__(self, name: str, apply: Callable[[Image.Image], Image.Image], prob: float = 1.0):
        self.name = name
        self.apply = apply
        self.prob = prob

    def should_run(self) -> bool:
        if self.prob >= 1.0:
            return True
        return random.random() < self.prob

class AugStepPipeline:

    # Canonical Solar augmentation pipeline expressed as discrete validated steps.

    def __init__(self, img_size: int, aug = None):
        self.img_size = img_size
        self.aug = aug or SolarTransforms(img_size).aug
        self.steps = self._build_steps()

    def _build_steps(self) -> list[AugStep]:
        sz = self.img_size
        a = self.aug
        return [
            AugStep("resize_pad", lambda im: transforms.functional.resize(im, [sz + a.crop_padding, sz + a.crop_padding]), prob = 1.0),
            AugStep("random_crop", transforms.RandomCrop(sz), prob = 1.0),
            AugStep("hflip", transforms.RandomHorizontalFlip(p = 1.0), prob = a.hflip_p),
            AugStep("vflip", transforms.RandomVerticalFlip(p = 1.0), prob = a.vflip_p),
            AugStep("rotation", transforms.RandomRotation(degrees = a.rotation), prob = 1.0),
            AugStep("color_jitter", transforms.ColorJitter(brightness = a.brightness, contrast = a.contrast, saturation = a.saturation, hue = a.hue), prob = 1.0),
            AugStep("grayscale", transforms.Grayscale(num_output_channels = 3), prob = a.grayscale_p)
        ]

    def __iter__(self):
        return iter(self.steps)

class AugStats:

    def __init__(self):
        self.attempts: dict[str, int] = {}
        self.rejects:  dict[str, int] = {}

    def record(self, step_name: str, accepted: bool) -> None:
        self.attempts[step_name] = self.attempts.get(step_name, 0) + 1
        if not accepted:
            self.rejects[step_name] = self.rejects.get(step_name, 0) + 1

    def reject_rate(self, step_name: str) -> float:
        attempts = self.attempts.get(step_name, 0)
        if attempts == 0:
            return 0.0
        return self.rejects.get(step_name, 0) / attempts

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "attempts": self.attempts[name],
                "rejects":  self.rejects.get(name, 0),
                "reject_rate": self.reject_rate(name),
            }
            for name in self.attempts
        }

class TensorNormalizer:

    # Wraps ToTensor + Normalize using project-wide mean/std + tensor RandomErasing.

    def __init__(self, img_size: int, erasing_p: float):
        self.img_size = img_size
        self._to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = SolarTransforms.MEAN, std = SolarTransforms.STD),
        ])
        self._random_erasing = transforms.RandomErasing(p = erasing_p)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != (self.img_size, self.img_size):
            img = transforms.functional.resize(img, [self.img_size, self.img_size])
        tensor = self._to_tensor_norm(img)
        return self._random_erasing(tensor)


class SafeAugDataset(Dataset):

    # Validates each augmentation step against the original image fingerprint.

    def __init__(self, base_dataset: Dataset, img_size: int, threshold: float = AugmentationValidator.DEFAULT_THRESHOLD, aug = None, stats: AugStats | None = None):
        self.base = base_dataset
        self.img_size = img_size
        self.threshold = threshold
        self.pipeline = AugStepPipeline(img_size, aug)
        self.stats = stats or AugStats()
        self.validator = AugmentationValidator(threshold = threshold)
        erasing_p = (aug.erasing_p if aug is not None else SolarTransforms(img_size).aug.erasing_p)
        self.normalizer = TensorNormalizer(img_size, erasing_p)

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        img, label = self._load_pil(index)
        self.validator.set_baseline(img)

        current = img
        for step in self.pipeline:
            if not step.should_run():
                continue
            candidate = step.apply(current)
            result = self.validator.validate(candidate)
            self.stats.record(step.name, result.accepted)
            if result.accepted:
                current = candidate

        tensor = self.normalizer(current)
        return tensor, label

    def _load_pil(self, index: int) -> tuple[Image.Image, int]:
        img, label = self.base[index]
        if isinstance(img, torch.Tensor):
            img = transforms.functional.to_pil_image(img)
        return img.convert("RGB"), int(label)
