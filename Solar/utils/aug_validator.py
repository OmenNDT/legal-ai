"""
Augmentation feature-similarity gate (OOP).
ImageFingerprint — extract compact visual feature (RGB histogram + Sobel edge density).
FeatureComparator — cosine similarity between two feature vectors.
AugmentationValidator — facade: build a baseline once, then validate each augmented candidate; reject if similarity drops below threshold.
Compares augmented image against its original to prevent transforms that destroy
class-defining features (e.g. a crop that removes a snowy region).
"""
from __future__ import annotations
import numpy as np
from PIL import Image

class SobelOperator:

    KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float32)
    KERNEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float32)

    @classmethod
    def magnitude(cls, gray: np.ndarray) -> np.ndarray:
        gx = cls._conv2d_valid(gray, cls.KERNEL_X)
        gy = cls._conv2d_valid(gray, cls.KERNEL_Y)
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _conv2d_valid(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k = kernel
        return (
            k[0, 0] * arr[0:-2, 0:-2] + k[0, 1] * arr[0:-2, 1:-1] + k[0, 2] * arr[0:-2, 2:] +
            k[1, 0] * arr[1:-1, 0:-2] + k[1, 1] * arr[1:-1, 1:-1] + k[1, 2] * arr[1:-1, 2:] +
            k[2, 0] * arr[2:, 0:-2] + k[2, 1] * arr[2:, 1:-1] + k[2, 2] * arr[2:, 2:]
        )

class ImageFingerprint:

    # 96-dim RGB histogram + 16-dim Sobel edge histogram = 112-dim fingerprint.

    def __init__(self, hist_bins: int = 32, edge_bins: int = 16, downsample_size: int = 96):
        self.hist_bins = hist_bins
        self.edge_bins = edge_bins
        self.downsample_size = downsample_size

    def extract(self, img: Image.Image) -> np.ndarray:
        arr = self._to_normalized_array(img)
        color_hist = self._color_histogram(arr)
        edge_hist  = self._edge_histogram(arr)
        return np.concatenate([color_hist, edge_hist])

    def _to_normalized_array(self, img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize(
            (self.downsample_size, self.downsample_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(small, dtype = np.float32) / 255.0

    def _color_histogram(self, arr: np.ndarray) -> np.ndarray:
        parts = []
        for channel in range(3):
            counts, _ = np.histogram(arr[..., channel], bins = self.hist_bins, range = (0.0, 1.0))
            parts.append(counts.astype(np.float32))
        hist = np.concatenate(parts)
        return self._normalize(hist)

    def _edge_histogram(self, arr: np.ndarray) -> np.ndarray:
        gray = arr.mean(axis = 2)
        magnitude = SobelOperator.magnitude(gray)
        counts, _ = np.histogram(magnitude, bins = self.edge_bins, range = (0.0, 2.0))
        return self._normalize(counts.astype(np.float32))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        return vec / (vec.sum() + 1e-8)

class FeatureComparator:

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

class ValidationResult:

    def __init__(self, accepted: bool, similarity: float, threshold: float):
        self.accepted = accepted
        self.similarity = similarity
        self.threshold = threshold

    def __bool__(self) -> bool:
        return self.accepted

    def __repr__(self) -> str:
        verdict = "ACCEPT" if self.accepted else "REJECT"
        return f"<ValidationResult {verdict} sim={self.similarity:.3f} thr={self.threshold:.2f}>"

class AugmentationValidator:

    # Facade: cache a baseline fingerprint, then validate candidates.

    DEFAULT_THRESHOLD = 0.85

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, fingerprint: ImageFingerprint | None = None, comparator: FeatureComparator | None = None):
        self.threshold = threshold
        self.fingerprint = fingerprint or ImageFingerprint()
        self.comparator  = comparator or FeatureComparator()
        self._baseline: np.ndarray | None = None

    def set_baseline(self, img: Image.Image) -> np.ndarray:
        self._baseline = self.fingerprint.extract(img)
        return self._baseline

    def validate(self, candidate: Image.Image) -> ValidationResult:
        if self._baseline is None:
            raise RuntimeError("Call set_baseline() before validate().")
        candidate_feat = self.fingerprint.extract(candidate)
        sim = self.comparator.cosine(self._baseline, candidate_feat)
        return ValidationResult(accepted = sim >= self.threshold, similarity = sim, threshold = self.threshold)
