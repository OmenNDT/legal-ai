from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass, field
from services.data_module import SolarPanelDataModule
import random
import logging
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

@dataclass
class AugmentConfig:
    target_count: int = 1500
    rotation_range: float = 15.0
    brightness_range: tuple[float, float] = (0.6, 1.6)
    sharpen_range: tuple[float, float] = (1.5, 3.0)
    blur_range: tuple[float, float] = (0.5, 1.5)
    seed: int = 42
    target_classes: list[str] = field(default_factory=lambda: ["Dusty", "Snow"])

class ImageAugmenter:

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        logger.debug("ImageAugmenter initialized | seed = %d | rotation = ±%.1f° | brightness = %s | sharpen = %s | blur = %s", cfg.seed, cfg.rotation_range, cfg.brightness_range, cfg.sharpen_range, cfg.blur_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self._rotate(img)
        img = self._brightness(img)
        img = self._sharpness(img)
        return img

    def _rotate(self, img: Image.Image) -> Image.Image:
        angle = self.rng.uniform(-self.cfg.rotation_range, self.cfg.rotation_range)
        logger.debug("Rotate: %.2f°", angle)
        return img.rotate(angle, expand = False)

    def _brightness(self, img: Image.Image) -> Image.Image:
        factor = self.rng.uniform(*self.cfg.brightness_range)
        logger.debug("Brightness factor: %.2f", factor)
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img: Image.Image) -> Image.Image:
        effect = self.rng.choice(["sharpen", "blur", "none"])
        if effect == "sharpen":
            factor = self.rng.uniform(*self.cfg.sharpen_range)
            logger.debug("Sharpness effect: sharpen | factor = %.2f", factor)
            return ImageEnhance.Sharpness(img).enhance(factor)
        if effect == "blur":
            radius = self.rng.uniform(*self.cfg.blur_range)
            logger.debug("Sharpness effect: blur | radius = %.2f", radius)
            return img.filter(ImageFilter.GaussianBlur(radius = radius))
        logger.debug("Sharpness effect: None")
        return img

class OfflineAugmentationPipeline:

    def __init__(self, train_paths_by_class: dict[str, list[Path]], cfg: AugmentConfig):
        self.train_paths_by_class = train_paths_by_class
        self.cfg = cfg
        self.augmenter = ImageAugmenter(cfg)
        logger.info("OfflineAugmentationPipeline ready | target_classes = %s | target_count = %d", cfg.target_classes, cfg.target_count)

    def run(self) -> None:
        logger.info("=== Offline Augmentation Pipeline START ===")
        for class_name in self.cfg.target_classes:
            train_paths = self.train_paths_by_class.get(class_name, [])
            if not train_paths:
                logger.warning("[%s] No train images found, skipping!", class_name)
                continue
            self._augment_class(class_name, train_paths)
        logger.info("=== Offline Augmentation Pipeline DONE ===")

    def _augment_class(self, class_name: str, train_paths: list[Path]) -> None:
        out_dir = train_paths[0].parent
        current = len([p for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        needed = max(0, self.cfg.target_count - current)
        logger.info("[%s] Output dir: %s", class_name, out_dir)
        logger.info("[%s] Original train images: %d | Total in dir: %d | Target: %d | Need to create: %d", class_name, len(train_paths), current, self.cfg.target_count, needed)

        if needed == 0:
            logger.info("[%s] Already at target count (%d), skipping!", class_name, current)
            return

        created = self._generate(train_paths, out_dir, needed)
        logger.info("[%s] Created %d augmented images. New total: %d!", class_name, created, current + created)

    def _generate(self, sources: list[Path], out_dir: Path, needed: int) -> int:
        created = 0
        cycle = 0
        logger.info("Starting generation: %d images needed from %d source(s)!", needed, len(sources))
        while created < needed:
            logger.debug("Cycle %d — %d/%d images created so far!", cycle, created, needed)
            for src in sources:
                if created >= needed:
                    break
                logger.debug("Augmenting source: %s", src.name)
                img = Image.open(src).convert("RGB")
                aug = self.augmenter(img)
                out_path = out_dir / f"aug_{cycle}_{src.stem}.jpg"
                aug.save(out_path, "JPEG", quality = 95)
                created = created + 1
                logger.debug("Saved: %s (%d/%d)", out_path.name, created, needed)
            cycle = cycle + 1
        return created

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    dm = SolarPanelDataModule()
    dm.setup()
    cfg = AugmentConfig()
    pipeline = OfflineAugmentationPipeline(dm.train_paths_by_class, cfg)
    pipeline.run()