from dataclasses import dataclass
from torchvision import transforms

@dataclass
class AugConfig:
    crop_padding: int = 32
    hflip_p: float = 0.5
    vflip_p: float = 0.2
    rotation: int = 20
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.1
    grayscale_p: float = 0.05
    erasing_p: float = 0.1

class SolarTransforms:

    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD  = [0.229, 0.224, 0.225]  # ImageNet std

    def __init__(self, img_size: int = 224, aug: AugConfig | None = None):
        self.img_size = img_size
        self.aug = aug or AugConfig()
        self._build()

    def _build(self) -> None:
        sz = self.img_size
        a = self.aug
        self.train = transforms.Compose([
            transforms.Resize((sz + a.crop_padding, sz + a.crop_padding)),
            transforms.RandomCrop(sz),
            transforms.RandomHorizontalFlip(p = a.hflip_p),
            transforms.RandomVerticalFlip(p = a.vflip_p),
            transforms.RandomRotation(degrees = a.rotation),
            transforms.ColorJitter(brightness = a.brightness, contrast = a.contrast, saturation = a.saturation, hue = a.hue),
            transforms.RandomGrayscale(p = a.grayscale_p),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.MEAN, std = self.STD),
            transforms.RandomErasing(p = a.erasing_p)
        ])

        self.eval = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        self.inverse = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.MEAN, self.STD)],
            std=[1 / s for s in self.STD]
        )

    @classmethod
    def tta_transforms(cls, img_size: int = 224) -> list:
        sz = img_size
        return [
            transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(), transforms.Normalize(mean = cls.MEAN, std = cls.STD)]),
            transforms.Compose([transforms.Resize((sz, sz)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean = cls.MEAN, std = cls.STD)]),
            transforms.Compose([transforms.Resize((sz + 32, sz + 32)), transforms.CenterCrop(sz), transforms.ToTensor(), transforms.Normalize(mean = cls.MEAN, std = cls.STD)]),
            transforms.Compose([transforms.Resize((sz, sz)), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize(mean = cls.MEAN, std = cls.STD)]),
            transforms.Compose([transforms.Resize((sz, sz)), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(mean = cls.MEAN, std = cls.STD)])
        ]