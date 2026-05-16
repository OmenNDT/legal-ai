from pathlib import Path
from utils.config import Config

class ProjectPaths:

    def __init__(self, base_dir: str | None = None):
        self.base = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.data = self.base / "data"
        self.models = self.base / "models"
        self.services = self.base / "services"
        self.utils = self.base / "utils"
        self.config = self.base / "config"
        self.demo = self.base / "DemoDoAn"

    @property
    def data_clean(self) -> Path:
        return self.data / "Clean"

    @property
    def data_dusty(self) -> Path:
        return self.data / "Dusty"

    @property
    def data_snow(self) -> Path:
        return self.data / "Snow"

    def get_class_dir(self, class_name: str) -> Path:
        mapping = {
            Config.CLASS_NAMES[0]: self.data_clean,
            Config.CLASS_NAMES[1]: self.data_dusty,
            Config.CLASS_NAMES[2]: self.data_snow
        }
        if class_name not in mapping:
            raise ValueError(f"Unknown class '{class_name}'. Choose from {Config.CLASS_NAMES}")
        return mapping[class_name]

    @property
    def demo_data(self) -> Path:
        return self.demo / "data"

    @property
    def demo_models(self) -> Path:
        return self.demo / "models"

    @property
    def demo_static(self) -> Path:
        return self.demo / "static"

    @property
    def demo_uploads(self) -> Path:
        return self.demo_static / "uploads"

    @property
    def demo_templates(self) -> Path:
        return self.demo / "templates"

    def all_paths(self) -> dict[str, Path]:
        return {
            "base": self.base,
            "data": self.data,
            "data_clean": self.data_clean,
            "data_dusty": self.data_dusty,
            "data_snow": self.data_snow,
            "models": self.models,
            "services": self.services,
            "utils": self.utils,
            "config": self.config,
            "demo": self.demo,
            "demo_data": self.demo_data,
            "demo_models": self.demo_models,
            "demo_static": self.demo_static,
            "demo_uploads": self.demo_uploads,
            "demo_templates": self.demo_templates
        }

    def check(self) -> None:
        for name, path in self.all_paths().items():
            status = "OK" if path.exists() else "MISSING"
            print(f"[{status}] {name}: {path}")

    def __repr__(self) -> str:
        return f"ProjectPaths(base = '{self.base}')"

paths = ProjectPaths()

if __name__ == "__main__":
    paths.check()