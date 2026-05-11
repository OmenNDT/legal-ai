from __future__ import annotations
from pathlib import Path

class ProjectPaths:
    _ROOT_MARKERS: tuple[str, ...] = ("PLAN.md", "README.md")
    _instance: "ProjectPaths | None" = None

    def __new__(cls) -> "ProjectPaths":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._project_root = self._find_project_root()
        self._initialized = True

    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in [current, *current.parents]:
            has_marker = any((parent / m).exists() for m in self._ROOT_MARKERS)
            has_source = (parent / "source_code").exists()
            if has_marker and has_source:
                return parent
        raise RuntimeError(f"Cannot find the project root. Markers: {self._ROOT_MARKERS}")

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def docs(self) -> Path:
        return self._project_root / "docs"

    @property
    def slides(self) -> Path:
        return self._project_root / "slides"

    @property
    def references(self) -> Path:
        return self._project_root / "references"

    @property
    def source_code(self) -> Path:
        return self._project_root / "source_code"

    @property
    def config_dir(self) -> Path:
        return self.source_code / "config"

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def jdbc_drivers(self) -> Path:
        return self.config_dir / "driver"

    @property
    def data_raw(self) -> Path:
        return self.source_code / "data_raw"

    @property
    def data_processed(self) -> Path:
        return self.source_code / "data_processed"

    @property
    def data_features(self) -> Path:
        return self.source_code / "data_features"

    @property
    def notebooks(self) -> Path:
        return self.source_code / "notebooks"

    @property
    def src(self) -> Path:
        return self.source_code / "src"

    @property
    def models(self) -> Path:
        return self.source_code / "models"

    @property
    def results(self) -> Path:
        return self.source_code / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def metrics(self) -> Path:
        return self.results / "metrics"

    def raw_csv(self, table_name: str) -> Path:
        return self.data_raw / f"{table_name}.csv"

    def processed_parquet(self, table_name: str) -> Path:
        return self.data_processed / f"{table_name}.parquet"

    def feature_parquet(self, split: str) -> Path:
        return self.data_features / f"{split}.parquet"

    def model_file(self, model_name: str, ext: str = "pkl") -> Path:
        return self.models / f"{model_name}.{ext}"

    def metrics_file(self, model_name: str, ext: str = "json") -> Path:
        return self.metrics / f"{model_name}.{ext}"

    def jdbc_jar(self, jar_name: str) -> Path:
        return self.jdbc_drivers / jar_name

    def ensure_dirs(self) -> None:
        for path in (
            self.data_processed,
            self.data_features,
            self.models,
            self.results,
            self.figures,
            self.metrics,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ProjectPaths(root='{self._project_root}')"

PATHS = ProjectPaths()