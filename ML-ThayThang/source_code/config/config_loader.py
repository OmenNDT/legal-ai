from __future__ import annotations
from typing import Any
import yaml
from .paths import PATHS

class ProjectConfig:
    _instance: "ProjectConfig | None" = None

    def __new__(cls) -> "ProjectConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._data: dict[str, Any] = self._load()
        self._initialized = True

    def _load(self) -> dict[str, Any]:
        config_file = PATHS.config_file
        if not config_file.exists():
            raise FileNotFoundError(f"Cannot find the config file: {config_file}")
        with open(config_file, "r", encoding = "utf-8") as f:
            return yaml.safe_load(f) or {}

    def reload(self) -> None:
        self._data = self._load()

    @property
    def database(self) -> dict[str, Any]:
        return self._data.get("database", {})

    @property
    def spark_connect(self) -> dict[str, Any]:
        return self._data.get("spark-connect", {})

    @property
    def jdbc_url(self) -> str:
        db = self.database
        return f"jdbc:{db.get('type')}://{db.get('host')}:{db.get('port')}/{db.get('database_name')}"

    @property
    def spark_connect_url(self) -> str:
        sc = self.spark_connect
        return f"sc://{sc.get('host')}:{sc.get('port')}"

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"ProjectConfig(keys={list(self._data.keys())})"

CONFIG = ProjectConfig()