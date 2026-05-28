import yaml
from pathlib import Path
from pyspark.sql import SparkSession

class SparkSessionBuilder:

    def __init__(self, config_path: str | Path | None = None, mode: str = "auto"):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with Path(config_path).open() as f:
            self._config = yaml.safe_load(f)
        self._mode = mode.lower()
        self._session: SparkSession | None = None

    def build(self) -> SparkSession:
        if self._session is not None:
            return self._session

        if self._mode == "connect":
            self._session = self._build_connect()
        elif self._mode == "local":
            self._session = self._build_local()
        elif self._mode == "auto":
            self._session = self._build_auto()
        else:
            raise ValueError(f"Unknown mode '{self._mode}'. Choose 'connect', 'local', or 'auto'.")

        return self._session

    def stop(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    def __enter__(self) -> SparkSession:
        return self.build()

    def __exit__(self, *_) -> None:
        self.stop()

    def _build_connect(self) -> SparkSession:
        cfg = self._config["spark-connect"]
        url = f"sc://{cfg['host']}:{cfg['port']}"
        return SparkSession.builder.appName("logistics-pipeline").remote(url).getOrCreate()

    def _build_local(self) -> SparkSession:
        return (
            SparkSession.builder
            .master("local[*]")
            .appName("logistics-pipeline")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )

    def _build_auto(self) -> SparkSession:
        try:
            session = self._build_connect()
            session.sql("SELECT 1").collect()
            return session
        except Exception:
            return self._build_local()