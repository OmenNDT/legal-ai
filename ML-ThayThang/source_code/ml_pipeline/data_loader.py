from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path
from typing import Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

TABLES = [
    "drivers",
    "trucks",
    "trailers",
    "customers",
    "facilities",
    "routes",
    "loads",
    "trips",
    "fuel_purchases",
    "maintenance_records",
    "delivery_events",
    "safety_incidents",
    "driver_monthly_metrics",
    "truck_utilization_metrics"
]

class PostgresLoader:
    def __init__(self, config: dict[str, Any], connect_timeout: int = 5) -> None:
        self._host = config["host"]
        self._port = config["port"]
        self._user = config["username"]
        self._password = config["password"]
        self._dbname = config["database_name"]
        self._connect_timeout = connect_timeout
        self._engine: Engine | None = None

    def connect(self) -> None:
        logger.info(
            "Connecting to PostgreSQL %s:%s db = %s user = %s (timeout = %ds)",
            self._host, self._port, self._dbname, self._user, self._connect_timeout,
        )
        url = (
            f"postgresql+psycopg2://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._dbname}"
        )
        self._engine = create_engine(url, connect_args = {"connect_timeout": self._connect_timeout})
        with self._engine.connect():
            pass
        logger.info("Connection established.")

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Connection closed.")

    def load_table(self, table_name: str) -> pd.DataFrame:
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table_name}")
        logger.info("Loading table: %s", table_name)
        query = text(f'SELECT * FROM "{table_name}"')
        with self._engine.connect() as conn:
            df = pd.read_sql(query, conn)
        logger.info("-> %d rows, %d cols", df.shape[0], df.shape[1])
        return df

    def load_all_tables(self) -> dict[str, pd.DataFrame]:
        logger.info("Loading all %d tables from PostgreSQL...", len(TABLES))
        tables: dict[str, pd.DataFrame] = {}
        for name in TABLES:
            tables[name] = self.load_table(name)
        logger.info("All tables loaded.")
        return tables

    def __enter__(self) -> "PostgresLoader":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

class CsvLoader:
    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def load_table(self, table_name: str) -> pd.DataFrame:
        path = self._data_dir / f"{table_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path)
        logger.info("-> %d rows, %d cols", df.shape[0], df.shape[1])
        return df

    def load_all_tables(self) -> dict[str, pd.DataFrame]:
        logger.info("Loading all %d tables from CSV directory %s...", len(TABLES), self._data_dir)
        tables: dict[str, pd.DataFrame] = {}
        for name in TABLES:
            tables[name] = self.load_table(name)
        logger.info("All tables loaded from CSV.")
        return tables

