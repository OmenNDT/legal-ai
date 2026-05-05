import os
import yaml
import psycopg2
from psycopg2 import pool
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if "postgresql" not in config:
            raise ValueError("PostgreSQL configuration not found in config.yaml")
        return config["postgresql"]

    def _resolve_env(self, value: str) -> str:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            inner = value[2:-1]
            if ":" in inner:
                var_name, default = inner.split(":", 1)
                return os.getenv(var_name, default)
            return os.getenv(inner, "")
        return str(value) if value is not None else ""

    def _build_connection_params(self) -> Dict[str, Any]:
        db_config = self.config
        url = self._resolve_env(db_config.get("url", ""))
        if url.startswith("jdbc:postgresql://"):
            url_parts = url.replace("jdbc:postgresql://", "").split("/")
            host_port = url_parts[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            db_name = url_parts[1] if len(url_parts) > 1 else ""
        else:
            host = self._resolve_env(db_config.get("host", "localhost"))
            port = int(self._resolve_env(str(db_config.get("port", 5432))))
            db_name = self._resolve_env(db_config.get("db_name", ""))
        return {
            "host": host,
            "port": port,
            "user": self._resolve_env(db_config.get("username", "")),
            "password": self._resolve_env(db_config.get("password", "")),
            "database": db_name,
        }

    def connect(self) -> None:
        if self.connection_pool is not None:
            return
        params = self._build_connection_params()
        try:
            self.connection_pool = pool.SimpleConnectionPool(1, 10, **params)
        except Exception as e:
            raise ConnectionError(f"Failed to create connection pool: {e}")

    def disconnect(self) -> None:
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None

    @contextmanager
    def get_connection(self):
        if self.connection_pool is None:
            raise ConnectionError("Database not connected. Call connect() first.")
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = True) -> Optional[List[Tuple]]:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query, params or ())
                    result = cursor.fetchall() if fetch else None
                    conn.commit()
                    return result
                except Exception as e:
                    conn.rollback()
                    raise e

    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.executemany(query, params_list)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise e

    def insert_one(self, table_name: str, data: Dict[str, Any]) -> None:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        self.execute_query(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", tuple(data.values()), fetch=False)

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> None:
        if not data_list:
            return
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(["%s"] * len(data_list[0]))
        self.execute_many(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", [tuple(d.values()) for d in data_list])

    def insert_data(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        if isinstance(data, dict):
            self.insert_one(table_name, data)
        else:
            self.insert_many(table_name, data)

    def delete_where(self, table_name: str, conditions: Dict[str, Any]) -> None:
        if not conditions:
            raise ValueError("Conditions cannot be empty for delete operation")
        where_clause = " AND ".join([f"{k} = %s" for k in conditions])
        self.execute_query(f"DELETE FROM {table_name} WHERE {where_clause}", tuple(conditions.values()), fetch=False)

    def delete_data(self, table_name: str, conditions: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        if isinstance(conditions, dict):
            self.delete_where(table_name, conditions)
        else:
            for c in conditions:
                self.delete_where(table_name, c)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
