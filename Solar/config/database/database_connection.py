import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from psycopg2 import pool

# Always load .env from the repo root before reading env vars.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")


class DatabaseConnection:

    def __init__(self, config_path: Any = None, **overrides: Any):
        # `config_path` is accepted only for backwards compatibility with
        # callers that used to pass a YAML path. Config now comes from env.
        del config_path
        self.params = self._build_params(overrides)
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None

    @staticmethod
    def _build_params(overrides: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "legal_ai"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
        }
        params.update(overrides)
        return params

    def connect(self) -> None:
        if self.connection_pool is not None:
            return
        try:
            self.connection_pool = pool.SimpleConnectionPool(1, 10, **self.params)
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
        self.execute_query(
            f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
            tuple(data.values()),
            fetch=False,
        )

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> None:
        if not data_list:
            return
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(["%s"] * len(data_list[0]))
        self.execute_many(
            f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
            [tuple(d.values()) for d in data_list],
        )

    def insert_data(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        if isinstance(data, dict):
            self.insert_one(table_name, data)
        else:
            self.insert_many(table_name, data)

    def delete_where(self, table_name: str, conditions: Dict[str, Any]) -> None:
        if not conditions:
            raise ValueError("Conditions cannot be empty for delete operation")
        where_clause = " AND ".join([f"{k} = %s" for k in conditions])
        self.execute_query(
            f"DELETE FROM {table_name} WHERE {where_clause}",
            tuple(conditions.values()),
            fetch=False,
        )

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
