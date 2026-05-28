from contextlib import contextmanager
from typing import Optional
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from .db_config import DBConfig

_pool: Optional[pool.ThreadedConnectionPool] = None

def init_pool(cfg: Optional[DBConfig] = None, minconn: int = 1, maxconn: int = 8) -> None:
    global _pool
    if _pool is not None:
        return
    cfg = cfg or DBConfig.from_env()
    _pool = pool.ThreadedConnectionPool(
        minconn, maxconn,
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.database,
        user=cfg.user,
        password=cfg.password,
    )

def _ensure_pool() -> pool.ThreadedConnectionPool:
    if _pool is None:
        init_pool()
    assert _pool is not None, "Connection pool failed to initialize"
    return _pool

@contextmanager
def get_conn():
    p = _ensure_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)

@contextmanager
def get_cursor(dict_rows: bool = True):
    with get_conn() as conn:
        cursor_factory = RealDictCursor if dict_rows else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
        finally:
            cur.close()
