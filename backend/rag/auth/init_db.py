from .db import get_conn
from .db_config import BASE_DIR

SCHEMA_FILE = BASE_DIR / "config" / "database" / "schema_auth.sql"

if __name__ == "__main__":
    sql = SCHEMA_FILE.read_text(encoding = "utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print(f"Applied schema from {SCHEMA_FILE}")