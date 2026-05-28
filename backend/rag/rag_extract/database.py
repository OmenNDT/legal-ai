"""RAG Extract Database (integrated into legal-ai)."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.rag_extract.config import DATABASE_URL


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_wal(dbapi_connection, connection_record):
    """Enable WAL mode for concurrent reads."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables on startup."""
    import backend.rag_extract.models.rag_document  # noqa: F401
    Base.metadata.create_all(bind=engine)
