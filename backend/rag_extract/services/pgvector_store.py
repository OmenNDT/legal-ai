"""
PostgreSQL Vector Store with pgvector extension.
Alternative to ChromaDB for production deployments.
"""
import os
import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)

Base = declarative_base()


class VectorDocument(Base):
    """Table storing document chunks as vectors."""
    __tablename__ = "vector_documents"

    id = Column(String(255), primary_key=True)
    doc_id = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    display_name = Column(String(500))
    standard_code = Column(String(100))
    section_id = Column(String(100))
    section_title = Column(String(500))
    chapter = Column(String(200))
    parent_sections = Column(String(1000))
    cross_references = Column(String(1000))
    heading_level = Column(Integer)
    content = Column(String(10000))  # Full text content
    embedding = Column(Vector(4096))  # Must match Ollama model dims
    metadata_json = Column(JSON)  # Extra metadata
    created_at = Column(DateTime, server_default=text("NOW()"))


class PostgresVectorStore:
    """
    PostgreSQL + pgvector vector store.
    
    Requires:
      - PostgreSQL 14+ with pgvector extension installed
      - CREATE EXTENSION IF NOT EXISTS vector;
    
    Environment:
      VECTOR_STORE_TYPE=postgres
      POSTGRES_URL=postgresql://user:pass@localhost:5432/rag_db
      POSTGRES_VECTOR_DIMS=4096
    """

    def __init__(self, connection_url: Optional[str] = None, dims: int = 4096):
        self.dims = dims
        self.url = connection_url or os.getenv(
            "POSTGRES_URL",
            "postgresql://rag_user:rag_pass@localhost:5432/rag_db"
        )
        self.engine = create_engine(self.url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self._init_extension()
        self._init_tables()
        logger.info(f"PostgresVectorStore initialized: {self.url.split('@')[-1]}")

    def _init_extension(self):
        """Ensure pgvector extension exists."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

    def _init_tables(self):
        """Create tables if not exist."""
        Base.metadata.create_all(self.engine)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding via Ollama (same as ChromaDB path)."""
        import httpx
        ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")
        try:
            resp = httpx.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add document chunks with embeddings."""
        session = self.Session()
        try:
            for i, doc_id in enumerate(ids):
                meta = metadatas[i]
                embedding = self._get_embedding(documents[i])
                vec = VectorDocument(
                    id=doc_id,
                    doc_id=meta.get("doc_id", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    display_name=meta.get("display_name", ""),
                    standard_code=meta.get("standard_code", ""),
                    section_id=meta.get("section_id", ""),
                    section_title=meta.get("section_title", ""),
                    chapter=meta.get("chapter", ""),
                    parent_sections=meta.get("parent_sections", ""),
                    cross_references=meta.get("cross_references", ""),
                    heading_level=meta.get("heading_level", 0),
                    content=documents[i][:9999],  # Truncate if too long
                    embedding=embedding,
                    metadata_json=meta,
                )
                session.merge(vec)  # Upsert by id
            session.commit()
            logger.info(f"Indexed {len(ids)} chunks to PostgreSQL")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add chunks: {e}")
            raise
        finally:
            session.close()

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Similarity search using cosine distance.
        Returns ChromaDB-compatible result format.
        """
        query_embedding = self._get_embedding(query_text)
        session = self.Session()
        try:
            # Build filter conditions
            filters = []
            if where:
                for key, val in where.items():
                    if key == "doc_id":
                        filters.append(f"doc_id = {val}")
                    elif key == "standard_code":
                        filters.append(f"standard_code = '{val}'")
            filter_sql = " AND ".join(filters) if filters else "TRUE"

            # Cosine similarity: 1 - cosine_distance
            sql = text(f"""
                SELECT
                    id,
                    doc_id,
                    display_name,
                    standard_code,
                    section_title,
                    chapter,
                    content,
                    metadata_json,
                    1 - (embedding <=> :query_vec) AS similarity
                FROM vector_documents
                WHERE {filter_sql}
                ORDER BY embedding <=> :query_vec
                LIMIT :limit
            """)
            results = session.execute(
                sql,
                {
                    "query_vec": str(query_embedding),
                    "limit": n_results,
                }
            ).fetchall()

            # Convert to ChromaDB-compatible format
            ids = []
            documents = []
            metadatas = []
            distances = []
            for row in results:
                ids.append(row.id)
                documents.append(row.content)
                metadatas.append(row.metadata_json or {})
                distances.append(1.0 - row.similarity)  # Convert back to distance

            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
            }
        finally:
            session.close()

    def delete_by_doc_id(self, doc_id: int) -> None:
        """Delete all chunks for a document."""
        session = self.Session()
        try:
            session.execute(
                text("DELETE FROM vector_documents WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            )
            session.commit()
            logger.info(f"Deleted vectors for doc_id={doc_id}")
        finally:
            session.close()

    def count(self) -> int:
        """Total vector count."""
        session = self.Session()
        try:
            result = session.execute(
                text("SELECT COUNT(*) FROM vector_documents")
            ).scalar()
            return result or 0
        finally:
            session.close()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return stats compatible with ChromaDB format."""
        session = self.Session()
        try:
            total = session.execute(
                text("SELECT COUNT(*) FROM vector_documents")
            ).scalar()
            doc_count = session.execute(
                text("SELECT COUNT(DISTINCT doc_id) FROM vector_documents")
            ).scalar()
            return {
                "total_chunks": total,
                "total_documents": doc_count,
                "store_type": "postgres",
            }
        finally:
            session.close()
