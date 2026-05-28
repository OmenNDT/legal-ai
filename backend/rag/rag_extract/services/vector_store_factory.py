"""
Vector Store Factory — ChromaDB (default) or PostgreSQL + pgvector.

Environment:
  VECTOR_STORE_TYPE=chromadb   # default
  VECTOR_STORE_TYPE=postgres   # use PostgreSQL + pgvector

When using postgres:
  POSTGRES_URL=postgresql://user:pass@localhost:5432/rag_db
  POSTGRES_VECTOR_DIMS=4096
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Lazy imports
_chromadb_available = False
try:
    import chromadb
    from chromadb.utils import embedding_functions
    _chromadb_available = True
except ImportError:
    pass

_pgvector_available = False
try:
    from backend.rag_extract.services.pgvector_store import PostgresVectorStore
    _pgvector_available = True
except ImportError as e:
    logger.debug(f"pgvector_store not available: {e}")
    PostgresVectorStore = None


class VectorStoreInterface:
    """Abstract interface for vector stores (ChromaDB-compatible API)."""

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def delete_by_doc_id(self, doc_id: int) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def get_collection_stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_all(self, include: List[str] = None) -> Dict[str, Any]:
        """Get all items (for 3D visualization). Optional for ChromaDB compat."""
        raise NotImplementedError


class ChromaDBWrapper(VectorStoreInterface):
    """Wraps ChromaDB collection to match VectorStoreInterface."""

    def __init__(self, collection):
        self._collection = collection

    def add_chunks(self, ids, documents, metadatas):
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self._collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def query(self, query_text, n_results=5, where=None):
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    def delete_by_doc_id(self, doc_id: int):
        self._collection.delete(where={"doc_id": doc_id})

    def count(self):
        return self._collection.count()

    def get_collection_stats(self):
        return {
            "total_chunks": self._collection.count(),
            "store_type": "chromadb",
        }

    def get_all(self, include=None):
        include = include or ["embeddings", "metadatas", "documents"]
        return self._collection.get(include=include)


def get_vector_store() -> VectorStoreInterface:
    """
    Factory: returns ChromaDB or PostgreSQL vector store based on env.
    """
    store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()

    if store_type == "postgres":
        if not _pgvector_available:
            raise RuntimeError(
                "PostgreSQL vector store requested but pgvector dependencies not installed. "
                "Install: pip install psycopg2-binary pgvector sqlalchemy"
            )
        url = os.getenv("POSTGRES_URL", "postgresql://rag_user:rag_pass@localhost:5432/rag_db")
        dims = int(os.getenv("POSTGRES_VECTOR_DIMS", "4096"))
        logger.info(f"Using PostgreSQL vector store: {url.split('@')[-1]}, dims={dims}")
        return PostgresVectorStore(connection_url=url, dims=dims)

    # Default: ChromaDB
    if not _chromadb_available:
        raise RuntimeError(
            "ChromaDB not installed. Install: pip install chromadb"
        )

    from pathlib import Path
    chroma_dir = Path(os.getenv("CHROMA_DIR", "./data/chroma"))
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")
    ef = embedding_functions.OllamaEmbeddingFunction(
        url=f"{ollama_url}/api/embeddings",
        model_name=model,
    )
    collection = client.get_or_create_collection(
        name="rag_documents_v2",
        embedding_function=ef,
        metadata={"description": "RAG Documents (Qwen3-embedding)"},
    )
    logger.info(f"Using ChromaDB vector store at {chroma_dir}")
    return ChromaDBWrapper(collection)
