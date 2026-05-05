import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, collection_name: str, embedding_model: str, host: str = "localhost", port: int = 8001):
        self._embedding_model_name = embedding_model
        self._encoder = None
        try:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB: {e}")
            self._client = None
            self._collection = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model_name)
        return self._encoder

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def add_chunks(self, chunks: list) -> list:
        if not chunks or self._collection is None:
            return []
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self._get_encoder().encode(texts, show_progress_bar=False).tolist()
        ids = [str(c["chunk_id"]) for c in chunks]
        metadatas = [
            {
                "chunk_id": int(c["chunk_id"]),
                "document_title": c.get("document_title", ""),
                "document_type": c.get("document_type", ""),
                "article_number": c.get("article_number", ""),
                "clause_number": c.get("clause_number", ""),
                "point_number": c.get("point_number", ""),
                "effective_status": c.get("effective_status", "active"),
                "version_id": int(c.get("version_id", 0)),
                "section_id": int(c.get("section_id", 0)),
            }
            for c in chunks
        ]
        self._collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        return ids

    def search(self, query: str, n_results: int = 20, where: dict = None) -> list:
        if self._collection is None:
            return []
        embedding = self._get_encoder().encode([query], show_progress_bar=False).tolist()[0]
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        output = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({
                "chunk_id": meta.get("chunk_id", 0),
                "chunk_text": doc,
                "content": doc,
                "vector_score": round(1.0 - float(dist), 4),
                **meta,
            })
        return output

    def delete_collection(self):
        if self._client is not None:
            self._client.delete_collection(self._collection.name)
