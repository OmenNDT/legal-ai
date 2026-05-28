"""
RAG Query Service (Refactored)
Chunks documents using section-aware hierarchical splitting,
embeds with Ollama, stores in vector store (ChromaDB or PostgreSQL),
and answers questions using Claude Haiku 4.5 with cross-reference enrichment.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from backend.rag_extract.models.rag_document import RagDocument
from backend.rag_extract.services.section_aware_chunker import chunk_by_sections
from backend.rag_extract.services.vector_store_factory import get_vector_store, VectorStoreInterface

logger = logging.getLogger(__name__)

# Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

# Lazy import lightrag to avoid import errors when not installed
_haiku_llm_func = None

def _get_haiku_llm():
    global _haiku_llm_func
    if _haiku_llm_func is None:
        from backend.rag_extract.services.lightrag_config import haiku_llm_func
        _haiku_llm_func = haiku_llm_func
    return _haiku_llm_func


class RagQueryService:
    """Handles document indexing and RAG queries."""

    def __init__(self):
        self._store: VectorStoreInterface = get_vector_store()
        self._system_prompt = (
            "Bạn là chuyên gia phân tích tài liệu kỹ thuật công nghiệp (ASME, API, ASTM, ASM). "
            "Luôn trả lời bằng tiếng Việt. Giữ nguyên thuật ngữ tiếng Anh chuyên ngành khi cần thiết."
        )

    def index_document(self, doc: RagDocument) -> int:
        """
        Chunk and index a single document into vector store using section-aware splitting.
        Returns number of chunks indexed.
        """
        if not doc.processed_content:
            logger.warning(f"Document {doc.id} has no processed content, skipping")
            return 0

        # Remove old chunks for this document
        self._store.delete_by_doc_id(doc.id)

        # Section-aware chunking (preserves hierarchy + cross-refs)
        section_chunks = chunk_by_sections(doc.processed_content)
        if not section_chunks:
            return 0

        # Prepare data for vector store with rich metadata
        ids = [f"doc{doc.id}_chunk{c.chunk_index}" for c in section_chunks]
        documents = [c.text for c in section_chunks]
        metadatas = [
            {
                "doc_id": doc.id,
                "display_name": doc.display_name,
                "standard_code": doc.standard_code or "",
                "chunk_index": c.chunk_index,
                "section_id": c.section_id,
                "section_title": c.section_title,
                "chapter": c.chapter,
                "parent_sections": ",".join(c.parent_sections),
                "cross_references": ",".join(c.cross_references),
                "heading_level": c.heading_level,
            }
            for c in section_chunks
        ]

        # Add to store in batches
        self._store.add_chunks(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(
            f"Indexed doc {doc.id} ({doc.display_name}): "
            f"{len(section_chunks)} section-aware chunks"
        )
        return len(section_chunks)

    def _has_chunks(self, doc_id: int) -> bool:
        """Check if a document already has chunks in vector store (fast lookup)."""
        try:
            result = self._store.query(
                query_text="",  # Empty query for existence check
                n_results=1,
                where={"doc_id": doc_id},
            )
            return len(result.get("ids", [[]])[0]) > 0
        except Exception:
            return False

    def index_all(self, db: Session, force: bool = False) -> dict:
        """Index ready documents. Only indexes new docs unless force=True."""
        docs = db.query(RagDocument).filter(RagDocument.status == "ready").all()
        total_chunks = 0
        indexed_docs = 0
        skipped_docs = 0

        for doc in docs:
            if not force and self._has_chunks(doc.id):
                skipped_docs += 1
                continue
            count = self.index_document(doc)
            if count > 0:
                total_chunks += count
                indexed_docs += 1

        stats = self._store.get_collection_stats()
        return {
            "indexed_documents": indexed_docs,
            "skipped_documents": skipped_docs,
            "total_chunks": total_chunks,
            "collection_size": stats.get("total_chunks", 0),
        }

    async def query(self, question: str, n_results: int = 5) -> dict:
        """
        RAG query with cross-reference enrichment.

        1. Semantic search for top chunks
        2. Pull cross-referenced sections from matched chunks
        3. Build enriched context → ask Haiku 4.5

        Returns:
            {answer, sources: [{text, doc_name, section_id, relevance}]}
        """
        store_count = self._store.count()
        if store_count == 0:
            return {
                "answer": "No documents indexed yet. Please index documents first.",
                "sources": [],
            }

        # Semantic search
        results = self._store.query(
            query_text=question,
            n_results=min(n_results, store_count),
        )

        # Build sources from top results
        sources = []
        context_parts = []
        seen_ids = set()
        xref_targets = set()

        for doc_text, distance, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            chunk_id = f"doc{metadata.get('doc_id')}_chunk{metadata.get('chunk_index')}"
            seen_ids.add(chunk_id)
            relevance = max(0, (1 - distance) * 100)

            section_id = metadata.get("section_id", "")
            section_label = f"§{section_id}" if section_id else ""

            sources.append({
                "text": doc_text[:300],
                "doc_name": metadata.get("display_name", "Unknown"),
                "standard_code": metadata.get("standard_code", ""),
                "section_id": section_id,
                "relevance": round(relevance, 1),
            })

            header = f"[{metadata.get('display_name', '')} {section_label}]".strip()
            context_parts.append(f"{header}\n{doc_text}")

            # Collect cross-references to fetch
            xrefs = metadata.get("cross_references", "")
            if xrefs:
                xref_targets.update(xrefs.split(","))

        # Enrich: pull cross-referenced sections
        xref_context = self._fetch_cross_references(xref_targets, seen_ids)
        if xref_context:
            context_parts.append("## Referenced Sections:\n" + xref_context)

        context = "\n\n---\n\n".join(context_parts)

        # Ask Haiku 4.5 with enriched context
        prompt = f"""Dựa trên các trích đoạn tài liệu kỹ thuật bên dưới, hãy trả lời câu hỏi.
Nếu không tìm thấy câu trả lời trong các trích đoạn, hãy nói rõ.
Các mục tham chiếu chéo (cross-reference) được bao gồm để bổ sung ngữ cảnh.
Trả lời bằng tiếng Việt. Giữ nguyên thuật ngữ tiếng Anh chuyên ngành.

## Trích đoạn tài liệu:
{context}

## Câu hỏi:
{question}

## Trả lời:"""

        try:
            haiku = _get_haiku_llm()
            answer = await haiku(prompt, system_prompt=self._system_prompt)
        except Exception as e:
            logger.error(f"Haiku query failed: {e}")
            answer = "LLM không khả dụng. Các trích đoạn liên quan:\n\n" + "\n\n".join(
                f"- [{s['doc_name']} §{s['section_id']}] ({s['relevance']}%): {s['text']}"
                for s in sources
            )

        return {"answer": answer, "sources": sources}

    def _fetch_cross_references(self, xref_targets: set, seen_ids: set) -> str:
        """Fetch chunks matching cross-referenced section IDs."""
        if not xref_targets:
            return ""

        xref_parts = []
        for ref in sorted(xref_targets):
            if not ref.strip():
                continue
            try:
                matches = self._store.query(
                    query_text="",  # Empty query, filter by section_id
                    n_results=2,
                    where={"section_id": ref.strip()},
                )
                docs = matches.get("documents", [[]])[0]
                metas = matches.get("metadatas", [[]])[0]
                if docs:
                    for doc_text, meta in zip(docs, metas):
                        chunk_id = f"doc{meta.get('doc_id')}_chunk{meta.get('chunk_index')}"
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            label = f"[§{ref} - {meta.get('display_name', '')}]"
                            xref_parts.append(f"{label}\n{doc_text[:600]}")
            except Exception:
                continue

        return "\n\n".join(xref_parts)

    def get_stats(self) -> dict:
        """Get indexing stats."""
        stats = self._store.get_collection_stats()
        return {
            "collection_name": "rag_documents_v2",
            "total_chunks": stats.get("total_chunks", 0),
            "embedding_model": f"ollama/{OLLAMA_EMBEDDING_MODEL}",
            "chunking_method": "section-aware-hierarchical",
            "store_type": stats.get("store_type", "unknown"),
        }

    def get_vectors_3d(self) -> dict:
        """Get all embeddings projected to 3D via PCA for visualization."""
        count = self._store.count()
        if count == 0:
            return {"traces": [], "total_chunks": 0, "dimensions": 0}

        try:
            data = self._store.get_all(include=["embeddings", "metadatas", "documents"])
        except NotImplementedError:
            logger.warning("Vector store does not support get_all (PostgreSQL). Returning empty.")
            return {"traces": [], "total_chunks": count, "dimensions": 0, "note": "PostgreSQL store does not support 3D visualization"}

        embeddings = data["embeddings"]
        metadatas = data["metadatas"]
        documents = data["documents"]

        import numpy as np
        emb_array = np.array(embeddings)
        n_dims = emb_array.shape[1]

        # PCA: center → SVD → project to 3D
        mean = emb_array.mean(axis=0)
        centered = emb_array - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:3].T
        total_var = (S ** 2).sum()
        variance = [(s ** 2 / total_var * 100) for s in S[:3]]

        # Group by document
        doc_groups = {}
        for i, (meta, doc_text) in enumerate(zip(metadatas, documents)):
            doc_id = meta.get("doc_id", 0)
            name = meta.get("display_name", "Unknown")
            code = meta.get("standard_code", "")
            key = f"{name} ({code})" if code else name

            if key not in doc_groups:
                doc_groups[key] = {"x": [], "y": [], "z": [], "text": [], "doc_id": doc_id}

            doc_groups[key]["x"].append(round(float(projected[i, 0]), 4))
            doc_groups[key]["y"].append(round(float(projected[i, 1]), 4))
            doc_groups[key]["z"].append(round(float(projected[i, 2]), 4))

            section = meta.get("section_id", "")
            label = f"§{section}" if section else f"chunk{meta.get('chunk_index', i)}"
            preview = doc_text[:120].replace("\n", " ")
            doc_groups[key]["text"].append(f"<b>{label}</b><br>{preview}...")

        return {
            "traces": [
                {"name": name, "doc_id": grp["doc_id"], "x": grp["x"], "y": grp["y"], "z": grp["z"], "text": grp["text"]}
                for name, grp in doc_groups.items()
            ],
            "total_chunks": count,
            "dimensions": n_dims,
            "variance": [round(v, 1) for v in variance],
        }
