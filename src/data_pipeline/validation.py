from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    total_chunks: int
    chunks_with_embeddings: int
    orphaned_chunks: int
    chroma_id_mismatches: int
    passed: bool
    errors: list = field(default_factory=list)

    @property
    def embedding_coverage(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return round(self.chunks_with_embeddings / self.total_chunks, 4)

class PipelineValidator:
    def __init__(self, db, vector_store):
        self._db = db
        self._vector_store = vector_store

    def validate(self) -> ValidationResult:
        total, with_embeddings = self._check_embedding_coverage()
        orphaned = self._check_orphaned_chunks()
        mismatches, errors = self._check_chroma_consistency()

        passed = (
            with_embeddings == total
            and orphaned == 0
            and mismatches == 0
        )
        return ValidationResult(
            total_chunks=total,
            chunks_with_embeddings=with_embeddings,
            orphaned_chunks=orphaned,
            chroma_id_mismatches=mismatches,
            passed=passed,
            errors=errors,
        )

    def _check_embedding_coverage(self) -> tuple:
        rows = self._db.execute_query(
            "SELECT COUNT(*), COUNT(embedding_id) FROM legal_chunks"
        )
        if rows:
            return int(rows[0][0]), int(rows[0][1])
        return 0, 0

    def _check_orphaned_chunks(self) -> int:
        rows = self._db.execute_query(
            "SELECT COUNT(*) FROM legal_chunks lc "
            "LEFT JOIN document_versions dv ON lc.version_id = dv.id "
            "WHERE dv.id IS NULL"
        )
        return int(rows[0][0]) if rows else 0

    def _check_chroma_consistency(self) -> tuple:
        rows = self._db.execute_query(
            "SELECT id FROM legal_chunks WHERE embedding_id IS NOT NULL"
        )
        if not rows:
            return 0, []

        pg_ids = {str(r[0]) for r in rows}
        try:
            chroma_result = self._vector_store._collection.get(ids=list(pg_ids), include=[])
            chroma_ids = set(chroma_result.get("ids", []))
        except Exception as e:
            return len(pg_ids), [f"ChromaDB query failed: {e}"]

        missing = pg_ids - chroma_ids
        errors = [f"chunk_id {cid} missing in ChromaDB" for cid in sorted(missing)[:20]]
        return len(missing), errors
