from dataclasses import dataclass, field
from src.pipeline.reasoning import StructuredAnswer

@dataclass
class FinalResponse:
    answer: str
    citations: list
    confidence: float
    intent: str
    domain: str
    entities: list
    legal_basis: str
    conclusion: str
    recommendation: str

class PostProcessing:
    CONFIDENCE_WEIGHTS = {
        "intent": 0.2,
        "retrieval": 0.3,
        "completeness": 0.5,
    }

    def __init__(self, db=None):
        self._db = db

    def process(self, answer: StructuredAnswer, candidates: list, structured_query) -> FinalResponse:
        verified = self._check_hallucination(answer, candidates)
        citations = self._inject_citations(verified, candidates)
        formatted = self._format_answer(verified, citations)
        confidence = self._compute_confidence(answer, candidates, structured_query)

        response = FinalResponse(
            answer=formatted,
            citations=citations,
            confidence=confidence,
            intent=structured_query.intent,
            domain=structured_query.domain,
            entities=structured_query.entities,
            legal_basis=answer.legal_basis,
            conclusion=answer.conclusion,
            recommendation=answer.recommendation,
        )

        if self._db and structured_query.query_id:
            self._persist(response, answer, structured_query.query_id, citations)

        return response

    def _check_hallucination(self, answer: StructuredAnswer, candidates: list) -> StructuredAnswer:
        if not candidates:
            return answer
        context_text = " ".join(c.content for c in candidates[:10]).lower()
        answer_words = set(answer.raw_answer.lower().split())
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        if overlap < 0.15:
            answer.raw_answer += "\n\n⚠️ Lưu ý: Câu trả lời có thể chứa thông tin không có trong văn bản pháp lý được cung cấp."
        return answer

    def _inject_citations(self, answer: StructuredAnswer, candidates: list) -> list:
        citations = []
        basis_text = (answer.legal_basis + " " + answer.analysis).lower()
        for chunk in candidates[:10]:
            if not chunk.document_title:
                continue
            title_words = set(chunk.document_title.lower().split())
            if title_words & set(basis_text.split()):
                citations.append({
                    "chunk_id": chunk.chunk_id,
                    "document_title": chunk.document_title,
                    "article": chunk.article_number,
                    "clause": chunk.clause_number,
                    "relevance_score": chunk.hybrid_score,
                    "excerpt": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                })
        return citations[:5]

    def _format_answer(self, answer: StructuredAnswer, citations: list) -> str:
        parts = []
        if answer.legal_basis:
            parts.append(f"**Căn cứ pháp lý:**\n{answer.legal_basis}")
        if answer.analysis:
            parts.append(f"**Phân tích:**\n{answer.analysis}")
        if answer.conclusion:
            parts.append(f"**Kết luận:**\n{answer.conclusion}")
        if answer.recommendation:
            parts.append(f"**Khuyến nghị:**\n{answer.recommendation}")
        if not parts:
            parts.append(answer.raw_answer)
        if citations:
            cite_lines = []
            for i, c in enumerate(citations):
                ref = c["document_title"]
                if c["article"]:
                    ref += f" - Điều {c['article']}"
                if c["clause"]:
                    ref += f", Khoản {c['clause']}"
                cite_lines.append(f"[{i+1}] {ref}")
            parts.append("**Tài liệu tham khảo:**\n" + "\n".join(cite_lines))
        return "\n\n".join(parts)

    def _compute_confidence(self, answer: StructuredAnswer, candidates: list, structured_query) -> float:
        intent_score = structured_query.intent_confidence
        retrieval_score = min(len(candidates) / 10.0, 1.0)
        completeness_score = (
            bool(answer.legal_basis) * 0.3 +
            bool(answer.analysis) * 0.3 +
            bool(answer.conclusion) * 0.2 +
            bool(answer.recommendation) * 0.2
        )
        confidence = (
            self.CONFIDENCE_WEIGHTS["intent"] * intent_score +
            self.CONFIDENCE_WEIGHTS["retrieval"] * retrieval_score +
            self.CONFIDENCE_WEIGHTS["completeness"] * completeness_score
        )
        return round(min(confidence, 1.0), 3)

    def _persist(self, response: FinalResponse, answer: StructuredAnswer, query_id: int, citations: list):
        try:
            rows = self._db.execute_query(
                "INSERT INTO legal_reasoning_results "
                "(query_id, facts, legal_basis, analysis, conclusion, recommendation, confidence_score) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (query_id, answer.facts, answer.legal_basis, answer.analysis,
                 answer.conclusion, answer.recommendation, response.confidence),
            )
            if rows and citations:
                reasoning_id = rows[0][0]
                for c in citations:
                    self._db.execute_query(
                        "INSERT INTO citations (reasoning_id, chunk_id, relevance_score) VALUES (%s, %s, %s)",
                        (reasoning_id, c["chunk_id"], c["relevance_score"]),
                        fetch=False,
                    )
        except Exception:
            pass
