"""
Sinh bộ câu hỏi - câu trả lời (Q&A) cho Luật Kế Toán 2015.

Chiến lược:
  - Với chunk loại `dinh_nghia` → sinh câu hỏi "X là gì?"
  - Với chunk loại `nghia_vu` → sinh câu hỏi "Ai/đơn vị nào phải ...?"
  - Với chunk loại `quyen_loi` → sinh câu hỏi "Được quyền ... không?"
  - Với chunk loại `cam_ket` → sinh câu hỏi "Có được phép ... không?"
  - Với chunk loại `che_tai` → sinh câu hỏi "Vi phạm ... bị xử lý thế nào?"
  - Với chunk loại `thu_tuc` → sinh câu hỏi "Thủ tục/hồ sơ ... gồm những gì?"
  - Với chunk loại `to_chuc` → sinh câu hỏi "Bộ máy kế toán ... được tổ chức ra sao?"
  - Với chunk loại `bao_cao` → sinh câu hỏi "Báo cáo ... phải lập như thế nào?"
  - Fallback (dạng khác) → sinh từ tên Điều

Q&A được lưu vào bảng `qa_data` với nhãn `source = 'ke_toan_generated'`.

Cách dùng:
    python -m chatbot.training.ke_toan_qa_generator
    # hoặc trong code:
    from chatbot.training.ke_toan_qa_generator import KeToaQaGenerator
    gen = KeToaQaGenerator(cfg)
    gen.run(doc_code = "LKT2015")
"""

import re
import psycopg2
from psycopg2.extras import execute_values
from dataclasses import dataclass
from ..data_pipeline.db_loader import DbConfig

DOC_CODE = "LKT2015"
SOURCE_LABEL = "ke_toan_generated"

def _clean(text: str) -> str:

    # Bỏ ký tự thừa, rút gọn về 1 dòng.
    return re.sub(r"\s+", " ", text).strip()

def _extract_subject(dieu: str) -> str:

    # Lấy phần nội dung Điều sau 'Điều N.' làm subject.
    m = re.match(r"Điều\s+\d+[\.\s]+(.+)", dieu, re.IGNORECASE)
    return m.group(1).strip().rstrip(".") if m else dieu

def _make_question(concept_type: str, dieu: str, khoan: str, content: str) -> str:
    subject = _extract_subject(dieu)
    ref = f"{dieu} khoản {khoan}" if khoan else dieu

    if concept_type == "dinh_nghia":

        # Tìm từ được định nghĩa: "X là ..."
        m = re.search(r"([A-ZĐÂĂÊÔƠƯÁÀẢÃẠẮẶẰẲẴẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ][^\n]{3,40}?)\s+là\s+", content)
        if m:
            term = m.group(1).strip().lower()
            return f'"{term}" trong Luật Kế toán được hiểu là gì?'
        return f"Luật Kế toán định nghĩa như thế nào về {subject.lower()}?"

    if concept_type == "pham_vi":
        return f"Luật Kế toán 2015 điều chỉnh những đối tượng nào tại {subject.lower()}?"

    if concept_type == "nguyen_tac":
        return f"Nguyên tắc kế toán quy định tại {ref} là gì?"

    if concept_type == "nghia_vu":

        # Thử trích đối tượng phải thực hiện
        m = re.search(r"(đơn vị kế toán|kế toán trưởng|người làm kế toán|doanh nghiệp|cơ quan|tổ chức)", content, re.IGNORECASE)
        subj = m.group(1).lower() if m else "đơn vị kế toán"
        return f"{subj.capitalize()} có nghĩa vụ gì theo quy định tại {ref}?"

    if concept_type == "quyen_loi":
        m = re.search(r"(đơn vị kế toán|kế toán trưởng|người làm kế toán|doanh nghiệp|cơ quan|kế toán viên)", content, re.IGNORECASE)
        subj = m.group(1).lower() if m else "đơn vị kế toán"
        return f"{subj.capitalize()} có quyền gì theo {ref}?"

    if concept_type == "cam_ket":
        return f"Theo {ref} Luật Kế toán, những hành vi nào bị nghiêm cấm?"

    if concept_type == "che_tai":
        return f"Vi phạm quy định tại {ref} Luật Kế toán bị xử lý như thế nào?"

    if concept_type == "thu_tuc":
        return f"Theo {ref}, thủ tục và hồ sơ liên quan đến {subject.lower()} gồm những gì?"

    if concept_type == "to_chuc":
        return f"Tổ chức bộ máy kế toán theo {ref} được quy định như thế nào?"

    if concept_type == "bao_cao":
        return f"Theo {ref} Luật Kế toán, báo cáo tài chính phải được lập và nộp như thế nào?"

    # fallback
    return f"{subject} được quy định như thế nào trong Luật Kế toán?" if subject else ""

def _trim_answer(content: str, max_chars: int = 800) -> str:
    """Rút gọn nội dung làm câu trả lời."""
    content = _clean(content)
    if len(content) <= max_chars:
        return content
    # Cắt tại câu hoàn chỉnh gần nhất
    cut = content[:max_chars].rfind(". ")
    return content[:cut + 1] if cut > 100 else content[:max_chars] + "..."

@dataclass
class QaPair:
    question: str
    answer: str
    source_chunk_id: int
    dieu: str

class KeToaQaGenerator:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port, dbname = c.db,
            user = c.user, password = c.password
        )

    def _load_chunks_with_concepts(self, doc_code: str) -> list[dict]:
        # Tải chunk kèm concept_type ưu tiên (confidence cao nhất).
        sql = """
            SELECT
                lc.id,
                lc.dieu,
                lc.khoan,
                lc.diem,
                lc.content,
                COALESCE(
                    (SELECT concept_type FROM concept_tags
                        WHERE chunk_id = lc.id
                        ORDER BY confidence DESC LIMIT 1),
                    'khac'
                ) AS primary_concept
            FROM law_chunks lc
            JOIN documents d ON d.id = lc.document_id
            WHERE (d.doc_code = %s OR d.short_code = %s) AND lc.content IS NOT NULL AND LENGTH(lc.content) > 50
            ORDER BY lc.id
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (doc_code, doc_code))
            cols = [d[0] for d in (cur.description or [])]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def generate(self, doc_code: str = DOC_CODE) -> list[QaPair]:
        chunks = self._load_chunks_with_concepts(doc_code)
        print(f"[KeToaQaGenerator] Loaded {len(chunks)} chunks từ {doc_code}")

        pairs: list[QaPair] = []
        seen_questions: set[str] = set()

        for ch in chunks:
            q = _make_question(
                concept_type = ch["primary_concept"],
                dieu = ch["dieu"] or "",
                khoan = ch["khoan"] or "",
                content = ch["content"]
            )
            if not q:
                continue

            # Tránh câu hỏi trùng lặp quá giống nhau
            q_norm = re.sub(r"\s+", " ", q.lower())
            if q_norm in seen_questions:
                continue
            seen_questions.add(q_norm)

            answer = _trim_answer(ch["content"])
            pairs.append(QaPair(
                question = q,
                answer = answer,
                source_chunk_id = ch["id"],
                dieu = ch["dieu"] or ""
            ))

        print(f"[KeToaQaGenerator] Sinh được {len(pairs)} cặp Q&A")
        return pairs

    def save(self, pairs: list[QaPair]) -> int:
        # Lưu vào bảng qa_data, trả về số dòng đã insert.
        sql = """
            INSERT INTO qa_data (question, answer, source)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        rows = [(p.question, p.answer, SOURCE_LABEL) for p in pairs]
        with self._connect() as conn, conn.cursor() as cur:
            # Thêm cột source nếu chưa có (graceful)
            try:
                cur.execute("ALTER TABLE qa_data ADD COLUMN IF NOT EXISTS source VARCHAR(64)")
            except Exception:
                pass
            execute_values(cur, sql, rows)
        print(f"[KeToaQaGenerator] Đã lưu {len(rows)} Q&A vào qa_data (source={SOURCE_LABEL!r})")
        return len(rows)

    def run(self, doc_code: str = DOC_CODE) -> None:
        pairs = self.generate(doc_code)
        if not pairs:
            print("[KeToaQaGenerator] Không có Q&A nào được sinh — hãy chạy ConceptTagger trước.")
            return
        self.save(pairs)

        # In mẫu 5 cặp đầu
        print("\n--- Mẫu Q&A ---")
        for p in pairs[:5]:
            print(f"Q: {p.question}")
            print(f"A: {p.answer[:120]}...")
            print(f"({p.dieu})\n")

if __name__ == "__main__":
    import os
    cfg = DbConfig(
        host = os.environ["POSTGRES_HOST"],
        port = int(os.environ.get("POSTGRES_PORT", 5432)),
        db = os.environ["POSTGRES_DB"],
        user = os.environ["POSTGRES_USER"],
        password = os.environ["POSTGRES_PASSWORD"]
    )
    gen = KeToaQaGenerator(cfg)
    gen.run(DOC_CODE)