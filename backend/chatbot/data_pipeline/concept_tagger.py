"""
Concept tagger cho văn bản luật Việt Nam.

Phân loại mỗi chunk (Điều/Khoản/Điểm) theo các loại khái niệm pháp lý:
  - dinh_nghia: Điều giải thích từ ngữ / định nghĩa
  - nghia_vu: Quy định bắt buộc, trách nhiệm, phải làm
  - quyen_loi: Quyền được hưởng, được làm
  - cam_ket: Cấm, không được phép
  - che_tai: Xử lý vi phạm, hình thức xử phạt
  - thu_tuc: Trình tự thực hiện, hồ sơ, thủ tục
  - to_chuc: Tổ chức bộ máy, nhân sự, phân công
  - pham_vi: Phạm vi điều chỉnh, đối tượng áp dụng
  - nguyen_tac: Nguyên tắc, chuẩn mực
  - bao_cao: Báo cáo, công khai, minh bạch

Kết quả lưu vào bảng `concept_tags`.

Cách dùng:
    from chatbot.data_pipeline.concept_tagger import ConceptTagger, ConceptDbSaver
    tagger = ConceptTagger()
    saver = ConceptDbSaver(cfg)
    tagger.run(cfg, saver, doc_code = "LKT2015")
"""

import re
import psycopg2
from psycopg2.extras import execute_values
from dataclasses import dataclass
from .db_loader import DbConfig

_CONCEPT_RULES: list[tuple[str, list[str]]] = [
    ("dinh_nghia", [
        r"giải thích từ ngữ",
        r"\blà\s+\w.{5,}", # "X là ..."  → định nghĩa
        r"được hiểu là",
        r"có nghĩa là",
        r"theo (luật|quy định) này.{0,20}(là|gồm)"
    ]),
    ("pham_vi", [
        r"phạm vi (điều chỉnh|áp dụng)",
        r"đối tượng áp dụng",
        r"luật này (quy định|điều chỉnh|áp dụng)",
        r"(không )?áp dụng (đối với|cho)"
    ]),
    ("nguyen_tac", [
        r"nguyên tắc (kế toán|lập|trình bày|ghi nhận)",
        r"chuẩn mực (kế toán|đạo đức)",
        r"yêu cầu kế toán",
        r"phải (phản ánh|ghi nhận|trình bày).{0,40}trung thực",
        r"nhất quán"
    ]),
    ("nghia_vu", [
        r"\bphải\b",
        r"\bcó trách nhiệm\b",
        r"\bchịu trách nhiệm\b",
        r"\bcó nghĩa vụ\b",
        r"bắt buộc",
        r"phải (lập|nộp|lưu|báo cáo|kiểm tra|thực hiện|tuân thủ)"
    ]),
    ("quyen_loi", [
        r"\bđược\b.{0,30}(quyền|hưởng|phép)",
        r"\bcó quyền\b",
        r"được (lựa chọn|sử dụng|yêu cầu|đề nghị|từ chối)",
        r"quyền (của|hành nghề|hành|kiểm tra)"
    ]),
    ("cam_ket", [
        r"\bkhông được\b",
        r"\bcấm\b",
        r"\bnghiêm cấm\b",
        r"không (được phép|có quyền)",
        r"cấm (làm|thực hiện|sử dụng|cung cấp)"
    ]),
    ("che_tai", [
        r"xử (lý|phạt|phạt vi phạm)",
        r"hình thức xử",
        r"vi phạm.{0,20}(bị|sẽ bị|phạt|xử lý)",
        r"bồi thường",
        r"thu hồi (giấy phép|chứng chỉ|quyết định)",
        r"đình chỉ",
        r"tước (quyền|chứng chỉ)"
    ]),
    ("thu_tuc", [
        r"hồ sơ",
        r"trình tự",
        r"thủ tục",
        r"(nộp|gửi|đăng ký|đăng nộp).{0,30}(cơ quan|bộ|sở)",
        r"thời hạn.{0,20}(ngày|tháng|năm)",
        r"cấp (giấy|chứng nhận|chứng chỉ)"
    ]),
    ("to_chuc", [
        r"bộ máy kế toán",
        r"kế toán trưởng",
        r"người làm kế toán",
        r"tổ chức (bộ máy|kế toán|kiểm tra|nghề nghiệp)",
        r"(bộ tài chính|cơ quan quản lý nhà nước).{0,30}(quy định|ban hành|hướng dẫn)",
        r"hội (kế toán|kiểm toán)"
    ]),
    ("bao_cao", [
        r"báo cáo tài chính",
        r"báo cáo kế toán",
        r"công khai",
        r"công bố",
        r"kiểm toán",
        r"kiểm tra kế toán",
        r"lập (và gửi|và nộp|báo cáo)"
    ]),
]

# Biên dịch các regex một lần
_COMPILED_RULES: list[tuple[str, list[re.Pattern]]] = [
    (concept, [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns])
    for concept, patterns in _CONCEPT_RULES
]


@dataclass
class ConceptTag:
    chunk_id: int
    concept_type: str
    confidence: float # 0.0 – 1.0 dựa trên số pattern khớp


def tag_chunk(chunk_id: int, text: str) -> list[ConceptTag]:
    
    # Trả về danh sách ConceptTag cho một chunk."""
    text_lower = text.lower()
    tags: list[ConceptTag] = []

    for concept, patterns in _COMPILED_RULES:
        match_count = sum(1 for p in patterns if p.search(text_lower))
        if match_count == 0:
            continue
        confidence = min(1.0, match_count / len(patterns) * 2)  # scale: 50% match → 1.0
        tags.append(ConceptTag(chunk_id = chunk_id, concept_type = concept, confidence = round(confidence, 3)))

    # Nếu không khớp gì → gán "khac"
    if not tags:
        tags.append(ConceptTag(chunk_id = chunk_id, concept_type = "khac", confidence = 0.5))

    return tags

class ConceptDbSaver:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port, dbname = c.db,
            user = c.user, password = c.password
        )

    def save(self, tags: list[ConceptTag]) -> None:
        sql = """
            INSERT INTO concept_tags (chunk_id, concept_type, confidence)
            VALUES %s
            ON CONFLICT (chunk_id, concept_type) DO UPDATE SET confidence = EXCLUDED.confidence
        """
        rows = [(t.chunk_id, t.concept_type, t.confidence) for t in tags]
        with self._connect() as conn, conn.cursor() as cur:
            execute_values(cur, sql, rows)
        print(f"[ConceptDbSaver] Đã lưu {len(rows)} concept tags")

class ConceptTagger:
    # Pipeline: tải chunks từ DB → gán nhãn khái niệm → lưu vào DB."""

    def _connect(self, cfg: DbConfig):
        return psycopg2.connect(
            host = cfg.host, port = cfg.port, dbname = cfg.db,
            user = cfg.user, password = cfg.password
        )

    def _load_chunks(self, cfg: DbConfig, doc_code: str | None) -> list[tuple[int, str]]:
        if doc_code:
            sql = """
                SELECT lc.id, lc.full_text
                FROM law_chunks lc
                JOIN documents d ON d.id = lc.document_id
                WHERE (d.doc_code = %s OR d.short_code = %s) AND lc.full_text IS NOT NULL
                ORDER BY lc.id
            """
            params = (doc_code, doc_code)
        else:
            sql = "SELECT id, full_text FROM law_chunks WHERE full_text IS NOT NULL ORDER BY id"
            params = ()
        with self._connect(cfg) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def run(self, cfg: DbConfig, saver: ConceptDbSaver, doc_code: str | None = None) -> None:
        label = doc_code or "ALL"
        print(f"[ConceptTagger] Tải chunks cho doc_code = {label!r} ...")
        chunks = self._load_chunks(cfg, doc_code)
        print(f"[ConceptTagger] Loaded {len(chunks)} chunks — đang tag ...")

        all_tags: list[ConceptTag] = []
        for cid, text in chunks:
            all_tags.extend(tag_chunk(cid, text))

        saver.save(all_tags)

        # In thống kê
        from collections import Counter
        stat = Counter(t.concept_type for t in all_tags)
        print("[ConceptTagger] Thống kê concept_type:")
        for ctype, cnt in stat.most_common():
            print(f"{ctype:20s}: {cnt:4d} chunks")
