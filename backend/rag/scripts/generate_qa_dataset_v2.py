"""Generate expanded QA dataset from parsed law text with paraphrasing.

Output: data/processed/qa_ke_toan_train_v2.json
"""

import json
import random
import re
from pathlib import Path
from typing import Optional

from src.common.config import INTENT_LABELS, NER_LABELS


def build_question_templates():
    """Expanded templates for generating diverse questions."""
    return {
        "hoi_dieu_khoan": [
            "Điều {id} Luật Kế toán quy định gì về {title}?",
            "Nội dung của Điều {id} về {title} là gì?",
            "Theo Điều {id} Luật Kế toán, {title}?",
            "Luật Kế toán quy định như thế nào về {title} tại Điều {id}?",
            "Điều {id} nói gì về {title}?",
            "Quy định tại Điều {id} về {title} là gì?",
            "Điều {id} của Luật Kế toán có nội dung gì?",
            "Theo Điều {id} của Luật Kế toán thì {title}?",
            "Nội dung chính của Điều {id} về {title}?",
            "Điều {id} Luật Kế toán quy định về {title} như thế nào?",
            "Điều {id} trong Luật Kế toán nêu rõ điều gì về {title}?",
            "Theo quy định tại Điều {id} Luật Kế toán thì {title}?",
            "Điều {id} Luật Kế toán có quy định gì liên quan đến {title}?",
            "Những quy định về {title} được nêu tại Điều {id} là gì?",
            "Theo Luật Kế toán tại Điều {id}, {title}?",
        ],
        "hoi_dinh_nghia": [
            "{title} được hiểu như thế nào theo Luật Kế toán?",
            "Điều {id} định nghĩa {title} là gì?",
            "Khái niệm {title} theo Luật Kế toán Điều {id}?",
            "{title} theo Luật Kế toán có nghĩa là gì?",
            "Định nghĩa về {title} được quy định tại Điều {id}?",
            "{title} được định nghĩa như thế nào trong Luật Kế toán?",
            "Theo Điều {id} Luật Kế toán, {title} là gì?",
            "Luật Kế toán quy định {title} có nghĩa là gì?",
            "Như thế nào được coi là {title} theo Luật Kế toán?",
            "{title} được hiểu là gì trong Luật Kế toán Điều {id}?",
            "Khái niệm {title} được định nghĩa ra sao tại Điều {id}?",
            "Theo Điều {id}, {title} được hiểu như sau?",
            "{title} có định nghĩa gì theo Luật Kế toán?",
            "Điều {id} giải thích {title} như thế nào?",
        ],
        "hoi_doi_tuong": [
            "Đối tượng áp dụng của {title} được quy định tại Điều {id}?",
            "Điều {id} quy định đối tượng nào?",
            "Ai là đối tượng của {title} theo Điều {id}?",
            "Những ai phải tuân thủ quy định về {title} tại Điều {id}?",
            "Đối tượng nào chịu sự điều chỉnh của {title} theo Điều {id}?",
            "Quy định tại Điều {id} áp dụng cho đối tượng nào?",
            "Ai phải thực hiện quy định về {title} tại Điều {id}?",
            "Đối tượng chịu trách nhiệm về {title} theo Điều {id} là ai?",
            "Những đối tượng nào liên quan đến {title} theo Điều {id}?",
            "Điều {id} quy định đối tượng áp dụng của {title} là gì?",
            "Các đối tượng thuộc phạm vi điều chỉnh của {title} tại Điều {id}?",
            "Theo Điều {id}, {title} áp dụng cho ai?",
            "Ai là người phải tuân thủ {title} theo Luật Kế toán Điều {id}?",
            "Đối tượng của quy định {title} tại Điều {id}?",
        ],
        "hoi_quyen_loi": [
            "Quyền lợi về {title} được quy định như thế nào tại Điều {id}?",
            "Điều {id} bảo vệ quyền lợi gì?",
            "Theo Điều {id}, quyền lợi về {title} là gì?",
            "Người lao động có quyền lợi gì về {title} tại Điều {id}?",
            "Quyền lợi liên quan đến {title} được quy định tại Điều {id}?",
            "Theo Luật Kế toán Điều {id}, quyền lợi về {title} gồm những gì?",
            "Điều {id} quy định quyền lợi nào về {title}?",
            "Những quyền lợi nào được đảm bảo trong {title} tại Điều {id}?",
            "Quyền lợi của đối tượng về {title} theo Điều {id} là gì?",
            "{title} tại Điều {id} đảm bảo quyền lợi gì?",
            "Điều {id} bảo đảm quyền lợi về {title} như thế nào?",
            "Quyền lợi hợp pháp về {title} theo Luật Kế toán Điều {id}?",
            "Theo Điều {id}, đối tượng có quyền lợi gì về {title}?",
        ],
        "hoi_nghia_vu": [
            "Nghĩa vụ về {title} theo Điều {id} Luật Kế toán?",
            "Điều {id} quy định nghĩa vụ gì về {title}?",
            "Theo Luật Kế toán Điều {id}, nghĩa vụ {title} là gì?",
            "Nghĩa vụ của người lao động về {title} tại Điều {id}?",
            "Những nghĩa vụ nào về {title} được quy định tại Điều {id}?",
            "Điều {id} quy định trách nhiệm gì về {title}?",
            "Nghĩa vụ phải thực hiện về {title} theo Điều {id}?",
            "Theo Luật Kế toán tại Điều {id}, nghĩa vụ về {title} là gì?",
            "Đối tượng có nghĩa vụ gì về {title} theo Điều {id}?",
            "Nghĩa vụ bắt buộc về {title} được quy định tại Điều {id}?",
            "Điều {id} yêu cầu thực hiện nghĩa vụ gì về {title}?",
            "Trách nhiệm và nghĩa vụ về {title} theo Điều {id}?",
            "Theo Điều {id}, nghĩa vụ của đối tượng về {title} là gì?",
        ],
        "hoi_thu_tuc": [
            "Thủ tục về {title} được quy định tại Điều {id} như thế nào?",
            "Điều {id} quy định thủ tục gì?",
            "Theo Luật Kế toán, thủ tục {title} tại Điều {id} là gì?",
            "Quy trình thực hiện {title} theo Điều {id} Luật Kế toán?",
            "Thủ tục thực hiện {title} được quy định tại Điều {id}?",
            "Các bước thủ tục về {title} theo Luật Kế toán Điều {id}?",
            "Điều {id} quy định quy trình {title} như thế nào?",
            "Thủ tục hành chính về {title} tại Điều {id}?",
            "Thực hiện {title} cần những thủ tục gì theo Điều {id}?",
            "Theo Điều {id}, thủ tục về {title} gồm các bước nào?",
            "Quy định về thủ tục {title} tại Điều {id} Luật Kế toán?",
            "Trình tự thực hiện {title} theo Luật Kế toán Điều {id}?",
            "Thủ tục cần tuân thủ khi thực hiện {title} tại Điều {id}?",
        ],
        "hoi_thoi_han": [
            "Thời hạn về {title} theo Điều {id} Luật Kế toán là bao lâu?",
            "Điều {id} quy định thời hạn như thế nào?",
            "Thời hạn của {title} được quy định tại Điều {id}?",
            "Thời gian thực hiện {title} theo Điều {id} là bao lâu?",
            "Kỳ hạn về {title} được quy định tại Điều {id}?",
            "Thời hạn hoàn thành {title} theo Luật Kế toán Điều {id}?",
            "Điều {id} quy định thời gian {title} là bao nhiêu?",
            "Thời hạn áp dụng cho {title} tại Điều {id}?",
            "Mốc thời gian về {title} theo Điều {id} Luật Kế toán?",
            "Theo Điều {id}, thời hạn {title} được quy định ra sao?",
            "Thời gian hiệu lực của {title} tại Điều {id}?",
            "Thời hạn thực hiện nghĩa vụ {title} theo Điều {id}?",
        ],
        "hoi_hieu_luc": [
            "{title} có hiệu lực từ khi nào theo Điều {id}?",
            "Điều {id} quy định hiệu lực của {title} như thế nào?",
            "Thời điểm có hiệu lực của {title} tại Điều {id}?",
            "{title} được áp dụng từ bao giờ theo Luật Kế toán?",
            "Hiệu lực thi hành của {title} theo Điều {id} Luật Kế toán?",
            "Điều {id} quy định thời điểm có hiệu lực của {title}?",
            "Từ khi nào {title} có hiệu lực theo Luật Kế toán?",
            "Thời gian bắt đầu hiệu lực của {title} tại Điều {id}?",
            "Khi nào {title} bắt đầu có hiệu lực theo Luật Kế toán?",
            "Thời điểm áp dụng của {title} được quy định tại Điều {id}?",
            "Điều {id} nêu rõ {title} có hiệu lực từ thời điểm nào?",
            "{title} có hiệu lực kể từ ngày nào theo Luật Kế toán?",
        ],
        "hoi_hinh_phat": [
            "Hình phạt/Chế tài vi phạm {title} được quy định tại Điều {id}?",
            "Điều {id} quy định chế tài nào cho {title}?",
            "Xử phạt vi phạm {title} theo Luật Kế toán Điều {id}?",
            "Hậu quả pháp lý khi vi phạm {title} tại Điều {id}?",
            "Chế tài xử lý vi phạm {title} theo Điều {id}?",
            "Mức phạt cho hành vi vi phạm {title} tại Điều {id}?",
            "Trách nhiệm pháp lý về {title} được quy định tại Điều {id}?",
            "Vi phạm {title} bị xử lý như thế nào theo Điều {id}?",
            "Chế tài đối với hành vi vi phạm {title} tại Điều {id} là gì?",
            "Điều {id} quy định hình thức xử phạt nào đối với {title}?",
            "Mức độ xử lý vi phạm {title} theo Luật Kế toán tại Điều {id}?",
        ],
        "hoi_sua_doi": [
            "Điều {id} có quy định sửa đổi, bổ sung gì về {title}?",
            "Nội dung sửa đổi {title} tại Điều {id}?",
            "Thay đổi về {title} được quy định tại Điều {id}?",
            "Điều {id} sửa đổi, bổ sung những gì về {title}?",
            "Quy định mới về {title} tại Điều {id} Luật Kế toán?",
            "Những điểm mới về {title} theo Điều {id}?",
        ],
    }


def detect_intent_matches(text: str) -> set[str]:
    """Return all intent names that match the given text."""
    text_lower = text.lower()
    matched = set()

    if any(k in text_lower for k in ["giải thích từ ngữ", "định nghĩa", "khái niệm", "hiểu là", "được hiểu là"]):
        matched.add("hoi_dinh_nghia")
    if any(k in text_lower for k in ["đối tượng áp dụng", "phạm vi điều chỉnh", "áp dụng cho"]):
        matched.add("hoi_doi_tuong")
    if any(k in text_lower for k in ["quyền", "được quyền", "quyền lợi", "được hưởng", "được phép", "benefit"]):
        matched.add("hoi_quyen_loi")
    if any(k in text_lower for k in ["nghĩa vụ", "trách nhiệm", "bắt buộc phải", "phải có", "phải thực hiện", "nộp", "báo cáo"]):
        matched.add("hoi_nghia_vu")
    if any(k in text_lower for k in ["thủ tục", "trình tự", "quy trình", "các bước", "hồ sơ", "chứng từ", "procedure"]):
        matched.add("hoi_thu_tuc")
    if any(k in text_lower for k in ["thời hạn", "thời hiệu", "kỳ hạn", "thời gian", "thời điểm", "deadline", "hiệu lực"]):
        matched.add("hoi_thoi_han")
    if any(k in text_lower for k in ["sửa đổi", "bổ sung", "thay thế", "điều chỉnh", "bãi bỏ"]):
        matched.add("hoi_sua_doi")
    if any(k in text_lower for k in ["vi phạm", "xử phạt", "chế tài", "trách nhiệm pháp lý", "hình phạt", "cấm", "xử lý"]):
        matched.add("hoi_hinh_phat")
    if any(k in text_lower for k in ["hiệu lực", "có hiệu lực", "thi hành", "áp dụng từ"]):
        matched.add("hoi_hieu_luc")

    return matched


def detect_all_intents(title: str, content: str) -> list[str]:
    """Detect all applicable intents for an article.

    Priority: title matches first (more reliable). If title yields nothing,
    fall back to content matches. If still nothing, default to hoi_dieu_khoan.
    """
    title_matches = detect_intent_matches(title)
    if title_matches:
        return sorted(title_matches)

    content_matches = detect_intent_matches(content)
    if content_matches:
        return sorted(content_matches)

    return ["hoi_dieu_khoan"]


def synonym_replace(text: str) -> str:
    """Simple synonym replacement for Vietnamese legal text."""
    synonyms = {
        "quy định": ["quy định", "nêu rõ", "đưa ra", "đặt ra"],
        "như thế nào": ["như thế nào", "ra sao", "thế nào"],
        "là gì": ["là gì", "có nghĩa là gì", "được hiểu là gì"],
        "bao lâu": ["bao lâu", "trong bao lâu", "bao nhiêu lâu"],
        "được": ["được", "có quyền"],
        "phải": ["phải", "bắt buộc", "có nghĩa vụ"],
        "gồm": ["gồm", "bao gồm", "có những"],
        "ai": ["ai", "đối tượng nào", "cá nhân/tổ chức nào"],
    }
    for key, reps in synonyms.items():
        if key in text:
            text = text.replace(key, random.choice(reps), 1)
    return text


def generate_questions(
    dieu_id: str,
    title: str,
    content: str,
    templates: dict,
    max_per_article: int = 12,
    max_per_intent: int = 6,
) -> list[dict]:
    """Generate balanced question variants for an article across multiple intents.

    - Detects ALL applicable intents (title-priority, content fallback).
    - Distributes the max_per_article budget evenly across matched intents.
    - Adds one synonym variant per base question.
    """
    intents = detect_all_intents(title, content)

    # Distribute budget: at least 2 per intent, at most max_per_intent
    quota = max(2, max_per_article // max(len(intents), 1))
    quota = min(quota, max_per_intent)

    questions = []
    for intent in intents:
        template_list = templates.get(intent, templates["hoi_dieu_khoan"])

        # Build base questions from templates (unique only)
        base_questions = []
        for t in template_list:
            q = t.format(id=dieu_id, title=title.lower())
            base_questions.append(q)

        # Add one synonym variant per base question
        variant_questions = []
        for q in base_questions:
            variant_questions.append(q)
            q2 = synonym_replace(q)
            if q2 != q:
                variant_questions.append(q2)

        # Shuffle and cap per-intent quota
        random.shuffle(variant_questions)
        selected = variant_questions[:quota]

        for q in selected:
            questions.append({"question": q, "intent": intent})

    # Final cap at max_per_article
    if len(questions) > max_per_article:
        questions = random.sample(questions, max_per_article)

    return questions


def extract_entities(text: str) -> list[dict]:
    """Extract legal entities from text using regex patterns."""
    patterns = {
        "LUAT": r"Luật\s+([\w\s]+?)(?:\s+số|\s+năm|\d{4}|$)",
        "THONG_TU": r"Thông\s+tư\s+(?:số\s+)?([\d/]+(?:/TT-\w+)?)",
        "NGHI_DINH": r"Nghị\s+định\s+(?:số\s+)?([\d/]+(?:/NĐ-CP)?)",
        "DIEU": r"Điều\s+(\d+)",
        "KHOAN": r"khoản\s+(\d+)",
        "DIEM": r"điểm\s+([a-zđ]+)",
        "CO_QUAN": r"(?:Bộ\s+\w+|Ngân\s+hàng\s+\w+|Quốc\s+hội|Chính\s+phủ|UBND)\s*(?:\w*\s*\w*)?",
        "NGAY_THANG": r"\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
    }
    entities = []
    for etype, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "entity": etype,
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
    entities.sort(key=lambda x: x["start"])
    return entities


def build_ner_labels(text: str, entities: list[dict]) -> list[str]:
    """Build BIO tags for NER training."""
    tokens = text.split()
    labels = ["O"] * len(tokens)
    char_to_token = {}
    pos = 0
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            char_to_token[pos + j] = i
        char_to_token[pos + len(token)] = i
        pos += len(token) + 1

    for ent in entities:
        start_char = ent["start"]
        end_char = ent["end"]
        start_token = char_to_token.get(start_char)
        end_token = None
        for c in range(end_char - 1, start_char, -1):
            if c in char_to_token:
                end_token = char_to_token[c]
                break
        if start_token is not None and end_token is not None:
            labels[start_token] = f"B-{ent['entity']}"
            for t in range(start_token + 1, end_token + 1):
                if labels[t] == "O":
                    labels[t] = f"I-{ent['entity']}"
    return labels


def build_answer(dieu: dict) -> str:
    """Build answer text from article content."""
    parts = []
    if dieu.get("content"):
        parts.append(dieu["content"])
    for k in dieu.get("khoans", []):
        parts.append(f"{k['id']}. {k['content']}")
        for di in k.get("diems", []):
            parts.append(f"  {di['id']}) {di['content']}")
    return "\n".join(parts)


def generate_dataset(structured_path: Path, output_path: Path, max_pairs: Optional[int] = None):
    """Generate expanded QA dataset from structured law."""
    with open(structured_path, encoding="utf-8") as f:
        data = json.load(f)

    templates = build_question_templates()
    dataset = []
    qa_id = 0

    for ch in data:
        for dieu in ch.get("dieus", []):
            dieu_id = dieu["id"]
            title = dieu["title"]
            answer_text = build_answer(dieu)
            entities = extract_entities(answer_text)
            ner_labels = build_ner_labels(answer_text, entities)

            questions = generate_questions(dieu_id, title, answer_text, templates, max_per_article=18, max_per_intent=8)
            for q in questions:
                dataset.append({
                    "id": qa_id,
                    "question": q["question"],
                    "intent": q["intent"],
                    "answer": answer_text,
                    "entities": entities,
                    "ner_labels": ner_labels,
                    "source": f"Điều {dieu_id} - {title}",
                })
                qa_id += 1
                if max_pairs and len(dataset) >= max_pairs:
                    break
            if max_pairs and len(dataset) >= max_pairs:
                break
        if max_pairs and len(dataset) >= max_pairs:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(dataset)} QA pairs")
    intent_counts = {}
    for d in dataset:
        intent_counts[d["intent"]] = intent_counts.get(d["intent"], 0) + 1
    print("Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(dataset) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")
    print(f"Saved → {output_path}")


def main():
    structured = Path("data/processed/luat_ke_toan_2025_structured.json")
    output = Path("data/processed/qa_ke_toan_train_v2.json")
    generate_dataset(structured, output)


if __name__ == "__main__":
    main()
