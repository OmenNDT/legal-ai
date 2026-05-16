"""Generate QA dataset automatically from parsed law text.

Creates question-answer pairs with intent labels and NER BIO tags.
Output: data/processed/qa_ke_toan_train.json
"""

import json
import random
import re
from pathlib import Path
from typing import Optional

from src.common.config import INTENT_LABELS, ENTITY_TYPES, NER_LABELS


def build_question_templates():
    """Return templates for generating questions from article titles."""
    return {
        "default": [
            "Điều {id} Luật Kế toán quy định gì về {title}?",
            "Nội dung của Điều {id} về {title} là gì?",
            "Theo Điều {id} Luật Kế toán, {title}?",
            "Luật Kế toán quy định như thế nào về {title} tại Điều {id}?",
            "Điều {id} nói gì về {title}?",
        ],
        "dinh_nghia": [
            "{title} được hiểu như thế nào theo Luật Kế toán?",
            "Điều {id} định nghĩa {title} là gì?",
            "Khái niệm {title} theo Luật Kế toán Điều {id}?",
        ],
        "doi_tuong": [
            "Đối tượng áp dụng của {title} được quy định tại Điều {id}?",
            "Điều {id} quy định đối tượng nào?",
            "Ai là đối tượng của {title} theo Điều {id}?",
        ],
        "quyen_loi": [
            "Quyền lợi về {title} được quy định như thế nào tại Điều {id}?",
            "Điều {id} bảo vệ quyền lợi gì?",
            "Theo Điều {id}, quyền lợi về {title} là gì?",
        ],
        "nghia_vu": [
            "Nghĩa vụ về {title} theo Điều {id} Luật Kế toán?",
            "Điều {id} quy định nghĩa vụ gì về {title}?",
            "Theo Luật Kế toán Điều {id}, nghĩa vụ {title} là gì?",
        ],
        "thu_tuc": [
            "Thủ tục về {title} được quy định tại Điều {id} như thế nào?",
            "Điều {id} quy định thủ tục gì?",
            "Theo Luật Kế toán, thủ tục {title} tại Điều {id} là gì?",
        ],
        "thoi_han": [
            "Thời hạn về {title} theo Điều {id} Luật Kế toán là bao lâu?",
            "Điều {id} quy định thời hạn như thế nào?",
            "Thời hạn của {title} được quy định tại Điều {id}?",
        ],
    }


def detect_intent(title: str) -> str:
    """Detect intent from article title using keyword matching."""
    title_lower = title.lower()
    if any(k in title_lower for k in ["giải thích", "định nghĩa", "khái niệm", "hiểu là"]):
        return "hoi_dinh_nghia"
    if any(k in title_lower for k in ["đối tượng", "áp dụng", "phạm vi", "subject"]):
        return "hoi_doi_tuong"
    if any(k in title_lower for k in ["quyền", "quyền lợi", "được quyền", "benefit", "right"]):
        return "hoi_quyen_loi"
    if any(k in title_lower for k in ["nghĩa vụ", "trách nhiệm", "bắt buộc", "phải", "obligation"]):
        return "hoi_nghia_vu"
    if any(k in title_lower for k in ["thủ tục", "trình tự", "quy trình", "procedure", "process"]):
        return "hoi_thu_tuc"
    if any(k in title_lower for k in ["thời hạn", "thời hiệu", "kỳ hạn", "deadline", "time limit"]):
        return "hoi_thoi_han"
    return "hoi_dieu_khoan"


def generate_questions(dieu_id: str, title: str, templates: dict) -> list[dict]:
    """Generate 3-5 question variants for an article."""
    intent = detect_intent(title)
    template_list = templates.get(intent, templates["default"])
    num_questions = min(random.randint(3, 5), len(template_list))
    selected = random.sample(template_list, num_questions)
    questions = []
    for t in selected:
        q = t.format(id=dieu_id, title=title.lower())
        questions.append({"question": q, "intent": intent})
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
    """Generate full QA dataset from structured law."""
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

            questions = generate_questions(dieu_id, title, templates)
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
        print(f"  {intent}: {count}")
    print(f"Saved → {output_path}")


def main():
    structured = Path("data/processed/luat_ke_toan_2025_structured.json")
    output = Path("data/processed/qa_ke_toan_train.json")
    generate_dataset(structured, output)


if __name__ == "__main__":
    main()
