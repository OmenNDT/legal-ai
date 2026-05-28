"""Helper functions for extractive answer generation."""

import re


def split_into_sentences(text: str) -> list[str]:
    """Tách văn bản thành câu."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def score_sentences(question: str, sentences: list[str]) -> list[tuple[str, float]]:
    """Score sentences by word overlap với question."""
    q_words = set(question.lower().split())
    scored = []
    for sent in sentences:
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words)
        score = overlap / max(len(q_words), 1)
        scored.append((sent, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_answer_from_sentences(question: str, sentences: list[str]) -> str:
    """Ghép các câu thành câu trả lời hoàn chỉnh với intro phù hợp."""
    if not sentences:
        return "Không tìm thấy thông tin phù hợp."

    question_lower = question.lower()
    if any(w in question_lower for w in ["có hiệu lực", "hết hiệu lực", "còn hiệu lực"]):
        intro = "Theo các quy định pháp luật liên quan:"
    elif any(w in question_lower for w in ["định nghĩa", "là gì", "khái niệm"]):
        intro = "Theo quy định của pháp luật:"
    elif any(w in question_lower for w in ["hình phạt", "xử phạt", "chế tài"]):
        intro = "Về chế tài xử phạt, pháp luật quy định:"
    elif any(w in question_lower for w in ["thủ tục", "quy trình", "làm thế nào"]):
        intro = "Về thủ tục thực hiện:"
    else:
        intro = "Theo quy định của pháp luật:"

    unique_sentences = []
    seen = set()
    for sent in sentences:
        normalized = re.sub(r'\s+', ' ', sent.lower().strip())
        if normalized not in seen and len(sent) > 20:
            unique_sentences.append(sent)
            seen.add(normalized)

    if not unique_sentences:
        return "Không tìm thấy thông tin đầy đủ để trả lời câu hỏi."

    answer = intro + "\n\n"
    for i, sent in enumerate(unique_sentences[:5], 1):
        answer += f"{i}. {sent}\n"

    answer += "\nLưu ý: Thông tin trên dựa trên các văn bản pháp luật hiện hành."
    return answer.strip()
