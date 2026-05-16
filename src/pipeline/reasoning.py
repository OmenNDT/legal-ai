from dataclasses import dataclass, field

@dataclass
class StructuredAnswer:
    legal_basis: str
    analysis: str
    conclusion: str
    recommendation: str
    raw_answer: str
    facts: str = ""
    confidence: float = 0.8

REASONING_SYSTEM = (
    "Bạn là chuyên gia pháp lý Việt Nam. Phân tích câu hỏi pháp lý theo 5 bước:\n"
    "Bước 1: Xác định sự kiện pháp lý liên quan\n"
    "Bước 2: Xác định vấn đề pháp lý cần giải quyết\n"
    "Bước 3: Tìm và trích dẫn căn cứ pháp lý từ văn bản được cung cấp\n"
    "Bước 4: Áp dụng quy định pháp luật vào tình huống cụ thể\n"
    "Bước 5: Đưa ra kết luận\n\n"
    "Sau đó trình bày câu trả lời theo đúng format:\n"
    "**Căn cứ pháp lý:**\n"
    "**Phân tích:**\n"
    "**Kết luận:**\n"
    "**Khuyến nghị:**"
)

class Reasoning:
    def __init__(self, llm_client, kg_reasoner=None):
        self._llm_client = llm_client
        self._kg_reasoner = kg_reasoner

    def generate(self, prompt: str, structured_query) -> StructuredAnswer:
        kg_context = self._get_kg_context(structured_query)
        full_prompt = prompt
        if kg_context:
            full_prompt += f"\n\nThông tin từ knowledge graph:\n{kg_context}"

        raw_answer = self._llm_client.complete(full_prompt, system=REASONING_SYSTEM)
        return self._parse_answer(raw_answer)

    def _get_kg_context(self, structured_query) -> str:
        if not self._kg_reasoner:
            return ""
        context_parts = []
        for entity in structured_query.entities:
            if entity["entity"] in ("THONG_TU", "NGHI_DINH", "LUAT"):
                doc_id = f"{entity['entity']}_{entity['text'].replace(' ', '_')}"
                result = self._kg_reasoner.check_validity(doc_id)
                context_parts.extend(result.reasoning_steps)
        return "\n".join(context_parts)

    def _parse_answer(self, raw: str) -> StructuredAnswer:
        sections = {"legal_basis": "", "analysis": "", "conclusion": "", "recommendation": "", "facts": ""}
        markers = {
            "**Căn cứ pháp lý:**": "legal_basis",
            "**Phân tích:**": "analysis",
            "**Kết luận:**": "conclusion",
            "**Khuyến nghị:**": "recommendation",
            "Bước 1:": "facts",
        }
        current_key = None
        for line in raw.split("\n"):
            matched = False
            for marker, key in markers.items():
                if marker in line:
                    current_key = key
                    remainder = line.replace(marker, "").strip()
                    if remainder:
                        sections[current_key] += remainder + "\n"
                    matched = True
                    break
            if not matched and current_key and line.strip():
                next_marker_found = any(m in line for m in markers)
                if not next_marker_found:
                    sections[current_key] += line + "\n"

        for key in sections:
            sections[key] = sections[key].strip()

        if not any(v for k, v in sections.items() if k != "facts"):
            sections["analysis"] = raw

        return StructuredAnswer(
            legal_basis=sections["legal_basis"],
            analysis=sections["analysis"],
            conclusion=sections["conclusion"],
            recommendation=sections["recommendation"],
            raw_answer=raw,
            facts=sections["facts"],
        )
