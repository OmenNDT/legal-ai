"""Legal Knowledge Graph reasoner.

Provides high-level query functions:
- is_valid(): Check if a document is still in force
- amendment_chain(): Trace all amendments
- related_docs(): Find related documents
- reasoning_path(): Find path between two legal entities
- explain(): Explain reasoning with evidence
"""

from dataclasses import dataclass
from typing import Optional

from src.knowledge.graph_builder import LegalKnowledgeGraph


@dataclass
class LegalAnswer:
    """Structured answer from the knowledge graph reasoner."""
    question: str
    answer: str
    evidence: list[dict]
    confidence: float
    source_entities: list[str]
    reasoning_steps: list[str]


class LegalReasoner:
    """Reasoning engine over the legal knowledge graph."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg

    def check_validity(self, doc_id: str) -> LegalAnswer:
        """Check if a legal document is still in force.

        Traces amendment chain and expiration status.
        """
        result = self.kg.is_valid(doc_id)
        doc_name = self.kg.G.nodes[doc_id].get("name", doc_id) if doc_id in self.kg.G else doc_id

        if result["status"] == "not_found":
            return LegalAnswer(
                question=f"{doc_name} còn hiệu lực không?",
                answer=f"Không tìm thấy văn bản '{doc_name}' trong cơ sở dữ liệu.",
                evidence=[], confidence=0.0, source_entities=[doc_id], reasoning_steps=[],
            )

        steps = [f"Kiểm tra trạng thái của {doc_name}"]
        evidence = []

        if result["expired_by"]:
            for item in result["expired_by"]:
                steps.append(f"→ Đã hết hiệu lực bởi: {self.kg.G.nodes[item['source']].get('name', item['source'])}")
                evidence.append({"type": "HET_HIEU_LUC", **item})
            answer = f"{doc_name} đã HẾT HIỆU LỰC."
        elif result["replaced_by"]:
            for item in result["replaced_by"]:
                steps.append(f"→ Đã được thay thế bởi: {self.kg.G.nodes[item['source']].get('name', item['source'])}")
                evidence.append({"type": "THAY_THE", **item})
            answer = f"{doc_name} đã được THAY THẾ."
        elif result["amended_by"]:
            for item in result["amended_by"]:
                steps.append(f"→ Đã được sửa đổi/bổ sung bởi: {self.kg.G.nodes[item['source']].get('name', item['source'])}")
                evidence.append({"type": "SUA_DOI_BO_SUNG", **item})
            answer = f"{doc_name} CÒN HIỆU LỰC (đã được sửa đổi/bổ sung)."
        else:
            steps.append("→ Không có văn bản hết hiệu lực, thay thế, hoặc sửa đổi.")
            answer = f"{doc_name} CÒN HIỆU LỰC."

        return LegalAnswer(
            question=f"{doc_name} còn hiệu lực không?",
            answer=answer,
            evidence=evidence,
            confidence=0.9 if result["status"] != "unknown" else 0.5,
            source_entities=[doc_id],
            reasoning_steps=steps,
        )

    def trace_amendments(self, doc_id: str) -> LegalAnswer:
        """Trace all amendments for a document."""
        chain = self.kg.amendment_chain(doc_id)
        doc_name = self.kg.G.nodes[doc_id].get("name", doc_id) if doc_id in self.kg.G else doc_id

        if not chain:
            return LegalAnswer(
                question=f"Lịch sử sửa đổi của {doc_name}",
                answer=f"{doc_name} chưa được sửa đổi/bổ sung.",
                evidence=[], confidence=1.0, source_entities=[doc_id],
                reasoning_steps=["Không tìm thấy chuỗi sửa đổi."],
            )

        steps = [f"Chuỗi sửa đổi của {doc_name}:"]
        evidence = []
        for i, item in enumerate(chain, 1):
            amending_name = self.kg.G.nodes[item["amending_doc"]].get("name", item["amending_doc"])
            relation = "sửa đổi/bổ sung" if item["relation"] == "SUA_DOI_BO_SUNG" else "thay thế"
            steps.append(f"  {i}. {amending_name} {relation} {item.get('amended_doc', '?')}")
            evidence.append(item)

        answer = f"{doc_name} có {len(chain)} văn bản sửa đổi/bổ sung."

        return LegalAnswer(
            question=f"Lịch sử sửa đổi của {doc_name}",
            answer=answer,
            evidence=evidence,
            confidence=0.85,
            source_entities=[doc_id] + [item["amending_doc"] for item in chain],
            reasoning_steps=steps,
        )

    def find_related(self, doc_id: str, max_depth: int = 2) -> LegalAnswer:
        """Find all documents related to a given document."""
        related = self.kg.related_docs(doc_id, max_depth)
        doc_name = self.kg.G.nodes[doc_id].get("name", doc_id) if doc_id in self.kg.G else doc_id

        if not related:
            return LegalAnswer(
                question=f"Văn bản liên quan đến {doc_name}",
                answer=f"Không tìm thấy văn bản liên quan đến {doc_name}.",
                evidence=[], confidence=0.3, source_entities=[doc_id],
                reasoning_steps=["Không có liên kết trong đồ thị."],
            )

        steps = [f"Văn bản liên quan đến {doc_name}:"]
        for item in related[:10]:
            relation = item["relation"].replace("_", " ")
            steps.append(f"  → {item['source_name']} --{relation}--> {item['target_name']}")

        answer = f"Tìm thấy {len(related)} văn bản liên quan đến {doc_name}."

        return LegalAnswer(
            question=f"Văn bản liên quan đến {doc_name}",
            answer=answer,
            evidence=related,
            confidence=0.8,
            source_entities=[doc_id] + [item["source"] for item in related[:10]],
            reasoning_steps=steps,
        )

    def find_reasoning_path(self, source_id: str, target_id: str) -> LegalAnswer:
        """Find the reasoning path between two legal entities."""
        path = self.kg.reasoning_path(source_id, target_id)
        source_name = self.kg.G.nodes[source_id].get("name", source_id) if source_id in self.kg.G else source_id
        target_name = self.kg.G.nodes[target_id].get("name", target_id) if target_id in self.kg.G else target_id

        if not path:
            return LegalAnswer(
                question=f"Mối liên hệ giữa {source_name} và {target_name}",
                answer=f"Không tìm thấy đường dẫn giữa {source_name} và {target_name}.",
                evidence=[], confidence=0.1, source_entities=[source_id, target_id],
                reasoning_steps=["Không có đường đi trong đồ thị."],
            )

        steps = [f"Đường dẫn từ {source_name} đến {target_name}:"]
        for i, edge in enumerate(path, 1):
            steps.append(f"  {i}. {edge['from_name']} --{edge['relation']}--> {edge['to_name']}")

        answer = f"Đường dẫn pháp lý: {' → '.join(e['relation'] for e in path)}"

        return LegalAnswer(
            question=f"Mối liên hệ giữa {source_name} và {target_name}",
            answer=answer,
            evidence=path,
            confidence=0.7,
            source_entities=[source_id, target_id],
            reasoning_steps=steps,
        )