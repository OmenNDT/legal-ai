from src.pipeline.retrieval import Chunk

class ReRanking:
    TOP_N = 10
    MAX_CONTEXT_TOKENS = 3000

    def __init__(self, cross_encoder=None, summarizer=None):
        self._cross_encoder = cross_encoder
        self._summarizer = summarizer

    def process(self, candidates: list, structured_query) -> str:
        reranked = self._rerank(candidates, structured_query.rewritten_query)
        top_n = reranked[:self.TOP_N]
        deduped = self._deduplicate(top_n)
        grouped = self._group_by_document(deduped)
        structured = self._add_legal_structure(grouped)
        injected = self._inject_metadata(structured)
        compressed = self._compress_context(injected)
        return self._build_prompt(compressed, structured_query)

    def _rerank(self, candidates: list, query: str) -> list:
        if not candidates:
            return []
        if self._cross_encoder is None:
            return sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)
        pairs = [(query, c.content) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.hybrid_score = float(score)
        return sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)

    def _deduplicate(self, chunks: list) -> list:
        seen_ids = set()
        seen_sigs = set()
        result = []
        for chunk in chunks:
            sig = chunk.content[:100]
            if chunk.chunk_id not in seen_ids and sig not in seen_sigs:
                seen_ids.add(chunk.chunk_id)
                seen_sigs.add(sig)
                result.append(chunk)
        return result

    def _group_by_document(self, chunks: list) -> list:
        doc_groups: dict = {}
        doc_order = []
        for chunk in chunks:
            key = chunk.document_title
            if key not in doc_groups:
                doc_groups[key] = []
                doc_order.append(key)
            doc_groups[key].append(chunk)
        grouped = []
        for key in doc_order:
            group = sorted(doc_groups[key], key=lambda c: (c.article_number, c.clause_number))
            grouped.extend(group)
        return grouped

    def _add_legal_structure(self, chunks: list) -> list:
        result = []
        for chunk in chunks:
            label_parts = []
            if chunk.article_number:
                label_parts.append(f"Điều {chunk.article_number}")
            if chunk.clause_number:
                label_parts.append(f"Khoản {chunk.clause_number}")
            if chunk.point_number:
                label_parts.append(f"Điểm {chunk.point_number}")
            label = ", ".join(label_parts) if label_parts else "Nội dung"
            result.append({
                "chunk": chunk,
                "label": label,
                "structured_content": f"[{label}] {chunk.content}",
            })
        return result

    def _inject_metadata(self, structured: list) -> list:
        for item in structured:
            chunk = item["chunk"]
            meta_parts = [f"Văn bản: {chunk.document_title}"]
            if chunk.document_type:
                meta_parts.append(f"Loại: {chunk.document_type}")
            if chunk.effective_status and chunk.effective_status != "active":
                meta_parts.append(f"Hiệu lực: {chunk.effective_status}")
            item["metadata_header"] = " | ".join(meta_parts)
            item["full_context"] = f"{item['metadata_header']}\n{item['structured_content']}"
        return structured

    def _compress_context(self, structured: list) -> list:
        total_tokens = 0
        result = []
        for item in structured:
            tokens = len(item["full_context"].split())
            if total_tokens + tokens > self.MAX_CONTEXT_TOKENS:
                if self._summarizer:
                    compressed = self._summarizer.summarize(
                        document=item["full_context"],
                        query="",
                        top_k=2,
                        use_model=False,
                    )["summary"]
                    item["full_context"] = compressed
                    tokens = len(compressed.split())
                else:
                    words = item["full_context"].split()
                    item["full_context"] = " ".join(words[:200]) + "..."
                    tokens = 200
            total_tokens += tokens
            result.append(item)
            if total_tokens >= self.MAX_CONTEXT_TOKENS:
                break
        return result

    def _build_prompt(self, structured: list, structured_query) -> str:
        system = (
            "Bạn là chuyên gia pháp lý Việt Nam. Hãy trả lời câu hỏi dựa trên các văn bản pháp luật "
            "được cung cấp. Trả lời theo cấu trúc: **Căn cứ pháp lý** → **Phân tích** → **Kết luận** → **Khuyến nghị**."
        )
        context_parts = [f"[{i+1}] {item['full_context']}" for i, item in enumerate(structured)]
        context = "\n\n".join(context_parts)
        return (
            f"SYSTEM: {system}\n\n"
            f"VĂN BẢN PHÁP LÝ:\n{context}\n\n"
            f"CÂU HỎI: {structured_query.rewritten_query}\n\n"
            f"TRẢ LỜI:"
        )
