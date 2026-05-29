from .retriever import RetrievedChunk
class PromptBuilder:

    MAX_CONTEXT_TOKENS = 400
    def build(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context_parts = [c.full_text for c in chunks]
        context = "\n".join(context_parts)
        max_chars = self.MAX_CONTEXT_TOKENS * 2
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return f"Ngữ cảnh: {context} | Câu hỏi: {question} | Trả lời:"
