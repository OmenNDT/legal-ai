from typing import List

# Chia văn bản dài thành các chunk vừa giới hạn token đầu vào của BART
class LongDocChunker:
    def __init__(self, tokenizer, max_tokens: int = 1024, overlap: int = 3):
        # tokenizer của HuggingFace dùng để đếm token chính xác
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        # Overlap nhỏ (3-5 câu) đủ giữ ngữ cảnh, tránh tạo quá nhiều chunk trùng lặp
        self.overlap = overlap

    # Chia theo câu trước, dồn câu vào chunk tới khi vượt giới hạn thì sang chunk mới
    def chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        chunks = []
        buf = []
        buf_tokens = 0
        for sent in sentences:
            n = len(self.tokenizer.encode(sent, add_special_tokens = False))
            # Câu quá dài đơn lẻ thì cắt cứng theo token
            if n >= self.max_tokens:
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                chunks.append(self._hard_cut(sent))
                continue
            if buf_tokens + n > self.max_tokens:
                chunks.append(" ".join(buf))
                # Giữ lại vài câu cuối làm overlap để giữ ngữ cảnh
                buf = buf[-self.overlap:] if self.overlap > 0 else []
                buf_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens = False)) for s in buf)
            buf.append(sent)
            buf_tokens += n
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    # Cắt cứng theo token khi một câu vượt max_tokens
    def _hard_cut(self, text: str) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens = False)[: self.max_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens = True)
