CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Danh mục văn bản pháp luật
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_code VARCHAR(100) UNIQUE NOT NULL,
    doc_name TEXT NOT NULL,
    doc_type VARCHAR(50),
    issue_year INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Kho Khoản/Điểm + Vector Embedding
CREATE TABLE IF NOT EXISTS law_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    hierarchy_path TEXT,
    phan VARCHAR(100),
    chuong VARCHAR(100),
    muc VARCHAR(100),
    dieu VARCHAR(100) NOT NULL,
    khoan VARCHAR(50),
    diem VARCHAR(50),
    content TEXT NOT NULL,
    full_text TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Dữ liệu Hỏi-Đáp (đã có sẵn, ghi bởi Spark)
CREATE TABLE IF NOT EXISTS qa_data (
    id BIGINT,
    id1 BIGINT,
    question TEXT,
    answer TEXT,
    idx BIGINT
);

-- 4. Ánh xạ Q&A ↔ Khoản luật + Prompt đóng gói
CREATE TABLE IF NOT EXISTS training_samples (
    id SERIAL PRIMARY KEY,
    qa_id_ref BIGINT,
    chunk_id_ref INT REFERENCES law_chunks(id) ON DELETE SET NULL,
    similarity_score FLOAT,
    formatted_prompt TEXT,
    target_answer TEXT,
    is_validated BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Nhật ký tương tác thực tế
CREATE TABLE IF NOT EXISTS inference_logs (
    id BIGSERIAL PRIMARY KEY,
    user_question TEXT NOT NULL,
    retrieved_chunk_ids INT[],
    generated_answer TEXT,
    latency_ms INT,
    user_feedback INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_law_chunks_embedding ON law_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_law_chunks_content_trgm ON law_chunks USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_law_chunks_doc_id ON law_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_law_chunks_dieu ON law_chunks(dieu);
CREATE INDEX IF NOT EXISTS idx_training_samples_qa ON training_samples(qa_id_ref);
CREATE INDEX IF NOT EXISTS idx_training_samples_chunk ON training_samples(chunk_id_ref);
CREATE INDEX IF NOT EXISTS idx_inference_logs_created ON inference_logs(created_at DESC);
