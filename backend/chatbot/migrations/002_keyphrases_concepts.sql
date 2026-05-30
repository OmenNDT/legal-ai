-- Migration 002: Thêm bảng keyphrases và concept_tags
-- Chạy: psql $DATABASE_URL -f migrations/002_keyphrases_concepts.sql

-- ──────────────────────────────────────────────────────
-- Bảng keyphrases: lưu keyphrase TF-IDF của từng chunk
-- ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keyphrases (
    id         SERIAL PRIMARY KEY,
    chunk_id   INTEGER NOT NULL REFERENCES law_chunks(id) ON DELETE CASCADE,
    phrase     TEXT    NOT NULL,
    rank       SMALLINT NOT NULL DEFAULT 1,  -- thứ hạng trong chunk (1 = quan trọng nhất)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunk_phrase UNIQUE (chunk_id, phrase)
);

-- Index tra cứu theo chunk
CREATE INDEX IF NOT EXISTS idx_keyphrases_chunk ON keyphrases(chunk_id);

-- Index trigram trên phrase để fuzzy search
CREATE INDEX IF NOT EXISTS idx_keyphrases_phrase_trgm
    ON keyphrases USING gin(phrase gin_trgm_ops);

-- ──────────────────────────────────────────────────────
-- Bảng concept_tags: nhãn loại khái niệm pháp lý
-- ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS concept_tags (
    id           SERIAL PRIMARY KEY,
    chunk_id     INTEGER NOT NULL REFERENCES law_chunks(id) ON DELETE CASCADE,
    concept_type VARCHAR(30) NOT NULL,  -- dinh_nghia | nghia_vu | quyen_loi | cam_ket | che_tai | thu_tuc | to_chuc | pham_vi | nguyen_tac | bao_cao | khac
    confidence   REAL NOT NULL DEFAULT 0.5,  -- 0.0 – 1.0
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunk_concept UNIQUE (chunk_id, concept_type)
);

-- Index tra cứu theo chunk
CREATE INDEX IF NOT EXISTS idx_concept_tags_chunk ON concept_tags(chunk_id);

-- Index tìm theo loại khái niệm
CREATE INDEX IF NOT EXISTS idx_concept_tags_type ON concept_tags(concept_type);

-- Index kết hợp (concept_type, confidence) cho query lọc ngưỡng
CREATE INDEX IF NOT EXISTS idx_concept_tags_type_conf ON concept_tags(concept_type, confidence DESC);

-- ──────────────────────────────────────────────────────
-- Cột source trên qa_data (nếu chưa có)
-- ──────────────────────────────────────────────────────
ALTER TABLE qa_data ADD COLUMN IF NOT EXISTS source VARCHAR(64);
CREATE INDEX IF NOT EXISTS idx_qa_data_source ON qa_data(source);

-- Kiểm tra các loại concept_type hợp lệ (CHECK không block INSERT nhưng document)
COMMENT ON COLUMN concept_tags.concept_type IS
    'dinh_nghia | nghia_vu | quyen_loi | cam_ket | che_tai | thu_tuc | to_chuc | pham_vi | nguyen_tac | bao_cao | khac';
