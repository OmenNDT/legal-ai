CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

CREATE TABLE public.legal_documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    document_number VARCHAR(50),
    document_type VARCHAR(50),
    issuing_body TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_legal_documents_001 ON public.legal_documents(document_number);

CREATE TABLE public.document_versions (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES public.legal_documents(id) ON DELETE CASCADE,
    version_number INT NOT NULL,
    effective_date DATE,
    expiry_date DATE,
    status VARCHAR(20),
    raw_text TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_id, version_number)
);

CREATE INDEX idx_document_versions_001 ON public.document_versions(document_id);
CREATE INDEX idx_document_versions_002 ON public.document_versions(status);

CREATE TABLE public.legal_sections (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES public.document_versions(id) ON DELETE CASCADE,
    article_number VARCHAR(20),
    clause_number VARCHAR(20),
    point VARCHAR(10),
    title TEXT,
    content TEXT,
    parent_id BIGINT,
    level INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_legal_sections_001 ON public.legal_sections(version_id);

CREATE TABLE public.legal_chunks (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES public.document_versions(id) ON DELETE CASCADE,
    section_id BIGINT REFERENCES public.legal_sections(id) ON DELETE SET NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INT,
    embedding_id VARCHAR(100),
    token_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_legal_chunks_001 ON public.legal_chunks(version_id);
CREATE INDEX idx_legal_chunks_002 ON public.legal_chunks(section_id);
CREATE INDEX idx_chunks_text_search ON public.legal_chunks USING GIN (to_tsvector('simple', chunk_text));

CREATE TABLE public.legal_metadata_tags (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES public.legal_chunks(id) ON DELETE CASCADE,
    tag_key VARCHAR(50),
    tag_value VARCHAR(100)
);

CREATE INDEX idx_metadata_tags_001 ON public.legal_metadata_tags(chunk_id);
CREATE INDEX idx_metadata_tags_002 ON public.legal_metadata_tags(tag_key, tag_value);

CREATE TABLE public.user_queries (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    normalized_query TEXT,
    intent VARCHAR(50),
    entities JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_user_queries_001 ON public.user_queries USING GIN (entities);

CREATE TABLE public.retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL REFERENCES public.user_queries(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES public.legal_chunks(id) ON DELETE CASCADE,
    score FLOAT,
    rank INT,
    retrieval_type VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_retrieval_logs_001 ON public.retrieval_logs(query_id);
CREATE INDEX idx_retrieval_logs_002 ON public.retrieval_logs(chunk_id);

CREATE TABLE public.legal_reasoning_results (
    id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL REFERENCES public.user_queries(id) ON DELETE CASCADE,
    facts TEXT,
    legal_basis TEXT,
    analysis TEXT,
    conclusion TEXT,
    recommendation TEXT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_reasoning_001 ON public.legal_reasoning_results(query_id);

CREATE TABLE public.citations (
    id BIGSERIAL PRIMARY KEY,
    reasoning_id BIGINT NOT NULL REFERENCES public.legal_reasoning_results(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES public.legal_chunks(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_citations_001 ON public.citations(reasoning_id);
CREATE INDEX idx_citations_002 ON public.citations(chunk_id);

CREATE TABLE public.evaluation_logs (
    id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL REFERENCES public.user_queries(id) ON DELETE CASCADE,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    feedback TEXT,
    evaluated_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eval_001 ON public.evaluation_logs(query_id);

CREATE TABLE public.chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    started_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP
);

CREATE TABLE public.chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20),
    message TEXT,
    query_id BIGINT,
    reasoning_id BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chat_msg_001 ON public.chat_messages(session_id);