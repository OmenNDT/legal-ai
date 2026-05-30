import axios from 'axios';

// Instance axios dùng chung. Mọi request sẽ qua /api/* và được Vite proxy về Flask 9010.
export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Lấy danh sách / tìm kiếm documents từ PostgreSQL
export const searchLaws = async (query = '') => {
  const { data } = await api.get('/laws', { params: query ? { q: query } : {} });
  return (data.documents || []).map(d => ({
    id: String(d.id),
    title: d.doc_name,
    type: d.doc_type,
    year: d.issue_year,
    doc_code: d.doc_code,
    chunk_count: d.chunk_count,
    content: `${d.doc_type} số ${d.doc_code}${d.issue_year ? ' năm ' + d.issue_year : ''}`,
  }));
};

export const getAllLaws = () => searchLaws('');

export const getLawOutline = async (id) => {
  const { data } = await api.get(`/laws/${id}`);
  if (!data.ok) throw new Error(data.error || 'Không tải được mục lục');
  const d = data.document;
  return {
    id: String(d.id),
    title: d.doc_name,
    type: d.doc_type,
    year: d.issue_year,
    doc_code: d.doc_code,
    chunk_count: d.chunk_count,
    outline: d.outline || [],
  };
};

export const getDieuContent = async (id, dieuKey) => {
  const { data } = await api.get(`/laws/${id}/dieu`, { params: { key: dieuKey } });
  if (!data.ok) throw new Error(data.error || 'Không tải được nội dung Điều');
  return data.content;
};

// ===== Eval metrics =====
export const fetchEvalMetrics = async () => {
  const { data } = await api.get('/eval/metrics', { validateStatus: () => true });
  return data; // {ok, data: {test_set_size, ks, retrievers, doc_names}, generated_at} | {ok: false, error}
};

// ===== Trợ lý pháp lý (RAG) =====
export const askLegalAssistant = async (question) => {
  const res = await api.post('/chat', { question }, { timeout: 60000, validateStatus: () => true });
  // Khi backend đang load model trả 503 với status: 'loading' — bubble lên cho UI hiển thị.
  if (res.status === 503) {
    const err = new Error(res.data?.message || res.data?.error || 'Mô hình đang khởi tạo');
    err.code = res.data?.status || 'loading';
    throw err;
  }
  if (!res.data?.ok) {
    throw new Error(res.data?.error || `HTTP ${res.status}`);
  }
  return res.data; // {found, answer, chunks, latency_ms, mode}
};

export const getChatStatus = async () => {
  const { data } = await api.get('/chat/status');
  return data; // {status, mode, error}
};
