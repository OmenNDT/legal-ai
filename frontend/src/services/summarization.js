import axios from 'axios';

// Instance riêng cho text summarisation — prefix /api (proxy → :9010)
const sumApi = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

// Đính token JWT nếu có (dùng cùng key với auth.js)
sumApi.interceptors.request.use((config) => {
  const token = localStorage.getItem('legal_ai_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Danh sách extractor hiển thị trong UI
export const EXTRACTORS = [
  { value: 'tfidf', label: 'TF-IDF', desc: 'Chấm điểm câu theo trọng số từ' },
  { value: 'textrank', label: 'TextRank', desc: 'PageRank trên đồ thị câu' },
  { value: 'kmeans', label: 'K-Means', desc: 'Gom cụm + chọn câu trung tâm' },
  { value: 'ensemble', label: 'Ensemble', desc: 'Kết hợp cả 3 thuật toán' },
];

// Lấy danh sách doc_id (có hỗ trợ search và phân trang)
export async function listDocuments({ q = '', page = 1, pageSize = 50 } = {}) {
  const { data } = await sumApi.get('/documents', {
    params: { q, page, page_size: pageSize },
  });
  return data;
}

// Lấy nội dung 1 doc + reference
export async function getDocument(docId) {
  const { data } = await sumApi.get(`/documents/${encodeURIComponent(docId)}`);
  return data;
}

// Gọi pipeline lai cho 1 văn bản hoặc 1 doc_id
export async function summarize({ text, docId, extractor = 'textrank', useAbstractive = true }) {
  const payload = {
    text,
    doc_id: docId,
    extractor,
    use_abstractive: useAbstractive,
  };
  const { data } = await sumApi.post('/summarize', payload);
  return data;
}

// Chạy chỉ extractive (nhanh hơn vì không cần BART)
export async function extractOnly({ text, docId, extractor = 'textrank' }) {
  const payload = { text, doc_id: docId, extractor };
  const { data } = await sumApi.post('/extract', payload);
  return data;
}

// Chạy song song 4 extractor (tfidf, textrank, kmeans, ensemble) để so sánh
export async function compareAll({ text, docId, useAbstractive = false }) {
  const methods = ['tfidf', 'textrank', 'kmeans', 'ensemble'];
  const results = await Promise.all(
    methods.map((m) =>
      summarize({ text, docId, extractor: m, useAbstractive })
        .then((data) => ({ method: m, ok: true, data }))
        .catch((err) => ({ method: m, ok: false, error: err?.response?.data?.error || err.message }))
    )
  );
  return results;
}

// Kiểm tra trạng thái backend
export async function health() {
  const { data } = await sumApi.get('/health');
  return data;
}
