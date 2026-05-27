import axios from 'axios';

const RAG_API_KEY = import.meta.env.VITE_RAG_API_KEY || '';

const ragApi = axios.create({
  baseURL: '/api/rag-extract',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': RAG_API_KEY,
  },
});

export const setRagApiKey = (key) => {
  ragApi.defaults.headers['X-API-Key'] = key;
};

export const getRagHealth = async () => {
  const { data } = await ragApi.get('/health');
  return data;
};

export const getRagStats = async () => {
  const { data } = await ragApi.get('/stats');
  return data;
};

export const queryRag = async (question, nResults = 5) => {
  const { data } = await ragApi.post('/query', { question, n_results: nResults });
  return data;
};

export const listRagDocuments = async (skip = 0, limit = 100) => {
  const { data } = await ragApi.get('/documents', { params: { skip, limit } });
  return data;
};

export const getRagDocument = async (docId) => {
  const { data } = await ragApi.get(`/documents/${docId}`);
  return data;
};

export const getRagIndexStats = async () => {
  const { data } = await ragApi.get('/index/stats');
  return data;
};

export const uploadDocuments = async (files, options = {}) => {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  if (options.displayName) formData.append('display_name', options.displayName);
  if (options.standardCode) formData.append('standard_code', options.standardCode);
  if (options.language) formData.append('language', options.language);

  const { data } = await ragApi.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const deleteDocument = async (docId) => {
  const { data } = await ragApi.delete(`/documents/${docId}`);
  return data;
};

export const processDocument = async (docId) => {
  const { data } = await ragApi.post(`/documents/${docId}/process`);
  return data;
};
