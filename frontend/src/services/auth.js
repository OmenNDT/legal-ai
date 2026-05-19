import { api } from './api';

const TOKEN_KEY = 'legal_token';
const USER_KEY = 'legal_user';

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const getStoredUser = () => {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

const persist = (user, token) => {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
};

export const clearAuth = () => {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
};

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

const extractError = (err, fallback = 'Có lỗi xảy ra. Vui lòng thử lại.') => {
  const msg = err?.response?.data?.error;
  return new Error(msg || fallback);
};

export const login = async (email, password) => {
  try {
    const { data } = await api.post('/auth/login', { email, password });
    persist(data.user, data.token);
    return data.user;
  } catch (err) {
    throw extractError(err, 'Đăng nhập thất bại.');
  }
};

export const register = async (name, email, password) => {
  try {
    const { data } = await api.post('/auth/register', { name, email, password });
    persist(data.user, data.token);
    return data.user;
  } catch (err) {
    throw extractError(err, 'Đăng ký thất bại.');
  }
};

export const requestPasswordReset = async (email) => {
  try {
    const { data } = await api.post('/auth/forgot-password', { email });
    return data; // { sent: true, dev_otp?: "123456" }
  } catch (err) {
    throw extractError(err, 'Không gửi được mã OTP.');
  }
};

export const resetPassword = async (email, otp, password) => {
  try {
    const { data } = await api.post('/auth/reset-password', { email, otp, password });
    return data;
  } catch (err) {
    throw extractError(err, 'Đặt lại mật khẩu thất bại.');
  }
};

export const fetchCurrentUser = async () => {
  try {
    const { data } = await api.get('/auth/me');
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    return data.user;
  } catch (err) {
    if (err?.response?.status === 401) clearAuth();
    throw extractError(err, 'Không lấy được thông tin tài khoản.');
  }
};

export const deleteAccount = async () => {
  try {
    await api.delete('/auth/me');
    clearAuth();
  } catch (err) {
    throw extractError(err, 'Xoá tài khoản thất bại.');
  }
};
