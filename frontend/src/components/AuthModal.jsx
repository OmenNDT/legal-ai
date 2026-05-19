import { useState } from 'react';
import { X, User, Lock, Mail, Eye, EyeOff, Scale, KeyRound, ArrowLeft } from 'lucide-react';
import { login, register, requestPasswordReset, resetPassword } from '../services/auth';

const MODES = {
  LOGIN: 'login',
  REGISTER: 'register',
  FORGOT: 'forgot',
  RESET: 'reset',
};

const titleFor = (mode) => ({
  [MODES.LOGIN]: 'Đăng nhập',
  [MODES.REGISTER]: 'Tạo tài khoản',
  [MODES.FORGOT]: 'Quên mật khẩu',
  [MODES.RESET]: 'Đặt lại mật khẩu',
}[mode]);

const subtitleFor = (mode) => ({
  [MODES.LOGIN]: 'Truy cập hệ thống tra cứu pháp luật',
  [MODES.REGISTER]: 'Tham gia cộng đồng pháp lý',
  [MODES.FORGOT]: 'Nhập email để nhận mã OTP',
  [MODES.RESET]: 'Nhập mã OTP và mật khẩu mới',
}[mode]);

const AuthModal = ({ isOpen, onClose, onLogin, dismissible = true }) => {
  const [mode, setMode] = useState(MODES.LOGIN);
  const [showPass, setShowPass] = useState(false);
  const [showPass2, setShowPass2] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [form, setForm] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: '',
    otp: '',
  });

  if (!isOpen) return null;

  const resetState = () => {
    setForm({ email: '', password: '', confirmPassword: '', name: '', otp: '' });
    setError('');
    setInfo('');
    setShowPass(false);
    setShowPass2(false);
  };

  const switchMode = (next) => {
    setMode(next);
    setError('');
    setInfo('');
  };

  const handleClose = () => {
    resetState();
    setMode(MODES.LOGIN);
    onClose();
  };

  const onField = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const validateLocal = () => {
    if (mode === MODES.REGISTER) {
      if (!form.name.trim()) return 'Vui lòng nhập họ và tên.';
      if (form.password.length < 6) return 'Mật khẩu phải có ít nhất 6 ký tự.';
      if (form.password !== form.confirmPassword) return 'Mật khẩu xác nhận không khớp.';
    }
    if (mode === MODES.RESET) {
      if (!/^\d{6}$/.test(form.otp.trim())) return 'Mã OTP phải gồm 6 chữ số.';
      if (form.password.length < 6) return 'Mật khẩu phải có ít nhất 6 ký tự.';
      if (form.password !== form.confirmPassword) return 'Mật khẩu xác nhận không khớp.';
    }
    return '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setInfo('');

    const localErr = validateLocal();
    if (localErr) { setError(localErr); return; }

    setLoading(true);
    try {
      if (mode === MODES.LOGIN) {
        const user = await login(form.email, form.password);
        onLogin(user);
        handleClose();
      } else if (mode === MODES.REGISTER) {
        const user = await register(form.name, form.email, form.password);
        onLogin(user);
        handleClose();
      } else if (mode === MODES.FORGOT) {
        const res = await requestPasswordReset(form.email);
        const devNote = res?.dev_otp ? ` (DEV OTP: ${res.dev_otp})` : '';
        setInfo(`Nếu email tồn tại, mã OTP đã được gửi.${devNote}`);
        setMode(MODES.RESET);
      } else if (mode === MODES.RESET) {
        await resetPassword(form.email, form.otp, form.password);
        setInfo('Đặt lại mật khẩu thành công. Vui lòng đăng nhập.');
        setMode(MODES.LOGIN);
        setForm({ email: form.email, password: '', confirmPassword: '', name: '', otp: '' });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitLabel = {
    [MODES.LOGIN]: 'Đăng nhập',
    [MODES.REGISTER]: 'Đăng ký',
    [MODES.FORGOT]: 'Gửi mã OTP',
    [MODES.RESET]: 'Đặt lại mật khẩu',
  }[mode];

  const showName = mode === MODES.REGISTER;
  const showOtp = mode === MODES.RESET;
  const showPassword = mode === MODES.LOGIN || mode === MODES.REGISTER || mode === MODES.RESET;
  const showConfirm = mode === MODES.REGISTER || mode === MODES.RESET;
  const showBack = mode === MODES.FORGOT || mode === MODES.RESET;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={dismissible ? handleClose : undefined}
      />

      <div className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200 mx-4">
        {/* Header */}
        <div
          className="px-8 pt-8 pb-12 text-center relative"
          style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #2d1f3e 50%, #722F37 100%)' }}
        >
          {showBack && (
            <button
              type="button"
              onClick={() => switchMode(MODES.LOGIN)}
              className="absolute top-4 left-4 text-white/60 hover:text-white transition-colors"
            >
              <ArrowLeft size={20} />
            </button>
          )}
          {dismissible && (
            <button
              type="button"
              onClick={handleClose}
              className="absolute top-4 right-4 text-white/60 hover:text-white transition-colors"
            >
              <X size={20} />
            </button>
          )}

          <div className="w-14 h-14 mx-auto mb-3 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-white/20">
            <Scale size={28} className="text-[#e8d5b7]" />
          </div>
          <h2 className="text-xl font-bold text-white font-['Playfair_Display']">{titleFor(mode)}</h2>
          <p className="text-white/70 text-sm mt-1">{subtitleFor(mode)}</p>
        </div>

        {/* Body */}
        <div className="px-8 py-6 -mt-6 bg-white rounded-t-2xl relative">
          <form onSubmit={handleSubmit} className="space-y-4">
            {showName && (
              <Field label="Họ và tên" icon={<User size={16} />}>
                <input
                  type="text" required value={form.name} onChange={onField('name')}
                  placeholder="Nguyễn Văn A"
                  className={inputCls}
                />
              </Field>
            )}

            <Field label="Email" icon={<Mail size={16} />}>
              <input
                type="email" required value={form.email} onChange={onField('email')}
                placeholder="you@example.com"
                disabled={mode === MODES.RESET}
                className={inputCls + (mode === MODES.RESET ? ' bg-gray-100' : '')}
              />
            </Field>

            {showOtp && (
              <Field label="Mã OTP (6 chữ số)" icon={<KeyRound size={16} />}>
                <input
                  type="text" inputMode="numeric" maxLength={6} required
                  value={form.otp} onChange={onField('otp')}
                  placeholder="123456"
                  className={inputCls + ' tracking-widest font-mono'}
                />
              </Field>
            )}

            {showPassword && (
              <Field label={mode === MODES.RESET ? 'Mật khẩu mới' : 'Mật khẩu'} icon={<Lock size={16} />}>
                <input
                  type={showPass ? 'text' : 'password'} required minLength={6}
                  value={form.password} onChange={onField('password')}
                  placeholder="••••••••"
                  className={inputCls + ' pr-10'}
                />
                <button
                  type="button" onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </Field>
            )}

            {showConfirm && (
              <Field label="Xác nhận mật khẩu" icon={<Lock size={16} />}>
                <input
                  type={showPass2 ? 'text' : 'password'} required minLength={6}
                  value={form.confirmPassword} onChange={onField('confirmPassword')}
                  placeholder="••••••••"
                  className={inputCls + ' pr-10'}
                />
                <button
                  type="button" onClick={() => setShowPass2(!showPass2)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPass2 ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </Field>
            )}

            {error && (
              <div className="px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs">
                {error}
              </div>
            )}
            {info && (
              <div className="px-3 py-2 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs">
                {info}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-[#722F37] !text-white text-sm font-semibold hover:bg-[#5a252c] active:scale-[0.98] transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : submitLabel}
            </button>
          </form>

          {/* Mode switches */}
          <div className="mt-5 text-center text-sm text-gray-500 space-y-1">
            {mode === MODES.LOGIN && (
              <>
                <p>
                  Chưa có tài khoản?{' '}
                  <button type="button" onClick={() => switchMode(MODES.REGISTER)}
                    className="text-[#722F37] font-semibold hover:underline">
                    Đăng ký ngay
                  </button>
                </p>
                <p>
                  <button type="button" onClick={() => switchMode(MODES.FORGOT)}
                    className="text-[#1e3a5f] text-xs hover:underline">
                    Quên mật khẩu?
                  </button>
                </p>
              </>
            )}
            {mode === MODES.REGISTER && (
              <p>
                Đã có tài khoản?{' '}
                <button type="button" onClick={() => switchMode(MODES.LOGIN)}
                  className="text-[#722F37] font-semibold hover:underline">
                  Đăng nhập
                </button>
              </p>
            )}
            {mode === MODES.FORGOT && (
              <p>
                Đã có mã OTP?{' '}
                <button type="button" onClick={() => switchMode(MODES.RESET)}
                  className="text-[#722F37] font-semibold hover:underline">
                  Đặt lại mật khẩu
                </button>
              </p>
            )}
            {mode === MODES.RESET && (
              <p>
                Chưa nhận được mã?{' '}
                <button type="button" onClick={() => switchMode(MODES.FORGOT)}
                  className="text-[#722F37] font-semibold hover:underline">
                  Gửi lại OTP
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const inputCls = "w-full pl-10 pr-4 py-2.5 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-[#1e3a5f]/20 focus:border-[#1e3a5f] bg-gray-50 transition-all";

const Field = ({ label, icon, children }) => (
  <div className="space-y-1.5">
    <label className="text-sm font-medium text-gray-700">{label}</label>
    <div className="relative">
      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">{icon}</span>
      {children}
    </div>
  </div>
);

export default AuthModal;
