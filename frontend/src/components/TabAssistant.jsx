import { useState, useEffect, useRef } from 'react';
import { Drawer, Tooltip } from 'antd';
import { MessageSquare, Plus, Trash2, Clock, BookOpen, ChevronDown, Paperclip, X, FileText, BookMarked, Loader2, BarChart3 } from 'lucide-react';
import { askLegalAssistant, getChatStatus } from '../services/api';
import TabEval from './TabEval';

const formatTime = (timestamp) => {
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'Vừa xong';
  if (minutes < 60) return `${minutes} phút trước`;
  const hours = Math.floor(diff / 3600000);
  if (hours < 24) return `${hours} giờ trước`;
  const days = Math.floor(diff / 86400000);
  if (days < 7) return `${days} ngày trước`;
  return new Date(timestamp).toLocaleDateString('vi-VN');
};

/* ============================================================
   CHAT CONTENT
   ============================================================ */
const ChatContent = ({ chat, onUpdateChat, defaultIntent }) => {
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [sending, setSending] = useState(false);
  const [modelStatus, setModelStatus] = useState({ status: 'idle', mode: 'extractive', error: null });
  const [evalOpen, setEvalOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Poll model status khi không ready
  useEffect(() => {
    let alive = true;
    let timer = null;
    const tick = async () => {
      try {
        const s = await getChatStatus();
        if (!alive) return;
        setModelStatus(s);
        if (s.status === 'ready' || s.status === 'error') return;
        timer = setTimeout(tick, 2500);
      } catch {
        if (!alive) return;
        timer = setTimeout(tick, 5000);
      }
    };
    tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, []);

  const chatId = chat?.id;
  const isEmptyChat = chat?.messages?.length === 0;
  // Apply the intent prefill exactly once per (chatId, defaultIntent) pair.
  const [appliedIntentKey, setAppliedIntentKey] = useState(null);
  const intentKey = `${chatId}::${defaultIntent}`;
  if (
    defaultIntent === 'summarize' &&
    isEmptyChat &&
    appliedIntentKey !== intentKey
  ) {
    setAppliedIntentKey(intentKey);
    setInput('Tóm tắt văn bản: ');
  }

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollToBottom(); }, [chat?.messages]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setFileName(selectedFile.name);
  };

  const clearFile = () => {
    setFile(null);
    setFileName('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSend = async () => {
    const question = input.trim();
    if (!question || sending) return;

    const userMsg = {
      role: 'user',
      text: question,
      hasFile: !!file,
      fileName,
      time: Date.now(),
    };

    const newMessages = [...(chat?.messages || []), userMsg];
    const isFirstMessage = (chat?.messages?.length || 0) === 0;
    const title = isFirstMessage
      ? question.slice(0, 40) + (question.length > 40 ? '...' : '')
      : (chat?.title || 'Cuộc trò chuyện mới');

    onUpdateChat({
      ...chat,
      title,
      messages: newMessages,
      timestamp: Date.now(),
    });
    setInput('');
    clearFile();
    setSending(true);

    try {
      const res = await askLegalAssistant(question);
      const aiMsg = {
        role: 'assistant',
        text: res.answer,
        chunks: res.chunks || [],
        found: res.found,
        mode: res.mode,
        latency_ms: res.latency_ms,
        time: Date.now(),
      };
      onUpdateChat(prev => ({ ...prev, messages: [...prev.messages, aiMsg], timestamp: Date.now() }));
    } catch (err) {
      const isLoading = err.code === 'loading';
      const aiMsg = {
        role: 'assistant',
        text: isLoading
          ? '⏳ Mô hình RAG đang khởi tạo (lần đầu cần ~30 giây để nạp BGE-M3). Vui lòng gửi lại câu hỏi sau ít phút.'
          : `Lỗi: ${err.message || err}`,
        isError: !isLoading,
        time: Date.now(),
      };
      onUpdateChat(prev => ({ ...prev, messages: [...prev.messages, aiMsg], timestamp: Date.now() }));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#faf9f7]">
      {/* Header */}
      <div className="px-6 py-4 bg-white border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-[#1e3a5f]">{chat?.title || 'Cuộc trò chuyện mới'}</h2>
            <p className="text-xs text-gray-400">RAG trên 12.000+ điều khoản · BGE-M3 + BARTpho</p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {modelStatus.status === 'ready' && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                Sẵn sàng · {modelStatus.mode === 'lora' ? 'BARTpho-LoRA' : 'Extractive (chưa train LoRA)'}
              </span>
            )}
            {modelStatus.status === 'loading' && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                <Loader2 size={11} className="animate-spin" /> Đang nạp BGE-M3...
              </span>
            )}
            {modelStatus.status === 'error' && (
              <span className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200" title={modelStatus.error}>
                Lỗi khởi tạo
              </span>
            )}
            {/* Eval button — tạm ẩn, bỏ comment để bật lại
            <Tooltip title="Xem đánh giá retrieval (Recall@K, MRR, NDCG, Answer-overlap)">
              <button
                type="button"
                onClick={() => setEvalOpen(true)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border border-gray-200 bg-white text-[#722F37] hover:bg-[#fdf4f5] hover:border-[#722F37] transition-colors"
              >
                <BarChart3 size={13} />
                Đánh giá Retrieval
              </button>
            </Tooltip>
            */}
          </div>
        </div>
      </div>

      {/* Eval Drawer — tạm ẩn, bỏ comment để bật lại
      <Drawer
        title={null}
        placement="right"
        width={Math.min(1280, typeof window !== 'undefined' ? window.innerWidth - 80 : 1280)}
        open={evalOpen}
        onClose={() => setEvalOpen(false)}
        styles={{ body: { padding: 0, background: '#faf9f7' } }}
        destroyOnClose
      >
        <TabEval />
      </Drawer>
      */}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {(!chat?.messages || chat.messages.length === 0) && (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 gap-3">
            <MessageSquare size={40} className="text-gray-300" />
            <p className="text-sm font-medium">Trợ lý pháp lý AI</p>
            <div className="text-xs text-gray-400 text-center space-y-1">
              <p>• Hỏi đáp về luật Việt Nam</p>
              <p>• Paste văn bản để tóm tắt</p>
              <p>• Upload file PDF/DOC/TXT</p>
            </div>
          </div>
        )}
        {chat?.messages?.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
              msg.role === 'user'
                ? 'bg-[#722F37] text-white rounded-br-md'
                : msg.isError
                  ? 'bg-red-50 text-red-700 border border-red-200 rounded-bl-md'
                  : 'bg-white text-gray-700 border border-gray-200 rounded-bl-md shadow-sm'
            }`}>
              {msg.hasFile && (
                <div className={`flex items-center gap-2 mb-1.5 pb-1.5 border-b ${msg.role === 'user' ? 'border-white/20' : 'border-gray-200'}`}>
                  <FileText size={14} />
                  <span className="text-xs font-medium">{msg.fileName}</span>
                </div>
              )}
              <div className="whitespace-pre-line">{msg.text}</div>
              {msg.role === 'assistant' && msg.chunks && msg.chunks.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-gray-400 mb-2">
                    <BookMarked size={12} /> Căn cứ pháp lý ({msg.chunks.length})
                  </div>
                  <ul className="space-y-1.5">
                    {msg.chunks.map((c, i) => (
                      <li key={i} className="text-xs text-gray-600 leading-relaxed">
                        <span className="font-semibold text-[#1e3a5f]">[{i + 1}]</span>{' '}
                        <span className="font-medium">{c.doc_name || 'Văn bản'}</span>
                        {c.dieu && <> · {c.dieu}</>}
                        {c.khoan && <>, {c.khoan}</>}
                        {c.diem && <>, {c.diem}</>}
                        <span className="text-gray-400"> · score {Number(c.similarity).toFixed(2)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {msg.role === 'assistant' && typeof msg.latency_ms === 'number' && (
                <div className="mt-2 text-[10px] text-gray-400">
                  {msg.latency_ms} ms · {msg.mode === 'lora' ? 'BARTpho-LoRA' : 'Extractive'}
                  {msg.found === false && ' · không tìm thấy thông tin'}
                </div>
              )}
            </div>
          </div>
        ))}
        {sending && (
          <div className="flex justify-start">
            <div className="px-4 py-2.5 rounded-2xl bg-white border border-gray-200 rounded-bl-md shadow-sm flex items-center gap-2 text-sm text-gray-500">
              <Loader2 size={14} className="animate-spin" />
              Đang truy vấn cơ sở pháp luật...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* file upload */}
      <div className="px-6 py-4 bg-white border-t border-gray-200">
        {file && (
          <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200 w-fit">
            <FileText size={14} className="text-[#722F37]" />
            <span className="text-xs text-gray-700">{fileName}</span>
            <button onClick={clearFile} className="p-0.5 rounded hover:bg-gray-200 ml-1">
              <X size={12} className="text-gray-500" />
            </button>
          </div>
        )}
        
        <div className="flex gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2.5 rounded-xl border border-gray-200 text-gray-500 hover:text-[#722F37] hover:border-[#722F37]/30 transition-all"
            title="Đính kèm file"
          >
            <Paperclip size={18} />
          </button>
          <input ref={fileInputRef} type="file" accept=".pdf,.doc,.docx,.txt" onChange={handleFileChange} className="hidden" />
          
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Nhập tình huống / câu hỏi pháp luật..."
            disabled={sending}
            className="flex-1 px-4 py-2.5 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-[#1e3a5f]/20 focus:border-[#1e3a5f] bg-gray-50 transition-all disabled:opacity-60"
          />
          <button
            onClick={handleSend}
            disabled={sending || !input.trim()}
            className="px-5 py-2.5 rounded-xl bg-[#722F37] text-white! text-sm font-medium hover:bg-[#5a252c] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {sending && <Loader2 size={14} className="animate-spin" />} Gửi
          </button>
        </div>
      </div>
    </div>
  );
};

/* ============================================================
   MAIN COMPONENT
   ============================================================ */
const buildEmptyChat = () => {
  const now = Date.now();
  return {
    id: 'chat-' + now,
    title: 'Cuộc trò chuyện mới',
    timestamp: now,
    messages: [],
  };
};

const HISTORY_STORAGE_KEY = 'legal-ai:chat-history';
const ACTIVE_STORAGE_KEY = 'legal-ai:active-chat-id';

const loadStoredHistory = () => {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0 && parsed.every(c => c && c.id && Array.isArray(c.messages))) {
      return parsed;
    }
  } catch (e) {
    console.warn('Failed to restore chat history:', e);
  }
  return null;
};

const loadStoredActiveId = () => {
  try {
    return localStorage.getItem(ACTIVE_STORAGE_KEY) || null;
  } catch {
    return null;
  }
};

const buildInitialChatHistory = () => loadStoredHistory() || [buildEmptyChat()];

const TabAssistant = ({ defaultIntent }) => {
  const [chatHistory, setChatHistory] = useState(buildInitialChatHistory);
  const [activeChatId, setActiveChatId] = useState(loadStoredActiveId);
  useEffect(() => {
    if (activeChatId === null && chatHistory[0]) setActiveChatId(chatHistory[0].id);
  }, [activeChatId, chatHistory]);

  // Persist chat history + active chat id whenever they change.
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(chatHistory));
    } catch (e) {
      console.warn('Failed to persist chat history:', e);
    }
  }, [chatHistory]);
  useEffect(() => {
    try {
      if (activeChatId) localStorage.setItem(ACTIVE_STORAGE_KEY, activeChatId);
    } catch {
      // ignore
    }
  }, [activeChatId]);
  const [historyExpanded, setHistoryExpanded] = useState(true);

  const activeChat = chatHistory.find(c => c.id === activeChatId);

  const createNewChat = () => {
    const newChat = buildEmptyChat();
    setChatHistory(prev => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat;
  };


  const updateChat = (updatedChatOrFn) => {
    setChatHistory(prev => {
      // Functional mode (dùng cho AI response tránh stale closure)
      if (typeof updatedChatOrFn === 'function') {
        return prev.map(c => {
          if (c.id !== activeChatId) return c;
          const updated = updatedChatOrFn(c);
          return updated;
        });
      }
      
      // Object mode: nếu chat chưa có trong history thì thêm mới
      const exists = prev.some(c => c.id === updatedChatOrFn.id);
      if (!exists) return [updatedChatOrFn, ...prev];
      return prev.map(c => c.id === updatedChatOrFn.id ? updatedChatOrFn : c);
    });
  };


  const deleteChat = (e, id) => {
    e.stopPropagation();
    const filtered = chatHistory.filter(c => c.id !== id);
    
    if (filtered.length === 0) {
      const newChat = buildEmptyChat();
      setChatHistory([newChat]);
      setActiveChatId(newChat.id);
    } else {
      setChatHistory(filtered);
      if (activeChatId === id) setActiveChatId(filtered[0].id);
    }
  };

  // Keep activeChatId in sync with the available history. Done in render
  // (store-previous pattern) to avoid setState inside an effect.
  const activeExists = chatHistory.some(c => c.id === activeChatId);
  if (!activeExists) {
    if (chatHistory.length > 0) {
      const firstId = chatHistory[0].id;
      if (activeChatId !== firstId) setActiveChatId(firstId);
    } else {
      const newChat = buildEmptyChat();
      setChatHistory([newChat]);
      setActiveChatId(newChat.id);
    }
  }

  return (
    <div className="h-[calc(100vh-64px)] flex overflow-hidden bg-[#faf9f7]">
      
      {/* Sidebar */}
      <aside className="w-64 shrink-0 flex flex-col border-r border-gray-200"
      style={{ 
        background: 'linear-gradient(180deg, #1e3a5f 0%, #2d1f3e 60%, #4a1520 100%)' 
      }}
    >
        
        {/* New Chat */}
        <div className="p-3">
          <button 
            onClick={createNewChat}
            className="w-full h-10 rounded-lg flex items-center justify-center gap-2 text-sm font-medium transition-all bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 hover:border-gray-300 active:scale-[0.98] shadow-sm"
          >
            <Plus size={16} className="text-[#2b23c0]" strokeWidth={2.5} />
            <span>Cuộc trò chuyện mới</span>
          </button>
        </div>

        {/* LỊCH SỬ */}
        <div className="flex-1 min-h-0 flex flex-col">
          <button 
            onClick={() => setHistoryExpanded(!historyExpanded)}
            className="w-full pb-2 pt-3 flex items-center justify-between hover:bg-gray-50 rounded-lg transition-colors px-3"
          >
            <div className="flex items-center gap-2">
              <span className="text-[#e8d5b7] text-[11px] uppercase tracking-wider font-bold">Lịch sử</span>
              <span className="text-[10px] text-gray-300 bg-gray-100 px-1.5 py-0.5 rounded-full">{chatHistory.length}</span>
            </div>
            <ChevronDown 
              size={16} 
              className={`text-gray-400 transition-transform duration-200 ${historyExpanded ? 'rotate-180' : ''}`}
            />
          </button>
          
          <div className={`flex-1 overflow-hidden transition-all duration-300 ${historyExpanded ? 'opacity-100' : 'max-h-0 opacity-0'}`}>
            <div className="h-full overflow-y-auto px-3 space-y-0.5">
              {chatHistory.map(chat => (
                <div
                  key={chat.id}
                  onClick={() => setActiveChatId(chat.id)}
                  className={`group flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
                    activeChatId === chat.id ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5'
                  }`}
                >
                  <MessageSquare size={15} className={activeChatId === chat.id ? 'text-[#e8d5b7]' : 'text-white/50'} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{chat.title}</div>
                    <div className="text-[11px] text-gray-400 flex items-center gap-1">
                      <Clock size={10} />
                      {formatTime(chat.timestamp)}
                    </div>
                  </div>
                  <button onClick={(e) => deleteChat(e, chat.id)} className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200">
                    <Trash2 size={13} className="text-gray-400 hover:text-red-500" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-100">
          <div className="flex items-center gap-2 text-gray-400">
            <BookOpen size={14} />
            <span className="text-xs">Hệ thống tra cứu pháp luật</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-hidden">
        <ChatContent 
          chat={activeChat || { id: 'temp', title: 'Cuộc trò chuyện mới', messages: [] }} 
          onUpdateChat={updateChat}
          defaultIntent={defaultIntent}
        />
      </main>
    </div>
  );
};

export default TabAssistant;