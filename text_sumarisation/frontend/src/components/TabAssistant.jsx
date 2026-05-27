import { useState, useEffect, useRef } from 'react';
import { MessageSquare, Plus, Trash2, Clock, BookOpen, ChevronDown, Paperclip, X, FileText } from 'lucide-react';

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
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

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

  const handleSend = () => {
    if (!input.trim() && !file) return;

    const userMsg = { 
      role: 'user', 
      text: input.trim() || `Tóm tắt file: ${fileName}`,
      hasFile: !!file,
      fileName,
      time: Date.now() 
    };
    
    const newMessages = [...(chat?.messages || []), userMsg];
    const title = chat?.messages?.length === 0 
      ? (input.trim() || fileName).slice(0, 30) + ((input.trim() || fileName).length > 30 ? '...' : '')
      : chat?.title;


    onUpdateChat({ ...chat, title, messages: newMessages });

    // Mock AI response
    setTimeout(() => {
      const isSummary = input.trim().toLowerCase().includes('tóm tắt') || !!file;
      const aiMsg = { 
        role: 'assistant', 
        text: isSummary 
          ? `📄 Tôi đã đọc nội dung. Đây là bản tóm tắt:\n\n1. **Quy định chung**: Nội dung chính của văn bản...\n2. **Điểm mới**: Các thay đổi so với trước...\n3. **Áp dụng**: Phạm vi và đối tượng...\n\nBạn có muốn hỏi thêm chi tiết gì không?`
          : `Đây là câu trả lời cho câu hỏi của bạn. Nếu cần tóm tắt văn bản nào, bạn có thể gửi nội dung hoặc upload file nhé!`,
        time: Date.now() 
      };
      
  
      onUpdateChat(prev => ({ ...prev, messages: [...prev.messages, aiMsg] }));
    }, 800);

    setInput('');
    clearFile();
  };

  return (
    <div className="h-full flex flex-col bg-[#faf9f7]">
      {/* Header */}
      <div className="px-6 py-4 bg-white border-b border-gray-200">
        <h2 className="text-base font-semibold text-[#1e3a5f]">{chat?.title || 'Cuộc trò chuyện mới'}</h2>
        <p className="text-xs text-gray-400">Hỏi đáp hoặc tóm tắt văn bản pháp luật</p>
      </div>

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
            <div className={`max-w-[75%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-[#722F37] text-white rounded-br-md' 
                : 'bg-white text-gray-700 border border-gray-200 rounded-bl-md shadow-sm'
            }`}>
              {msg.hasFile && (
                <div className={`flex items-center gap-2 mb-1.5 pb-1.5 border-b ${msg.role === 'user' ? 'border-white/20' : 'border-gray-200'}`}>
                  <FileText size={14} />
                  <span className="text-xs font-medium">{msg.fileName}</span>
                </div>
              )}
              <div className="whitespace-pre-line">{msg.text}</div>
            </div>
          </div>
        ))}
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
            placeholder="Nhập câu hỏi hoặc yêu cầu tóm tắt..."
            className="flex-1 px-4 py-2.5 rounded-xl border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-[#1e3a5f]/20 focus:border-[#1e3a5f] bg-gray-50 transition-all"
          />
          <button 
            onClick={handleSend}
            className="px-5 py-2.5 rounded-xl bg-[#722F37] text-white! text-sm font-medium hover:bg-[#5a252c] transition-colors"
          >
            Gửi
          </button>
        </div>
      </div>
    </div>
  );
};

/* ============================================================
   MAIN COMPONENT
   ============================================================ */
const buildInitialChatHistory = () => {
  const now = Date.now();
  return [
    {
      id: 'demo-1',
      title: 'Luật đất đai 2024 có gì mới?',
      timestamp: now - 3600000,
      messages: [
        { role: 'user', text: 'Luật đất đai 2024 có gì mới?', time: now - 3600000 },
        { role: 'assistant', text: 'Luật Đất đai 2024 có nhiều điểm mới về thời hạn sử dụng đất...', time: now - 3590000 },
      ],
    },
  ];
};

const buildEmptyChat = () => {
  const now = Date.now();
  return {
    id: 'chat-' + now,
    title: 'Cuộc trò chuyện mới',
    timestamp: now,
    messages: [],
  };
};

const TabAssistant = ({ defaultIntent }) => {
  const [chatHistory, setChatHistory] = useState(buildInitialChatHistory);
  const [activeChatId, setActiveChatId] = useState('demo-1');
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