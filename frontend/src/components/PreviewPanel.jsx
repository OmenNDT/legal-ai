import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Button, Tag, Tooltip, Badge, Spin, Empty } from 'antd';
import { ChevronLeft, ChevronRight, X, FileText, Highlighter, List } from 'lucide-react';
import { getDieuContent } from '../services/api';

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const stripDiacritics = (s) => (s || '').normalize('NFD').replace(/[̀-ͯ]/g, '');

const PreviewPanel = ({ document, keyword, onClose }) => {
  const contentRef = useRef(null);
  const [currentMatch, setCurrentMatch] = useState(0);
  const [activeDieu, setActiveDieu] = useState(null);
  const [dieuContent, setDieuContent] = useState(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [contentError, setContentError] = useState(null);

  const outline = document?.outline;
  const outlineLoading = document && outline === undefined;

  // Auto-pick first Điều khi outline đã load
  useEffect(() => {
    if (!outline || outline.length === 0) {
      setActiveDieu(null);
      return;
    }
    // Nếu có keyword, ưu tiên Điều đầu tiên có tiêu đề khớp
    if (keyword) {
      const kw = stripDiacritics(keyword.toLowerCase());
      const hit = outline.find(o => stripDiacritics(o.title.toLowerCase()).includes(kw));
      if (hit) {
        setActiveDieu(hit.dieu_key);
        return;
      }
    }
    setActiveDieu(outline[0].dieu_key);
  }, [document?.id, outline, keyword]);

  // Fetch nội dung khi đổi Điều
  useEffect(() => {
    if (!document?.id || !activeDieu) {
      setDieuContent(null);
      return;
    }
    let cancelled = false;
    setLoadingContent(true);
    setContentError(null);
    getDieuContent(document.id, activeDieu)
      .then(text => { if (!cancelled) setDieuContent(text); })
      .catch(err => { if (!cancelled) setContentError(err.message); })
      .finally(() => { if (!cancelled) setLoadingContent(false); });
    return () => { cancelled = true; };
  }, [document?.id, activeDieu]);

  const matches = useMemo(() => {
    if (!dieuContent || !keyword) return [];
    const regex = new RegExp(`(${escapeRegex(keyword)})`, 'gi');
    const indices = [];
    let m;
    while ((m = regex.exec(dieuContent)) !== null) indices.push(m.index);
    return indices;
  }, [dieuContent, keyword]);

  const totalMatches = matches.length;

  const scrollToMatch = useCallback((index) => {
    if (!contentRef.current || totalMatches === 0) return;
    const marks = contentRef.current.querySelectorAll('mark[data-match]');
    if (marks[index]) {
      marks[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
      setCurrentMatch(index);
    }
  }, [totalMatches]);

  const goToNext = useCallback(() => {
    if (totalMatches === 0) return;
    scrollToMatch((currentMatch + 1) % totalMatches);
  }, [currentMatch, totalMatches, scrollToMatch]);

  const goToPrev = useCallback(() => {
    if (totalMatches === 0) return;
    scrollToMatch((currentMatch - 1 + totalMatches) % totalMatches);
  }, [currentMatch, totalMatches, scrollToMatch]);

  useEffect(() => { setCurrentMatch(0); }, [activeDieu, keyword]);

  useEffect(() => {
    if (totalMatches > 0) {
      const t = setTimeout(() => scrollToMatch(0), 80);
      return () => clearTimeout(t);
    }
  }, [totalMatches, scrollToMatch]);

  if (!document) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400 bg-[#f5f2ed]">
        <FileText size={48} className="mb-4 opacity-30" />
        <p className="text-sm">Chọn một văn bản để xem trước</p>
      </div>
    );
  }

  const renderHighlighted = () => {
    if (!dieuContent) return null;
    if (!keyword) return dieuContent;
    const parts = dieuContent.split(new RegExp(`(${escapeRegex(keyword)})`, 'gi'));
    let counter = -1;
    return parts.map((part, i) => {
      if (part.toLowerCase() === keyword.toLowerCase()) {
        counter++;
        const isActive = counter === currentMatch;
        const idx = counter;
        return (
          <mark
            key={i}
            data-match={idx}
            className={`rounded px-1 cursor-pointer transition-all ${
              isActive
                ? 'bg-yellow-300 text-black ring-2 ring-yellow-500'
                : 'bg-yellow-100 text-black hover:bg-yellow-200'
            }`}
            onClick={() => scrollToMatch(idx)}
            title={`Match #${idx + 1}`}
          >
            {part}
          </mark>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="h-full flex flex-col bg-white border-l border-[#e8e4e0]">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-[#e8e4e0] bg-[#faf8f5] shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <FileText size={18} style={{ color: '#1e3a5f' }} />
          <div className="min-w-0">
            <h3 className="font-semibold text-sm truncate text-[#2d2d2d]">{document.title}</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <Tag className="rounded-full text-xs border-none" style={{ background: '#1e3a5f', color: 'white' }}>
                {document.type}
              </Tag>
              <span className="text-xs text-gray-400">{document.year}</span>
              {outline && outline.length > 0 && (
                <span className="text-xs text-gray-400">• {outline.length} điều</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {totalMatches > 0 && (
            <div className="flex items-center gap-1 bg-white rounded-lg border border-[#e8e4e0] px-2 py-1">
              <Tooltip title="Match trước">
                <Button type="text" size="small" icon={<ChevronLeft size={14} />} onClick={goToPrev} />
              </Tooltip>
              <Badge count={`${currentMatch + 1}/${totalMatches}`} style={{ backgroundColor: '#1e3a5f', fontSize: '11px' }} />
              <Tooltip title="Match tiếp">
                <Button type="text" size="small" icon={<ChevronRight size={14} />} onClick={goToNext} />
              </Tooltip>
            </div>
          )}
          <Button type="text" size="small" icon={<X size={16} />} onClick={onClose} />
        </div>
      </div>

      {keyword && dieuContent && (
        <div className="px-5 py-2 bg-yellow-50 border-b border-yellow-100 flex items-center gap-2 shrink-0">
          <Highlighter size={14} className="text-yellow-600" />
          <span className="text-xs text-yellow-700">
            <strong>{totalMatches}</strong> vị trí của "<strong>{keyword}</strong>" trong điều đang xem
          </span>
        </div>
      )}

      {/* Body: outline + content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Outline */}
        <div className="w-72 shrink-0 border-r border-[#e8e4e0] bg-[#fafaf8] flex flex-col">
          <div className="px-4 py-2 border-b border-[#e8e4e0] flex items-center gap-2 text-xs uppercase tracking-wide text-[#9a8478] shrink-0">
            <List size={14} /> Mục lục
          </div>
          <div className="flex-1 overflow-y-auto">
            {outlineLoading ? (
              <div className="flex justify-center py-6"><Spin size="small" /></div>
            ) : !outline || outline.length === 0 ? (
              <div className="p-4">
                <Empty
                  description={<span className="text-xs text-gray-500">Văn bản chưa có nội dung</span>}
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              </div>
            ) : (
              <ul className="py-1">
                {outline.map(o => {
                  const isActive = o.dieu_key === activeDieu;
                  return (
                    <li key={o.dieu_key}>
                      <button
                        onClick={() => setActiveDieu(o.dieu_key)}
                        className={`w-full text-left px-4 py-2 text-xs leading-snug border-l-2 transition-colors ${
                          isActive
                            ? 'bg-white border-[#1e3a5f] text-[#1e3a5f] font-semibold'
                            : 'border-transparent text-[#4a4a4a] hover:bg-white/60'
                        }`}
                      >
                        {o.title}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        {/* Content */}
        <div
          ref={contentRef}
          className="flex-1 overflow-y-auto p-6 text-sm leading-relaxed"
          style={{ color: '#4a4a4a', fontFamily: "'Inter', sans-serif" }}
        >
          {!activeDieu ? (
            <div className="text-gray-400 italic">Chọn một điều để xem nội dung.</div>
          ) : loadingContent ? (
            <div className="flex items-center gap-2 text-gray-400"><Spin size="small" /> Đang tải điều…</div>
          ) : contentError ? (
            <div className="text-red-500 italic">Lỗi: {contentError}</div>
          ) : (
            <div className="whitespace-pre-line">
              {renderHighlighted()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviewPanel;
