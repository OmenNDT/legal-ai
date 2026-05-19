import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Button, Tag, Tooltip, Badge } from 'antd';
import { ChevronLeft, ChevronRight, X, FileText, Highlighter } from 'lucide-react';

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const PreviewPanel = ({ document, keyword, onClose }) => {
  const contentRef = useRef(null);
  const [currentMatch, setCurrentMatch] = useState(0);
  const [prevResetKey, setPrevResetKey] = useState(`${document?.id}::${keyword}`);

  const fullContent = document?.fullContent;

  const matches = useMemo(() => {
    if (!fullContent || !keyword) return [];
    const regex = new RegExp(`(${escapeRegex(keyword)})`, 'gi');
    const indices = [];
    let match;
    while ((match = regex.exec(fullContent)) !== null) {
      indices.push(match.index);
    }
    return indices;
  }, [fullContent, keyword]);

  const totalMatches = matches.length;

  const scrollToMatch = useCallback((index) => {
    if (!contentRef.current || totalMatches === 0) return;
    const markElements = contentRef.current.querySelectorAll('mark[data-match]');
    if (markElements[index]) {
      markElements[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
      setCurrentMatch(index);
    }
  }, [totalMatches]);

  const goToNext = useCallback(() => {
    if (totalMatches === 0) return;
    const next = (currentMatch + 1) % totalMatches;
    scrollToMatch(next);
  }, [currentMatch, totalMatches, scrollToMatch]);

  const goToPrev = useCallback(() => {
    if (totalMatches === 0) return;
    const prev = (currentMatch - 1 + totalMatches) % totalMatches;
    scrollToMatch(prev);
  }, [currentMatch, totalMatches, scrollToMatch]);

  // Reset currentMatch when document or keyword changes. This uses the
  // store-previous-prop pattern (https://react.dev/reference/react/useState#storing-information-from-previous-renders)
  // to avoid setState inside an effect.
  const currentResetKey = `${document?.id}::${keyword}`;
  if (prevResetKey !== currentResetKey) {
    setPrevResetKey(currentResetKey);
    setCurrentMatch(0);
  }

  useEffect(() => {
    if (totalMatches > 0) {
      const t = setTimeout(() => scrollToMatch(0), 100);
      return () => clearTimeout(t);
    }
  }, [document?.id, keyword, totalMatches, scrollToMatch]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'F3' || (e.key === 'Enter' && e.shiftKey)) {
        e.preventDefault();
        goToNext();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToNext]);

  if (!document) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400 bg-[#f5f2ed]">
        <FileText size={48} className="mb-4 opacity-30" />
        <p className="text-sm">Chọn một văn bản để xem trước</p>
      </div>
    );
  }

  const renderHighlightedContent = () => {
    if (!fullContent || !keyword) return fullContent;

    const parts = fullContent.split(new RegExp(`(${escapeRegex(keyword)})`, 'gi'));
    let matchCounter = -1;

    return parts.map((part, i) => {
      if (part.toLowerCase() === keyword.toLowerCase()) {
        matchCounter++;
        const isActive = matchCounter === currentMatch;
        const idx = matchCounter;
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
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {totalMatches > 0 && (
            <div className="flex items-center gap-1 bg-white rounded-lg border border-[#e8e4e0] px-2 py-1">
              <Tooltip title="Match trước">
                <Button type="text" size="small" icon={<ChevronLeft size={14} />} onClick={goToPrev} />
              </Tooltip>
              <Badge
                count={`${currentMatch + 1}/${totalMatches}`}
                style={{ backgroundColor: '#1e3a5f', fontSize: '11px' }}
              />
              <Tooltip title="Match tiếp theo">
                <Button type="text" size="small" icon={<ChevronRight size={14} />} onClick={goToNext} />
              </Tooltip>
            </div>
          )}
          <Button type="text" size="small" icon={<X size={16} />} onClick={onClose} />
        </div>
      </div>

      {/* Toolbar */}
      {keyword && (
        <div className="px-5 py-2 bg-yellow-50 border-b border-yellow-100 flex items-center gap-2 shrink-0">
          <Highlighter size={14} className="text-yellow-600" />
          <span className="text-xs text-yellow-700">
            Đang hiển thị <strong>{totalMatches}</strong> vị trí của "<strong>{keyword}</strong>"
          </span>
        </div>
      )}

      {/* Content */}
      <div
        ref={contentRef}
        className="flex-1 overflow-y-auto p-6 text-sm leading-relaxed"
        style={{ color: '#4a4a4a', fontFamily: "'Inter', sans-serif" }}
      >
        <div className="whitespace-pre-line">
          {renderHighlightedContent()}
        </div>
      </div>
    </div>
  );
};

export default PreviewPanel;
