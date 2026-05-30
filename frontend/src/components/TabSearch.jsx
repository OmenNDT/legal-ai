import { useState, useEffect } from 'react';
import { Input, Card, Tag, Spin, Empty, Typography } from 'antd';
import { Search, BookOpen, ArrowRight } from 'lucide-react';
import { searchLaws, getAllLaws, getLawOutline } from '../services/api';
import ResultHighlight from './ResultHighlight';
import PreviewPanel from './PreviewPanel';

const { Search: AntSearch } = Input;
const { Title, Text } = Typography;

const TabSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [isResizing, setIsResizing] = useState(false);
  const [featuredDocs, setFeaturedDocs] = useState([]);
  const [featuredLoading, setFeaturedLoading] = useState(true);

  useEffect(() => {
    getAllLaws()
      .then(docs => setFeaturedDocs(docs))
      .catch(() => {})
      .finally(() => setFeaturedLoading(false));
  }, []);

  const docId = selectedDoc?.id;
  const hasOutline = selectedDoc?.outline !== undefined;
  useEffect(() => {
    if (!docId || hasOutline) return;
    let cancelled = false;
    console.log('[TabSearch] fetching outline for doc', docId);
    getLawOutline(docId)
      .then(full => {
        if (cancelled) return;
        console.log('[TabSearch] outline loaded', full.outline?.length, 'điều');
        setSelectedDoc(prev => {
          if (!prev || String(prev.id) !== String(full.id)) return prev;
          return { ...prev, outline: full.outline, chunk_count: full.chunk_count };
        });
      })
      .catch(err => {
        if (cancelled) return;
        console.error('[TabSearch] outline error', err);
        setSelectedDoc(prev => {
          if (!prev || String(prev.id) !== String(docId)) return prev;
          return { ...prev, outline: [], outlineError: err.message };
        });
      });
    return () => { cancelled = true; };
  }, [docId, hasOutline]);

  const handleSearch = async (value) => {
    setLoading(true);
    setHasSearched(true);
    setSelectedDoc(null);
    const data = await searchLaws(value);
        setResults(data);
    setLoading(false);
    if (data && data.length > 0) {
      setSelectedDoc(data[0]);
    } else {
      setSelectedDoc(null);
    }
  };

  const suggestions = featuredDocs.slice(0, 4).map(d => d.title);

    // Resize sidebar
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;
      const newWidth = Math.max(320, Math.min(600, e.clientX));
      setSidebarWidth(newWidth);
    };
    const handleMouseUp = () => setIsResizing(false);
    
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);


  return (
  <div className="h-[calc(100vh-64px)] flex flex-col">
    {hasSearched ? (
      <div className="flex-1 flex overflow-hidden">
        
        {/* Sidebar trái - search + list kết quả */}
        <div 
          className="shrink-0 flex flex-col overflow-hidden"
          style={{ 
            width: `${sidebarWidth}px`,
            background: 'linear-gradient(180deg, #1e3a5f 0%, #2d1f3e 60%, #4a1520 100%)' 
          }}
        >
          <div className="p-5 shrink-0">
            <div className="text-white font-bold text-base mb-5 flex items-center gap-2">
              <BookOpen size={20} style={{ color: '#e8d5b7' }} />
              <span className="font-['Playfair_Display']">Tra cứu Luật</span>
            </div>
            
            <AntSearch
              placeholder="Nhập từ khóa..."
              allowClear
              enterButton={<Search size={14} />}
              size="middle"
              value={query}
              onSearch={handleSearch}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full"
            />
            
            <div className="mt-3 text-white/50 text-xs">
              {loading ? 'Đang tìm...' : `${results.length} kết quả cho "${query}"`}
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto px-5 pb-5">
            <Spin spinning={loading} size="small">
              {results.length > 0 ? (
                <div className="flex flex-col gap-4">
                  {results.map((item, idx) => (
                    <Card 
                      key={idx}
                      className={`w-full hover:shadow-lg transition-all duration-200 border-none rounded-xl overflow-hidden cursor-pointer ${selectedDoc?.title === item.title ? 'ring-2 ring-[#e8d5b7]' : ''}`}
                      style={{ background: 'rgba(255,255,255,0.95)' }}
                      bodyStyle={{ padding: 0 }}
                      onClick={() => setSelectedDoc(item)}
                    >
                      <div className="p-4">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1.5">
                              <Tag 
                                className="rounded-full text-xs font-medium border-none"
                                style={{ background: '#1e3a5f', color: 'white' }}
                              >
                                {item.type}
                              </Tag>
                              <span className="text-gray-400 text-xs">{item.year}</span>
                            </div>
                            <h3 
                              className="text-sm font-bold truncate"
                              style={{ color: '#2d2d2d' }}
                            >
                              <ResultHighlight text={item.title} keyword={query} />
                            </h3>
                          </div>
                          <ArrowRight 
                            size={16} 
                            className={`shrink-0 ml-2 mt-1 transition-all ${selectedDoc?.title === item.title ? 'translate-x-1 text-[#1e3a5f]' : 'text-[#9a8478]'}`} 
                          />
                        </div>
                        
                        <div className="line-clamp-2 text-xs leading-relaxed" style={{ color: '#6b5b5e' }}>
                          <ResultHighlight text={item.content} keyword={query} />
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                !loading && (
                  <div className="text-center py-8">
                    <Empty 
                      description={<span className="text-white/60 text-xs">Không tìm thấy kết quả</span>}
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                    />
                  </div>
                )
              )}
            </Spin>
          </div>
        </div>

        {/* Thanh kéo thả resize */}
        <div
          className="w-1.5 shrink-0 cursor-col-resize hover:bg-[#1e3a5f]/30 active:bg-[#1e3a5f]/50 transition-colors relative group"
          style={{ background: 'transparent' }}
          onMouseDown={() => setIsResizing(true)}
          title="Kéo để điều chỉnh độ rộng"
        >
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-0.5 bg-[#1e3a5f]/20 group-hover:bg-[#1e3a5f]/40" />
          
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="flex flex-col gap-0.5">
              <div className="w-1 h-1 rounded-full bg-[#1e3a5f]" />
              <div className="w-1 h-1 rounded-full bg-[#1e3a5f]" />
              <div className="w-1 h-1 rounded-full bg-[#1e3a5f]" />
            </div>
          </div>
        </div>
        
        {/* Preview bên phải */}
        <div className="flex-1 bg-[#faf9f7] overflow-hidden">
          {selectedDoc ? (
            <PreviewPanel 
              document={selectedDoc} 
              keyword={query} 
              onClose={() => setSelectedDoc(null)} 
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <BookOpen size={48} className="mb-4 opacity-20" />
              <p className="text-sm">Chọn một văn bản để xem trước</p>
            </div>
          )}
        </div>
      </div>
    ) : (
      /* ========== CHƯA SEARCH ========== */
      <div className="flex-1 overflow-y-auto">
        
        {/* Hero Section */}
        <div 
          className="relative overflow-hidden pb-6 pt-8"
          style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #2d1f3e 50%, #8b1a2b 100%)' }}
        >
          
          <div className="relative z-10 max-w-3xl mx-auto px-6 pt-6 pb-8 text-center">
            <div 
              className="inline-flex items-center gap-2 backdrop-blur-sm px-4 py-2 rounded-full mb-3 border -translate-y-5"
              style={{ background: 'rgba(255,255,255,0.1)', borderColor: 'rgba(255,255,255,0.2)' }}
            >
              <Search size={16} style={{ color: '#e8d5b7' }} />
              <span className="text-white/90 text-sm font-medium">Kho văn bản pháp luật Việt Nam</span>
            </div>
            
            <Title level={2} className="text-white mb-1 font-['Playfair_Display'] text-3xl! md:text-4xl!" style={{ color: '#ffffff' }}>
              Tra cứu văn bản pháp luật
            </Title>
            <Text className="text-white/90 text-base block mb-5" style={{ color: 'rgba(255,255,255,0.9)' }}>
              Tìm kiếm trong hệ thống văn bản Hiến pháp, Bộ luật và Luật
            </Text>

            <div className="max-w-2xl mx-auto mt-4 translate-y-5">
              <AntSearch
                placeholder="Nhập từ khóa tra cứu..."
                allowClear
                enterButton={
                  <span className="flex items-center gap-2 font-medium">
                    <Search size={16} /> Tra cứu
                  </span>
                }
                size="large"
                onSearch={handleSearch}
                onChange={(e) => setQuery(e.target.value)}
                className="shadow-2xl"
              />
              
              <div className="mt-6 flex flex-wrap justify-center gap-3 translate-y-4">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setQuery(s);
                      handleSearch(s);
                    }}
                    className="px-4 py-2.5 rounded-full text-sm transition-all flex items-center gap-2 border"
                    style={{ 
                      background: 'rgba(255,255,255,0.1)', 
                      color: 'rgba(255,255,255,0.8)',
                      borderColor: 'rgba(255,255,255,0.15)'
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = 'rgba(255,255,255,0.2)';
                      e.target.style.color = 'white';
                      e.target.style.borderColor = 'rgba(255,255,255,0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = 'rgba(255,255,255,0.1)';
                      e.target.style.color = 'rgba(255,255,255,0.8)';
                      e.target.style.borderColor = 'rgba(255,255,255,0.15)';
                    }}
                  >
                    <Search size={14} /> {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Featured Laws */}
        <div className="max-w-5xl mx-auto px-6 pb-12" style={{ background: '#faf9f7' }}>
          <div className="pt-8">
            <Text
              className="text-sm uppercase tracking-widest mb-6 block text-center font-semibold"
              style={{ color: '#9a8478', borderBottom: 'none' }}
            >
              Văn bản nổi bật
            </Text>
            <Spin spinning={featuredLoading}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {featuredDocs.map((law, i) => (
                  <Card
                    key={law.id || i}
                    className="hover:shadow-lg transition-all cursor-pointer rounded-xl border-none"
                    style={{ background: '#ffffff' }}
                    bodyStyle={{ padding: '14px 16px' }}
                    onClick={() => {
                      setQuery(law.title);
                      handleSearch(law.title);
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)';
                    }}
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
                        style={{ background: 'rgba(232,213,183,0.3)' }}
                      >
                        <BookOpen size={20} style={{ color: '#c9a96e' }} />
                      </div>
                      <div className="min-w-0">
                        <div className="text-[#2d2d2d] font-semibold text-sm truncate">{law.title}</div>
                        <div className="text-xs mt-0.5" style={{ color: '#9a8478' }}>
                          {law.type}{law.year ? ` ${law.year}` : ''}
                          {law.chunk_count > 0 ? ` • ${law.chunk_count.toLocaleString()} điều khoản` : ''}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Spin>
          </div>
        </div>
      </div>
    )}
  </div>
  );
};

export default TabSearch;