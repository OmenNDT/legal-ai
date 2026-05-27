import { Search, MessageSquare, FileText, ArrowRight, Scale } from 'lucide-react';


const LandingPage = ({ onNavigate }) => {
  const features = [
    {
      icon: <Search size={18} className="text-[#8b1a2b]" />,
      title: 'Tra cứu Luật',
      description: 'Tìm kiếm văn bản pháp luật Việt Nam với tính năng highlight từ khóa.',
      action: 'search',
    },
    {
      icon: <MessageSquare size={18} className="text-[#8b1a2b]" />,
      title: 'Trợ lý pháp lý',
      description: 'Trợ lý ảo thông minh, giải đáp thắc mắc về các lĩnh vực pháp luật cơ bản.',
      action: 'ai-assistant',
      intent: 'chat', 
    },
    {
      icon: <FileText size={18} className="text-[#8b1a2b]" />,
      title: 'Tóm tắt văn bản',
      description: 'Tóm tắt nội dung chính của các văn bản pháp luật quan trọng nhất.',
      action: 'ai-assistant',
      intent: 'summarize', 
    }
  ];

  return (
    <div className="font-sans"> 
      {/* Hero Section */}
      <div 
        className="relative overflow-hidden pb-8 md:pb-12"
        style={{ 
          background: 'linear-gradient(135deg, #1e3a5f 0%, #2d1f3e 50%, #8b1a2b 100%)',
        }}
      >
        
        <div className="relative z-10 max-w-6xl mx-auto px-6 pt-6 pb-10 md:pt-8 md:pb-12">
          <div className="max-w-2xl">
            <div 
              className="inline-flex items-center gap-2 backdrop-blur-sm px-4 py-2 rounded-full mb-8 border"
              style={{ background: 'rgba(255,255,255,0.1)', borderColor: 'rgba(255,255,255,0.2)' }}
            >
              <Scale size={16} style={{ color: '#e8d5b7' }} />
              <span className="text-white/90 text-sm font-medium">Hệ thống tra cứu pháp luật Việt Nam</span>
            </div>
            
            <h1 
              className="text-4xl md:text-6xl font-extrabold text-white tracking-normal"
              style={{ 
                fontFamily: '"Playfair Display", Georgia, serif', // [FIX] Thêm fallback font
                textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
                lineHeight: 1.4,
                display: 'block',
                marginBottom: '1.5rem',
              }}
            >
              Đồng hành pháp lý,<br />
              <span style={{ color: '#e8d5b7' }}>Kiến tạo niềm tin</span>
            </h1>
            
            <p className="text-lg text-white/80 mb-8 leading-relaxed max-w-lg">
              Tra cứu nhanh chóng, chính xác các văn bản pháp luật Hiến pháp, Dân sự, Hình sự, Lao động và Đất đai.
            </p>
            
            <div className="flex gap-12 flex-wrap items-center">
              <button 
                type="button"
                className="h-14 px-10 text-base font-bold rounded-lg flex items-center gap-3 shadow-xl transition-all hover:scale-105 active:scale-95"
                style={{ background: '#1e3a5f', color: '#ffffff' }}
                onClick={() => onNavigate('search')}
              >
                Bắt đầu tra cứu <ArrowRight size={18} />
              </button>
              
              <button 
                type="button"
                className="h-14 px-10 text-base font-bold rounded-lg flex items-center gap-3 shadow-xl transition-all hover:scale-105 active:scale-95 border border-white/20"
                style={{ 
                  background: '#8b1a2b', 
                  color: 'white',
                  fontFamily: 'sans-serif',
                }}
                onClick={() => onNavigate('ai-assistant')}
              >
                Trợ lý pháp lý
              </button>
            </div>
          </div>
          
          <div className="absolute right-8 top-1/2 -translate-y-1/2 opacity-10 hidden lg:block">
            <Scale size={280} className="text-white" />
          </div>
        </div>
        
        {/* Wave SVG */}
        <div 
          className="absolute bottom-0 left-0 right-0 z-10" 
          style={{ 
            bottom: '-2px',
            lineHeight: 0 
          }}
        >
          <svg 
            viewBox="0 0 1440 80" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg" 
            className="w-full h-auto block"
          >
            <path 
              d="M0 60 
                C 200 30, 400 120, 600 70 
                C 800 80, 1100 120, 1440 60 
                V 150 H 0 Z" 
              fill="#faf9f7"/>
          </svg>
        </div>
      </div>

      {/* Feature Section */}
      <div className="max-w-7xl mx-auto px-6 -mt-10 relative z-20">
        <div className="bg-white rounded-xl shadow-xl border border-[#e8e4e0] p-4 grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature, idx) => (
            <div 
              key={idx} 
              className="flex items-start gap-4 p-4 py-0 rounded-lg hover:bg-[#faf9f7] transition-colors"
            >
              <div className="w-10 h-10 shrink-0 rounded-full bg-[#1e3a5f]/10 flex items-center justify-center text-[#1e3a5f]">
                {feature.icon}
              </div>
              
              <div className="flex flex-col h-full flex-1">
                <div className="flex-1">
                  <div className="text-xl font-bold text-[#2d2d2d] leading-none mb-3">
                    {feature.title}
                  </div>
                  
                  <p className="text-[13px] font-medium text-[#6b5b5e] leading-relaxed mb-3">
                    {feature.description}
                  </p>
                </div>

                
                <button
                  type="button"
                  className="text-[#1e3a5f] font-bold text-sm flex items-center gap-1 mt-auto cursor-pointer hover:underline text-left bg-transparent border-none p-0"
                  onClick={() => onNavigate(feature.action, feature.intent)}
                > 
                  Khám phá <ArrowRight size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LandingPage;