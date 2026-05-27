import { Menu, Layout, Tooltip, Dropdown } from 'antd';
import { Search, MessageSquare, Scale, Home, User, LogOut, ChevronDown, Cpu, Database } from 'lucide-react';

const { Header } = Layout;

const AppHeader = ({ activeTab, onTabChange, user, onOpenAuth, onLogout }) => {
  const mainMenuItems = [
    {
      key: 'home',
      icon: <Home size={16} />,
      label: 'Trang chủ',
    },
    {
      key: 'search',
      icon: <Search size={16} />,
      label: 'Tra cứu Luật',
    },
    {
      key: 'ai-assistant',
      icon: <MessageSquare size={16} />,
      label: 'Trợ lý pháp lý',
    },
    {
      key: 'string-matching',
      icon: <Cpu size={16} />,
      label: 'Tìm kiếm văn bản',
    },
    {
      key: 'rag-extract',
      icon: <Database size={16} />,
      label: 'RAG Extract',
    },
  ];

  const userMenuItems = [
    {
      key: 'profile',
      label: 'Hồ sơ cá nhân',
      icon: <User size={14} />,
    },
    {
      key: 'logout',
      label: 'Đăng xuất',
      icon: <LogOut size={14} className="text-red-500" />,
    },
  ];

  const handleUserMenuClick = ({ key }) => {
    if (key === 'profile') onTabChange('profile');
    if (key === 'logout') onLogout();
  };

  return (
    <Header 
      className="shadow-sm px-6 flex items-center justify-between sticky top-0 z-50"
      style={{ 
        height: 64, 
        lineHeight: '64px',
        background: '#faf9f7'
      }}
    >
      {/* Logo Section */}
      <div 
        className="flex items-center gap-3 cursor-pointer shrink-0"
        onClick={() => onTabChange('home')}
        style={{ minWidth: 180 }}
      >
        <div className="p-2 rounded-lg shadow-md shrink-0" style={{ background: '#722F37' }}>
          <Scale className="text-white" size={22} />
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-lg font-bold whitespace-nowrap font-['Playfair_Display']" style={{ color: '#2d2d2d' }}>
            Tra cứu Luật
          </span>
          <span className="text-[10px] uppercase tracking-widest font-medium" style={{ color: '#722F37' }}>
            Việt Nam
          </span>
        </div>
      </div>
      
      {/* Menu & Auth Section */}
      <div className="flex-1 flex justify-end items-center gap-4">
        <Menu
          mode="horizontal"
          selectedKeys={[activeTab]}
          onClick={(e) => onTabChange(e.key)}
          items={mainMenuItems}
          className="border-none font-medium text-sm bg-transparent"
          style={{ 
            justifyContent: 'flex-end',
            lineHeight: '62px',
            minWidth: 0,
            flex: 1
          }}
          overflowedIndicator={null} 
        />
        
        {/* USER */}
        {user ? (
          <Dropdown 
            menu={{ items: userMenuItems, onClick: handleUserMenuClick }} 
            placement="bottomRight"
            arrow
          >
            <div className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 px-2 py-1 rounded-lg transition-colors">
              <div 
                className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold shrink-0"
                style={{ background: '#1e3a5f' }}
              >
                {user.avatar ? (
                  <img src={user.avatar} alt="" className="w-full h-full rounded-full object-cover" />
                ) : (
                  user.name?.charAt(0).toUpperCase()
                )}
              </div>
              <span className="text-sm text-gray-700 hidden md:block max-w-25 truncate">
                {user.name}
              </span>
              <ChevronDown size={14} className="text-gray-400" />
            </div>
          </Dropdown>
        ) : (
          <Tooltip title="Đăng nhập / Đăng ký" placement="bottomRight">
            <button
              onClick={onOpenAuth}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-gray-600 hover:text-[#722F37] hover:bg-[#722F37]/5 transition-all border-0 border-gray-200 hover:border-[#722F37]/30"
            >
              <User size={18} />
              <span className="hidden sm:inline">Đăng nhập</span>
            </button>
          </Tooltip>
        )}
      </div>

      <style>
        {`
          .ant-menu-horizontal {
            border-bottom: none !important;
          }
        `}
      </style>
    </Header>
  );
};

export default AppHeader;