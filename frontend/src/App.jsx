import { useState, useEffect } from 'react';
import { Layout } from 'antd';
import { Scale } from 'lucide-react';
import AppHeader from './components/header';
import LandingPage from './components/LandingPage';
import TabSearch from './components/TabSearch';
import TabAssistant from './components/TabAssistant';
import TabStringMatching from './components/TabStringMatching';
import TabRagExtract from './components/TabRagExtract';
import AuthModal from './components/AuthModal';
import { clearAuth, getStoredUser } from './services/auth';

const { Content } = Layout;

const VALID_TABS = ['home', 'search', 'ai-assistant', 'string-matching', 'rag-extract'];
const tabFromHash = () => {
  const t = window.location.hash.replace(/^#\/?/, '');
  return VALID_TABS.includes(t) ? t : 'home';
};

const App = () => {
  const [activeTab, setActiveTab] = useState(tabFromHash);
  const [chatIntent, setChatIntent] = useState(null);
  const [user, setUser] = useState(getStoredUser);
  const [authOpen, setAuthOpen] = useState(false);

  // Sync tab → URL hash (pushes history entry so browser back works).
  useEffect(() => {
    const targetHash = `#/${activeTab}`;
    if (window.location.hash !== targetHash) {
      window.history.pushState({ tab: activeTab }, '', targetHash);
    }
  }, [activeTab]);

  // Listen to browser back/forward.
  useEffect(() => {
    const onPop = () => setActiveTab(tabFromHash());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    setAuthOpen(false);
  };

  const handleLogout = () => {
    setUser(null);
    clearAuth();
  };

  const handleTabChange = (key, intent = null) => {
    setActiveTab(key);
    setChatIntent(intent);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <LandingPage onNavigate={handleTabChange} />;
      case 'search':
        return <TabSearch />;
      case 'ai-assistant':
        return <TabAssistant defaultIntent={chatIntent} />;
      case 'string-matching':
        return <TabStringMatching />;
      case 'rag-extract':
        return <TabRagExtract />;
      default:
        return <LandingPage onNavigate={handleTabChange} />;
    }
  };

  // Chưa đăng nhập → màn auth bắt buộc, không cho đóng modal.
  if (!user) {
    return (
      <div
        className="min-h-screen flex items-center justify-center relative overflow-hidden"
        style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #2d1f3e 50%, #8b1a2b 100%)' }}
      >
        <div className="relative z-10 text-center px-6">
          <div
            className="inline-flex items-center justify-center w-16 h-16 rounded-2xl shadow-2xl mb-5"
            style={{ background: '#722F37' }}
          >
            <Scale className="text-white" size={32} />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2 font-['Playfair_Display']">
            Tra cứu Luật Việt Nam
          </h1>
          <p className="text-white/80 text-base mb-8">
            Vui lòng đăng nhập để tiếp tục sử dụng hệ thống
          </p>
        </div>
        <AuthModal
          isOpen={true}
          onClose={() => {}}
          onLogin={handleLogin}
          dismissible={false}
        />
      </div>
    );
  }

  return (
    <Layout className="min-h-screen bg-[#faf9f7]">
      <AppHeader
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={user}
        onOpenAuth={() => setAuthOpen(true)}
        onLogout={handleLogout}
      />
      <Content>
        {renderContent()}
      </Content>

      <AuthModal
        isOpen={authOpen}
        onClose={() => setAuthOpen(false)}
        onLogin={handleLogin}
      />
    </Layout>
  );
};

export default App;
