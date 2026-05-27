import { useState, useEffect, useRef } from 'react';
import { Card, Statistic, Input, Button, List, Tag, Spin, Empty, Alert, Tabs, Typography, Badge, Popconfirm, message } from 'antd';
import { Database, Search, FileText, Activity, BookOpen, Upload as UploadIcon, Trash2, PlayCircle, RefreshCw, FileUp, X, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { getRagHealth, getRagStats, queryRag, listRagDocuments, getRagIndexStats, uploadDocuments, deleteDocument, processDocument } from '../services/ragExtractApi';

const { Search: AntSearch } = Input;
const { Text, Title } = Typography;

const TabRagExtract = () => {
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [processingId, setProcessingId] = useState(null);
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [indexStats, setIndexStats] = useState(null);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [docTotal, setDocTotal] = useState(0);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadOverview();
  }, []);

  const loadOverview = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, s, idx] = await Promise.all([
        getRagHealth(),
        getRagStats(),
        getRagIndexStats().catch(() => null),
      ]);
      setHealth(h);
      setStats(s);
      setIndexStats(idx);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Không thể kết nối RAG Extract');
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listRagDocuments(0, 100);
      setDocuments(data.items || []);
      setDocTotal(data.total || 0);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Không thể tải danh sách tài liệu');
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async (value) => {
    if (!value.trim()) return;
    setLoading(true);
    setError(null);
    setQueryResult(null);
    try {
      const result = await queryRag(value, 5);
      setQueryResult(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Query thất bại');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDocuments(selectedFiles);
      message.success(`Đã tải lên ${result.total_uploaded} tài liệu thành công!`);
      if (result.errors?.length > 0) {
        message.warning(`${result.total_errors} file lỗi: ${result.errors.map(e => e.filename).join(', ')}`);
      }
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await Promise.all([loadOverview(), loadDocuments()]);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Tải lên thất bại');
    } finally {
      setUploading(false);
    }
  };

  const handleProcess = async (docId) => {
    setProcessingId(docId);
    setError(null);
    try {
      const result = await processDocument(docId);
      message.success(`Tài liệu "${result.filename}" đã xử lý xong!`);
      await loadDocuments();
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Xử lý thất bại');
    } finally {
      setProcessingId(null);
    }
  };

  const handleDelete = async (docId) => {
    setError(null);
    try {
      await deleteDocument(docId);
      message.success('Đã xóa tài liệu');
      await Promise.all([loadOverview(), loadDocuments()]);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Xóa thất bại');
    }
  };

  const handleTabChange = (key) => {
    setActiveTab(key);
    setError(null);
    if (key === 'documents' && documents.length === 0) {
      loadDocuments();
    }
  };

  const statusColor = (status) => {
    switch (status) {
      case 'ready': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'processing';
      default: return 'default';
    }
  };

  const statusIcon = (status) => {
    switch (status) {
      case 'ready': return <CheckCircle size={14} className="text-green-500" />;
      case 'failed': return <AlertCircle size={14} className="text-red-500" />;
      case 'processing': return <Clock size={14} className="text-blue-500" />;
      default: return null;
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // ── Upload Section ──
  const uploadContent = (
    <div className="space-y-6">
      <Card className="shadow-sm">
        <div className="text-center py-6">
          <UploadIcon size={48} className="text-[#1e3a5f] mx-auto mb-4" />
          <Title level={4} className="!mb-2">Tải lên tài liệu</Title>
          <Text type="secondary" className="block mb-6">
            Hỗ trợ PDF, DOCX, TXT, MD, XLSX. Tối đa 10 file mỗi lần.
          </Text>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.doc,.txt,.md,.xlsx,.xls"
            onChange={(e) => setSelectedFiles(Array.from(e.target.files))}
            className="hidden"
            id="rag-file-input"
          />
          <label
            htmlFor="rag-file-input"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg cursor-pointer transition-all hover:scale-105"
            style={{ background: '#1e3a5f', color: 'white' }}
          >
            <FileUp size={18} />
            Chọn file
          </label>

          {selectedFiles.length > 0 && (
            <div className="mt-6 text-left max-w-md mx-auto">
              <div className="flex items-center justify-between mb-3">
                <Text strong>{selectedFiles.length} file đã chọn</Text>
                <Button
                  size="small"
                  danger
                  icon={<X size={14} />}
                  onClick={() => { setSelectedFiles([]); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                >
                  Xóa hết
                </Button>
              </div>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {selectedFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                    <FileText size={16} className="text-[#1e3a5f] shrink-0" />
                    <span className="flex-1 truncate text-sm">{f.name}</span>
                    <span className="text-xs text-gray-400 shrink-0">{formatFileSize(f.size)}</span>
                  </div>
                ))}
              </div>
              <Button
                type="primary"
                size="large"
                block
                className="mt-4"
                loading={uploading}
                onClick={handleUpload}
                style={{ background: '#722F37' }}
                icon={<UploadIcon size={16} />}
              >
                {uploading ? 'Đang tải lên...' : `Tải lên ${selectedFiles.length} file`}
              </Button>
            </div>
          )}
        </div>
      </Card>

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="shadow-sm">
            <Statistic
              title="Tổng tài liệu"
              value={stats?.total_documents || 0}
              prefix={<Database size={18} className="text-[#1e3a5f]" />}
            />
          </Card>
          <Card className="shadow-sm">
            <Statistic
              title="Sẵn sàng"
              value={stats?.ready_documents || 0}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircle size={18} className="text-green-600" />}
            />
          </Card>
          <Card className="shadow-sm">
            <Statistic
              title="Đang xử lý"
              value={(stats?.total_documents || 0) - (stats?.ready_documents || 0) - (stats?.failed_documents || 0)}
              prefix={<Clock size={18} className="text-blue-500" />}
            />
          </Card>
        </div>
      )}
    </div>
  );

  const overviewContent = (
    <div className="space-y-6">
      {health && (
        <Alert
          message={`RAG Extract Status: ${health.status}`}
          type={health.status === 'ok' ? 'success' : 'warning'}
          showIcon
          className="mb-4"
        />
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="shadow-sm">
          <Statistic
            title="Tổng tài liệu"
            value={stats?.total_documents || 0}
            prefix={<Database size={18} className="text-[#1e3a5f]" />}
          />
        </Card>
        <Card className="shadow-sm">
          <Statistic
            title="Sẵn sàng"
            value={stats?.ready_documents || 0}
            valueStyle={{ color: '#52c41a' }}
            prefix={<Activity size={18} className="text-green-600" />}
          />
        </Card>
        <Card className="shadow-sm">
          <Statistic
            title="Lỗi"
            value={stats?.failed_documents || 0}
            valueStyle={{ color: '#ff4d4f' }}
            prefix={<Activity size={18} className="text-red-600" />}
          />
        </Card>
      </div>

      {indexStats && (
        <Card title="Vector Store Stats" className="shadow-sm">
          <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-auto">
            {JSON.stringify(indexStats, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );

  const queryContent = (
    <div className="space-y-6">
      <Card className="shadow-sm">
        <div className="flex gap-2">
          <AntSearch
            placeholder="Nhập câu hỏi để truy vấn RAG..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onSearch={handleQuery}
            enterButton={
              <Button type="primary" icon={<Search size={16} />} loading={loading}>
                Truy vấn
              </Button>
            }
            size="large"
            className="flex-1"
          />
        </div>
      </Card>

      {queryResult && (
        <Card title="Kết quả" className="shadow-sm">
          <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-auto max-h-[500px]">
            {JSON.stringify(queryResult, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );

  const documentsContent = (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Title level={5} className="!mb-0">
          Danh sách tài liệu <Badge count={docTotal} showZero />
        </Title>
        <div className="flex gap-2">
          <Button onClick={() => { loadDocuments(); loadOverview(); }} icon={<RefreshCw size={14} />}>
            Làm mới
          </Button>
          <Button type="primary" onClick={() => setActiveTab('upload')} icon={<UploadIcon size={14} />} style={{ background: '#722F37' }}>
            Tải lên
          </Button>
        </div>
      </div>

      {documents.length === 0 ? (
        <Empty description="Chưa có tài liệu nào. Hãy tải lên tài liệu đầu tiên!" />
      ) : (
        <List
          grid={{ gutter: 16, xs: 1, sm: 1, md: 2, lg: 2, xl: 3 }}
          dataSource={documents}
          renderItem={(doc) => (
            <List.Item>
              <Card className="shadow-sm h-full" size="small">
                <div className="flex items-start gap-3">
                  <FileText size={20} className="text-[#1e3a5f] shrink-0 mt-1" />
                  <div className="flex-1 min-w-0">
                    <Text strong className="block truncate">
                      {doc.display_name || doc.filename}
                    </Text>
                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                      <Tag color={statusColor(doc.status)} icon={statusIcon(doc.status)}>
                        {doc.status}
                      </Tag>
                      {doc.standard_code && <Tag>{doc.standard_code}</Tag>}
                      {doc.language && <Tag>{doc.language}</Tag>}
                    </div>
                    <Text type="secondary" className="text-xs mt-1 block">
                      ID: {doc.id} | {formatFileSize(doc.file_size_bytes)}
                      {doc.total_chunks && ` | ${doc.total_chunks} chunks`}
                    </Text>
                    <div className="flex gap-2 mt-2">
                      {doc.status === 'processing' && (
                        <Button
                          size="small"
                          type="primary"
                          icon={<PlayCircle size={14} />}
                          loading={processingId === doc.id}
                          onClick={() => handleProcess(doc.id)}
                        >
                          Xử lý
                        </Button>
                      )}
                      <Popconfirm
                        title="Xóa tài liệu này?"
                        description="Hành động này không thể hoàn tác."
                        onConfirm={() => handleDelete(doc.id)}
                        okText="Xóa"
                        cancelText="Hủy"
                        okButtonProps={{ danger: true }}
                      >
                        <Button size="small" danger icon={<Trash2 size={14} />}>
                          Xóa
                        </Button>
                      </Popconfirm>
                    </div>
                  </div>
                </div>
              </Card>
            </List.Item>
          )}
        />
      )}
    </div>
  );

  return (
    <div className="h-[calc(100vh-64px)] flex flex-col bg-[#faf9f7]">
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-6">
            <Database size={28} className="text-[#1e3a5f]" />
            <div>
              <Title level={3} className="!mb-0">RAG Extract</Title>
              <Text type="secondary">Truy hồi và trích xuất thông tin tài liệu</Text>
            </div>
          </div>

          {error && (
            <Alert
              message="Lỗi"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
              className="mb-4"
            />
          )}

          <Tabs
            activeKey={activeTab}
            onChange={handleTabChange}
            items={[
              {
                key: 'upload',
                label: (
                  <span className="flex items-center gap-1">
                    <UploadIcon size={14} />
                    Tải lên
                  </span>
                ),
                children: uploadContent,
              },
              {
                key: 'overview',
                label: 'Tổng quan',
                children: overviewContent,
              },
              {
                key: 'query',
                label: 'Truy vấn',
                children: queryContent,
              },
              {
                key: 'documents',
                label: `Tài liệu (${docTotal})`,
                children: documentsContent,
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
};

export default TabRagExtract;
