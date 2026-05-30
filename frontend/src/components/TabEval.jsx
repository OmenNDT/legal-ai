import { useEffect, useState, useMemo } from 'react';
import { Card, Table, Tag, Spin, Alert, Tooltip, Statistic, Row, Col, Empty, Button, Segmented } from 'antd';
import { BarChart3, RefreshCw, Activity, Target, Clock, FileText } from 'lucide-react';
import { fetchEvalMetrics } from '../services/api';

const RETRIEVER_LABEL = {
  pure_vector: 'Pure Vector (HNSW)',
  hybrid_rrf: 'Hybrid RRF (Production)',
  vector_rerank_pretrained: 'Vector + BGE-reranker-v2-m3',
  vector_rerank_finetuned: 'Vector + Reranker (fine-tuned)'
};
const RETRIEVER_COLOR = {
  pure_vector: '#1f77b4',
  hybrid_rrf: '#722F37',
  vector_rerank_pretrained: '#2ca02c',
  vector_rerank_finetuned: '#ff7f0e'
};

const Bar = ({ value, max = 1, color = '#722F37', label }) => {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded h-4 overflow-hidden relative">
        <div className="h-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="text-xs font-mono w-14 text-right" style={{ color: '#2d2d2d' }}>
        {label ?? value.toFixed(3)}
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, suffix, hint, icon: Icon, color = '#722F37' }) => (
  <Card size="small" className="shadow-sm">
    <div className="flex items-start justify-between">
      <div>
        <div className="text-xs text-gray-500 uppercase tracking-wider">{title}</div>
        <div className="text-2xl font-bold mt-1" style={{ color }}>
          {value}
          {suffix && <span className="text-sm ml-1 font-normal text-gray-500">{suffix}</span>}
        </div>
        {hint && <div className="text-xs text-gray-400 mt-1">{hint}</div>}
      </div>
      {Icon && <Icon size={20} color={color} />}
    </div>
  </Card>
);

const TabEval = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedK, setSelectedK] = useState(5);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchEvalMetrics();
      if (!res?.ok) {
        setError(res?.error || 'Không tải được metrics');
        setData(null);
      } else {
        setData(res.data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const retrievers = useMemo(() => {
    if (!data?.retrievers) return [];
    return Object.entries(data.retrievers).map(([key, val]) => ({
      key,
      label: RETRIEVER_LABEL[key] || key,
      color: RETRIEVER_COLOR[key] || '#666',
      ...val.summary,
      per_doc: val.per_doc
    }));
  }, [data]);

  const ks = data?.ks || [1, 3, 5, 10, 20];
  const maxRecall = Math.max(0.001, ...retrievers.flatMap(r => ks.map(k => r[`recall@${k}`] || 0)));

  if (loading) {
    return <div className="flex justify-center items-center min-h-[60vh]"><Spin size="large" tip="Đang tải metrics..." /></div>;
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto py-12 px-6">
        <Alert
          type="warning"
          showIcon
          message="Chưa có kết quả eval"
          description={
            <div>
              <p>{error}</p>
              <p className="mt-2 text-xs font-mono bg-gray-50 p-2 rounded">
                # Trên máy có GPU:<br/>
                python3 backend/chatbot/eval/build_test_set.py<br/>
                python3 backend/chatbot/eval/embed_queries.py<br/>
                python3 backend/chatbot/eval/run_baselines.py<br/>
                python3 backend/chatbot/eval/run_reranker.py
              </p>
            </div>
          }
          action={<Button icon={<RefreshCw size={14} />} onClick={load}>Tải lại</Button>}
        />
      </div>
    );
  }

  const docNames = data?.doc_names || {};
  const docTableRows = (() => {
    const docIds = new Set();
    retrievers.forEach(r => Object.keys(r.per_doc || {}).forEach(d => docIds.add(d)));
    return Array.from(docIds).map(d => {
      const row = {
        key: d,
        doc_id: d,
        doc_code: docNames[d]?.code || `doc#${d}`,
        doc_name: docNames[d]?.name || '',
      };
      retrievers.forEach(r => {
        const stats = r.per_doc?.[d];
        row[`n_${r.key}`] = stats?.n || 0;
        row[`recall_${r.key}`] = stats ? stats[`recall@${selectedK}`] : null;
      });
      return row;
    }).sort((a, b) => (b[`n_${retrievers[0]?.key}`] || 0) - (a[`n_${retrievers[0]?.key}`] || 0));
  })();

  return (
    <div className="max-w-7xl mx-auto py-6 px-4 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2" style={{ color: '#722F37' }}>
            <BarChart3 size={26} /> Đánh giá Retrieval
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Test set: <b>{data?.test_set_size || 0}</b> câu hỏi · Gold labels: <b>{data?.gold_source || 'qwen-2.5-7b-instruct'}</b> (3-level scoring: 0/1/2)
          </p>
        </div>
        <Button icon={<RefreshCw size={14} />} onClick={load}>Tải lại</Button>
      </div>

      {/* Top KPI cards */}
      <Row gutter={[16, 16]} className="mb-6">
        {retrievers.map(r => (
          <Col xs={24} md={12} lg={24 / retrievers.length} key={r.key}>
            <Card size="small" className="shadow-sm" styles={{ body: { padding: 14 } }}>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-3 h-3 rounded-full" style={{ background: r.color }} />
                <span className="font-semibold text-sm">{r.label}</span>
              </div>
              <Row gutter={8}>
                <Col span={8}>
                  <Statistic title="Recall@5" value={r['recall@5']} precision={3} valueStyle={{ fontSize: 18, color: r.color }} />
                </Col>
                <Col span={8}>
                  <Statistic title="MRR@10" value={r['mrr@10']} precision={3} valueStyle={{ fontSize: 18 }} />
                </Col>
                <Col span={8}>
                  <Statistic title="p50 ms" value={r['latency_ms_p50']} precision={0} valueStyle={{ fontSize: 18 }} />
                </Col>
              </Row>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Section 1: Recall@K comparison */}
      <Card title={<span className="flex items-center gap-2"><Target size={16}/> Recall@K — Tỷ lệ chunk đúng trong top-K</span>} className="mb-6 shadow-sm">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-4 text-gray-600">Retriever</th>
              {ks.map(k => <th key={k} className="py-2 pr-4 text-gray-600">Recall@{k}</th>)}
              <th className="py-2 pr-4 text-gray-600">MRR@10</th>
              <th className="py-2 pr-4 text-gray-600">NDCG@10</th>
            </tr>
          </thead>
          <tbody>
            {retrievers.map(r => (
              <tr key={r.key} className="border-b last:border-0">
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full shrink-0" style={{ background: r.color }} />
                    <span className="font-medium">{r.label}</span>
                  </div>
                </td>
                {ks.map(k => (
                  <td key={k} className="py-3 pr-4" style={{ minWidth: 140 }}>
                    <Bar value={r[`recall@${k}`] || 0} max={maxRecall} color={r.color} />
                  </td>
                ))}
                <td className="py-3 pr-4 font-mono">{(r['mrr@10'] ?? 0).toFixed(3)}</td>
                <td className="py-3 pr-4 font-mono">{(r['ndcg@10'] ?? 0).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Section 2: Answer-overlap (unbiased) */}
      <Card title={<span className="flex items-center gap-2"><Activity size={16}/> Answer-Overlap — Câu trả lời có substring 4-gram trùng chunk?</span>}
            className="mb-6 shadow-sm"
            extra={<Tooltip title="Metric không phụ thuộc weak gold — đánh giá tính khả dụng thực của chunk được retrieve">
              <Tag color="blue">Unbiased</Tag>
            </Tooltip>}>
        <Row gutter={[16, 16]}>
          {retrievers.map(r => (
            <Col xs={24} md={12} lg={24 / retrievers.length} key={r.key}>
              <Card size="small" type="inner">
                <div className="text-xs font-semibold mb-2" style={{ color: r.color }}>{r.label}</div>
                <div className="mb-2">
                  <div className="text-xs text-gray-500 mb-1">overlap @ top-5</div>
                  <Bar value={r.overlap_in_top5 || 0} max={1} color={r.color} label={`${((r.overlap_in_top5 || 0) * 100).toFixed(1)}%`} />
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">overlap @ top-10</div>
                  <Bar value={r.overlap_in_top10 || 0} max={1} color={r.color} label={`${((r.overlap_in_top10 || 0) * 100).toFixed(1)}%`} />
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* Section 3: Latency */}
      <Card title={<span className="flex items-center gap-2"><Clock size={16}/> Latency (ms / query)</span>} className="mb-6 shadow-sm">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left">
              <th className="py-2 pr-4 text-gray-600">Retriever</th>
              <th className="py-2 pr-4 text-gray-600">p50</th>
              <th className="py-2 pr-4 text-gray-600">p95</th>
              <th className="py-2 pr-4 text-gray-600">avg</th>
              <th className="py-2 pr-4 text-gray-600">retrieve / rerank</th>
            </tr>
          </thead>
          <tbody>
            {retrievers.map(r => (
              <tr key={r.key} className="border-b last:border-0">
                <td className="py-3 pr-4 font-medium">{r.label}</td>
                <td className="py-3 pr-4 font-mono">{r['latency_ms_p50']?.toFixed?.(0) ?? '-'}</td>
                <td className="py-3 pr-4 font-mono">{r['latency_ms_p95']?.toFixed?.(0) ?? '-'}</td>
                <td className="py-3 pr-4 font-mono">{r['latency_ms_avg']?.toFixed?.(0) ?? '-'}</td>
                <td className="py-3 pr-4 font-mono text-xs">
                  {r['retrieve_ms_p50'] != null ? `${r['retrieve_ms_p50'].toFixed(0)} / ${r['rerank_ms_p50']?.toFixed(0) ?? '-'}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Section 4: Per-document breakdown */}
      <Card
        title={<span className="flex items-center gap-2"><FileText size={16}/> Recall theo văn bản luật</span>}
        extra={
          <Segmented
            options={ks.map(k => ({ label: `K=${k}`, value: k }))}
            value={selectedK}
            onChange={setSelectedK}
            size="small"
          />
        }
        className="shadow-sm">
        {docTableRows.length === 0 ? <Empty /> : (
          <Table
            size="small"
            pagination={{ pageSize: 10 }}
            dataSource={docTableRows}
            columns={[
              {
                title: 'Văn bản',
                dataIndex: 'doc_name',
                key: 'doc_name',
                render: (txt, row) => (
                  <div>
                    <Tag color="purple">{row.doc_code}</Tag>
                    <span className="text-sm">{txt || `Doc #${row.doc_id}`}</span>
                  </div>
                ),
                ellipsis: true,
              },
              {
                title: 'N test',
                dataIndex: `n_${retrievers[0]?.key}`,
                key: 'n',
                width: 80,
                align: 'right',
              },
              ...retrievers.map(r => ({
                title: <span style={{ color: r.color }}>{r.label.split('(')[0].trim()}</span>,
                dataIndex: `recall_${r.key}`,
                key: r.key,
                width: 160,
                render: (v) => v == null ? '—' : <Bar value={v} max={1} color={r.color} label={v.toFixed(3)} />
              })),
            ]}
          />
        )}
      </Card>
    </div>
  );
};

export default TabEval;
