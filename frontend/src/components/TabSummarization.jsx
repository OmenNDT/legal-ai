import { useEffect, useMemo, useState } from 'react';
import {
  Input, Select, Button, Card, Tag, Switch, Tooltip, Slider, Progress,
  Typography, Upload, message, Empty, AutoComplete, Tabs, Statistic,
} from 'antd';
import {
  FileText, Play, Sparkles, Layers, Cpu, Activity, Upload as UploadIcon,
  Download, Search as SearchIcon, BookOpen, Gauge, BarChart3, RefreshCw,
} from 'lucide-react';
import {
  EXTRACTORS, listDocuments, getDocument, summarize, extractOnly, health,
} from '../services/summarization';
import TextRankGraph from './TextRankGraph';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;

const COLORS = {
  bg: '#faf9f7',
  navy: '#1e3a5f',
  wine: '#722F37',
  wineDeep: '#4a1520',
  purple: '#2d1f3e',
  gold: '#e8d5b7',
  goldDark: '#c9a96e',
  dark: '#2d2d2d',
  muted: '#9a8478',
  subtle: '#6b5b5e',
};

const DEFAULT_SAMPLE = `This Co-Branding and Advertising Agreement is made as of June 21, 1999 between I-ESCROW, INC. and 2THEMART.COM, INC. The parties agree to launch a co-branded website where i-Escrow's escrow services will be made available to 2TheMart customers. This agreement is governed by California law and shall remain in effect for an initial term of two years.`;

const TabSummarization = () => {
  // Cấu hình đầu vào
  const [mode, setMode] = useState('doc'); // 'doc' | 'text'
  const [docList, setDocList] = useState([]);
  const [docId, setDocId] = useState(null);
  const [docPreview, setDocPreview] = useState(null); // {text, word_count, reference}
  const [text, setText] = useState(DEFAULT_SAMPLE);
  const [extractor, setExtractor] = useState('textrank');
  const [useAbstractive, setUseAbstractive] = useState(true);
  const [searchQ, setSearchQ] = useState('');
  const [serverInfo, setServerInfo] = useState(null);

  // Trạng thái thực thi
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState(null); // 'clean' | 'split' | 'extract' | 'abstract' | 'done'
  const [result, setResult] = useState(null);

  // Lấy thông tin backend
  useEffect(() => {
    health().then(setServerInfo).catch(() => setServerInfo(null));
  }, []);

  // Tải danh sách doc khi gõ tìm kiếm
  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        const data = await listDocuments({ q: searchQ, pageSize: 100 });
        setDocList(data.items || []);
      } catch {
        setDocList([]);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [searchQ]);

  // Khi chọn doc -> tải preview
  useEffect(() => {
    if (!docId) return;
    getDocument(docId)
      .then(setDocPreview)
      .catch(() => setDocPreview(null));
  }, [docId]);

  // Chạy pipeline
  const run = async () => {
    if (mode === 'text' && !text.trim()) {
      message.warning('Vui lòng nhập văn bản hoặc upload file');
      return;
    }
    if (mode === 'doc' && !docId) {
      message.warning('Chọn một hợp đồng trong danh sách');
      return;
    }
    setRunning(true);
    setResult(null);
    setStage('clean');
    try {
      // Mô phỏng tiến trình từng bước cho người xem
      const stages = useAbstractive
        ? ['clean', 'split', 'extract', 'abstract']
        : ['clean', 'split', 'extract'];
      let idx = 0;
      const ticker = setInterval(() => {
        idx = Math.min(idx + 1, stages.length - 1);
        setStage(stages[idx]);
      }, 700);
      const data = await summarize({
        text: mode === 'text' ? text : undefined,
        docId: mode === 'doc' ? docId : undefined,
        extractor,
        useAbstractive,
      });
      clearInterval(ticker);
      setStage('done');
      setResult(data);
    } catch (e) {
      message.error('Chạy tóm tắt thất bại: ' + (e?.response?.data?.error || e.message));
      setStage(null);
    } finally {
      setRunning(false);
    }
  };

  // Tải file txt lên
  const handleUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setText(String(e.target?.result ?? ''));
      setMode('text');
      message.success(`Đã nạp ${file.name}`);
    };
    reader.onerror = () => message.error('Không đọc được file');
    reader.readAsText(file, 'utf-8');
    return false;
  };

  // Xuất kết quả ra .txt
  const exportResult = () => {
    if (!result) return;
    const lines = [];
    lines.push('================ KẾT QUẢ TÓM TẮT LAI ================');
    lines.push(`Doc: ${result.doc_id || '(văn bản tự nhập)'}`);
    lines.push(`Extractor: ${result.extractive?.method}`);
    lines.push(`Số câu sau split: ${result.num_sentences}`);
    lines.push(`Số câu giữ lại: ${result.extractive?.sentences?.length}`);
    lines.push('');
    lines.push('--- Câu được trích xuất ---');
    (result.extractive?.sentences || []).forEach((s, i) => {
      lines.push(`[${i + 1}] (#${s.idx}) ${s.text}`);
    });
    lines.push('');
    if (result.abstractive) {
      lines.push('--- Bản viết lại (BART) ---');
      lines.push(result.abstractive.text);
    }
    if (result.rouge) {
      lines.push('');
      lines.push('--- ROUGE so với reference ---');
      lines.push(JSON.stringify(result.rouge, null, 2));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `summary_${result.doc_id || 'text'}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Các "step" của pipeline để render trực quan
  const stageList = useMemo(() => ([
    { key: 'clean', label: 'Làm sạch', icon: <RefreshCw size={14} /> },
    { key: 'split', label: 'Tách câu', icon: <Layers size={14} /> },
    { key: 'extract', label: `Extractive (${extractor})`, icon: <BarChart3 size={14} /> },
    { key: 'abstract', label: 'Abstractive (BART)', icon: <Sparkles size={14} /> },
  ]), [extractor]);

  const stageIndex = stageList.findIndex((s) => s.key === stage);

  return (
    <div className="min-h-[calc(100vh-64px)]" style={{ background: COLORS.bg }}>
      {/* Hero */}
      <div
        className="relative overflow-hidden pb-6 pt-8"
        style={{ background: `linear-gradient(135deg, ${COLORS.navy} 0%, ${COLORS.purple} 50%, #8b1a2b 100%)` }}
      >
        <div className="relative z-10 max-w-5xl mx-auto px-6 pt-2 pb-6 text-center">
          <div
            className="inline-flex items-center gap-2 backdrop-blur-sm px-4 py-2 rounded-full mb-3 border"
            style={{ background: 'rgba(255,255,255,0.1)', borderColor: 'rgba(255,255,255,0.2)' }}
          >
            <Sparkles size={16} style={{ color: COLORS.gold }} />
            <span className="text-white/90 text-sm font-medium">Hybrid Summarization · Extractive → Abstractive</span>
          </div>
          <Title level={2} className="text-white mb-1 font-['Playfair_Display'] text-3xl! md:text-4xl!" style={{ color: '#ffffff' }}>
            Tóm tắt văn bản
          </Title>
          <Text className="text-white/90 text-base block" style={{ color: 'rgba(255,255,255,0.9)' }}>
            Lọc câu cốt lõi bằng TF-IDF/TextRank/KMeans, rồi để BART viết lại mượt mà.
          </Text>
          {serverInfo && (
            <div className="mt-3 inline-flex items-center gap-2 text-white/80 text-xs">
              <Tag color="default" style={{ background: 'rgba(255,255,255,0.12)', color: 'white', border: 'none' }}>
                device: {serverInfo.device}
              </Tag>
              <Tag color="default" style={{ background: 'rgba(255,255,255,0.12)', color: 'white', border: 'none' }}>
                model: {serverInfo.model}
              </Tag>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 -mt-2">
        {/* Controls */}
        <Card className="rounded-2xl border-none shadow-md mb-6" bodyStyle={{ padding: 20 }}>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Cột 1: chọn nguồn */}
            <div>
              <label className="text-xs uppercase tracking-widest font-semibold mb-2 block" style={{ color: COLORS.muted }}>
                Nguồn đầu vào
              </label>
              <Tabs
                activeKey={mode}
                onChange={setMode}
                size="small"
                items={[
                  { key: 'doc', label: 'Chọn từ CUAD (510 file)' },
                  { key: 'text', label: 'Nhập / upload văn bản' },
                ]}
              />
              {mode === 'doc' ? (
                <>
                  <AutoComplete
                    placeholder="Gõ để tìm tên hợp đồng..."
                    value={searchQ}
                    onChange={setSearchQ}
                    options={docList.map((d) => ({ value: d }))}
                    style={{ width: '100%' }}
                    onSelect={(v) => { setDocId(v); setSearchQ(v); }}
                  />
                  {docPreview && (
                    <div className="mt-2 text-xs" style={{ color: COLORS.subtle }}>
                      <BookOpen size={12} className="inline mr-1" />
                      {docPreview.word_count.toLocaleString()} từ
                      {docPreview.reference && (
                        <Tag color="gold" style={{ marginLeft: 8 }}>có reference (CUAD)</Tag>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex justify-end mb-1">
                    <Upload
                      accept=".txt,.md"
                      showUploadList={false}
                      beforeUpload={handleUpload}
                    >
                      <Button size="small" icon={<UploadIcon size={12} />} style={{ borderRadius: 8, fontSize: 12, height: 26 }}>
                        Upload (.txt)
                      </Button>
                    </Upload>
                  </div>
                  <TextArea
                    rows={6}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Dán văn bản cần tóm tắt..."
                    style={{ borderRadius: 10 }}
                  />
                </>
              )}
            </div>

            {/* Cột 2: thuật toán */}
            <div>
              <label className="text-xs uppercase tracking-widest font-semibold mb-2 block" style={{ color: COLORS.muted }}>
                Thuật toán Extractive
              </label>
              <Select
                value={extractor}
                onChange={setExtractor}
                options={EXTRACTORS.map((e) => ({ value: e.value, label: `${e.label} — ${e.desc}` }))}
                style={{ width: '100%' }}
              />
              <div className="mt-4 flex items-center gap-2">
                <Switch checked={useAbstractive} onChange={setUseAbstractive} />
                <span className="text-sm" style={{ color: COLORS.dark }}>Bật bước Abstractive (BART)</span>
              </div>
              <div className="mt-2 text-xs" style={{ color: COLORS.subtle }}>
                Tắt để chỉ chạy extractive (nhanh, không cần GPU).
              </div>
            </div>

            {/* Cột 3: tiến trình */}
            <div>
              <label className="text-xs uppercase tracking-widest font-semibold mb-2 block" style={{ color: COLORS.muted }}>
                Luồng xử lý
              </label>
              <div className="space-y-2">
                {stageList.map((s, i) => {
                  const active = i === stageIndex;
                  const done = stageIndex > i || stage === 'done';
                  if (!useAbstractive && s.key === 'abstract') return null;
                  return (
                    <div
                      key={s.key}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
                      style={{
                        background: active ? COLORS.gold : done ? 'rgba(134,199,155,0.18)' : '#faf6f2',
                        color: active ? COLORS.wineDeep : COLORS.dark,
                        fontWeight: active ? 700 : 500,
                      }}
                    >
                      <span style={{ color: active ? COLORS.wine : done ? '#3b9b5a' : COLORS.muted }}>{s.icon}</span>
                      <span className="text-sm">{s.label}</span>
                      {done && !active && <Tag color="green" style={{ marginLeft: 'auto' }}>xong</Tag>}
                      {active && <Tag color="gold" style={{ marginLeft: 'auto' }}>đang chạy</Tag>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Button
              type="primary"
              icon={<Play size={14} />}
              loading={running}
              onClick={run}
              style={{ background: COLORS.wine, borderColor: COLORS.wine, borderRadius: 8, height: 38 }}
            >
              Chạy pipeline lai
            </Button>
            <Button
              icon={<Download size={14} />}
              disabled={!result}
              onClick={exportResult}
              style={{ borderRadius: 8, height: 38, borderColor: COLORS.goldDark, color: COLORS.wineDeep }}
            >
              Xuất kết quả
            </Button>
            {result?.timings && (
              <div className="ml-auto flex items-center gap-3 text-xs" style={{ color: COLORS.subtle }}>
                <Gauge size={14} style={{ color: COLORS.goldDark }} />
                {Object.entries(result.timings).map(([k, v]) => (
                  <Tag key={k} color="default" style={{ borderRadius: 999 }}>{k}: {v}s</Tag>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Kết quả */}
        {!result ? (
          <Card className="rounded-2xl border-none shadow-sm" bodyStyle={{ padding: 40 }}>
            <Empty description={<span style={{ color: COLORS.muted }}>Chưa có kết quả. Chọn nguồn và bấm <b>Chạy pipeline lai</b>.</span>} />
          </Card>
        ) : (
          <ResultView result={result} />
        )}
      </div>
    </div>
  );
};

// -------- Khung kết quả --------
const ResultView = ({ result }) => {
  const { extractive, abstractive, rouge, num_sentences, raw_word_count } = result;
  return (
    <div className="space-y-5">
      {/* Hàng thống kê */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Số từ gốc" value={raw_word_count?.toLocaleString()} color={COLORS.navy} />
        <StatCard label="Số câu sau split" value={num_sentences} color={COLORS.purple} />
        <StatCard label="Số câu giữ lại" value={extractive?.sentences?.length || 0} color={COLORS.wine} />
        <StatCard
          label="Tỉ lệ nén"
          value={
            raw_word_count
              ? `${Math.round(((abstractive?.text?.split(' ').length || extractive?.sentences?.reduce((a, s) => a + s.words, 0) || 0) / raw_word_count) * 100)}%`
              : '—'
          }
          color={COLORS.wineDeep}
        />
      </div>

      {/* Extractive panel */}
      <Card
        className="rounded-2xl border-none shadow-md overflow-hidden"
        bodyStyle={{ padding: 0 }}
      >
        <div
          className="px-5 py-3 flex items-center justify-between"
          style={{ background: `linear-gradient(135deg, ${COLORS.navy} 0%, ${COLORS.purple} 100%)` }}
        >
          <div className="flex items-center gap-2 text-white">
            <BarChart3 size={16} style={{ color: COLORS.gold }} />
            <span className="font-semibold text-sm font-['Playfair_Display']">Bước Extractive · {extractive.method}</span>
          </div>
          <Tag style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: 'none', borderRadius: 999 }}>
            {extractive.sentences.length} câu
          </Tag>
        </div>
        <div className="p-5">
          <ol className="space-y-2 text-sm">
            {extractive.sentences.map((s, i) => (
              <li
                key={i}
                className="px-3 py-2 rounded-lg border-l-4"
                style={{ borderLeftColor: COLORS.wine, background: '#faf6f2', color: COLORS.dark }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <Tag color="default" style={{ marginRight: 6 }}>#{s.idx}</Tag>
                    {s.text}
                  </div>
                  <Tag color="gold" style={{ flexShrink: 0 }}>
                    {extractive.scores[i]?.toFixed?.(3) ?? '—'}
                  </Tag>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </Card>

      {/* Đồ thị câu (chỉ khi extractor là TextRank và backend trả về graph) */}
      {extractive?.method?.toLowerCase().includes('textrank') && extractive?.extra?.graph && (
        <TextRankGraph graph={extractive.extra.graph} />
      )}

      {/* Abstractive panel */}
      {abstractive && (
        <Card className="rounded-2xl border-none shadow-md overflow-hidden" bodyStyle={{ padding: 0 }}>
          <div
            className="px-5 py-3 flex items-center justify-between"
            style={{ background: `linear-gradient(135deg, ${COLORS.wine} 0%, ${COLORS.wineDeep} 100%)` }}
          >
            <div className="flex items-center gap-2 text-white">
              <Sparkles size={16} style={{ color: COLORS.gold }} />
              <span className="font-semibold text-sm font-['Playfair_Display']">Bước Abstractive · {abstractive.method}</span>
            </div>
            <Tag style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: 'none', borderRadius: 999 }}>
              {abstractive.num_chunks} chunk
            </Tag>
          </div>
          <div className="p-5">
            <Paragraph className="text-base leading-relaxed" style={{ color: COLORS.dark }}>
              {abstractive.text}
            </Paragraph>
            {abstractive.chunk_summaries?.length > 1 && (
              <details className="mt-3 text-xs" style={{ color: COLORS.subtle }}>
                <summary className="cursor-pointer font-semibold">Xem từng chunk</summary>
                <ol className="mt-2 space-y-1 list-decimal pl-5">
                  {abstractive.chunk_summaries.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ol>
              </details>
            )}
          </div>
        </Card>
      )}

      {/* ROUGE */}
      {rouge && (
        <Card className="rounded-2xl border-none shadow-md">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={16} style={{ color: COLORS.wine }} />
            <span className="font-semibold text-sm" style={{ color: COLORS.dark }}>
              ROUGE so với reference (clauses từ master_clauses.csv)
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <Statistic title="ROUGE-1 F" value={(rouge.rouge1_f * 100).toFixed(2)} suffix="%" />
            <Statistic title="ROUGE-2 F" value={(rouge.rouge2_f * 100).toFixed(2)} suffix="%" />
            <Statistic title="ROUGE-L F" value={(rouge.rougeL_f * 100).toFixed(2)} suffix="%" />
            <Statistic title="R1 Precision" value={(rouge.rouge1_p * 100).toFixed(2)} suffix="%" />
            <Statistic title="R1 Recall" value={(rouge.rouge1_r * 100).toFixed(2)} suffix="%" />
          </div>
        </Card>
      )}
    </div>
  );
};

const StatCard = ({ label, value, color }) => (
  <div
    className="rounded-xl px-4 py-3 shadow-sm"
    style={{ background: 'white', borderLeft: `4px solid ${color}` }}
  >
    <div className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: COLORS.muted }}>{label}</div>
    <div className="text-xl font-bold mt-1" style={{ color: COLORS.dark }}>{value ?? '—'}</div>
  </div>
);

export default TabSummarization;
