import { useState, useEffect, useRef, useMemo } from 'react';
import { Input, Select, Button, Slider, Card, Tag, Typography, Switch, Tooltip, Segmented, Upload, message, Checkbox } from 'antd';
import { Play, Pause, RotateCcw, SkipForward, SkipBack, ChevronsRight, Cpu, Zap, Search, Activity, Upload as UploadIcon, Download } from 'lucide-react';
import { ALGORITHMS, runStringMatching } from '../services/stringMatching';
import { api } from '../services/api';

const { TextArea } = Input;
const { Title, Text } = Typography;

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

const DEFAULT_TEXT = 'Bộ luật Dân sự 2015 quy định về quan hệ dân sự giữa các chủ thể bình đẳng. Quan hệ dân sự là nền tảng của pháp luật dân sự.';
const DEFAULT_PATTERN = 'dân sự';

const TabStringMatching = () => {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [pattern, setPattern] = useState(DEFAULT_PATTERN);
  const [selected, setSelected] = useState(['naive']); // subset of ['naive','kmp','boyer_moore']
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [results, setResults] = useState(null); // {naive?, kmp?, boyer_moore?}
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(300); // ms per step
  const [exporting, setExporting] = useState(false);
  const [running, setRunning] = useState(false);
  const timerRef = useRef(null);

  const ALGO_SHORT = { naive: 'Naïve', kmp: 'KMP', boyer_moore: 'BM' };

  const runAll = async () => {
    if (!pattern || pattern.length === 0) {
      message.warning('Vui lòng nhập pattern.');
      return;
    }
    if (selected.length === 0) {
      message.warning('Chọn ít nhất một thuật toán.');
      return;
    }
    try {
      setRunning(true);
      const out = await runStringMatching(text, pattern, selected, { caseSensitive, trace: true });
      setResults(out);
      setStepIndex(0);
      setPlaying(false);
    } catch (e) {
      message.error('Chạy thuật toán thất bại: ' + (e?.response?.data?.error || e.message));
    } finally {
      setRunning(false);
    }
  };

  const formatMs = (ms) => {
    if (ms == null) return '—';
    if (ms < 1) return `${(ms * 1000).toFixed(1)} µs`;
    if (ms < 1000) return `${ms.toFixed(3)} ms`;
    return `${(ms / 1000).toFixed(3)} s`;
  };

  // Big-O với n,m thực tế. n = độ dài text, m = độ dài pattern.
  const complexityLabel = (algoKey) => {
    const n = text.length;
    const m = pattern.length;
    const nm = n * m;
    const nPlusM = n + m;
    const nDivM = m > 0 ? Math.ceil(n / m) : n;
    if (algoKey === 'naive') {
      return {
        best: `O(n) = O(${n})`,
        worst: `O(n × m) = O(${n} × ${m}) = O(${nm})`,
      };
    }
    if (algoKey === 'kmp') {
      return {
        best: `O(n + m) = O(${n} + ${m}) = O(${nPlusM})`,
        worst: `O(n + m) = O(${n} + ${m}) = O(${nPlusM})`,
      };
    }
    if (algoKey === 'boyer_moore') {
      return {
        best: `O(n / m) = O(${n} / ${m}) = O(${nDivM})`,
        worst: `O(n × m) = O(${n} × ${m}) = O(${nm})`,
      };
    }
    return { best: '—', worst: '—' };
  };

  // Total step count: max across all algorithms (when compare mode)
  const maxSteps = useMemo(() => {
    if (!results) return 0;
    return Math.max(...Object.values(results).map((r) => r.steps.length));
  }, [results]);

  useEffect(() => {
    if (!playing) {
      clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setStepIndex((s) => {
        if (s >= maxSteps - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, speed);
    return () => clearInterval(timerRef.current);
  }, [playing, speed, maxSteps]);

  const reset = () => {
    setStepIndex(0);
    setPlaying(false);
  };

  const exportResults = async () => {
    if (!results) {
      message.warning('Hãy chạy thuật toán trước khi xuất kết quả.');
      return;
    }
    // Use positions from any algorithm (all should agree on found positions).
    const firstKey = Object.keys(results)[0];
    const positions = results[firstKey]?.steps?.slice(-1)[0]?.positions ?? [];
    const complexities = {};
    const comparisons = {};
    for (const [k, r] of Object.entries(results)) {
      complexities[k] = complexityLabel(k).worst;
      const lastStep = r.steps[r.steps.length - 1];
      comparisons[k] = lastStep?.comparisons ?? 0;
    }
    try {
      setExporting(true);
      const { data } = await api.post('/string-matching/export', {
        text,
        pattern,
        case_sensitive: caseSensitive,
        positions,
        occurrences: positions.length,
        complexities,
        comparisons,
      });
      if (data?.content && data?.filename) {
        const blob = new Blob([data.content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }
      message.success(`Đã xuất: ${data.filename}`);
    } catch (e) {
      message.error('Xuất kết quả thất bại: ' + (e?.response?.data?.error || e.message));
    } finally {
      setExporting(false);
    }
  };

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
            <Cpu size={16} style={{ color: COLORS.gold }} />
            <span className="text-white/90 text-sm font-medium">Trực quan hóa thuật toán</span>
          </div>
          <Title level={2} className="text-white mb-1 font-['Playfair_Display'] text-3xl! md:text-4xl!" style={{ color: '#ffffff' }}>
            Tìm kiếm văn bản
          </Title>
          <Text className="text-white/90 text-base block" style={{ color: 'rgba(255,255,255,0.9)' }}>
            Quan sát Naive, KMP và Boyer-Moore tìm pattern trong text.
          </Text>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 -mt-2">
        {/* Controls Card */}
        <Card
          className="rounded-2xl border-none shadow-md mb-6"
          style={{ background: '#ffffff' }}
          bodyStyle={{ padding: 20 }}
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs uppercase tracking-widest font-semibold block" style={{ color: COLORS.muted }}>
                  Text (chuỗi nguồn)
                </label>
                <Upload
                  accept=".txt,.md"
                  showUploadList={false}
                  beforeUpload={(file) => {
                    const name = (file.name || '').toLowerCase();
                    if (!name.endsWith('.txt') && !name.endsWith('.md')) {
                      message.error('Chỉ hỗ trợ file .txt hoặc .md');
                      return Upload.LIST_IGNORE;
                    }
                    const reader = new FileReader();
                    reader.onload = (e) => {
                      setText(String(e.target?.result ?? ''));
                      message.success(`Đã nạp ${file.name}`);
                    };
                    reader.onerror = () => message.error('Không đọc được file');
                    reader.readAsText(file, 'utf-8');
                    return false;
                  }}
                >
                  <Button
                    size="small"
                    icon={<UploadIcon size={12} />}
                    style={{ borderRadius: 8, fontSize: 12, height: 26 }}
                  >
                    Upload file (.txt, .md)
                  </Button>
                </Upload>
              </div>
              <TextArea
                rows={4}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Nhập đoạn văn bản hoặc tải file..."
                style={{ borderRadius: 10 }}
              />
            </div>
            <div>
              <label className="text-xs uppercase tracking-widest font-semibold mb-2 block" style={{ color: COLORS.muted }}>
                Pattern (chuỗi cần tìm)
              </label>
              <Input
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                placeholder="Nhập pattern..."
                style={{ borderRadius: 10 }}
              />
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Checkbox.Group
                  value={selected}
                  onChange={(vals) => setSelected(vals)}
                  options={Object.entries(ALGORITHMS).map(([k, v]) => ({ value: k, label: v.name }))}
                />
                <div className="flex items-center gap-2 text-xs" style={{ color: COLORS.subtle }}>
                  <span>Case-sensitive</span>
                  <Switch size="small" checked={caseSensitive} onChange={setCaseSensitive} />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Button
              type="primary"
              icon={<Search size={14} />}
              onClick={runAll}
              style={{ background: COLORS.wine, borderColor: COLORS.wine, borderRadius: 8, height: 38 }}
            >
              Chạy thuật toán
            </Button>
            <Button
              icon={playing ? <Pause size={14} /> : <Play size={14} />}
              onClick={() => setPlaying((p) => !p)}
              disabled={!results || stepIndex >= maxSteps - 1}
              style={{ borderRadius: 8, height: 38 }}
            />
            <Button
              icon={<SkipBack size={14} />}
              onClick={() => setStepIndex((s) => Math.max(0, s - 1))}
              disabled={!results}
              style={{ borderRadius: 8, height: 38 }}
            />
            <Button
              icon={<SkipForward size={14} />}
              onClick={() => setStepIndex((s) => Math.min(maxSteps - 1, s + 1))}
              disabled={!results}
              style={{ borderRadius: 8, height: 38 }}
            />
            <Button
              icon={<ChevronsRight size={14} />}
              onClick={() => setStepIndex(maxSteps - 1)}
              disabled={!results}
              style={{ borderRadius: 8, height: 38 }}
            >
              Đến cuối
            </Button>
            <Button icon={<RotateCcw size={14} />} onClick={reset} disabled={!results} style={{ borderRadius: 8, height: 38 }}>
              Reset
            </Button>
            <Button
              icon={<Download size={14} />}
              onClick={exportResults}
              loading={exporting}
              disabled={!results}
              style={{ borderRadius: 8, height: 38, borderColor: COLORS.goldDark, color: COLORS.wineDeep }}
            >
              Xuất kết quả
            </Button>
            <div className="flex items-center gap-2 ml-2 min-w-[200px]">
              <Zap size={14} style={{ color: COLORS.goldDark }} />
              <span className="text-xs" style={{ color: COLORS.subtle }}>Tốc độ</span>
              <Slider
                min={10}
                max={1590}
                step={10}
                value={1600 - speed}
                onChange={(v) => setSpeed(1600 - v)}
                style={{ width: 140 }}
              />
            </div>
            {results && (
              <Tag color="default" style={{ borderRadius: 999, marginLeft: 'auto' }}>
                Bước {stepIndex + 1} / {maxSteps}
              </Tag>
            )}
          </div>
        </Card>

        {/* Visualization */}
        {results ? (
          <div
            className="grid gap-5 grid-cols-1"
            style={{
              gridTemplateColumns:
                Object.keys(results).length > 1
                  ? `repeat(${Object.keys(results).length}, minmax(0, 1fr))`
                  : '1fr',
            }}
          >
            {Object.entries(results).map(([algoKey, res]) => (
              <AlgorithmPanel
                key={algoKey}
                algoKey={algoKey}
                result={res}
                text={text}
                pattern={pattern}
                stepIndex={Math.min(stepIndex, res.steps.length - 1)}
                isCompare={Object.keys(results).length > 1}
                complexityLabel={complexityLabel(algoKey)}
                elapsedLabel={formatMs(res.elapsedMs)}
              />
            ))}
          </div>
        ) : (
          <Card className="rounded-2xl border-none shadow-sm" style={{ background: '#ffffff' }} bodyStyle={{ padding: 40 }}>
            <div className="text-center" style={{ color: COLORS.muted }}>
              <Activity size={36} className="mx-auto mb-3 opacity-50" />
              <p className="text-sm">Nhập text + pattern và bấm <b>Chạy thuật toán</b> để xem mô phỏng.</p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

// ---------------------- AlgorithmPanel ----------------------
const AlgorithmPanel = ({ algoKey, result, text, pattern, stepIndex, isCompare, complexityLabel, elapsedLabel }) => {
  const step = result.steps[stepIndex] || result.steps[result.steps.length - 1];
  const algo = ALGORITHMS[algoKey];

  return (
    <Card
      className="rounded-2xl border-none shadow-md overflow-hidden"
      style={{ background: '#ffffff' }}
      bodyStyle={{ padding: 0 }}
    >
      {/* Header */}
      <div
        className="px-5 py-3 flex items-center justify-between"
        style={{ background: `linear-gradient(135deg, ${COLORS.navy} 0%, ${COLORS.purple} 100%)` }}
      >
        <div className="flex items-center gap-2">
          <Cpu size={16} style={{ color: COLORS.gold }} />
          <span className="text-white font-semibold text-sm font-['Playfair_Display']">{algo.name}</span>
        </div>
        <Tag style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: 'none', borderRadius: 999, fontSize: 11 }}>
          {algo.complexity}
        </Tag>
      </div>

      <div className="p-5">
        {/* Stats — hiển thị tổng kết cuối cùng để thấy ngay sau khi chạy */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          <StatBox label="So sánh" value={result.comparisons} color={COLORS.navy} />
          <StatBox label="Tìm thấy" value={result.positions.length} color={COLORS.wine} />
        </div>
        <div className="grid grid-cols-2 gap-2 mb-3">
          <StatBox label="Best case" value={complexityLabel?.best ?? '—'} color={COLORS.navy} small />
          <StatBox label="Worst case" value={complexityLabel?.worst ?? '—'} color={COLORS.wineDeep} small />
        </div>
        <div className="mb-4">
          <StatBox label="Vị trí" value={result.positions.join(', ') || '—'} color={COLORS.goldDark} small />
        </div>

        {/* Text visualization */}
        <TextStrip text={text} pattern={pattern} step={step} algoKey={algoKey} />

        {/* Pattern row */}
        <PatternStrip text={text} pattern={pattern} step={step} algoKey={algoKey} />

        {/* Step message */}
        <div
          className="mt-4 px-3 py-2 rounded-lg text-xs leading-relaxed border-l-4"
          style={{
            background: stepBgColor(step.type),
            borderLeftColor: stepBorderColor(step.type),
            color: COLORS.dark,
          }}
        >
          <span className="font-semibold mr-1" style={{ color: stepBorderColor(step.type) }}>
            [{step.type.toUpperCase()}]
          </span>
          {step.message}
        </div>

        {/* Algorithm-specific extras */}
        {algoKey === 'kmp' && step.lps && step.lps.length > 0 && (
          <LpsTable pattern={pattern} lps={step.lps} highlight={step.lpsI} />
        )}
        {algoKey === 'boyer_moore' && step.badChar && (
          <BadCharTable badChar={step.badChar} />
        )}

        {/* Mini progress bar */}
        {!isCompare && (
          <div className="mt-3 h-1.5 rounded-full overflow-hidden" style={{ background: '#f0e9e4' }}>
            <div
              className="h-full transition-all duration-200"
              style={{
                width: `${((stepIndex + 1) / result.steps.length) * 100}%`,
                background: `linear-gradient(90deg, ${COLORS.wine}, ${COLORS.goldDark})`,
              }}
            />
          </div>
        )}
      </div>
    </Card>
  );
};

// ---------------------- Sub components ----------------------
const StatBox = ({ label, value, color, small }) => (
  <div className="rounded-lg px-3 py-2" style={{ background: '#faf6f2', borderLeft: `3px solid ${color}` }}>
    <div className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: COLORS.muted }}>
      {label}
    </div>
    <div
      className={`font-bold ${small ? 'text-xs' : 'text-lg'}`}
      style={{ color: COLORS.dark, whiteSpace: 'normal', wordBreak: 'break-word' }}
    >
      {value}
    </div>
  </div>
);

const TextStrip = ({ text, pattern, step }) => {
  const m = pattern.length;
  // For naive & BM: the pattern shift base is step.i (the offset in text where pattern[0] lines up).
  // For KMP: step.i represents (textCursor - j), and step.textCursor is the current text index.
  const shiftBase = step.i;
  const currentTextIdx =
    step.textCursor != null ? step.textCursor : shiftBase + (step.j ?? 0);

  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest font-semibold mb-1.5" style={{ color: COLORS.muted }}>
        Text
      </div>
      <div className="flex flex-wrap gap-0.5 font-mono text-sm p-3 rounded-lg" style={{ background: '#faf6f2' }}>
        {[...text].map((ch, idx) => {
          const inWindow = idx >= shiftBase && idx < shiftBase + m;
          const isCurrent = idx === currentTextIdx;
          const isMatched = step.positions.some((p) => idx >= p && idx < p + m);
          let bg = 'transparent';
          let color = COLORS.dark;
          let border = '1px solid transparent';
          if (isMatched) {
            bg = COLORS.gold;
            color = COLORS.wineDeep;
          }
          if (inWindow && !isMatched) {
            bg = 'rgba(30,58,95,0.10)';
          }
          if (isCurrent) {
            if (step.type === 'match' || step.type === 'found') {
              bg = '#86c79b';
              color = 'white';
            } else if (step.type === 'mismatch') {
              bg = '#d97a7a';
              color = 'white';
            } else {
              border = `1px solid ${COLORS.navy}`;
            }
          }
          return (
            <span
              key={idx}
              className="inline-flex items-center justify-center transition-all duration-200"
              style={{
                minWidth: 18,
                padding: '4px 3px',
                borderRadius: 4,
                background: bg,
                color,
                border,
                fontWeight: isCurrent ? 700 : 500,
              }}
              title={`idx=${idx}`}
            >
              {ch === ' ' ? '␣' : ch}
            </span>
          );
        })}
      </div>
    </div>
  );
};

const PatternStrip = ({ text, pattern, step, algoKey }) => {
  const shiftBase = step.i;
  // Each text char is rendered with minWidth 18px + gap 2px = ~20px; we pad with empty cells to align.
  return (
    <div className="mt-3">
      <div className="text-[10px] uppercase tracking-widest font-semibold mb-1.5" style={{ color: COLORS.muted }}>
        Pattern{' '}
        {algoKey === 'boyer_moore' && (
          <span className="normal-case font-normal" style={{ color: COLORS.subtle }}>
            (so sánh phải → trái)
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-0.5 font-mono text-sm p-3 rounded-lg" style={{ background: 'rgba(114,47,55,0.06)' }}>
        {/* leading spacers */}
        {Array.from({ length: Math.max(0, shiftBase) }).map((_, idx) => (
          <span key={`sp-${idx}`} style={{ minWidth: 18, padding: '4px 3px' }} />
        ))}
        {[...pattern].map((ch, idx) => {
          const isCurrent = idx === step.j;
          let bg = 'rgba(114,47,55,0.10)';
          let color = COLORS.wineDeep;
          if (isCurrent) {
            if (step.type === 'match' || step.type === 'found') {
              bg = '#86c79b';
              color = 'white';
            } else if (step.type === 'mismatch') {
              bg = '#d97a7a';
              color = 'white';
            } else {
              bg = COLORS.wine;
              color = 'white';
            }
          }
          return (
            <span
              key={idx}
              className="inline-flex items-center justify-center transition-all duration-200"
              style={{
                minWidth: 18,
                padding: '4px 3px',
                borderRadius: 4,
                background: bg,
                color,
                fontWeight: isCurrent ? 700 : 500,
              }}
              title={`pattern[${idx}]`}
            >
              {ch === ' ' ? '␣' : ch}
            </span>
          );
        })}
      </div>
    </div>
  );
};

const LpsTable = ({ pattern, lps, highlight }) => (
  <div className="mt-4">
    <div className="text-[10px] uppercase tracking-widest font-semibold mb-1.5" style={{ color: COLORS.muted }}>
      Mảng LPS (Longest Prefix-Suffix)
    </div>
    <div className="overflow-x-auto rounded-lg" style={{ background: '#faf6f2' }}>
      <table className="font-mono text-xs w-full">
        <tbody>
          <tr>
            <td className="px-2 py-1 font-semibold" style={{ color: COLORS.muted, width: 50 }}>idx</td>
            {pattern.split('').map((_, i) => (
              <td key={i} className="px-2 py-1 text-center" style={{ color: COLORS.muted }}>{i}</td>
            ))}
          </tr>
          <tr>
            <td className="px-2 py-1 font-semibold" style={{ color: COLORS.dark }}>p</td>
            {pattern.split('').map((ch, i) => (
              <td key={i} className="px-2 py-1 text-center font-semibold" style={{ color: COLORS.wineDeep }}>{ch === ' ' ? '␣' : ch}</td>
            ))}
          </tr>
          <tr>
            <td className="px-2 py-1 font-semibold" style={{ color: COLORS.dark }}>lps</td>
            {lps.map((v, i) => (
              <td
                key={i}
                className="px-2 py-1 text-center font-bold transition-all"
                style={{
                  background: i === highlight ? COLORS.gold : 'transparent',
                  color: i === highlight ? COLORS.wineDeep : COLORS.navy,
                  borderRadius: 4,
                }}
              >
                {v}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  </div>
);

const BadCharTable = ({ badChar }) => {
  const entries = Object.entries(badChar);
  if (entries.length === 0) return null;
  return (
    <div className="mt-4">
      <div className="text-[10px] uppercase tracking-widest font-semibold mb-1.5" style={{ color: COLORS.muted }}>
        Bảng Bad-Character (last occurrence trong pattern)
      </div>
      <div className="flex flex-wrap gap-1.5 p-2 rounded-lg" style={{ background: '#faf6f2' }}>
        {entries.map(([ch, idx]) => (
          <Tooltip key={ch} title={`Vị trí cuối của '${ch}' trong pattern`}>
            <div className="flex items-center gap-1 px-2 py-1 rounded font-mono text-xs" style={{ background: 'white', border: `1px solid ${COLORS.gold}` }}>
              <span style={{ color: COLORS.wineDeep, fontWeight: 700 }}>{ch === ' ' ? '␣' : ch}</span>
              <span style={{ color: COLORS.muted }}>→</span>
              <span style={{ color: COLORS.navy, fontWeight: 700 }}>{idx}</span>
            </div>
          </Tooltip>
        ))}
      </div>
    </div>
  );
};

// ---------------------- step colors ----------------------
function stepBgColor(type) {
  switch (type) {
    case 'match': return 'rgba(134,199,155,0.15)';
    case 'mismatch': return 'rgba(217,122,122,0.12)';
    case 'found': return 'rgba(232,213,183,0.35)';
    case 'shift': return 'rgba(30,58,95,0.08)';
    case 'lps': return 'rgba(45,31,62,0.06)';
    default: return '#faf6f2';
  }
}
function stepBorderColor(type) {
  switch (type) {
    case 'match': return '#3b9b5a';
    case 'mismatch': return '#c14545';
    case 'found': return COLORS.goldDark;
    case 'shift': return COLORS.navy;
    case 'lps': return COLORS.purple;
    default: return COLORS.muted;
  }
}

export default TabStringMatching;
