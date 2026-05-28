import { useEffect, useMemo, useRef, useState } from 'react';
import { Card, Tag, Slider, Switch, Tooltip } from 'antd';
import { Network, Info } from 'lucide-react';

// Vẽ đồ thị câu của TextRank bằng SVG + force-directed layout tự cài.
// Nodes = câu (kích thước theo PageRank score), edges = cosine similarity TF-IDF.
// Không cần thêm thư viện - dùng SVG thuần và useEffect chạy vài iteration để xếp nút.
const TextRankGraph = ({ graph }) => {
  const [hoverId, setHoverId] = useState(null);
  const [minEdge, setMinEdge] = useState(graph?.edge_threshold ?? 0.05);
  const [showLabels, setShowLabels] = useState(true);
  const svgRef = useRef(null);

  const width = 880;
  const height = 520;

  // Lọc cạnh theo ngưỡng người dùng kéo
  const visibleEdges = useMemo(
    () => (graph?.edges || []).filter((e) => e.weight >= minEdge),
    [graph, minEdge]
  );

  // Tính vị trí node bằng force layout (Fruchterman-Reingold rút gọn).
  // Chạy 1 lần khi graph đổi, kết quả lưu vào state để hover/slider không tính lại.
  const [positions, setPositions] = useState({});
  useEffect(() => {
    if (!graph?.nodes?.length) return;
    const nodes = graph.nodes;
    const edges = graph.edges || [];
    const N = nodes.length;
    // Khởi tạo vị trí ngẫu nhiên trên hình tròn
    const pos = {};
    const cx = width / 2;
    const cy = height / 2;
    const R = Math.min(width, height) / 2 - 60;
    nodes.forEach((n, i) => {
      const a = (2 * Math.PI * i) / N;
      pos[n.id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
    });
    // Tham số force layout
    const area = width * height;
    const k = Math.sqrt(area / Math.max(N, 1)) * 0.6;
    const iterations = 180;
    let temperature = width / 8;
    const cool = temperature / iterations;

    for (let it = 0; it < iterations; it++) {
      const disp = {};
      nodes.forEach((n) => { disp[n.id] = { x: 0, y: 0 }; });
      // Lực đẩy giữa các node (repulsive)
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = pos[a.id].x - pos[b.id].x;
          const dy = pos[a.id].y - pos[b.id].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
          const force = (k * k) / dist;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          disp[a.id].x += fx; disp[a.id].y += fy;
          disp[b.id].x -= fx; disp[b.id].y -= fy;
        }
      }
      // Lực kéo theo cạnh (attractive), tỉ lệ thuận với weight
      edges.forEach((e) => {
        const a = pos[e.source];
        const b = pos[e.target];
        if (!a || !b) return;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const force = ((dist * dist) / k) * (0.5 + e.weight);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        disp[e.source].x -= fx; disp[e.source].y -= fy;
        disp[e.target].x += fx; disp[e.target].y += fy;
      });
      // Lực gravity nhẹ kéo tất cả nút về center (giữ nút cô lập không bay góc)
      nodes.forEach((n) => {
        const dx = cx - pos[n.id].x;
        const dy = cy - pos[n.id].y;
        disp[n.id].x += dx * 0.02;
        disp[n.id].y += dy * 0.02;
      });
      // Giới hạn dịch chuyển theo temperature + clamp trong khung
      nodes.forEach((n) => {
        const d = disp[n.id];
        const dl = Math.sqrt(d.x * d.x + d.y * d.y) || 0.01;
        const lim = Math.min(dl, temperature);
        pos[n.id].x += (d.x / dl) * lim;
        pos[n.id].y += (d.y / dl) * lim;
        pos[n.id].x = Math.max(30, Math.min(width - 30, pos[n.id].x));
        pos[n.id].y = Math.max(30, Math.min(height - 30, pos[n.id].y));
      });
      temperature = Math.max(0.5, temperature - cool);
    }
    setPositions(pos);
  }, [graph]);

  if (!graph?.nodes?.length) return null;

  // Tỉ lệ score -> kích thước node để câu quan trọng to hơn
  const scores = graph.nodes.map((n) => n.score);
  const sMin = Math.min(...scores);
  const sMax = Math.max(...scores);
  const sizeOf = (s) => {
    if (sMax === sMin) return 10;
    const t = (s - sMin) / (sMax - sMin);
    return 6 + t * 18; // 6..24 px
  };

  const maxW = Math.max(...visibleEdges.map((e) => e.weight), 0.0001);

  return (
    <Card
      className="rounded-2xl border-none shadow-md overflow-hidden"
      bodyStyle={{ padding: 0 }}
    >
      <div
        className="px-5 py-3 flex items-center justify-between"
        style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #2d1f3e 100%)' }}
      >
        <div className="flex items-center gap-2 text-white">
          <Network size={16} style={{ color: '#e8d5b7' }} />
          <span className="font-semibold text-sm font-['Playfair_Display']">
            Đồ thị câu TextRank · PageRank α={graph.damping}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Tag style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: 'none' }}>
            {graph.nodes.length} nút
          </Tag>
          <Tag style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: 'none' }}>
            {visibleEdges.length}/{(graph.edges || []).length} cạnh
          </Tag>
        </div>
      </div>

      <div className="px-5 py-3 flex flex-wrap items-center gap-4 border-b" style={{ borderColor: '#eee' }}>
        <div className="flex items-center gap-2 text-xs" style={{ color: '#6b5b5e' }}>
          <span>Ngưỡng cạnh (cosine ≥)</span>
          <div style={{ width: 180 }}>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={minEdge}
              onChange={setMinEdge}
              tooltip={{ formatter: (v) => v?.toFixed(2) }}
            />
          </div>
          <Tag color="default">{minEdge.toFixed(2)}</Tag>
        </div>
        <div className="flex items-center gap-2 text-xs" style={{ color: '#6b5b5e' }}>
          <Switch checked={showLabels} onChange={setShowLabels} size="small" />
          <span>Hiện số thứ tự câu</span>
        </div>
        <Tooltip title="Nút to = PageRank cao. Nút vàng = câu được chọn vào extractive. Cạnh đậm = cosine TF-IDF cao.">
          <span className="text-xs flex items-center gap-1 ml-auto" style={{ color: '#9a8478' }}>
            <Info size={12} /> hover lên nút để xem nội dung câu
          </span>
        </Tooltip>
      </div>

      <div className="relative" style={{ background: '#faf6f2' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', display: 'block' }}
        >
          {/* Vẽ cạnh trước, node sau để node nằm trên */}
          {visibleEdges.map((e, i) => {
            const a = positions[e.source];
            const b = positions[e.target];
            if (!a || !b) return null;
            const opacity = 0.15 + 0.55 * (e.weight / maxW);
            const stroke = 0.5 + 2.5 * (e.weight / maxW);
            const dim = hoverId !== null && hoverId !== e.source && hoverId !== e.target;
            return (
              <line
                key={i}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke="#722F37"
                strokeOpacity={dim ? 0.05 : opacity}
                strokeWidth={stroke}
              />
            );
          })}
          {graph.nodes.map((n) => {
            const p = positions[n.id];
            if (!p) return null;
            const r = sizeOf(n.score);
            const isHover = hoverId === n.id;
            const fill = n.picked ? '#c9a96e' : '#1e3a5f';
            const stroke = isHover ? '#722F37' : (n.picked ? '#8b6b3d' : '#0d2640');
            return (
              <g
                key={n.id}
                onMouseEnter={() => setHoverId(n.id)}
                onMouseLeave={() => setHoverId(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle
                  cx={p.x} cy={p.y} r={r}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={isHover ? 3 : 1.5}
                  opacity={hoverId !== null && !isHover ? 0.35 : 1}
                />
                {showLabels && (
                  <text
                    x={p.x}
                    y={p.y + 3}
                    textAnchor="middle"
                    fontSize={Math.max(9, r * 0.55)}
                    fill={n.picked ? '#4a1520' : '#ffffff'}
                    fontWeight="bold"
                    style={{ pointerEvents: 'none' }}
                  >
                    {n.idx}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Tooltip khi hover lên node */}
        {hoverId !== null && positions[hoverId] && (() => {
          const n = graph.nodes.find((x) => x.id === hoverId);
          if (!n) return null;
          return (
            <div
              className="absolute pointer-events-none rounded-lg shadow-lg text-xs px-3 py-2"
              style={{
                left: Math.min(Math.max(8, positions[hoverId].x * (svgRef.current?.clientWidth / width || 1) + 12), (svgRef.current?.clientWidth || width) - 320),
                top: Math.max(4, positions[hoverId].y * ((svgRef.current?.clientHeight || height) / height) - 30),
                background: 'white',
                border: '1px solid #e8d5b7',
                maxWidth: 320,
                zIndex: 10,
              }}
            >
              <div className="flex items-center gap-2 mb-1">
                <Tag color={n.picked ? 'gold' : 'blue'} style={{ marginRight: 0 }}>#{n.idx}</Tag>
                <span style={{ color: '#722F37', fontWeight: 600 }}>
                  PR = {n.score.toFixed(5)}
                </span>
                <span style={{ color: '#9a8478' }}>· {n.words} từ</span>
              </div>
              <div style={{ color: '#2d2d2d' }}>{n.preview}</div>
            </div>
          );
        })()}
      </div>

      {/* Chú thích */}
      <div className="px-5 py-3 flex flex-wrap items-center gap-4 text-xs" style={{ color: '#6b5b5e', borderTop: '1px solid #eee' }}>
        <span className="flex items-center gap-1">
          <span style={{ width: 12, height: 12, background: '#c9a96e', borderRadius: '50%', display: 'inline-block' }} />
          Câu được chọn (top-K)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ width: 12, height: 12, background: '#1e3a5f', borderRadius: '50%', display: 'inline-block' }} />
          Câu còn lại
        </span>
        <span>Kích thước nút ∝ điểm PageRank · Độ đậm cạnh ∝ cosine TF-IDF</span>
      </div>
    </Card>
  );
};

export default TextRankGraph;
