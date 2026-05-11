const COLORS = {
  bg: '#0F172A', bg2: '#1E293B', text: '#F1F5F9', muted: '#94A3B8',
  accent: '#6366F1', accent2: '#A855F7', success: '#10B981',
  warning: '#F59E0B', danger: '#EF4444', blue: '#3B82F6', cyan: '#06B6D4'
};

Chart.defaults.color = COLORS.muted;
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.font.size = 11;

const charts = {};
let cachedHistory = [];
let lastPrediction = null;

document.querySelectorAll('.topnav a').forEach(a => {
  a.addEventListener('click', () => {
    document.querySelectorAll('.topnav a').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    const tab = a.dataset.tab;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById('page-' + tab).classList.add('active');
    if (tab === 'compare') ensureCompare();
    if (tab === 'history') loadHistory();
    setTimeout(() => Object.values(charts).forEach(c => c && c.resize()), 50);
  });
});

initDashboard();
initPredict();

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

/* ---------- DASHBOARD ---------- */
async function initDashboard() {
  try {
    const data = await fetchJSON('/api/dashboard');
    renderDashboard(data);
  } catch (e) {
    console.error('Dashboard load failed', e);
  }
}

function renderDashboard(data) {
  const k = data.kpis;
  document.getElementById('dash-subtitle').textContent =
    `${k.total_trips.toLocaleString()} chuyến · ${k.total_drivers} tài xế · ${k.total_routes} tuyến · ${data.date_range.from} → ${data.date_range.to}`;

  const kpis = document.getElementById('dash-kpis');
  kpis.innerHTML = `
    <div class="card kpi accent"><div class="label">Tổng chuyến</div><div class="value">${k.total_trips.toLocaleString()}</div></div>
    <div class="card kpi success"><div class="label">On-time rate</div><div class="value">${k.on_time_rate}%</div></div>
    <div class="card kpi warning"><div class="label">Avg delay</div><div class="value">${k.avg_delay} <span style="font-size:13px;color:var(--text-muted)">phút</span></div></div>
    <div class="card kpi"><div class="label">Drivers active</div><div class="value">${k.active_drivers} <span style="font-size:13px;color:var(--text-muted)">/ ${k.total_drivers}</span></div></div>
  `;

  charts.histogram = makeChart('chart-histogram', {
    type: 'bar',
    data: {
      labels: data.delay_histogram.labels,
      datasets: [{
        label: 'Số chuyến',
        data: data.delay_histogram.values,
        backgroundColor: gradientColor('chart-histogram', COLORS.accent, COLORS.accent2),
        borderRadius: 6
      }]
    },
    options: {
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 12 } }
      },
      scales: {
        x: {
          grid: { display: false },
          title: { display: true, text: 'Khoảng trễ (phút)', color: COLORS.muted }
        },
        y: {
          title: { display: true, text: 'Số chuyến', color: COLORS.muted }
        }
      }
    }
  });

  charts.monthly = makeChart('chart-monthly', {
    type: 'line',
    data: {
      labels: data.on_time_monthly.labels,
      datasets: [{
        label: 'On-time rate (%)',
        data: data.on_time_monthly.values,
        borderColor: COLORS.success,
        backgroundColor: 'rgba(16, 185, 129, 0.15)',
        fill: true, tension: 0.35, pointRadius: 3, borderWidth: 2
      }]
    },
    options: {
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 12 } }
      },
      scales: {
        x: {
          title: { display: true, text: 'Tháng', color: COLORS.muted }
        },
        y: {
          suggestedMin: 0, suggestedMax: 100,
          title: { display: true, text: 'Tỷ lệ đúng giờ (%)', color: COLORS.muted }
        }
      }
    }
  });

  document.getElementById('dash-routes').innerHTML = data.top_routes.map(r => `
    <tr>
      <td>${r.route}</td>
      <td class="mono">+${r.avg_delay}</td>
      <td><div>${r.on_time_pct}%</div><div class="progress"><div style="width:${r.on_time_pct}%"></div></div></td>
    </tr>`).join('') || '<tr><td colspan="3" class="empty">Không có dữ liệu</td></tr>';

  document.getElementById('dash-drivers').innerHTML = data.top_drivers.map(d => `
    <tr>
      <td>${d.driver_id}${d.name ? ' — ' + d.name : ''}</td>
      <td class="mono">${d.trips}</td>
      <td><div>${d.on_time_pct}%</div><div class="progress"><div style="width:${d.on_time_pct}%"></div></div></td>
    </tr>`).join('') || '<tr><td colspan="3" class="empty">Không có dữ liệu</td></tr>';
}

/* ---------- PREDICT ---------- */
async function initPredict() {
  const opts = await fetchJSON('/api/options');
  fillSelect('sel-driver', opts.drivers);
  fillSelect('sel-truck', opts.trucks);
  fillSelect('sel-route', opts.routes);
  fillSelectSimple('sel-loadtype', opts.load_types);
  fillSelectSimple('sel-booking', opts.bookings);

  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  document.getElementById('inp-sched').value = now.toISOString().slice(0, 16);

  document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-predict');
    btn.disabled = true; btn.textContent = '⏳ Đang dự đoán...';

    const fd = new FormData(e.target);
    const payload = Object.fromEntries(fd.entries());
    payload.weight_lbs = +payload.weight_lbs;
    payload.pieces = +payload.pieces;
    payload.revenue = +payload.revenue;
    payload.detention_minutes = +payload.detention_minutes;
    payload.route_label = document.getElementById('sel-route').selectedOptions[0]?.textContent || '';

    try {
      const res = await fetchJSON('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      lastPrediction = res;
      renderPrediction(res);
      document.getElementById('actual-section').style.display = 'block';
      document.getElementById('accuracy-result').style.display = 'none';
      document.getElementById('inp-actual').value = '';
    } catch (err) {
      alert('Predict failed: ' + err.message);
    } finally {
      btn.disabled = false; btn.textContent = '⚡ DỰ ĐOÁN DELAY';
    }
  });

  document.getElementById('btn-confirm-actual').addEventListener('click', async () => {
    if (!lastPrediction) return;
    const actualStr = document.getElementById('inp-actual').value;
    if (actualStr === '' || isNaN(+actualStr)) {
      alert('Nhập actual delay (phút)');
      return;
    }
    const actual = +actualStr;
    try {
      await fetchJSON(`/api/history/${lastPrediction.history_id}/actual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actual_delay_minutes: actual })
      });
    } catch (err) {
      console.warn('Update history failed', err);
    }
    renderAccuracy(lastPrediction.predicted_delay_minutes, actual);
  });
}

function renderAccuracy(predicted, actual) {
  const err = predicted - actual;
  const absErr = Math.abs(err);
  const denom = Math.max(Math.abs(actual), Math.abs(predicted), 1);
  const pct = Math.max(0, (1 - absErr / Math.max(denom, 30)) * 100);

  let grade, color;
  if (absErr <= 30) { grade = '🎯 Đạt (within 30m)'; color = 'var(--success)'; }
  else if (absErr <= 60) { grade = '⚠️ Chưa đạt'; color = 'var(--warning)'; }
  else if (absErr <= 120) { grade = '⚠️ Sai số lớn'; color = 'var(--warning)'; }
  else { grade = '✗ Kém'; color = 'var(--danger)'; }

  document.getElementById('accuracy-result').style.display = 'block';
  const accEl = document.getElementById('acc-pct');
  accEl.textContent = pct.toFixed(1) + '%';
  accEl.style.color = color;
  document.getElementById('acc-pred').textContent = (predicted >= 0 ? '+' : '') + predicted.toFixed(1) + ' min';
  document.getElementById('acc-actual').textContent = (actual >= 0 ? '+' : '') + actual.toFixed(1) + ' min';
  document.getElementById('acc-err').textContent = absErr.toFixed(1) + ' min';
  const gradeEl = document.getElementById('acc-grade');
  gradeEl.textContent = grade;
  gradeEl.style.color = color;
}

function fillSelect(id, items) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(i => `<option value="${i.id}">${i.label}</option>`).join('');
}
function fillSelectSimple(id, items) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(i => `<option value="${i}">${i}</option>`).join('');
}

function renderPrediction(res) {
  const delay = res.predicted_delay_minutes;
  const num = document.getElementById('gauge-num');
  const lbl = document.getElementById('gauge-lbl');
  const needle = document.getElementById('gauge-needle');

  let color = COLORS.success;
  let label = '✓ Đúng giờ';
  if (delay < -15) { color = COLORS.blue; label = '⏪ Sớm hơn lịch'; }
  else if (delay > 60) { color = COLORS.danger; label = '🚨 Trễ nghiêm trọng'; }
  else if (delay > 15) { color = COLORS.warning; label = '⚠️  Trễ vừa — Có rủi ro'; }

  num.textContent = (delay >= 0 ? '+' : '') + delay.toFixed(1) + ' min';
  num.style.color = color;
  lbl.textContent = label;

  if (res.scheduled_datetime && res.predicted_arrival_datetime) {
    document.getElementById('time-section').style.display = 'block';
    document.getElementById('time-scheduled').textContent = res.scheduled_datetime;
    const arrEl = document.getElementById('time-arrival');
    arrEl.textContent = res.predicted_arrival_datetime;
    arrEl.style.color = color;
  }

  const clamped = Math.max(-180, Math.min(360, delay));
  const angle = ((clamped + 180) / 540) * 180 - 90;
  needle.style.transform = `translateX(-50%) rotate(${angle}deg)`;

  document.getElementById('stat-conf').textContent = `±${(res.metrics.MAE || 15).toFixed(0)}m`;
  document.getElementById('stat-model').textContent = res.best_model.slice(0, 5).toUpperCase();
  const r2 = res.metrics.R2 || 0;
  const rel = r2 >= 0.8 ? 'HIGH' : (r2 >= 0.6 ? 'MED' : 'LOW');
  document.getElementById('stat-rel').textContent = rel;
  document.getElementById('stat-rel').style.color = r2 >= 0.8 ? COLORS.success : (r2 >= 0.6 ? COLORS.warning : COLORS.danger);

  const list = document.getElementById('contrib-list');
  if (!res.feature_contributions || res.feature_contributions.length === 0) {
    list.innerHTML = '<div class="empty">Model không cung cấp feature importance</div>';
  } else {
    const max = Math.max(...res.feature_contributions.map(c => c.importance));
    list.innerHTML = res.feature_contributions.map(c => {
      const w = max > 0 ? (c.importance / max) * 100 : 0;
      const hasShap = c.shap !== undefined && c.direction !== 'neutral';
      const sv = c.shap || 0;
      const barColor = hasShap
        ? (sv > 0 ? COLORS.danger : COLORS.success)
        : COLORS.accent;
      const valStr = hasShap
        ? (sv >= 0 ? '+' : '') + sv.toFixed(2) + 'm'
        : c.importance.toFixed(3);
      const valColor = hasShap ? barColor : 'var(--text)';
      return `
        <div class="imp-row">
          <span class="name" title="value: ${c.value}">${c.name}</span>
          <div class="bar-wrap"><div class="bar" style="width:${w}%; background:${barColor}"></div></div>
          <span class="val" style="color:${valColor}">${valStr}</span>
        </div>`;
    }).join('');
  }
}

/* ---------- COMPARE ---------- */
let compareLoaded = false;
async function ensureCompare() {
  if (compareLoaded) return;
  compareLoaded = true;
  const data = await fetchJSON('/api/compare');
  renderCompare(data);
}

function renderCompare(data) {
  const m = data.metrics;
  const best = data.best_model;
  const cards = document.getElementById('model-cards');
  cards.innerHTML = ['LinearRegression', 'RandomForest', 'XGBoost'].map(name => {
    const v = m[name] || {};
    const winner = name === best;
    return `
      <div class="model-card ${winner ? 'winner' : ''}">
        ${winner ? '<div class="star">⭐</div>' : ''}
        <div class="name" ${winner ? 'style="color: var(--accent);"' : ''}>${name}</div>
        <div class="metric-row r2"><span>R²</span><span class="v">${(v.R2 || 0).toFixed(4)}</span></div>
        <div class="metric-row"><span>MAE</span><span class="v">${(v.MAE || 0).toFixed(2)} min</span></div>
        <div class="metric-row"><span>RMSE</span><span class="v">${(v.RMSE || 0).toFixed(2)} min</span></div>
        <div class="metric-row"><span>Within 30m</span><span class="v">${(v.Within30min || 0).toFixed(1)}%</span></div>
        <div class="metric-row"><span>Train time</span><span class="v">${(v.TrainTime || 0).toFixed(1)}s</span></div>
      </div>`;
  }).join('');

  const names = ['LinearRegression', 'RandomForest', 'XGBoost'];
  const colors = [COLORS.blue, COLORS.accent2, COLORS.accent];

  charts.compare = makeChart('chart-compare', {
    type: 'bar',
    data: {
      labels: ['R²', 'Within 30m (÷100)'],
      datasets: names.map((n, i) => ({
        label: n,
        data: [m[n].R2, m[n].Within30min / 100],
        backgroundColor: colors[i],
        borderRadius: 4
      }))
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } },
      scales: {
        x: { title: { display: true, text: 'Chỉ số', color: COLORS.muted } },
        y: { suggestedMax: 1, title: { display: true, text: 'Giá trị', color: COLORS.muted } }
      }
    }
  });

  charts.errors = makeChart('chart-errors', {
    type: 'bar',
    data: {
      labels: ['MAE', 'RMSE'],
      datasets: names.map((n, i) => ({
        label: n, data: [m[n].MAE, m[n].RMSE], backgroundColor: colors[i], borderRadius: 4
      }))
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } },
      scales: {
        x: { title: { display: true, text: 'Chỉ số lỗi', color: COLORS.muted } },
        y: { title: { display: true, text: 'Phút (min)', color: COLORS.muted } }
      }
    }
  });

  charts.tradeoff = makeChart('chart-tradeoff', {
    type: 'scatter',
    data: {
      datasets: names.map((n, i) => ({
        label: n,
        data: [{ x: m[n].TrainTime, y: m[n].R2 }],
        backgroundColor: colors[i],
        pointRadius: 10,
        pointHoverRadius: 12
      }))
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } },
      scales: {
        x: { title: { display: true, text: 'Thời gian huấn luyện (giây)', color: COLORS.muted } },
        y: { title: { display: true, text: 'R² (độ chính xác)', color: COLORS.muted }, suggestedMin: 0.75, suggestedMax: 0.9 }
      }
    }
  });
}

/* ---------- HISTORY ---------- */
async function loadHistory() {
  const data = await fetchJSON('/api/history');
  cachedHistory = data.items;
  renderHistory(data);
  document.getElementById('btn-refresh-history').onclick = loadHistory;
  document.getElementById('filter-status').onchange = applyHistoryFilter;
  document.getElementById('filter-search').oninput = applyHistoryFilter;
  document.getElementById('btn-export').onclick = exportHistoryCSV;
}

function renderHistory(data) {
  const s = data.stats;
  document.getElementById('hist-subtitle').textContent =
    `${s.total} predictions · ${s.within_30_pct}% within 30 min · ${s.pending} đang chờ feedback`;

  document.getElementById('hist-kpis').innerHTML = `
    <div class="card kpi"><div class="label">Total predictions</div><div class="value">${s.total}</div></div>
    <div class="card kpi success"><div class="label">Within 30m</div><div class="value">${s.within_30_pct}%</div></div>
    <div class="card kpi warning"><div class="label">Avg abs error</div><div class="value">${s.avg_error} <span style="font-size:13px;color:var(--text-muted)">min</span></div></div>
    <div class="card kpi"><div class="label">Pending</div><div class="value">${s.pending}</div></div>
  `;
  applyHistoryFilter();
  renderHistoryChart();
}

function applyHistoryFilter() {
  const status = document.getElementById('filter-status').value;
  const q = document.getElementById('filter-search').value.toLowerCase();
  let items = cachedHistory;
  if (status) items = items.filter(i => i.status === status);
  if (q) items = items.filter(i =>
    (i.driver_id || '').toLowerCase().includes(q) ||
    (i.route_label || '').toLowerCase().includes(q)
  );
  renderHistoryTable(items);
}

function renderHistoryTable(items) {
  const body = document.getElementById('hist-tbody');
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">Chưa có prediction nào — thử thêm chuyến ở tab Predict</td></tr>';
    return;
  }
  body.innerHTML = items.slice(0, 50).map(it => {
    const pred = it.predicted_delay_minutes;
    const predStr = (pred >= 0 ? '+' : '') + pred.toFixed(0);
    const predColor = pred > 60 ? 'var(--danger)' : (pred > 15 ? 'var(--warning)' : (pred < -15 ? 'var(--blue)' : 'var(--success)'));
    let actual = '―', err = '―', pill = '<span class="pill wait">Đang chờ</span>';
    if (it.actual_delay_minutes !== null) {
      actual = (it.actual_delay_minutes >= 0 ? '+' : '') + it.actual_delay_minutes.toFixed(0);
      const e = Math.abs(it.error_minutes);
      err = e.toFixed(0);
      if (e <= 30) pill = '<span class="pill ok">Within 30m</span>';
      else if (e <= 60) pill = '<span class="pill warn">Off 30m+</span>';
      else pill = '<span class="pill bad">Off 60m+</span>';
    }
    const route = (it.route_label || '').replace(/\s*\([^)]*\)\s*/, '');
    return `<tr>
      <td class="mono">#${it.id}</td>
      <td>${it.timestamp.slice(5, 16)}</td>
      <td>${it.driver_id || ''}</td>
      <td>${route}</td>
      <td class="mono" style="color:${predColor}">${predStr}</td>
      <td class="mono">${actual}</td>
      <td class="mono">${err}</td>
      <td>${pill}</td>
    </tr>`;
  }).join('');
}

function renderHistoryChart() {
  const counts = { ON_TIME: 0, EARLY: 0, MODERATE_DELAY: 0, SEVERE_DELAY: 0 };
  cachedHistory.forEach(it => { if (counts[it.status] !== undefined) counts[it.status]++; });
  charts.history = makeChart('chart-history', {
    type: 'doughnut',
    data: {
      labels: ['Đúng giờ (≤30m)', 'Sớm hơn (<-15m)', 'Trễ vừa (30–60m)', 'Trễ nặng (>60m)'],
      datasets: [{
        label: 'Số dự đoán',
        data: [counts.ON_TIME, counts.EARLY, counts.MODERATE_DELAY, counts.SEVERE_DELAY],
        backgroundColor: [COLORS.success, COLORS.blue, COLORS.warning, COLORS.danger],
        borderWidth: 0
      }]
    },
    options: {
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, padding: 8 } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed} chuyến`
          }
        },
        datalabels: {
          display: true,
          formatter: (value, ctx) => {
            const total = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
            if (!total || value === 0) return '';
            return ((value / total) * 100).toFixed(1) + '%';
          },
          color: '#fff',
          font: { weight: 'bold', size: 12 }
        }
      },
      cutout: '60%'
    }
  });
}

function exportHistoryCSV() {
  const headers = ['id', 'timestamp', 'driver_id', 'route_label', 'predicted_delay_minutes', 'actual_delay_minutes', 'error_minutes', 'status', 'model'];
  const rows = cachedHistory.map(it => headers.map(h => JSON.stringify(it[h] ?? '')).join(','));
  const csv = headers.join(',') + '\n' + rows.join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'predictions_history.csv'; a.click();
  URL.revokeObjectURL(url);
}

/* ---------- helpers ---------- */
function makeChart(id, config) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (charts[id.replace('chart-', '')]) charts[id.replace('chart-', '')].destroy();
  config.options = config.options || {};
  config.options.responsive = true;
  config.options.maintainAspectRatio = false;
  config.options.plugins = config.options.plugins || {};
  if (!config.options.plugins.datalabels) {
    config.options.plugins.datalabels = { display: false };
  }
  return new Chart(el, config);
}

function gradientColor(canvasId, c1, c2) {
  return (ctx) => {
    const chart = ctx.chart;
    const { ctx: c, chartArea } = chart;
    if (!chartArea) return c1;
    const g = c.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    g.addColorStop(0, c1);
    g.addColorStop(1, c2);
    return g;
  };
}
