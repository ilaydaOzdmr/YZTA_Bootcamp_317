const TRY_FMT = new Intl.NumberFormat('tr-TR', { maximumFractionDigits: 0 });
const PCT_FMT = (v) => (v === null || v === undefined ? '—' : `%${Math.round(v * 100)}`);
const money = (v) => (v === null || v === undefined ? '—' : `${TRY_FMT.format(v)} TL`);
const num = (v) => (v === null || v === undefined ? '—' : TRY_FMT.format(v));

function fmtDate(iso) {
  if (!iso) return '—';
  const [y, m, d] = String(iso).split('-');
  if (!d) return iso;
  return `${d}.${m}.${y}`;
}

function riskZone(prob) {
  if (prob === null || prob === undefined) return 'warn';
  if (prob < 0.25) return 'safe';
  if (prob < 0.6) return 'warn';
  return 'crisis';
}

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API hatası (${res.status}) — ${path}`);
  }
  return res.json();
}

/* ==================== HERO / GAUGE / STATS (cash-forecast) ==================== */

function setVerdict(data) {
  const zone = riskZone(data.kriz_olasiligi);
  const word = document.getElementById('verdict-word');
  const detail = document.getElementById('verdict-detail');
  const labels = { safe: 'GÜVENLİ', warn: 'RİSKLİ', crisis: 'KRİZ RİSKİ' };
  word.textContent = labels[zone];
  word.className = `verdict-word ${zone}`;

  const cukurTumce = data.goes_negative
    ? `Kasa ${fmtDate(data.trough_date)} tarihinde eksiye düşebilir.`
    : `Kasa ${fmtDate(data.trough_date)} tarihinde ${money(data.trough)} seviyesine inebilir.`;
  detail.textContent =
    `${cukurTumce} Bu, ${money(data.buffer)} güvenlik tamponunun ` +
    `${data.trough < data.buffer ? 'altında' : 'üstünde'} kalır. ` +
    `Monte Carlo simülasyonuna göre 30 gün içinde tampon ihlali olasılığı ${PCT_FMT(data.tampon_ihlali_olasiligi)}.`;
}

function setGauge(prob) {
  const zone = riskZone(prob);
  document.getElementById('gauge-pct').textContent = PCT_FMT(prob);
  document.getElementById('gauge-pct').style.color =
    zone === 'safe' ? '#33B682' : zone === 'warn' ? '#E8A33D' : '#E24C4C';
  const angleDeg = 180 - prob * 180;
  const rad = (angleDeg * Math.PI) / 180;
  const x = 100 + 70 * Math.cos(rad);
  const y = 100 - 70 * Math.sin(rad);
  const needle = document.getElementById('gauge-needle');
  needle.setAttribute('x2', x.toFixed(1));
  needle.setAttribute('y2', y.toFixed(1));
}

function setStats(data) {
  document.getElementById('stat-start').textContent = money(data.start_balance);
  const troughCard = document.getElementById('stat-trough-card');
  document.getElementById('stat-trough').textContent = money(data.trough);
  document.getElementById('stat-trough-date').textContent = fmtDate(data.trough_date);
  troughCard.className = 'stat-card ' + riskZone(data.kriz_olasiligi);
  document.getElementById('stat-buffer-prob').textContent = PCT_FMT(data.tampon_ihlali_olasiligi);
  document.getElementById('stat-buffer-amount').textContent = `tampon: ${money(data.buffer)}`;
  document.getElementById('stat-crisis-date').textContent = fmtDate(data.en_olasi_kriz_tarihi);
  document.getElementById('stat-ar-ap').textContent = `${money(data.acik_alacak_tl)} / ${money(data.acik_borc_tl)}`;
}

function setStockBridge(data) {
  const section = document.getElementById('stockbridge-section');
  if (!data.stok_koprusu) { section.hidden = true; return; }
  const sk = data.stok_koprusu;
  document.getElementById('stockbridge-text').textContent =
    `Kritik stok seviyelerindeki ürünler için ${money(sk.acil_stok_ihtiyaci)} tutarında ` +
    `plansız acil sipariş gerekebilir. Bu ek yük hesaba katıldığında kriz olasılığı ` +
    `${PCT_FMT(sk.kriz_olasiligi_3ayak)} olarak güncellenir.`;
  section.hidden = false;
}

function buildLegend() {
  const legend = document.getElementById('chart-legend');
  legend.innerHTML = '';
  [
    { label: 'Gerçekleşen', color: '#EDEAE2' },
    { label: 'İyimser', color: '#33B682' },
    { label: 'Normal', color: '#6C8CFF' },
    { label: 'Kötümser', color: '#E24C4C' },
    { label: 'Tampon eşiği', color: '#5F6890' },
  ].forEach(({ label, color }) => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${label}`;
    legend.appendChild(item);
  });
}

function renderChart(data) {
  const rows = data.scenario_curves || [];
  const labels = rows.map((r) => fmtDate(r.date));
  const buffer = data.buffer;
  buildLegend();

  const ctx = document.getElementById('cashflow-chart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Gerçekleşen', data: rows.map((r) => r.gerceklesen), borderColor: '#EDEAE2', borderWidth: 2, pointRadius: 0, spanGaps: true },
        { label: 'İyimser', data: rows.map((r) => r.iyimser), borderColor: '#33B682', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
        { label: 'Normal', data: rows.map((r) => r.normal), borderColor: '#6C8CFF', borderWidth: 2.5, pointRadius: 0 },
        { label: 'Kötümser', data: rows.map((r) => r.kotumser), borderColor: '#E24C4C', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
        { label: 'Tampon eşiği', data: rows.map(() => buffer), borderColor: '#5F6890', borderDash: [2, 3], borderWidth: 1, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: '#2A3358' }, ticks: { color: '#5F6890', font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 8 } },
        y: { grid: { color: '#2A3358' }, ticks: { color: '#5F6890', font: { family: 'IBM Plex Mono', size: 10 }, callback: (v) => TRY_FMT.format(v) } },
      },
    },
  });
}

async function loadCashForecast() {
  const data = await getJSON('/api/cash-forecast');
  document.getElementById('cutoff-date').textContent = fmtDate(data.cutoff);
  applySourceBadge(data.source);
  setVerdict(data);
  setGauge(data.kriz_olasiligi);
  setStats(data);
  setStockBridge(data);
  renderChart(data);
}

/* ==================== OVERVIEW (topbar) ==================== */

async function loadOverview() {
  const data = await getJSON('/api/overview');
  document.getElementById('ov-musteri').textContent = data.musteri ?? '—';
  document.getElementById('ov-kasa').textContent = money(data.guncel_kasa);
}

/* ==================== INVOICE RISK TABLE ==================== */

function riskBadgeFromDaysLate(days) {
  if (days >= 14) return { cls: 'kritik', label: 'KRİTİK' };
  if (days >= 7) return { cls: 'yuksek', label: 'YÜKSEK' };
  if (days > 0) return { cls: 'orta', label: 'ORTA' };
  return { cls: 'guvenli', label: 'GÜVENLİ' };
}

async function loadInvoiceRisk() {
  const { items } = await getJSON('/api/invoice-risk?top_n=8');
  const body = document.getElementById('invoice-risk-body');
  if (!items || items.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="table-loading">Veri yok</td></tr>';
    return;
  }
  body.innerHTML = items.map((r) => {
    const badge = r.risk_label
      ? { cls: r.risk_label.toLowerCase(), label: r.risk_label }
      : riskBadgeFromDaysLate(r.pred_days_late);
    return `<tr>
      <td class="mono">#${r.invoice_id}${r.disputed ? ' ⚠' : ''}</td>
      <td class="mono">#${r.customer_id}</td>
      <td class="mono">${money(r.amount)}</td>
      <td class="mono">${fmtDate(r.due_date)}</td>
      <td class="mono">${r.pred_days_late.toFixed(1)} gün</td>
      <td><span class="badge ${badge.cls}">${badge.label}</span></td>
    </tr>`;
  }).join('');
}

/* ==================== STOCK RISK TABLE ==================== */

async function loadStockRisk() {
  const { items } = await getJSON('/api/stock-risk?top_n=8');
  const body = document.getElementById('stock-risk-body');
  if (!items || items.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="table-loading">Veri yok</td></tr>';
    return;
  }
  body.innerHTML = items.map((r) => {
    const badge = { cls: (r.risk_level || 'orta').toLowerCase(), label: r.risk_level || '—' };
    return `<tr>
      <td>${r.product_name ?? ('#' + r.product_id)}</td>
      <td class="mono">${r.category ?? '—'}</td>
      <td class="mono">${num(r.current_stock)}</td>
      <td class="mono">${r.days_to_zero !== undefined ? Number(r.days_to_zero).toFixed(0) + ' gün' : '—'}</td>
      <td class="mono">${fmtDate(r.order_by_date)}</td>
      <td class="mono">${money(r.stockout_cost_30d)}</td>
      <td><span class="badge ${badge.cls}">${badge.label}</span></td>
    </tr>`;
  }).join('');
}

/* ==================== SIMULATOR: scenarios + simulate + recommend ==================== */

function statusClass(status) {
  if (status === 'GUVENLI') return 'status-guvenli';
  if (status === 'RISKLI') return 'status-riskli';
  return 'status-kriz';
}

function renderSimResult(r) {
  const panel = document.getElementById('sim-result');
  panel.hidden = false;
  panel.innerHTML = `
    <div class="sim-title">${r.label}</div>
    <div class="sim-field"><span class="sim-label">DURUM</span><span class="sim-value ${statusClass(r.status)}">${r.status}</span></div>
    <div class="sim-field"><span class="sim-label">EN DÜŞÜK NOKTA</span><span class="sim-value">${money(r.trough)}</span></div>
    <div class="sim-field"><span class="sim-label">TARİH</span><span class="sim-value">${fmtDate(r.trough_date)}</span></div>
    <div class="sim-field"><span class="sim-label">KRİZ OLASILIĞI</span><span class="sim-value">${PCT_FMT(r.kriz_olasiligi)}</span></div>
  `;
}

async function runSimulation(key, chipEl) {
  document.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
  if (chipEl) chipEl.classList.add('active');
  try {
    const r = await getJSON(`/api/simulate/${encodeURIComponent(key)}`);
    renderSimResult(r);
  } catch (err) {
    showError(err.message);
  }
}

function renderChips(containerId, items, bestKey) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  items.forEach((item) => {
    const chip = document.createElement('button');
    chip.className = 'chip' + (item.key === bestKey ? ' best' : '');
    chip.textContent = item.label;
    chip.addEventListener('click', () => runSimulation(item.key, chip));
    container.appendChild(chip);
  });
}

async function loadSimulator() {
  const [{ items: scenarioItems }, rec] = await Promise.all([
    getJSON('/api/scenarios'),
    getJSON('/api/recommend'),
  ]);

  const shocks = scenarioItems.filter((s) => !s.is_action && s.key !== '0_baseline');
  const actions = scenarioItems.filter((s) => s.is_action);
  const bestKey = rec.en_iyi ? rec.en_iyi.scenario : null;

  renderChips('shock-chips', shocks, null);
  renderChips('action-chips', actions, bestKey);

  const banner = document.getElementById('recommend-banner');
  if (rec.en_iyi) {
    banner.className = 'recommend-banner resolved';
    banner.innerHTML =
      `Aksiyon alınmazsa kriz olasılığı <span class="rb-metric" style="color:var(--crisis)">${PCT_FMT(rec.baseline_kriz_olasiligi)}</span>. ` +
      `En iyi aksiyon: <strong>${rec.en_iyi.label}</strong> → kasa ${money(rec.en_iyi.trough)} seviyesine çıkar, ` +
      `kriz olasılığı <span class="rb-metric">${PCT_FMT(rec.en_iyi.kriz_olasiligi)}</span>'e düşer.`;
  } else {
    banner.className = 'recommend-banner';
    banner.innerHTML = `Mevcut aksiyonlardan hiçbiri kriz olasılığını (${PCT_FMT(rec.baseline_kriz_olasiligi)}) tek başına düşürmüyor — kombinasyon değerlendirilmeli.`;
  }
}

/* ==================== Ortak yardımcılar ==================== */

function applySourceBadge(source) {
  const badge = document.getElementById('source-badge');
  badge.textContent = source === 'live' ? 'CANLI VERİ' : 'OFFLINE / DEMO VERİSİ';
  badge.className = `source-badge ${source === 'live' ? 'live' : 'offline'}`;
  document.getElementById('footer-source').textContent =
    source === 'live'
      ? 'Kaynak: canlı pipeline (digital_twin.db)'
      : 'Kaynak: reports/ altındaki önceden hesaplanmış çıktılar';
}

function showError(message) {
  const banner = document.getElementById('error-banner');
  banner.textContent = message;
  banner.hidden = false;
}

async function init() {
  const tasks = [
    ['nakit akışı', loadCashForecast],
    ['genel durum', loadOverview],
    ['fatura riski', loadInvoiceRisk],
    ['stok riski', loadStockRisk],
    ['simülatör', loadSimulator],
  ];
  for (const [label, fn] of tasks) {
    try {
      await fn();
    } catch (err) {
      showError(`${label} yüklenemedi: ${err.message}`);
    }
  }
}

init();
