const TRY_FMT = new Intl.NumberFormat('tr-TR', { maximumFractionDigits: 0 });
const PCT_FMT = (v) => `%${Math.round(v * 100)}`;
const money = (v) => (v === null || v === undefined ? '—' : `${TRY_FMT.format(v)} TL`);

function fmtDate(iso) {
  if (!iso) return '—';
  const [y, m, d] = iso.split('-');
  return `${d}.${m}.${y}`;
}

function riskZone(prob) {
  if (prob < 0.25) return 'safe';
  if (prob < 0.6) return 'warn';
  return 'crisis';
}

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

  // 0 -> needle points left (180deg), 1 -> needle points right (0deg)
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

  const troughEl = document.getElementById('stat-trough');
  const troughCard = document.getElementById('stat-trough-card');
  troughEl.textContent = money(data.trough);
  document.getElementById('stat-trough-date').textContent = fmtDate(data.trough_date);
  troughCard.className = 'stat-card ' + riskZone(data.kriz_olasiligi);

  document.getElementById('stat-buffer-prob').textContent = PCT_FMT(data.tampon_ihlali_olasiligi);
  document.getElementById('stat-buffer-amount').textContent = `tampon: ${money(data.buffer)}`;

  document.getElementById('stat-crisis-date').textContent = fmtDate(data.en_olasi_kriz_tarihi);

  document.getElementById('stat-ar-ap').textContent =
    `${money(data.acik_alacak_tl)} / ${money(data.acik_borc_tl)}`;
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

function buildLegend(rows) {
  const legend = document.getElementById('chart-legend');
  legend.innerHTML = '';
  rows.forEach(({ label, color }) => {
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

  buildLegend([
    { label: 'Gerçekleşen', color: '#EDEAE2' },
    { label: 'İyimser', color: '#33B682' },
    { label: 'Normal', color: '#6C8CFF' },
    { label: 'Kötümser', color: '#E24C4C' },
    { label: 'Tampon eşiği', color: '#5F6890' },
  ]);

  const ctx = document.getElementById('cashflow-chart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Gerçekleşen',
          data: rows.map((r) => r.gerceklesen),
          borderColor: '#EDEAE2',
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: true,
        },
        {
          label: 'İyimser',
          data: rows.map((r) => r.iyimser),
          borderColor: '#33B682',
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: 'Normal',
          data: rows.map((r) => r.normal),
          borderColor: '#6C8CFF',
          borderWidth: 2.5,
          pointRadius: 0,
        },
        {
          label: 'Kötümser',
          data: rows.map((r) => r.kotumser),
          borderColor: '#E24C4C',
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: 'Tampon eşiği',
          data: rows.map(() => buffer),
          borderColor: '#5F6890',
          borderDash: [2, 3],
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          grid: { color: '#2A3358' },
          ticks: { color: '#5F6890', font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 8 },
        },
        y: {
          grid: { color: '#2A3358' },
          ticks: {
            color: '#5F6890',
            font: { family: 'IBM Plex Mono', size: 10 },
            callback: (v) => TRY_FMT.format(v),
          },
        },
      },
    },
  });
}

function showError(message) {
  const banner = document.getElementById('error-banner');
  banner.textContent = message;
  banner.hidden = false;
}

async function init() {
  try {
    const res = await fetch('/api/cash-forecast');
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `API hatası (${res.status})`);
    }
    const data = await res.json();

    document.getElementById('cutoff-date').textContent = fmtDate(data.cutoff);
    const badge = document.getElementById('source-badge');
    badge.textContent = data.source === 'live' ? 'CANLI VERİ' : 'OFFLINE / DEMO VERİSİ';
    badge.className = `source-badge ${data.source === 'live' ? 'live' : 'offline'}`;
    document.getElementById('footer-source').textContent =
      data.source === 'live'
        ? 'Kaynak: canlı pipeline (digital_twin.db)'
        : 'Kaynak: reports/ altındaki önceden hesaplanmış çıktılar';

    setVerdict(data);
    setGauge(data.kriz_olasiligi);
    setStats(data);
    setStockBridge(data);
    renderChart(data);
  } catch (err) {
    showError(
      `Veri yüklenemedi: ${err.message}. Backend'in çalıştığından ` +
      `(uvicorn src.api.server:app) ve reports/ klasörünün mevcut olduğundan emin olun.`
    );
  }
}

init();
