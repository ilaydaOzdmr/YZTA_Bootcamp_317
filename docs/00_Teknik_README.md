<p align="center">
  <img src="banner.png" alt="ResilienceOS Banner" width="860"/>
</p>

# ResilienceOS
### KOBİ Finansal & Operasyonel Dijital İkiz + Otonom Yapay Zekâ Orkestrası
**YZTA Bootcamp 2026 — Grup 317**

KOBİ'lerin faturalarını, banka hareketlerini, stoklarını ve tedarikçi verilerini tek bir
ortak veri yapısında birleştirip; **nakit akışı, tahsilat ve stok risklerini gerçekleşmeden
tahmin eden**, şok senaryolarını simüle edip **aksiyon planı üreten** yapay zekâ destekli
karar platformu.

> **Headline demo:** 31 Mayıs'ta sistem, 18 Haziran'da bir nakit darboğazı öngörür
> (kasa −59.780 TL, kriz olasılığı %56, tarih sapması 0 gün). Sebep: ABC Tekstil'in 180K
> faturası ~21 gün gecikecek + aynı hafta büyük ham madde ödemesi + kritik stok tükenmesi.
> Sistem çözüm üretir: erken ödeme indirimi + tedarikçi ödemesini bölme → kombine aksiyonla
> kriz olasılığı %0'a iner.

---

## Proje Yapısı

```
src/
├── data_generator/     Sentetik dijital ikiz üretici (Faker) + doğrulama (13 kontrol)
│   ├── generate_digital_twin.py    → data/digital_twin.db (8 FK-tutarlı tablo)
│   └── verify_digital_twin.py
├── data_ingest/        Gerçek veri çekiciler
│   ├── fetch_evds.py    → TCMB EVDS makro (USD/TRY, TÜFE, faiz)
│   └── fetch_kaggle.py  → 4 gerçek veri seti
├── models/             ML modelleri (LightGBM)
│   ├── train_invoice_delay.py     → fatura gecikme tahmini (RMSE 4.30 gün)
│   ├── forecast_cashflow.py       → nakit akışı ileri projeksiyonu ⭐
│   └── forecast_demand_stock.py   → talep tahmini + stok tükenme
├── simulation/         Şok & counterfactual motoru ("Aksiyonları Uygula")
│   └── shock_simulation.py
├── analysis/           Kalibrasyon + backtest doğrulaması (sentetik ↔ gerçek)
│   ├── validate_calibration.py
│   ├── rolling_backtest.py
│   └── validate_real_transfer.py
├── core/               twin_api.py — tüm motorları tek temiz arayüzde toplar
├── api/                server.py — twin_api'yi HTTP'ye açar (FastAPI, 8 endpoint)
└── agents/             CrewAI multi-agent katmanı
    └── agents.py · tasks.py · tools.py · main_crew.py

data/      digital_twin.db · kaggle/ (gerçek veri setleri)
models/    eğitilmiş modeller (*.txt)
reports/   metrikler (*.json) + tahminler (*.csv) + agent_report.md
frontend/  statik dashboard (index.html, dashboard.html, app.js, style.css)
docs/      dataset araştırması + teknik doküman + sprint planları
```

---

## Kurulum ve Hazırlık

```bash

# Projenin yerel ortamda sorunsuz çalışması için aşağıdaki adımları sırasıyla uygulayın:
# Repoyu Klonlayın:

git clone https://github.com/ilaydaOzdmr/YZTA_Bootcamp_317.git
cd YZTA_Bootcamp_317

# Sanal ortamı oluştur (Proje ana dizinindeyken)
python -m venv venv

# Sanal ortamı aktif et (Windows için)
venv\Scripts\activate
# (Mac/Linux için: source venv/bin/activate)

# Çevresel Değişkenleri Ayarlayın: Proje kök dizininde bir ".env" dosyası oluşturun ve Gemini API anahtarınızı ekleyin. 
#(API Key almak için: https://aistudio.google.com/api-keys) 
GEMINI_API_KEY=sizin_api_anahtariniz_buraya_gelecek

# Gerekli paketleri kur
pip install -r requirements.txt

# 1) Sentetik veriyi üret + doğrula
python src/data_generator/generate_digital_twin.py
python src/data_generator/verify_digital_twin.py       # 13/13 geçmeli

# 2) (Opsiyonel) Gerçek veri  — kimlik gerekir (.env / ~/.kaggle/kaggle.json)
python src/data_ingest/fetch_evds.py                   # gerçek makroyu yükler
python src/data_ingest/fetch_kaggle.py                 # 4 veri seti indirir

# 3) Modelleri eğit + tahmin
python src/models/train_invoice_delay.py
python src/models/forecast_cashflow.py
python src/models/forecast_demand_stock.py

# 4) Şok senaryoları
python src/simulation/shock_simulation.py

# 5) Kalibrasyonu gerçek veriyle doğrula
python src/analysis/validate_calibration.py
```

> ⚠️ **Önemli:** `generate_digital_twin.py`, `macro_daily`'yi placeholder ile yeniden yazar.
> Gerçek makro istiyorsan **generate'den sonra `fetch_evds.py`'yi tekrar çalıştır.**

Kimlikler için `.env.example` → `.env` (ayrıntı: `src/data_ingest/*.py` başlıkları).

---

## Mevcut Durum (Sprint 1)

| Bileşen | Sonuç |
|---------|-------|
| Sentetik dijital ikiz | ✅ 8 tablo, 13/13 doğrulama |
| Gerçek veri | ✅ EVDS makro + 4 Kaggle veri seti |
| Kalibrasyon | ✅ Sentetik gecikme gerçek IBM dağılımına kalibre (validate_calibration ile doğrulandı) |
| Fatura gecikme modeli | ✅ RMSE 4.30 gün (%41.3 iyileşme) |
| Nakit projeksiyonu | ✅ Krizi 0 gün sapmayla öngörür (−59.780 @ 18 Haz, kriz olasılığı %56) |
| Talep/stok modeli | ✅ RMSE 2.78 adet (%53.4 iyileşme) |
| Şok simülasyonu | ✅ Kombine çözüm krizi çözer (kriz olasılığı → %0) |

**Sonraki:** ML modellerinin iyileştirilmesi + simülasyon motoru (Sprint 2), ardından
arayüz (dashboard) ve multi-agent katman (Sprint 3).

---

## Mevcut Durum (Sprint 2 — ML Modelleri)

| Bileşen | Detay | Sonuç |
|---------|-------|-------|
| Fatura gecikme tahmini | LightGBM, 5-fold CV, 20 özellik (RFM + tarihsel + anomali; sızıntısız) | ✅ RMSE 4.30 gün, %91.1 geç/zamanında doğruluk |
| Nakit akışı projeksiyonu | LightGBM tahsilat zamanlaması + direct-method defter, 3 senaryo + Monte Carlo | ✅ Krizi 0 gün sapmayla öngörür, kriz olasılığı %56 |
| Talep & stok tükenme | LightGBM, EWMA + tedarikçi lead time + maliyet analizi | ✅ RMSE 2.78 adet, %53.4 iyileşme |

**Demo çıktıları:**
- ABC Tekstil #4218 (180.000 TL) → Risk **86.0 / KRİTİK**, tahmini gecikme ~21 gün
- Ham Kumaş - ND-7234 → **KRİTİK** (stok = 0, 30 Haz tükeniyor), 30g kesinti maliyeti: 5.505 TL

**Sonraki:** `shock_simulation.py` ("Aksiyonları Uygula" motoru) + CrewAI multi-agent katmanı.

---

## Mevcut Durum (Sprint 3 — Arayüz & Multi-Agent)

| Bileşen | Detay | Sonuç |
|---------|-------|-------|
| FastAPI backend | `twin_api`'yi HTTP'ye açar; 8 endpoint, her biri canlı/offline iki modlu | ✅ overview, cash-forecast, invoice-risk, stock-risk, scenarios, simulate, recommend |
| Dashboard (frontend) | Statik HTML/JS + Chart.js; kriz göstergesi, 30 günlük projeksiyon grafiği, fatura/stok risk tabloları, şok simülatörü | ✅ `index.html` (açılış) + `dashboard.html` |
| Multi-agent katman | CrewAI, 4 ajan (CFO / Tahsilat / Tedarik / Risk Orkestratör), LLM: Gemini 2.0 Flash | ✅ `src/agents/` — `twin_api`'yi tool olarak çağırır, rapor → `reports/agent_report.md` |

**Çalıştırma:**
- Dashboard: `uvicorn src.api.server:app --port 8000` → http://localhost:8000
- Ajanlar: kök dizine `.env` (`GEMINI_API_KEY=...`) + `python src/agents/main_crew.py`

---

Ayrıntılı veri seti araştırması: [`01_Dataset_Arastirmasi_ve_Veri_Workflow.md`](01_Dataset_Arastirmasi_ve_Veri_Workflow.md)
