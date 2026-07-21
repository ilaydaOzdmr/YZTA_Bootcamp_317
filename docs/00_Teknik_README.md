<p align="center">
  <img src="docs/banner.png" alt="ResilienceOS Banner" width="860"/>
</p>

# ResilienceOS
### KOBİ Finansal & Operasyonel Dijital İkiz + Otonom Yapay Zekâ Orkestrası
**YZTA Bootcamp 2026 — Grup 317**

KOBİ'lerin faturalarını, banka hareketlerini, stoklarını ve tedarikçi verilerini tek bir
ortak veri yapısında birleştirip; **nakit akışı, tahsilat ve stok risklerini gerçekleşmeden
tahmin eden**, şok senaryolarını simüle edip **aksiyon planı üreten** yapay zekâ destekli
karar platformu.

> **Headline demo:** 31 Mayıs'ta sistem, 17-19 Haziran'da bir nakit darboğazı öngörür
> (kasa −99K TL). Sebep: ABC Tekstil'in 180K faturası 17 gün gecikecek + aynı hafta büyük
> ham madde ödemesi + kritik stok tükenmesi. Sistem çözüm üretir: ABC'ye %2 erken ödeme
> indirimi + tedarikçi ödemesini bölme → kasa +89K'ya döner (kriz atlatılır).

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
│   ├── train_invoice_delay.py     → fatura gecikme tahmini (RMSE 4.65 gün)
│   ├── forecast_cashflow.py       → nakit akışı ileri projeksiyonu ⭐
│   └── forecast_demand_stock.py   → talep tahmini + stok tükenme
├── simulation/         Şok & counterfactual motoru ("Aksiyonları Uygula")
│   └── shock_simulation.py
└── analysis/           Kalibrasyon doğrulaması (sentetik ↔ gerçek)
    └── validate_calibration.py

data/     digital_twin.db · kaggle/ (gerçek veri setleri)
models/   eğitilmiş modeller (*.txt)
reports/  metrikler (*.json) + tahminler/projeksiyonlar (*.csv)
docs/     dataset araştırması + ekip özeti
```

---

## Hızlı Başlangıç

```bash
# Sanal ortamı oluştur (Proje ana dizinindeyken)
python -m venv venv

# Sanal ortamı aktif et (Windows için)
venv\Scripts\activate
# (Mac/Linux için: source venv/bin/activate)

# ".env" dosyası oluştur ve içerisine şunu ekle
GEMINI_API_KEY=senin_api_anahtarin_buraya

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
| Kalibrasyon | ✅ Sentetik gecikme gerçek IBM dağılımına uyumlu (ort 2.71g, medyan 0, p90 13) |
| Fatura gecikme modeli | ✅ RMSE 4.65 gün (%38.8 iyileşme) |
| Nakit projeksiyonu | ✅ Krizi 2 gün sapmayla öngörür (−99K @ 19 Haz) |
| Talep/stok modeli | ✅ RMSE 3.90 adet (%27.9) |
| Şok simülasyonu | ✅ ABC erken ödeme kurtarır (+77K), kombine en iyi (+89K) |

**Sonraki (Sprint 2):** Multi-agent katman (CFO / Tahsilat / Tedarik / Risk Denetçisi) —
bu motorları çağırıp otonom aksiyon üretir. **Sprint 3:** özel arayüz (dashboard).

---

## Mevcut Durum (Sprint 2 — ML Modelleri)

| Bileşen | Detay | Sonuç |
|---------|-------|-------|
| Fatura gecikme tahmini | LightGBM, 5-fold CV, 22 özellik (RFM + tarihsel + anomali) | ✅ RMSE 4.86 gün, %89.7 geç/zamanında doğruluk |
| Nakit akışı projeksiyonu | LightGBM, net_flow hedef, 3 senaryo (iyimser/normal/kötümser) | ✅ CV RMSE 10.627 TL/gün |
| Talep & stok tükenme | LightGBM, EWMA + tedarikçi lead time + maliyet analizi | ✅ RMSE 0.30 adet, %92.3 iyileşme |

**Demo çıktıları:**
- ABC Tekstil INV-4173 (180.000 TL) → Risk **85.0 / KRİTİK**, tahmini gecikme 20 gün
- Ham Kumaş - ND-7234 → **KRİTİK** (stok = 0), 30g kesinti riski: 11.827 TL

**Sonraki:** `shock_simulation.py` ("Aksiyonları Uygula" motoru) + CrewAI multi-agent katmanı.

---

Ayrıntılı veri seti araştırması: [`docs/01_Dataset_Arastirmasi_ve_Veri_Workflow.md`](docs/01_Dataset_Arastirmasi_ve_Veri_Workflow.md)
