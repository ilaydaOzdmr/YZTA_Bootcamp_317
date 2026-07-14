# Şok & Counterfactual Simülasyon Motoru

Nakit projeksiyonunun (`forecast_cashflow`) üzerine **aksiyon senaryoları** uygular ve
darboğazı çözer — rapordaki "Aksiyonları Uygula" motoru.

```bash
python src/models/train_invoice_delay.py    # önce (gecikme modeli gerekir)
python src/models/forecast_cashflow.py
python src/simulation/shock_simulation.py
```

## Senaryolar (bugün = 31 Mayıs 2025, ufuk 30 gün)

| Senaryo | En düşük kasa | Durum | Δ |
|---------|--------------:|-------|--:|
| Baseline (kriz) | **-46.701 TL** | 🔴 KRİZ | — |
| ABC'ye %2 erken ödeme indirimi | +129.699 TL | 🟢 GÜVENLİ | +176.400 |
| Tedarikçi ödemesini 2 taksite böl (+10g) | +198.221 TL | 🟢 GÜVENLİ | +244.922 |
| **Kur +%10 şok (downside)** | **-313.901 TL** | 🔴 KRİZ↓ | -267.200 |
| **KOMBİNE ÇÖZÜM (1+2)** | **+374.621 TL** | 🟢 GÜVENLİ | +421.322 |

Durum kademeleri: 🔴 KRİZ (kasa < 0) · 🟡 KURTARILDI (pozitif, 100K tampon altı) · 🟢 GÜVENLİ.

## Kritik tasarım notu: ölçüm penceresi

Erteleme süresi **tahmin penceresi içinde kalmalıdır.** Aksi halde ertelenen taksit
pencerenin dışına düşer ve senaryo krizi *çözmüş gibi* görünür — oysa sadece ödemeyi
görüş alanından çıkarmıştır (ölçüm artefaktı).

Bu yüzden `SUPPLIER_DEFER_DAYS = 10`: büyük ödeme 17 Haziran'da, +10 gün → **27 Haziran**,
pencere 30 Haziran'da bitiyor → **her iki taksit de sayılıyor**, iyileşme gerçek.
(Tedarikçilerin sözleşmesel esneklik payı 3-15 gün olduğundan 10 gün gerçekçidir.)

## Mekanik

- **Hedefleme dayanıklı:** sabit tutar eşiği yerine **en büyük** alacak/ödeme hedeflenir.
- **Erken ödeme:** en büyük riskli alacak (ABC, 180K) → tahmini gecikme yerine **vadesinde**
  tahsil, tutar ×0.98.
- **Tedarikçi bölme:** büyük ödeme → 2 eşit taksit, 2. taksit +10 gün; **toplam korunur**.
- **Kur şoku:** ithal ham madde ödemesi ×1.10.

## Çıktılar
`reports/shock_scenarios.json` · `reports/shock_projection_curves.csv`

## Sonraki
Sprint 2'de agent katmanı (CFO / Tahsilat / Tedarik / Risk Denetçisi) bu senaryoları
`twin_api.simulate()` ve `twin_api.recommend()` üzerinden çağıracak.
