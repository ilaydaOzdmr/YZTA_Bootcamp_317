# Şok & Counterfactual Simülasyon Motoru

Nakit projeksiyonunun üzerine **aksiyon ve şok senaryoları** uygular; her birinin
**kriz olasılığını** (Monte Carlo, sistemik korelasyonlu) hesaplar.

## Baseline (aksiyon yok)
Kriz olasılığı **%56** · beklenen kasa +7.717 TL

## Aksiyonlar (kriz olasılığını düşürür)
| Aksiyon | Beklenen kasa | Kriz olasılığı |
|---------|--------------:|---------------:|
| ABC erken odeme (%2 indirim) | +184.117 TL | **%12** |
| Tahsilat kampanyasi (en riskli 3 fatura) | +193.157 TL | **%10** |
| Buyuk faturayi faktore sat (%5 iskonto) | +178.717 TL | **%12** |
| Tedarikci odemesini 2 taksite bol (+10g) | +289.730 TL | **%2** |
| KOMBINE COZUM (erken odeme + tedarikci bolme) | +466.130 TL | **%0** |

## Şoklar / downside (kriz olasılığını artırır)
| Şok | Beklenen kasa | Kriz olasılığı |
|-----|--------------:|---------------:|
| SOK: genel tahsilat yavaslamasi (tum alacaklar +15g) | -894.786 TL | **%100** |
| SOK: kur +%10 (ithal odeme, EVDS'den) | -262.155 TL | **%98** |

## Kalibrasyon (rolling backtest ile KANITLANDI)
`src/analysis/rolling_backtest.py` — 9 ay-sonu cutoff'ta model **out-of-time** eğitilip
30 gün ileri projeksiyon gerçekleşenle kıyaslanır:
- **Coverage %100** (gerçekleşen P5–P95 içinde) · **PIT ort 0.386**
- Kriz olasılığı ≥%50 → gerçek kriz %100 · <%50 → %0

**Dürüstlük notları (jüri için):** kriz ayrımı 1 gerçek krize dayanıyor (n=1, crunch haftası);
sistemik korelasyon rho **0.1–0.7 platosundan** seçildi (0.35, tek noktaya kilitli değil);
kalibrasyon in-sample; aralıklar muhafazakâr-geniş; medyan projeksiyon hafif iyimser (P5/olasılığa bakılmalı).

## Belirsizlik modeli
- **Heteroskedastik σ:** modelin out-of-fold artıklarından (erken ödeyende ~2.5g, geç ödeyende ~7.6g).
- **Sistemik korelasyon (rho=0.35):** gecikmeler birlikte hareket eder ("kötü günler üst üste biner").
- **Pazarlıklı ödemeler** (erken ödeme/faktoring) deterministik (`sigma_scale=0`).

## Aksiyonlar
Erken ödeme indirimi · tahsilat kampanyası (en riskli 3) · faktoring (%5 iskonto) ·
tedarikçi bölme (+10g) · kombine.

## Şoklar
Genel tahsilat yavaşlaması (+15g) · kur şoku (**gerçek EVDS USD/TRY volatilitesinden**).

## Çıktılar
`reports/shock_scenarios.json` · `reports/rolling_backtest.json` · `reports/shock_projection_curves.csv`
